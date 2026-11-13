"""Check 4 - calibration drift.

The differentiator. On frames where the arm is holding still, action
(commanded position) and observation.state (measured position) should
agree. A steady offset means leader/follower calibration drifted - the
real-world case behind LeRobot PR #3762 was a 17-degree offset on one
joint that survived recording AND training. No shipped tool checks this.

The comparison only means something when action[j] and state[j] are the
same quantity. Equal dims do not imply that: many datasets pair an
end-effector pose state with joint (or delta) actions. A dim is compared
only when (a) info.json names do not contradict it, and (b) the data
agrees - a position command tracks its measured position (high
correlation, similar range). A constant calibration offset changes
neither, so the guard does not hide real drift.

Grippers are excluded: a gripper commanded shut on an object never
reaches the commanded position, so a steady offset is normal there.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.stats import canonical, gripper_dims, is_generic


def _comparable_dims(ctx, cfg) -> tuple[list[int], dict[int, str]]:
    """Dims where action and state are the same physical quantity, plus a
    reason for every excluded dim."""
    n = ctx.state.shape[1]
    sn, an = ctx.state_names, ctx.action_names
    excluded: dict[int, str] = {}
    grip = gripper_dims(sn, n) | gripper_dims(an, n)
    named = (
        sn is not None and an is not None and len(sn) == n and len(an) == n
        and not is_generic(sn) and not is_generic(an)
    )
    keep = []
    for j in range(n):
        if j in grip:
            excluded[j] = "gripper"
            continue
        if named and canonical(sn[j]) != canonical(an[j]):
            excluded[j] = f"state {sn[j]!r} vs action {an[j]!r}"
            continue
        s, a = ctx.state[:, j], ctx.action[:, j]
        ok = np.isfinite(s) & np.isfinite(a)
        if ok.sum() < 20 or np.ptp(s[ok]) < 1e-8 or np.ptp(a[ok]) < 1e-8:
            excluded[j] = "no motion"
            continue
        corr = float(np.corrcoef(s[ok], a[ok])[0, 1])
        ratio = float(np.ptp(a[ok]) / np.ptp(s[ok]))
        if not (corr >= cfg.calib_min_corr and 0.5 <= ratio <= 2.0):
            excluded[j] = f"action does not track state (corr {corr:.2f}, range ratio {ratio:.2f})"
            continue
        keep.append(j)
    return keep, excluded


class CalibrationDriftCheck(Check):
    id = "calibration-drift"
    title = "Calibration drift (action vs state offset on still frames)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        state, action = ctx.state, ctx.action
        if state is None or action is None:
            return self.skip("observation.state or action column missing")
        if state.shape[1] != action.shape[1]:
            return self.skip(
                f"state dim {state.shape[1]} != action dim {action.shape[1]} "
                "(cannot compare joints)"
            )

        n_joints = state.shape[1]
        joints, excluded = _comparable_dims(ctx, cfg)
        if not joints:
            reasons = sorted(set(r.split(" (")[0] for r in excluded.values()))
            return self.skip(
                "no dim where action is a position command for the same joint "
                f"({'; '.join(reasons)})"
            )
        state_range = np.maximum(1e-8, np.nanmax(state, axis=0) - np.nanmin(state, axis=0))

        # Per joint, measure the mean action-state offset on the frames where
        # THAT joint is holding still (its stillest quartile of motion).
        # Requiring all joints still at once throws away nearly every frame.
        # Offsets are also split by where in its range the joint sits: a
        # calibration error is position-independent, contact/load is not.
        offsets = np.zeros(n_joints)
        counts = np.zeros(n_joints, dtype=np.int64)
        mid = np.nanmedian(state, axis=0)
        half_sum = np.zeros((n_joints, 2))
        half_n = np.zeros((n_joints, 2), dtype=np.int64)
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 10:
                continue
            s = state[rows]
            ds = np.abs(np.diff(s, axis=0))
            vel_thresh = np.maximum(1e-8, np.nanpercentile(ds, 25, axis=0))
            still = ds < vel_thresh  # (n-1, joints), per-joint stillness
            idx = rows[1:]
            diff = action[idx] - state[idx]
            for j in joints:
                m = still[:, j] & np.isfinite(diff[:, j])
                if m.sum() >= 5:
                    offsets[j] += diff[m, j].sum()
                    counts[j] += int(m.sum())
                    hi = state[idx, j] >= mid[j]
                    for h, sel in enumerate((m & ~hi, m & hi)):
                        half_sum[j, h] += diff[sel, j].sum()
                        half_n[j, h] += int(sel.sum())

        usable = counts >= 20
        if not usable.any():
            return self.skip("not enough still frames to measure calibration")

        mean_offset = np.where(usable, offsets / np.maximum(1, counts), 0.0)
        frac = np.abs(mean_offset) / state_range
        still_share = float(counts.max()) / max(1, ctx.row_count)
        half_mean = half_sum / np.maximum(1, half_n)

        worst_j = int(np.argmax(np.where(usable, frac, 0.0)))
        for j in joints:
            if not usable[j]:
                continue
            f = float(frac[j])
            if f > cfg.calib_fail_range_frac:
                sev = Severity.FAIL
            elif f > cfg.calib_warn_range_frac:
                sev = Severity.WARN
            else:
                continue
            consistent = True
            if (half_n[j] >= 10).all():
                lo, hi = half_mean[j]
                consistent = (np.sign(lo) == np.sign(hi)) and (
                    min(abs(lo), abs(hi)) >= 0.5 * max(abs(lo), abs(hi))
                )
            if not consistent:
                # offset only in part of the range: load, contact or a joint
                # limit - not a calibration shift. Report, do not gate.
                sev = Severity.INFO
            res.findings.append(
                Finding(
                    self.id,
                    sev,
                    f"joint {j}: mean action-state offset {mean_offset[j]:+.4g} on still "
                    f"frames = {f:.1%} of its range. "
                    + (
                        "Leader/follower calibration has drifted; the policy is learning "
                        "a shifted target."
                        if consistent
                        else f"The offset depends on position ({half_mean[j, 0]:+.3g} low vs "
                        f"{half_mean[j, 1]:+.3g} high), which points to load or contact, "
                        "not calibration."
                    ),
                    details={
                        "joint": j,
                        "mean_offset": float(mean_offset[j]),
                        "range_fraction": f,
                        "still_frames": int(counts[j]),
                        "position_independent": bool(consistent),
                    },
                )
            )
        res.details = {
            "mean_offset_per_joint": [float(x) for x in mean_offset],
            "range_fraction_per_joint": [float(x) for x in frac],
            "still_frame_share": still_share,
            "compared_joints": joints,
            "excluded_joints": {str(k): v for k, v in excluded.items()},
        }
        note = f"; {len(excluded)} dim(s) not comparable" if excluded else ""
        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = (
                f"max joint offset {float(frac[worst_j]):.1%} of range "
                f"(joint {worst_j}) over {len(joints)} comparable joints{note}"
            )
        else:
            res.summary = (
                f"worst joint {worst_j}: offset {float(frac[worst_j]):.1%} of its range{note}"
            )
        return res

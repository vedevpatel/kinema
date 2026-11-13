"""Check 4 - calibration drift.

The differentiator. On frames where the arm is holding still, action
(commanded position) and observation.state (measured position) should
agree. A steady offset means leader/follower calibration drifted - the
real-world case behind LeRobot PR #3762 was a 17-degree offset on one
joint that survived recording AND training. No shipped tool checks this.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity


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
        state_range = np.maximum(1e-8, state.max(axis=0) - state.min(axis=0))

        # Per joint, measure the mean action-state offset on the frames where
        # THAT joint is holding still (its stillest quartile of motion).
        # Requiring all joints still at once throws away nearly every frame.
        offsets = np.zeros(n_joints)
        counts = np.zeros(n_joints, dtype=np.int64)
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 10:
                continue
            s = state[rows]
            ds = np.abs(np.diff(s, axis=0))
            vel_thresh = np.maximum(1e-8, np.percentile(ds, 25, axis=0))
            still = ds < vel_thresh  # (n-1, joints), per-joint stillness
            idx = rows[1:]
            diff = action[idx] - state[idx]
            for j in range(n_joints):
                m = still[:, j]
                if m.sum() >= 5:
                    offsets[j] += diff[m, j].sum()
                    counts[j] += int(m.sum())

        usable = counts >= 20
        if not usable.any():
            return self.skip("not enough still frames to measure calibration")

        mean_offset = np.where(usable, offsets / np.maximum(1, counts), 0.0)
        frac = np.abs(mean_offset) / state_range
        still_share = float(counts.max()) / max(1, ctx.row_count)

        worst_j = int(np.argmax(np.where(usable, frac, 0.0)))
        for j in range(n_joints):
            if not usable[j]:
                continue
            f = float(frac[j])
            if f > cfg.calib_fail_range_frac:
                sev = Severity.FAIL
            elif f > cfg.calib_warn_range_frac:
                sev = Severity.WARN
            else:
                continue
            res.findings.append(
                Finding(
                    self.id,
                    sev,
                    f"joint {j}: mean action-state offset {mean_offset[j]:+.4g} on still "
                    f"frames = {f:.1%} of its range. Leader/follower calibration has "
                    "drifted; the policy is learning a shifted target.",
                    details={
                        "joint": j,
                        "mean_offset": float(mean_offset[j]),
                        "range_fraction": f,
                        "still_frames": int(counts[j]),
                    },
                )
            )
        res.details = {  # type: ignore[attr-defined]
            "mean_offset_per_joint": [float(x) for x in mean_offset],
            "range_fraction_per_joint": [float(x) for x in frac],
            "still_frame_share": still_share,
        }
        if not res.findings:
            res.summary = (
                f"max joint offset {float(frac[worst_j]):.1%} of range "
                f"(joint {worst_j}); measured on {still_share:.0%} of frames"
            )
        else:
            res.summary = (
                f"worst joint {worst_j}: offset {float(frac[worst_j]):.1%} of its range"
            )
        return res

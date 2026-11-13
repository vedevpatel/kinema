"""Check 4 - calibration drift.

On frames where a joint holds still, its command (action) and measured
position (observation.state) should agree. The real-world case behind
LeRobot PR #3762 was a 17-degree offset on one joint that survived
recording AND training.

Per comparable joint, the still-frame offset (action - state) is fit as a
line over the joint's position (both in units of its range):

- intercept: a constant offset - leader/follower calibration drift.
- slope: the offset grows across the range - a scale (gear ratio / unit)
  calibration error, where command and measurement disagree on how far a
  unit of motion is.
- a large offset the line does not explain (low R^2) is load, contact or
  a joint limit: INFO, not a calibration verdict.

Which dims are comparable: action[j] must be a position command for the
same joint as state[j]. Equal dims do not imply that - many datasets pair
an end-effector pose state with joint or delta actions. A dim is compared
when its names do not contradict the pairing and the data agrees
(correlation, similar range); a calibration error changes neither. Every
other dim is classified (delta / velocity command, different space, ...)
so a skip says why.

Grippers get their own test: a gripper commanded shut on an object never
gets there, so only the FREE end (where nothing blocks it - the end with
the smaller still-frame offset) says anything about calibration.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.stats import action_gripper_dims, canonical, gripper_dims, is_generic, is_velocity


def _space_of(ctx, i: int, j: int) -> str:
    """Best guess at what action[j] is when it does not track state[i]."""
    an = ctx.action_names
    name = an[j] if an and len(an) == ctx.action.shape[1] else ""
    if is_velocity(name):
        return "velocity command"
    s, a = ctx.state[:, i], ctx.action[:, j]
    # delta commands correlate with the step in state - the next step
    # (target = state + action) or the one that produced this frame
    for lag in (0, 1):
        xs, ys = [], []
        for ep in ctx.episode_ids()[:200]:
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            d = np.diff(s[rows])
            xs.append(a[rows][1:] if lag else a[rows][:-1])
            ys.append(d)
        if not xs:
            break
        x, y = np.concatenate(xs), np.concatenate(ys)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() > 20 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            if abs(float(np.corrcoef(x[ok], y[ok])[0, 1])) >= 0.7:
                return "delta command"
    return "different space"


def _pairs(ctx) -> list[tuple[int, int]]:
    """(state index, action index) candidates: same index when the vectors
    match in size, otherwise by name (an 8-dim pose+quaternion state still
    shares x/y/z/gripper with a 7-dim action)."""
    ns, na = ctx.state.shape[1], ctx.action.shape[1]
    if ns == na:
        return [(j, j) for j in range(ns)]
    sn, an = ctx.state_names, ctx.action_names
    if not (sn and an and len(sn) == ns and len(an) == na):
        return []
    by_name: dict[str, int] = {}
    for i, nm in enumerate(sn):
        if not is_generic([nm]):
            by_name.setdefault(canonical(nm), i)
    out = []
    for j, nm in enumerate(an):
        i = by_name.get(canonical(nm)) if not is_generic([nm]) else None
        if i is not None:
            out.append((i, j))
    return out


def _comparable_dims(ctx, cfg):
    """(joint pairs, gripper pairs, excluded-with-reason keyed by action index)."""
    sn, an = ctx.state_names, ctx.action_names
    ns, na = ctx.state.shape[1], ctx.action.shape[1]
    have = sn is not None and an is not None and len(sn) == ns and len(an) == na
    named_grip_s, named_grip_a = gripper_dims(sn, ns), gripper_dims(an, na)
    excluded: dict[int, str] = {}
    joints, grippers = [], []
    for i, j in _pairs(ctx):
        if have and not is_generic([sn[i]]) and not is_generic([an[j]]) \
                and canonical(sn[i]) != canonical(an[j]):
            excluded[j] = f"different quantities (state {sn[i]!r}, action {an[j]!r})"
            continue
        s, a = ctx.state[:, i], ctx.action[:, j]
        ok = np.isfinite(s) & np.isfinite(a)
        if ok.sum() < 20 or np.ptp(s[ok]) < 1e-8 or np.ptp(a[ok]) < 1e-8:
            excluded[j] = "no motion"
            continue
        corr = float(np.corrcoef(s[ok], a[ok])[0, 1])
        ratio = float(np.ptp(a[ok]) / np.ptp(s[ok]))
        tracks = abs(corr) >= cfg.calib_min_corr and 0.5 <= ratio <= 2.0
        if i in named_grip_s or j in named_grip_a:
            # grippers track loosely (blocked on objects); require same units
            if 0.5 <= ratio <= 2.0 and corr >= 0.5:
                grippers.append((i, j))
            else:
                excluded[j] = "gripper in different units"
            continue
        if not tracks or corr < 0:
            excluded[j] = _space_of(ctx, i, j)
            continue
        joints.append((i, j))
    return joints, grippers, excluded


def _still_rows(ctx, i: int) -> np.ndarray:
    """Row indices where state dim i holds still (its stillest ~40% of steps)."""
    out = []
    for ep in ctx.episode_ids():
        rows = ctx.rows_for(ep)
        if len(rows) < 10:
            continue
        ds = np.abs(np.diff(ctx.state[rows, i]))
        thr = np.nanpercentile(ds, 40)
        out.append(rows[1:][ds <= thr])
    return np.concatenate(out) if out else np.empty(0, dtype=int)


class CalibrationDriftCheck(Check):
    id = "calibration-drift"
    title = "Calibration drift (action vs state on still frames)"

    def _joint(self, ctx, cfg, pair, inferred_grip):
        i, j = pair
        s, a = ctx.state[:, i], ctx.action[:, j]
        rng = max(float(np.nanmax(s) - np.nanmin(s)), 1e-8)
        rows = _still_rows(ctx, i)
        rows = rows[np.isfinite(s[rows]) & np.isfinite(a[rows])]
        if len(rows) < 20:
            return None, None
        u = (s[rows] - np.nanmedian(s)) / rng  # position, in units of the range
        off = (a[rows] - s[rows]) / rng  # offset, in units of the range
        # least-squares line off = b0 + b1 * u
        A = np.stack([np.ones_like(u), u], axis=1)
        (b0, b1), *_ = np.linalg.lstsq(A, off, rcond=None)
        resid = off - (b0 + b1 * u)
        var = float(np.var(off))
        r2 = 1.0 - float(np.var(resid)) / var if var > 1e-12 else 1.0
        mean_off = float(np.mean(off))
        info = {"joint": j, "offset_fraction": mean_off, "scale_error": float(b1),
                "r2": r2, "still_frames": int(len(rows))}
        name = f" ({ctx.action_names[j]})" if ctx.action_names and len(ctx.action_names) == ctx.action.shape[1] else ""
        findings = []
        # constant offset: large mean offset that does not vary across the range
        if abs(mean_off) > cfg.calib_warn_range_frac and abs(b1) < max(cfg.calib_scale_warn, abs(mean_off)):
            sev = Severity.FAIL if abs(mean_off) > cfg.calib_fail_range_frac else Severity.WARN
            findings.append((sev, f"joint {j}{name}: commanded and measured position disagree by "
                             f"{mean_off:+.1%} of its range on still frames, the same across its "
                             "range - leader/follower calibration drift; the policy learns a "
                             "shifted target", {**info, "kind": "offset"}))
        # scale error: offset grows linearly with position, and the line explains it
        elif abs(b1) > cfg.calib_scale_warn and r2 >= cfg.calib_scale_min_r2:
            sev = Severity.FAIL if abs(b1) > cfg.calib_scale_fail else Severity.WARN
            findings.append((sev, f"joint {j}{name}: command and measurement drift apart across "
                             f"the range ({b1:+.1%} per full range, R^2 {r2:.2f}) - a scale "
                             "calibration error (gear ratio or units), not a fixed offset",
                             {**info, "kind": "scale"}))
        elif abs(mean_off) > cfg.calib_warn_range_frac or abs(b1) > cfg.calib_scale_warn:
            findings.append((Severity.INFO, f"joint {j}{name}: still-frame offset {mean_off:+.1%} "
                             f"of range varies irregularly with position (R^2 {r2:.2f}) - load, "
                             "contact or a joint limit rather than calibration",
                             {**info, "kind": "irregular"}))
        if j in inferred_grip:  # unnamed dim in a gripper slot: do not gate on it
            findings = [(Severity.INFO, m + " [dim looks like an unnamed gripper: not gated]", d)
                        for sev, m, d in findings]
        return findings, info

    def _gripper(self, ctx, cfg, pair):
        i, j = pair
        s, a = ctx.state[:, i], ctx.action[:, j]
        rng = max(float(np.nanmax(s) - np.nanmin(s)), 1e-8)
        lo, hi = np.nanpercentile(a, [1, 99])
        band = 0.15 * (hi - lo)
        rows = _still_rows(ctx, i)
        rows = rows[np.isfinite(s[rows]) & np.isfinite(a[rows])]
        sides = []
        for label, sel in (("low", a[rows] <= lo + band), ("high", a[rows] >= hi - band)):
            r = rows[sel]
            if len(r) >= 20:
                sides.append((label, float(np.median((a[r] - s[r]) / rng)), len(r)))
        if not sides:
            return None, None
        label, off, nfr = min(sides, key=lambda x: abs(x[1]))  # the free end
        info = {"joint": j, "free_end": label, "offset_fraction": off, "still_frames": nfr,
                "kind": "gripper"}
        if abs(off) > cfg.calib_gripper_warn:
            name = f" ({ctx.action_names[j]})" if ctx.action_names else ""
            return [(Severity.WARN, f"gripper {j}{name}: even at its free ({label}) end the "
                     f"measured position sits {off:+.1%} of its range off the command - gripper "
                     "calibration drift", info)], info
        return [], info

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        state, action = ctx.state, ctx.action
        if state is None or action is None:
            return self.skip("observation.state or action column missing")
        if state.shape[1] != action.shape[1] and not _pairs(ctx):
            return self.skip(
                f"state dim {state.shape[1]} != action dim {action.shape[1]} and no "
                "names to pair them by"
            )
        joints, grippers, excluded = _comparable_dims(ctx, cfg)
        _, inferred_grip = action_gripper_dims(ctx)
        if not joints and not grippers:
            kinds = sorted({r.split(" (")[0] for r in excluded.values()})
            return self.skip("no dim where action is a position command for the same joint: "
                             + "; ".join(f"{sum(1 for r in excluded.values() if r.startswith(k))} "
                                         f"{k}" for k in kinds))
        per_joint = {}
        for pair in joints:
            fs, info = self._joint(ctx, cfg, pair, inferred_grip)
            if info:
                per_joint[str(pair[1])] = info
            for sev, msg, det in fs or []:
                res.findings.append(Finding(self.id, sev, msg, details=det))
        for pair in grippers:
            fs, info = self._gripper(ctx, cfg, pair)
            if info:
                per_joint[str(pair[1])] = info
            for sev, msg, det in fs or []:
                res.findings.append(Finding(self.id, sev, msg, details=det))
        if not per_joint:
            return self.skip("not enough still frames to measure calibration")

        res.details = {
            "per_joint": per_joint,
            "compared_joints": [j for _, j in joints],
            "compared_grippers": [j for _, j in grippers],
            "excluded_joints": {str(k): v for k, v in excluded.items()},
        }
        worst = max(per_joint.values(), key=lambda d: abs(d.get("offset_fraction", 0)))
        kinds = sorted({r.split(" (")[0] for r in excluded.values()})
        note = ("; not compared: " + ", ".join(
            f"{sum(1 for r in excluded.values() if r.startswith(k))} {k}" for k in kinds)) if kinds else ""
        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = (f"{len(joints)} joint(s) + {len(grippers)} gripper(s) agree; worst offset "
                           f"{worst['offset_fraction']:+.1%} of range (joint {worst['joint']}){note}")
        else:
            n_bad = sum(1 for f in res.findings if f.severity >= Severity.WARN)
            res.summary = f"{n_bad} joint(s) miscalibrated{note}"
        return res

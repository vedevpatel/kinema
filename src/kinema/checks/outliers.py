"""Check 10 - outlier episodes.

Per-episode metrics (duration, jerk, peak velocity, static fraction,
end-pose deviation) flagged by IQR, the method from LeRobot PR #3762.
The worst episodes get a ranked list so they can be reviewed or
deleted before training.

Only the harmful direction is flagged (high jerk is a glitch, low jerk is
smooth; only duration is suspicious both ways). Velocity and jerk are
measured per dim in units of that dim's std, without grippers or velocity
commands, so a dim with large units, an open/shut toggle or a base that
starts and stops does not dominate. Duration and
end pose are compared within a task: a multi-task dataset legitimately
ends in different places. An episode is a WARN when independent kinds of
evidence agree, or when one metric is far out (Tukey k=3).
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.stats import action_gripper_dims, velocity_dims

# metric -> (which side is harmful, evidence family)
_METRICS = {
    "duration_s": ("both", "duration"),
    "p95_jerk": ("high", "motion"),
    "peak_velocity": ("high", "motion"),
    "static_fraction": ("high", "activity"),
    "end_pose_deviation": ("high", "outcome"),
}


# absolute materiality floor: a static fraction of 0.05 vs a median of 0 is
# not an outlier however tight the band (idling itself is idle-segments' job)
_ABS_FLOOR = {"static_fraction": 0.15}


_MIN_MOTION_FRAMES = 22  # >= 20 jerk samples, so p95 is not just the max


def _is_toggle(ctx, j: int, std: float) -> bool:
    """A two-level command (an unnamed gripper, an on/off switch): its large
    steps are all the same size. A glitch varies in size and snaps back; a
    toggle jumps by exactly the gap between its levels, every time."""
    big = []
    for ep in ctx.episode_ids():
        rows = ctx.rows_for(ep)
        if len(rows) < 2:
            continue
        z = np.abs(np.diff(ctx.action[rows, j])) / std
        big.extend(z[z > 1.0].tolist())
    if len(big) < 3:
        return False
    arr = np.asarray(big)
    return float(arr.std() / arr.mean()) < 0.05


def _iqr_flags(values: np.ndarray, k: float, side: str = "both",
               rel: float = 0.5, floor: float = 0.0) -> np.ndarray:
    """Flag values outside the IQR fence (on the harmful side) that also
    differ materially from the median: by more than `rel` x median (and
    `floor` in absolute terms). NaN values are never flagged."""
    ok = np.isfinite(values)
    out = np.zeros(len(values), dtype=bool)
    if ok.sum() < 8:
        return out
    v = values[ok]
    med = np.median(v)
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    material = np.abs(v - med) > np.maximum(rel * np.maximum(np.abs(med), 1e-9), floor)
    if iqr <= 0:
        # degenerate band (all values ~equal): flag only real deviations
        hi_flag, lo_flag = material & (v > med), material & (v < med)
    else:
        hi_flag = (v > q3 + k * iqr) & material
        lo_flag = (v < q1 - k * iqr) & material
    flags = hi_flag | lo_flag if side == "both" else hi_flag
    out[ok] = flags
    return out


def _far_out(values: np.ndarray, side: str, floor: float) -> np.ndarray:
    """Beyond Tukey's k=3 fence AND at least 3x (or, two-sided, under 1/3 of)
    the median: one metric alone only earns a WARN when it is extreme."""
    fenced = _iqr_flags(values, 3.0, side, rel=0.0, floor=floor)
    ok = np.isfinite(values)
    if not fenced.any() or ok.sum() == 0:
        return fenced
    med = np.median(values[ok])
    v = np.where(ok, values, med)
    extreme = v > 3 * med if med > 0 else v > med + floor
    if side == "both":
        extreme |= v < med / 3
    return fenced & extreme


def _task_key(ctx, ep: int, rows: np.ndarray):
    meta = ctx.meta_for(ep)
    if meta and meta.tasks:
        return tuple(meta.tasks)
    if ctx.task_index is not None and len(rows):
        vals, counts = np.unique(ctx.task_index[rows], return_counts=True)
        return int(vals[np.argmax(counts)])
    return None


class OutlierEpisodesCheck(Check):
    id = "outlier-episodes"
    title = "Outlier episodes (duration, jerk, velocity, static, end-pose)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        fps = ctx.fps or 30.0

        motion_dims = None
        if action is not None:
            n = action.shape[1]
            std = np.nanstd(action, axis=0)
            named_grip, inferred_grip = action_gripper_dims(ctx)
            # grippers toggle and velocity commands step on/off by design:
            # neither belongs in a jerk / peak-velocity outlier score
            grip = named_grip | inferred_grip | velocity_dims(ctx.action_names, n)
            motion_dims = [j for j in range(n) if std[j] > 1e-8 and j not in grip
                           and len(np.unique(action[:, j])) > 10 and not _is_toggle(ctx, j, std[j])]
            scale = np.where(std > 1e-8, std, 1.0)
            eps_idle = np.maximum(1e-8, 0.01 * std)

        eps: list[int] = []
        tasks: list = []
        vals: dict[str, list] = {m: [] for m in _METRICS}
        end_poses = []
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 5:
                continue
            eps.append(ep)
            tasks.append(_task_key(ctx, ep, rows))
            vals["duration_s"].append(len(rows) / fps)
            if action is not None:
                da = np.diff(action[rows], axis=0)
                vals["static_fraction"].append(float((np.abs(da) < eps_idle).all(axis=1).mean()))
                # on a handful of frames the 95th-percentile jerk IS the maximum,
                # and a peak over 8 frames is not comparable to one over 300:
                # short episodes sit out the motion metrics (duration still counts)
                if motion_dims and len(rows) >= _MIN_MOTION_FRAMES:
                    z = da[:, motion_dims] / scale[motion_dims]
                    vel = np.nanmax(np.abs(z), axis=1)
                    jerk = np.nanmax(np.abs(np.diff(z, axis=0)), axis=1) if len(z) > 1 else np.zeros(1)
                    vals["p95_jerk"].append(float(np.percentile(jerk, 95)))
                    vals["peak_velocity"].append(float(vel.max()))
                else:
                    vals["p95_jerk"].append(np.nan)
                    vals["peak_velocity"].append(np.nan)
            else:
                for m in ("p95_jerk", "peak_velocity", "static_fraction"):
                    vals[m].append(np.nan)
            end_poses.append(ctx.state[rows][-1] if ctx.state is not None else None)

        if len(eps) < 8:
            return self.skip("fewer than 8 episodes - IQR flagging needs a population")

        # duration and end pose only mean something within a task
        groups: dict = {}
        for i, t in enumerate(tasks):
            groups.setdefault(t, []).append(i)
        comparable = np.zeros(len(eps), dtype=bool)
        for idx in groups.values():
            if len(idx) >= 5:
                comparable[idx] = True
        dur = np.asarray(vals["duration_s"], dtype=np.float64)
        dur_rel = np.full(len(eps), np.nan)
        end_dev = np.full(len(eps), np.nan)
        have_state = all(p is not None for p in end_poses)
        if have_state:
            end_arr = np.asarray(end_poses, dtype=np.float64)
            rng = np.maximum(1e-8, np.nanmax(end_arr, axis=0) - np.nanmin(end_arr, axis=0))
        for idx in groups.values():
            if len(idx) < 5:
                continue
            idx = np.asarray(idx)
            dur_rel[idx] = dur[idx] / max(np.median(dur[idx]), 1e-9)
            if have_state:
                med = np.nanmedian(end_arr[idx], axis=0)
                end_dev[idx] = np.nanmax(np.abs(end_arr[idx] - med) / rng, axis=1)

        metrics = {
            "duration_s": dur_rel,  # relative to the task's median duration
            "p95_jerk": np.asarray(vals["p95_jerk"], dtype=np.float64),
            "peak_velocity": np.asarray(vals["peak_velocity"], dtype=np.float64),
            "static_fraction": np.asarray(vals["static_fraction"], dtype=np.float64),
            "end_pose_deviation": end_dev,
        }
        flags: dict[int, list[str]] = {ep: [] for ep in eps}
        far: dict[int, bool] = {ep: False for ep in eps}
        for name, v in metrics.items():
            side, _ = _METRICS[name]
            floor = _ABS_FLOOR.get(name, 0.0)
            f = _iqr_flags(v, cfg.outlier_iqr_k, side, floor=floor)
            f3 = _far_out(v, side, floor)
            for ep, a, b in zip(eps, f, f3):
                if a:
                    flags[ep].append(name)
                if b:
                    far[ep] = True

        worst = sorted(
            ((ep, m) for ep, m in flags.items() if m),
            key=lambda x: (-len({_METRICS[n][1] for n in x[1]}), -far[x[0]], -len(x[1])),
        )
        n_warn = 0
        for ep, names in worst:
            families = {_METRICS[n][1] for n in names}
            warn = len(families) >= 2 or far[ep]
            n_warn += warn
            res.findings.append(
                Finding(
                    self.id,
                    Severity.WARN if warn else Severity.INFO,
                    f"episode {ep}: IQR outlier on {', '.join(names)}"
                    + (" (far out)" if far[ep] else ""),
                    episode_index=ep,
                    details={"metrics": names, "far_out": far[ep]},
                )
            )

        note = "" if comparable.all() else (
            f"; duration/end pose skipped for {int((~comparable).sum())} episodes whose task "
            "has < 5 episodes")
        if worst:
            top = [ep for ep, _ in worst[:10]]
            res.summary = (
                f"{n_warn} episode(s) to review, {len(worst) - n_warn} mild, of {len(eps)}; "
                f"worst: {top}{note}"
            )
        else:
            res.summary = f"no IQR outliers across {len(eps)} episodes{note}"
        res.details = {
            "metrics": {name: {str(ep): (None if not np.isfinite(x) else float(x)) for ep, x in zip(eps, v)}
                        for name, v in metrics.items()},
            "flagged": {str(ep): names for ep, names in flags.items() if names},
        }
        return res

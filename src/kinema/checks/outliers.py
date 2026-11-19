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
evidence agree (static stillness excluded - that is idle-segments' job),
or when one metric is far out (Tukey k=3, 6x the median).

Two metrics cannot earn a single-metric WARN on their own:
- static_fraction: a partly-still episode is idle-segments' job, and on a
  dataset whose median stillness is ~5%, "4x the median" is a pause, not a
  glitch (that one path produced most of this check's noise on real data).
  It no longer counts toward multi-metric agreement either: on real teleop,
  spike+static pairs were a paused operator, not a glitch.
- end_pose_deviation: only when the deviation is also >= half the task's
  end-pose range - on tight-range tasks 3x the median is centimetres.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.checks.stats_drift import noise_floor
from kinema.stats import action_gripper_dims, is_toggle, structural_velocity_dims, velocity_dims

# metric -> (which side is harmful, evidence family)
_METRICS = {
    "duration_s": ("both", "duration"),
    "p95_jerk": ("high", "motion"),
    "peak_velocity": ("high", "motion"),
    "velocity_spike": ("high", "motion"),
    "static_fraction": ("high", "activity"),
    "end_pose_deviation": ("high", "outcome"),
}


# absolute materiality floor: a static fraction of 0.05 vs a median of 0 is
# not an outlier however tight the band (idling itself is idle-segments' job)
_ABS_FLOOR = {"static_fraction": 0.15}


_MIN_MOTION_FRAMES = 22  # >= 20 jerk samples, so p95 is not just the max


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


def _far_out(values: np.ndarray, side: str, floor: float, abs_min: float = 0.0) -> np.ndarray:
    """Beyond Tukey's k=3 fence AND at least _FAR_OUT_REL times (or, two-sided,
    under 1/_FAR_OUT_REL of) the median AND past abs_min in absolute terms:
    one metric alone only earns a WARN when it is extreme."""
    fenced = _iqr_flags(values, 3.0, side, rel=0.0, floor=floor)
    ok = np.isfinite(values)
    if not fenced.any() or ok.sum() == 0:
        return fenced
    med = np.median(values[ok])
    v = np.where(ok, values, med)
    extreme = v > _FAR_OUT_REL * med if med > 0 else v > med + floor
    if side == "both":
        extreme |= v < med / _FAR_OUT_REL
    return fenced & extreme & (np.abs(v) >= abs_min)


# metrics that can earn a WARN on their own when far out, and the absolute
# bar they must also clear. static_fraction is absent on purpose (docstring).
# one metric alone only earns a WARN when it is extreme: past Tukey k=3 AND
# at least _FAR_OUT_REL times the median. 6x, not 3x: measured on the calibration
# halves 2026-11-15 - known-good datasets' far-out episodes cluster at 3.3-4.1x
# the median (wide-outcome tasks), while an injected one-frame 30-std spike lands
# at 10-198x. 3x WARNed on 8 of 14 untouched quick datasets; 6x keeps full teeth
# on the spike floor and clears the known-good noise.
_FAR_OUT_REL = 6.0

_FAR_OUT_ABS_MIN = {"end_pose_deviation": 0.5}
# a partly-still episode is idle-segments' job: static_fraction earns no WARN
# here, neither alone nor by counting toward multi-metric agreement
_FAR_OUT_ELIGIBLE = frozenset(_METRICS) - {"static_fraction"}


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
            # neither belongs in a jerk / peak-velocity outlier score. A
            # one-frame SPIKE in a velocity command is still a glitch, so
            # those dims get their own metric: the episode's largest step in
            # units of the dim's own p99 step (stepping dims sit near 1).
            grip = named_grip | inferred_grip
            vel = (velocity_dims(ctx.action_names, n)
                   | structural_velocity_dims(ctx)) - grip

            # float32 noise floor relative to each dim's magnitude, not an
            # absolute 1e-8: a channel jittering at 4e-6 around 0.9 is not
            # motion in metres or millimetres alike (tests/test_invariance.py)
            with np.errstate(invalid="ignore"):
                nf = noise_floor({"min": np.nanmin(action, axis=0), "max": np.nanmax(action, axis=0)})

            def _cont(j: int) -> bool:
                return (std[j] > nf[j] and len(np.unique(action[:, j])) > 10
                        and not is_toggle(ctx, j, std[j]))

            vel_dims = [j for j in sorted(vel) if _cont(j)]
            motion_dims = [j for j in range(n) if _cont(j) and j not in grip and j not in vel]
            da_all = np.abs(np.diff(action, axis=0))
            step_scale = (np.nanpercentile(da_all[:, vel_dims], 99, axis=0)
                          if vel_dims else None)
            scale = np.where(std > nf, std, 1.0)
            eps_idle = np.maximum(np.maximum(nf, 0.01 * std), 1e-300)

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
                    vel_f = np.nanmax(np.abs(z), axis=1)
                    jerk = np.nanmax(np.abs(np.diff(z, axis=0)), axis=1) if len(z) > 1 else np.zeros(1)
                    vals["p95_jerk"].append(float(np.percentile(jerk, 95)))
                    vals["peak_velocity"].append(float(vel_f.max()))
                else:
                    vals["p95_jerk"].append(np.nan)
                    vals["peak_velocity"].append(np.nan)
                if vel_dims and step_scale is not None and len(rows) >= 2:
                    spike = np.abs(da[:, vel_dims]) / np.maximum(step_scale, 1e-8)
                    vals["velocity_spike"].append(float(np.nanmax(spike)))
                else:
                    vals["velocity_spike"].append(np.nan)
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
            lo_e, hi_e = np.nanmin(end_arr, axis=0), np.nanmax(end_arr, axis=0)
            rng = np.maximum(np.maximum(hi_e - lo_e, noise_floor({"min": lo_e, "max": hi_e})), 1e-300)
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
            "velocity_spike": np.asarray(vals["velocity_spike"], dtype=np.float64),
            "static_fraction": np.asarray(vals["static_fraction"], dtype=np.float64),
            "end_pose_deviation": end_dev,
        }
        flags: dict[int, list[str]] = {ep: [] for ep in eps}
        far: dict[int, bool] = {ep: False for ep in eps}
        for name, v in metrics.items():
            side, _ = _METRICS[name]
            floor = _ABS_FLOOR.get(name, 0.0)
            f = _iqr_flags(v, cfg.outlier_iqr_k, side, floor=floor)
            f3 = (_far_out(v, side, floor, _FAR_OUT_ABS_MIN.get(name, 0.0))
                  if name in _FAR_OUT_ELIGIBLE else np.zeros(len(v), dtype=bool))
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
            # static_fraction counts toward nothing on its own: a partly-still episode is
            # idle-segments' job (a pause on a low-stillness dataset reads as "4x
            # the median"), so it no longer earns agreement either - measured on
            # real teleop (berkeley_mvp), where spike+static pairs were noise
            families = {_METRICS[n][1] for n in names if n != "static_fraction"}
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

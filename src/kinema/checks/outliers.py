"""Check 10 - outlier episodes.

Per-episode metrics (duration, jerk, peak velocity, static fraction,
end-pose deviation) flagged by IQR, the method from LeRobot PR #3762.
The worst episodes get a ranked list so they can be reviewed or
deleted before training.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity


def _iqr_flags(values: np.ndarray, k: float) -> np.ndarray:
    med = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        # degenerate band (all values ~equal): flag only real deviations
        return np.abs(values - med) > np.maximum(0.5 * np.abs(med), 1e-9)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    outside = (values < lo) | (values > hi)
    # materiality floor: natural jitter inside a narrow band is not an outlier
    material = np.abs(values - med) > 0.5 * np.maximum(np.abs(med), 1e-9)
    return outside & material


class OutlierEpisodesCheck(Check):
    id = "outlier-episodes"
    title = "Outlier episodes (duration, jerk, velocity, static, end-pose)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        fps = ctx.fps or 30.0

        eps: list[int] = []
        duration, p95_jerk, peak_vel, static_frac = [], [], [], []
        end_poses = []
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 5:
                continue
            eps.append(ep)
            duration.append(len(rows) / fps)
            if action is not None:
                a = action[rows]
                da = np.diff(a, axis=0)
                vel = np.abs(da).max(axis=1)
                jerk = np.abs(np.diff(da, axis=0)).max(axis=1) if len(da) > 1 else np.zeros(1)
                p95_jerk.append(float(np.percentile(jerk, 95)))
                peak_vel.append(float(vel.max()))
                eps_idle = np.maximum(1e-8, 0.01 * action.std(axis=0))
                static_frac.append(float((da < eps_idle).all(axis=1).mean()))
            else:
                p95_jerk.append(0.0)
                peak_vel.append(0.0)
                static_frac.append(0.0)
            if ctx.state is not None:
                end_poses.append(ctx.state[rows][-1])

        if len(eps) < 8:
            return self.skip("fewer than 8 episodes - IQR flagging needs a population")

        metrics = {
            "duration_s": np.asarray(duration),
            "p95_jerk": np.asarray(p95_jerk),
            "peak_velocity": np.asarray(peak_vel),
            "static_fraction": np.asarray(static_frac),
        }
        flags: dict[int, list[str]] = {ep: [] for ep in eps}
        for name, vals in metrics.items():
            flagged = _iqr_flags(vals, cfg.outlier_iqr_k)
            for ep, f in zip(eps, flagged):
                if f:
                    flags[ep].append(name)

        if end_poses:
            end_arr = np.asarray(end_poses)
            med = np.median(end_arr, axis=0)
            rng = np.maximum(1e-8, end_arr.max(axis=0) - end_arr.min(axis=0))
            dev = (np.abs(end_arr - med) / rng).max(axis=1)
            flagged = _iqr_flags(dev, cfg.outlier_iqr_k)
            for ep, f, d in zip(eps, flagged, dev):
                if f:
                    flags[ep].append("end_pose_deviation")
            metrics["end_pose_deviation"] = dev

        worst = sorted(((ep, m) for ep, m in flags.items() if m), key=lambda x: -len(x[1]))
        for ep, names in worst:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.WARN if len(names) >= 2 else Severity.INFO,
                    f"episode {ep}: IQR outlier on {', '.join(names)}",
                    episode_index=ep,
                    details={"metrics": names},
                )
            )

        if worst:
            top = [ep for ep, _ in worst[:10]]
            res.summary = (
                f"{len(worst)}/{len(eps)} episodes flagged; worst: {top}. "
                f"Review or drop with: dataset.delete_episodes({top})"
            )
        else:
            res.summary = f"no IQR outliers across {len(eps)} episodes"
        res.details = {
            "metrics": {name: {str(ep): float(v) for ep, v in zip(eps, vals)} for name, vals in metrics.items()},
            "flagged": {str(ep): names for ep, names in flags.items() if names},
        }
        return res

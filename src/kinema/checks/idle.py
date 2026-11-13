"""Checks 5 & 6 - idle segments and stuck actuators.

Idle stretches teach the policy to do nothing (the classic "robot stays
still at inference" postmortem). A joint that never moves is a broken
actuator, a disabled motor, or a recording bug - and the policy learns
to ignore that joint.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity

_PICK_WORDS = ("pick", "place", "grasp", "grab", "lift")


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end) runs of True in a boolean array."""
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(mask) - 1]
    keep = mask[starts]
    return [(int(s), int(e) + 1) for s, e, k in zip(starts, ends, keep) if k]


class IdleSegmentsCheck(Check):
    id = "idle-segments"
    title = "Idle segments (policy learns to do nothing)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        fps = ctx.fps or 30.0
        eps = np.maximum(1e-8, cfg.idle_eps_frac * action.std(axis=0))
        min_run = max(2, int(round(cfg.idle_min_seconds * fps)))

        n_flagged = 0
        worst_frac = 0.0
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            a = action[rows]
            da = np.abs(np.diff(a, axis=0))
            idle = np.r_[False, (da < eps).all(axis=1)]
            runs = [(s, e) for s, e in _runs(idle) if e - s >= min_run]
            idle_frames = sum(e - s for s, e in runs)
            frac = idle_frames / len(rows)
            worst_frac = max(worst_frac, frac)
            ends_idle = bool(runs and runs[-1][1] == len(rows))
            if frac > cfg.idle_warn_fraction or ends_idle:
                n_flagged += 1
                parts = []
                if frac > cfg.idle_warn_fraction:
                    parts.append(
                        f"{frac:.0%} of frames idle in runs >= {cfg.idle_min_seconds:g}s"
                    )
                if ends_idle:
                    parts.append("episode ends idle (teaches stopping early)")
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"episode {ep}: " + "; ".join(parts),
                        episode_index=ep,
                        details={"idle_fraction": frac, "idle_runs": len(runs), "ends_idle": ends_idle},
                    )
                )
        if not res.findings:
            res.summary = f"no idle runs >= {cfg.idle_min_seconds:g}s at >= {cfg.idle_warn_fraction:.0%} of an episode"
        else:
            res.summary = f"{n_flagged}/{ctx.total_episodes} episodes flagged; worst {worst_frac:.0%} idle"
        return res


class StuckActuatorCheck(Check):
    id = "stuck-actuator"
    title = "Stuck actuators (joints that never move)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        eps = np.maximum(1e-8, cfg.idle_eps_frac * action.std(axis=0))

        flagged_eps: set[int] = set()
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            a = action[rows]
            da = np.abs(np.diff(a, axis=0))
            static_frac = (da < eps).mean(axis=0)
            for j, f in enumerate(static_frac):
                if f > cfg.stuck_warn_fraction:
                    flagged_eps.add(ep)
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"episode {ep}: joint {j} static for {f:.0%} of frames - "
                            "dead actuator or disabled motor?",
                            episode_index=ep,
                            details={"joint": j, "static_fraction": float(f)},
                        )
                    )
            # gripper heuristic: pick/place task, last action dim never fires
            meta = ctx.meta_for(ep)
            if meta and any(w in " ".join(meta.tasks).lower() for w in _PICK_WORDS):
                g = action.shape[1] - 1
                if static_frac[g] > 0.95:
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"episode {ep}: gripper (action dim {g}) never actuates on a "
                            "pick/place task",
                            episode_index=ep,
                            details={"joint": g, "static_fraction": float(static_frac[g])},
                        )
                    )
        if not res.findings:
            res.summary = "every joint moves in every episode"
        else:
            res.summary = f"{len(flagged_eps)}/{ctx.total_episodes} episodes with stuck joints"
        return res

"""Checks 5 & 6 - idle segments and stuck actuators.

Idle stretches teach the policy to do nothing (the classic "robot stays
still at inference" postmortem). A joint that never moves is a broken
actuator, a disabled motor, or a recording bug - and the policy learns
to ignore that joint.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.stats import gripper_dims

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

    # A joint is stuck in an episode when its commanded range there is a
    # sliver of its range across the dataset. Frame-to-frame differences
    # are the wrong statistic: piecewise-constant commands (quantized
    # servos, low-rate teleop, binary grippers) read as "static" on most
    # frames while sweeping the joint's whole range.

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        n = action.shape[1]
        grip = gripper_dims(ctx.action_names, n)
        lo = np.nanpercentile(action, 1, axis=0)
        hi = np.nanpercentile(action, 99, axis=0)
        glob = hi - lo
        live = [j for j in range(n) if glob[j] > 1e-8]  # constant dims: check 8
        if not live:
            return self.skip("no action dim varies (see nan-and-normalization)")
        eps_ids = [ep for ep in ctx.episode_ids() if len(ctx.rows_for(ep)) >= 3]
        if not eps_ids:
            return self.skip("no episode long enough")
        ep_range = np.stack(
            [np.nanmax(action[ctx.rows_for(ep)], axis=0) - np.nanmin(action[ctx.rows_for(ep)], axis=0)
             for ep in eps_ids]
        )  # (episodes, dims)
        still = ep_range < cfg.stuck_range_frac * np.maximum(glob, 1e-12)
        n_eps = len(eps_ids)

        # 1. episodes where nothing moves at all: aborted/empty recordings
        dead = [eps_ids[i] for i in range(n_eps) if still[i, live].all()]
        dead_set = set(dead)
        if dead:
            res.findings.append(Finding(
                self.id, Severity.WARN,
                f"{len(dead)}/{n_eps} episodes where no joint moves at all: {dead[:10]}"
                f"{'...' if len(dead) > 10 else ''} - aborted or empty recordings "
                "that teach the policy to do nothing",
                details={"episodes": dead},
            ))

        # 2. a joint that moves in most episodes but not in a few: the motor
        #    died or was disabled during those recordings. A joint idle in
        #    (nearly) every episode is an unused DOF for this task - INFO.
        for j in live:
            if j in grip:
                continue
            stuck_eps = [eps_ids[i] for i in range(n_eps) if still[i, j] and eps_ids[i] not in dead_set]
            if not stuck_eps:
                continue
            share = len(stuck_eps) / n_eps
            if share >= cfg.stuck_unused_share:
                sev, why = Severity.INFO, "an unused degree of freedom for this task?"
            elif share <= cfg.stuck_warn_share:
                sev, why = Severity.WARN, "it moves in the others - dead or disabled motor during those recordings?"
            else:
                sev, why = Severity.INFO, "the task may not need it in every episode"
            name = f" ({ctx.action_names[j]})" if ctx.action_names and len(ctx.action_names) == n else ""
            res.findings.append(Finding(
                self.id, sev,
                f"joint {j}{name} does not move in {len(stuck_eps)}/{n_eps} episodes "
                f"{stuck_eps[:8]}{'...' if len(stuck_eps) > 8 else ''} - {why}",
                details={"joint": j, "episodes": stuck_eps, "share": share},
            ))

        # 3. pick/place task where the gripper never actuates
        g_dims = sorted(grip) if grip else ([n - 1] if ctx.action_names is None else [])
        for g in g_dims:
            if g not in live:
                continue
            idle_g = []
            for i, ep in enumerate(eps_ids):
                meta = ctx.meta_for(ep)
                if ep in dead_set or not meta:
                    continue
                if any(w in " ".join(meta.tasks).lower() for w in _PICK_WORDS) and \
                        ep_range[i, g] < 0.05 * glob[g]:
                    idle_g.append(ep)
            if idle_g:
                unused = len(idle_g) / n_eps >= cfg.stuck_unused_share
                res.findings.append(Finding(
                    self.id, Severity.INFO if unused else Severity.WARN,
                    f"gripper (action dim {g}) never actuates in {len(idle_g)}/{n_eps} "
                    f"pick/place episode(s): {idle_g[:10]}{'...' if len(idle_g) > 10 else ''}"
                    + (" - that arm looks unused in this task" if unused else ""),
                    details={"joint": g, "episodes": idle_g},
                ))

        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = "every joint moves wherever the dataset uses it"
        else:
            flagged = {e for f in res.findings if f.severity >= Severity.WARN
                       for e in f.details.get("episodes", [])}
            res.summary = f"{len(flagged)}/{n_eps} episodes with stuck joints"
        return res

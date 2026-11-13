"""Checks 5 & 6 - idle segments and stuck actuators.

Idle stretches teach the policy to do nothing (the classic "robot stays
still at inference" postmortem). A joint that never moves is a broken
actuator, a disabled motor, or a recording bug - and the policy learns
to ignore that joint.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.stats import action_gripper_dims, canonical, is_generic

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

    # A frame is idle when the command does not change AND (when state is
    # recorded) the robot does not move: a held command while the arm is
    # still travelling, or a constant velocity command, is not idling.
    # Where the idle sits matters: a long still start teaches the policy to
    # wait at t=0 (the "robot never starts" postmortem); stopping after the
    # task is done is correct behaviour unless it dominates the episode.

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        fps = ctx.fps or 30.0
        eps = np.maximum(1e-8, cfg.idle_eps_frac * np.nanstd(action, axis=0))
        state = ctx.state
        # the robot "did not move" over a run when every state dim stays
        # within a sliver of its dataset range (robust to sensor jitter,
        # which defeats a per-frame test)
        srange = None
        if state is not None:
            srange = np.nanpercentile(state, 99, axis=0) - np.nanpercentile(state, 1, axis=0)
            live_s = srange > 1e-8
        min_run = max(2, int(round(cfg.idle_min_seconds * fps)))

        n_flagged = 0
        worst_frac = 0.0
        n_eps = 0
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            n_eps += 1
            still = (np.abs(np.diff(action[rows], axis=0)) < eps).all(axis=1)
            idle = np.r_[still[0], still]  # frame 0 idles iff the first step does
            runs = [(s, e) for s, e in _runs(idle) if e - s >= min_run]
            if srange is not None and live_s.any():
                st = state[rows]

                def robot_still(s0: int, e0: int) -> bool:
                    seg = st[s0:e0][:, live_s]
                    moved = np.nanmax(seg, axis=0) - np.nanmin(seg, axis=0)
                    return bool((moved <= cfg.idle_state_frac * srange[live_s]).all())

                runs = [(s0, e0) for s0, e0 in runs if robot_still(s0, e0)]
            n = len(rows)
            idle_frames = sum(e - s for s, e in runs)
            frac = idle_frames / n
            worst_frac = max(worst_frac, frac)
            lead_s = (runs[0][1] / fps) if runs and runs[0][0] == 0 else 0.0
            trail_s = ((runs[-1][1] - runs[-1][0]) / fps) if runs and runs[-1][1] == n else 0.0
            heavy = frac > cfg.idle_warn_fraction
            starts_idle = lead_s > 0 and lead_s < n / fps  # all-idle episodes are stuck-actuator's
            if not (heavy or starts_idle or trail_s):
                continue
            parts = []
            if heavy:
                parts.append(f"{frac:.0%} of frames idle in runs >= {cfg.idle_min_seconds:g}s")
            if starts_idle:
                parts.append(f"starts idle for {lead_s:.1f}s (teaches waiting at the start)")
            if trail_s:
                parts.append(f"ends with {trail_s:.1f}s idle")
            sev = Severity.WARN if (heavy or starts_idle) else Severity.INFO
            if sev == Severity.WARN:
                n_flagged += 1
            res.findings.append(
                Finding(
                    self.id,
                    sev,
                    f"episode {ep}: " + "; ".join(parts)
                    + ("" if sev == Severity.WARN else " - trim if you want shorter episodes"),
                    episode_index=ep,
                    details={"idle_fraction": frac, "idle_runs": len(runs),
                             "leading_idle_s": lead_s, "trailing_idle_s": trail_s},
                )
            )
        n_tail = sum(1 for f in res.findings if f.severity == Severity.INFO)
        basis = "" if state is not None else " (no state recorded: judged from commands alone)"
        if not n_flagged:
            res.summary = (
                f"no episode starts idle or idles >= {cfg.idle_warn_fraction:.0%}"
                + (f"; {n_tail} end with a still tail" if n_tail else "")
            )
        else:
            res.summary = f"{n_flagged}/{n_eps} episodes flagged; worst {worst_frac:.0%} idle"
        res.summary += basis
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
        named_grip, inferred_grip = action_gripper_dims(ctx)
        grip = named_grip | inferred_grip
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
        g_dims = sorted(grip)
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

        # 4. commanded but not followed: the leader moves, the follower's
        #    measured position does not (dead / unpowered follower motor).
        #    Action-only checks cannot see this - the commands look healthy.
        res.findings.extend(self._not_followed(ctx, cfg, eps_ids, dead_set))

        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = "every joint moves wherever the dataset uses it"
        else:
            flagged = {e for f in res.findings if f.severity >= Severity.WARN
                       for e in f.details.get("episodes", [])}
            res.summary = f"{len(flagged)}/{n_eps} episodes with stuck joints"
        return res

    def _not_followed(self, ctx, cfg, eps_ids, dead_set) -> list[Finding]:
        state, action = ctx.state, ctx.action
        if state is None or state.shape[1] != action.shape[1]:
            return []
        n = action.shape[1]
        sn, an = ctx.state_names, ctx.action_names
        have_names = sn is not None and an is not None and len(sn) == n == len(an)
        a_rng = np.nanpercentile(action, 99, axis=0) - np.nanpercentile(action, 1, axis=0)
        s_rng = np.nanpercentile(state, 99, axis=0) - np.nanpercentile(state, 1, axis=0)
        live = a_rng[a_rng > 1e-8]
        typical = float(np.median(live)) if len(live) else 0.0
        out: list[Finding] = []
        pairs = []
        for j in range(n):
            # a joint commanded over a sliver of the robot's typical joint range
            # is noise, not motion the follower should reproduce
            if a_rng[j] <= 1e-8 or a_rng[j] < 0.01 * typical:
                continue
            # per dim: a meaningful name on both sides vouches for the pairing;
            # "motor_3" vouches for nothing and needs the data to agree
            named = have_names and not is_generic([sn[j]]) and not is_generic([an[j]])
            if named:
                if canonical(sn[j]) != canonical(an[j]):
                    continue
                if s_rng[j] < 0.01 * a_rng[j]:
                    # same joint by name, commanded everywhere, never measured moving
                    out.append(Finding(
                        self.id, Severity.WARN,
                        f"joint {j} ({an[j]}): commanded across the dataset but its measured "
                        "position never changes - follower motor dead or unpowered, or the "
                        "state is not actually recorded",
                        details={"joint": j, "kind": "never_followed"},
                    ))
                    continue
                pairs.append(j)
            else:
                # unnamed: only dims where action demonstrably tracks state
                ok = np.isfinite(state[:, j]) & np.isfinite(action[:, j])
                if ok.sum() < 20 or s_rng[j] <= 1e-8:
                    continue
                corr = float(np.corrcoef(state[ok, j], action[ok, j])[0, 1])
                if corr >= cfg.calib_min_corr and 0.5 <= a_rng[j] / s_rng[j] <= 2.0:
                    pairs.append(j)
        for j in pairs:
            bad = []
            for ep in eps_ids:
                if ep in dead_set:
                    continue
                rows = ctx.rows_for(ep)
                if len(rows) < 10:
                    continue
                # p5-p95: a command held for a couple of frames is a blip the
                # follower may legitimately lag, not sustained motion
                a5, a95 = np.nanpercentile(action[rows, j], [5, 95])
                s5, s95 = np.nanpercentile(state[rows, j], [5, 95])
                a_ep = (a95 - a5) / a_rng[j]
                s_ep = (s95 - s5) / max(s_rng[j], 1e-12)
                if a_ep >= cfg.follow_min_command and s_ep < cfg.follow_ratio * a_ep:
                    bad.append((ep, a_ep, s_ep))
            if not bad:
                continue
            name = f" ({an[j]})" if an and len(an) == n else ""
            ep0, a0, s0 = max(bad, key=lambda x: x[1] - x[2])
            out.append(Finding(
                self.id, Severity.WARN,
                f"joint {j}{name}: commanded to move but the follower did not in "
                f"{len(bad)}/{len(eps_ids)} episode(s) {[b[0] for b in bad][:8]}"
                f"{'...' if len(bad) > 8 else ''} (e.g. episode {ep0}: commanded {a0:.0%} of "
                f"its range, moved {s0:.0%}) - dead or disabled follower motor, or blocked joint",
                details={"joint": j, "kind": "not_followed", "episodes": [b[0] for b in bad]},
            ))
        return out


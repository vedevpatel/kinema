"""Check 9 - duplicate episodes and train/eval leakage.

Exact-copy episodes silently inflate metrics; near-duplicates do the
same with plausible deniability. LeRobot datasets do not declare
train/eval splits in the format, so straddling-split detection only
runs when the caller supplies split membership; duplicates themselves
are always reported.
"""

from __future__ import annotations

import hashlib

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity


class DuplicateCheck(Check):
    id = "duplicate-and-leakage"
    title = "Duplicate episodes / train-eval leakage"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")

        hashes: dict[str, list[int]] = {}
        trajectories: dict[int, np.ndarray] = {}
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 2:
                continue
            a = action[rows]
            h = hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()
            hashes.setdefault(h, []).append(ep)
            trajectories[ep] = a

        exact_groups = [eps for eps in hashes.values() if len(eps) > 1]
        for eps in exact_groups:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.WARN,
                    f"episodes {eps}: identical action trajectories - exact duplicates",
                    details={"episodes": eps, "kind": "exact"},
                )
            )

        # near-duplicates: downsample each trajectory, compare pairwise
        eps = sorted(trajectories)
        if 1 < len(eps) <= cfg.near_dup_max_pairs:
            grid = 32
            feats = {}
            for ep in eps:
                a = trajectories[ep]
                idx = np.linspace(0, len(a) - 1, min(grid, len(a))).astype(int)
                v = a[idx].ravel()
                scale = v.std()
                feats[ep] = v / scale if scale > 1e-12 else v * 0.0
            flagged: set[tuple[int, int]] = set()
            for i in range(len(eps)):
                for k in range(i + 1, len(eps)):
                    e1, e2 = eps[i], eps[k]
                    if {e1, e2} <= set(sum(exact_groups, [])):
                        continue  # already reported as exact
                    if len(feats[e1]) != len(feats[e2]):
                        continue
                    rmse = float(np.sqrt(((feats[e1] - feats[e2]) ** 2).mean()))
                    if rmse < 0.02:
                        flagged.add((e1, e2))
            for e1, e2 in sorted(flagged):
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.INFO,
                        f"episodes {e1} and {e2}: near-identical trajectories "
                        "(downsampled RMSE < 0.02) - possible re-recordings of the same run",
                        details={"episodes": [e1, e2], "kind": "near"},
                    )
                )
        elif len(eps) > cfg.near_dup_max_pairs:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    f"{len(eps)} episodes: near-duplicate scan skipped "
                    f"(>{cfg.near_dup_max_pairs} episodes); exact-duplicate scan still ran",
                )
            )

        if not res.findings:
            res.summary = "no exact or near duplicates"
        else:
            n_exact = sum(len(g) - 1 for g in exact_groups)
            res.summary = f"{n_exact} exact duplicate(s), {len(res.findings) - len(exact_groups)} other note(s)"
        return res

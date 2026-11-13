"""Check 9 - duplicate episodes and train/eval leakage.

Exact-copy episodes silently inflate metrics; near-duplicates do the
same with plausible deniability, and partial copies (a stretch of one
episode pasted into another) are what botched merges and replay tools
leave behind.

- exact: identical action trajectories.
- partial: episodes sharing identical 8-frame windows of motion. Windows,
  not single frames: clipped or quantized commands repeat individual
  values by chance, a run of 8 moving frames does not.
- near: downsampled trajectories within 0.05 std per dim (z-scored with
  dataset-wide stats, so no single large-unit dim decides).
- leakage: any of the above spanning two splits declared in info.json
  ("splits": {"train": "0:80", "val": "80:100"}) - eval metrics would be
  measured on training data.
"""

from __future__ import annotations

import hashlib

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity

_WINDOW = 8
_NEAR_GRID = 32
_NEAR_EXACT_MAX = 100_000  # above this the near-dup scan runs on an even sample
_NEAR_BLOCK_BYTES = 64 << 20  # distance-matrix budget per block
_COMMON_WINDOW_EPISODES = 4  # a window in more episodes than this is a shared motion, not a copy


def _splits(info: dict) -> dict[int, str]:
    """episode -> split name, from info.json "splits" ("start:end" ranges)."""
    out: dict[int, str] = {}
    for name, spec in (info.get("splits") or {}).items():
        try:
            lo, hi = (int(x) for x in str(spec).split(":"))
        except ValueError:
            continue
        for e in range(lo, hi):
            out.setdefault(e, name)
    return out


def _windows(a: np.ndarray, eps: np.ndarray) -> dict[bytes, int]:
    """Byte key -> start frame, for every _WINDOW-frame window with motion that
    occurs ONCE in the episode. A motion the episode itself repeats is part of
    the task by design (retrace-the-pattern benchmarks replay their demo) and
    says nothing about copies between episodes."""
    out: dict[bytes, int] = {}
    repeated: set[bytes] = set()
    if len(a) < _WINDOW:
        return out
    moving = np.r_[False, (np.abs(np.diff(a, axis=0)) > eps).any(axis=1)]
    raw = np.ascontiguousarray(a)
    for s in range(0, len(a) - _WINDOW + 1):
        if moving[s + 1: s + _WINDOW].sum() >= _WINDOW // 2:
            k = raw[s: s + _WINDOW].tobytes()
            if k in out:
                repeated.add(k)
            else:
                out[k] = s
    for k in repeated:
        del out[k]
    return out


class DuplicateCheck(Check):
    id = "duplicate-and-leakage"
    title = "Duplicate episodes / train-eval leakage"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        split_of = _splits(ctx.info)
        multi_split = len(set(split_of.values())) > 1

        trajectories: dict[int, np.ndarray] = {}
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) >= 2:
                trajectories[ep] = action[rows]
        eps = sorted(trajectories)
        pairs: dict[tuple[int, int], str] = {}  # pair -> strongest kind

        # 1. exact copies
        hashes: dict[str, list[int]] = {}
        for ep in eps:
            h = hashlib.sha1(np.ascontiguousarray(trajectories[ep]).tobytes()).hexdigest()
            hashes.setdefault(h, []).append(ep)
        exact_groups = [g for g in hashes.values() if len(g) > 1]
        exact_eps = {e for g in exact_groups for e in g}
        for g in exact_groups:
            res.findings.append(Finding(
                self.id, Severity.WARN,
                f"episodes {g}: identical action trajectories - exact duplicates",
                details={"episodes": g, "kind": "exact"},
            ))
            for i in range(len(g)):
                for k in range(i + 1, len(g)):
                    pairs[(g[i], g[k])] = "exact"

        # 2. partial copies: shared windows of motion. WHERE they sit matters:
        #    scripted sims replay the same opening motion (aligned, from frame
        #    0) - a copied beginning or a deterministic start, reported as
        #    INFO; a shared stretch anywhere else is a splice or a botched
        #    merge, and even ~17 frames of identical motion is no accident.
        std = np.nanstd(action, axis=0)
        move_eps = np.maximum(1e-8, 0.01 * std)
        owner: dict[bytes, dict[int, int]] = {}
        n_windows: dict[int, int] = {}
        for ep in eps:
            if ep in exact_eps:
                continue
            wins = _windows(trajectories[ep], move_eps)
            n_windows[ep] = len(wins)
            for win, start in wins.items():
                owner.setdefault(win, {})[ep] = start
        partial_found: list[Finding] = []
        shared: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for owners in owner.values():
            if 2 <= len(owners) <= _COMMON_WINDOW_EPISODES:
                o = sorted(owners)
                for i in range(len(o)):
                    for k in range(i + 1, len(o)):
                        shared.setdefault((o[i], o[k]), []).append((owners[o[i]], owners[o[k]]))
        for (e1, e2), pos in sorted(shared.items()):
            n = len(pos)
            smaller = max(1, min(n_windows.get(e1, 0), n_windows.get(e2, 0)))
            share = n / smaller
            aligned_opening = all(p1 == p2 for p1, p2 in pos) and min(p1 for p1, _ in pos) <= 2
            if share >= 0.8:
                sev, what = Severity.WARN, "a near-complete copy"
            elif aligned_opening and n >= 5:
                sev, what = Severity.INFO, "an identical opening (scripted start, or a copied beginning)"
            elif n >= 10 or (n >= 5 and share >= 0.10):
                # the data cannot tell copied data from a scripted generator
                # replaying a primitive - say both; neither adds diversity
                sev, what = Severity.WARN, ("copied data (splice, botched merge) or a scripted "
                                            "generator replaying a primitive")
            else:
                continue
            pairs.setdefault((e1, e2), "partial")
            partial_found.append(Finding(
                self.id, sev,
                f"episodes {e1} and {e2}: share {n} identical {_WINDOW}-frame stretches of motion "
                f"({share:.0%} of the shorter) - {what}; those segments add no diversity",
                details={"episodes": [e1, e2], "kind": "partial", "shared_windows": n,
                         "aligned_opening": aligned_opening},
            ))
        warn_pairs = [f for f in partial_found if f.severity == Severity.WARN]
        if len(warn_pairs) > 5:
            # systemic: one dataset-level finding, not one per pair
            eps_involved = sorted({e for f in warn_pairs for e in f.details["episodes"]})
            ex = ", ".join(f"{f.details['episodes'][0]}~{f.details['episodes'][1]}" for f in warn_pairs[:6])
            res.findings.append(Finding(
                self.id, Severity.WARN,
                f"{len(warn_pairs)} episode pairs ({len(eps_involved)} episodes) share identical "
                f"stretches of motion (e.g. {ex}) - copied data, or a scripted generator replaying "
                "primitives; those segments add no diversity",
                details={"kind": "partial", "pairs": [f.details["episodes"] for f in warn_pairs],
                         "episodes": eps_involved},
            ))
            res.findings.extend(f for f in partial_found if f.severity != Severity.WARN)
        else:
            res.findings.extend(partial_found)

        # 3. near-duplicates, vectorized over all episodes. Exact up to
        # _NEAR_EXACT_MAX episodes (blocked matmul sized to _NEAR_BLOCK_BYTES,
        # so a 90k-episode scan is minutes, not a skipped check); beyond that
        # an evenly-spaced sample with an INFO note saying so.
        live = std > 1e-8
        if len(eps) > 1 and live.any():
            scan_eps = eps
            if len(eps) > _NEAR_EXACT_MAX:
                pos = np.unique(np.round(np.linspace(0, len(eps) - 1, _NEAR_EXACT_MAX)).astype(int))
                scan_eps = [eps[p] for p in pos]
                res.findings.append(Finding(
                    self.id, Severity.INFO,
                    f"{len(eps)} episodes: near-duplicate scan ran on an evenly-spaced sample "
                    f"of {len(scan_eps)} (exact up to {_NEAR_EXACT_MAX}); "
                    "exact and partial-copy scans covered every episode",
                ))
            mu = np.nanmean(action, axis=0)[live]
            sd = std[live]
            feats = np.stack([
                ((trajectories[e][np.linspace(0, len(trajectories[e]) - 1, _NEAR_GRID).astype(int)][:, live]
                  - mu) / sd).ravel()
                for e in scan_eps
            ])
            feats = np.nan_to_num(feats)
            thresh = (0.05 ** 2) * feats.shape[1]  # mean squared diff < 0.05^2
            sq = (feats ** 2).sum(axis=1)
            block = max(64, min(1024, _NEAR_BLOCK_BYTES // (8 * len(scan_eps))))
            for i0 in range(0, len(scan_eps), block):
                fi = feats[i0:i0 + block]
                d2 = sq[i0:i0 + block, None] + sq[None, :] - 2 * fi @ feats.T
                ii, kk = np.nonzero(d2 < thresh)
                for a_, b_ in zip(ii + i0, kk):
                    if a_ >= b_:
                        continue
                    e1, e2 = scan_eps[a_], scan_eps[b_]
                    if (e1, e2) in pairs:
                        continue
                    pairs[(e1, e2)] = "near"
                    res.findings.append(Finding(
                        self.id, Severity.INFO,
                        f"episodes {e1} and {e2}: near-identical trajectories (downsampled, "
                        "within 0.05 std per dim) - possible re-recordings of the same run",
                        details={"episodes": [e1, e2], "kind": "near"},
                    ))

        # 4. leakage across declared splits
        if multi_split:
            leaks = [(p, k) for p, k in pairs.items()
                     if split_of.get(p[0]) and split_of.get(p[1]) and split_of[p[0]] != split_of[p[1]]]
            if leaks:
                ex = ", ".join(f"{a}({split_of[a]})~{b}({split_of[b]}) [{k}]" for (a, b), k in leaks[:5])
                res.findings.append(Finding(
                    self.id, Severity.WARN,
                    f"{len(leaks)} duplicate pair(s) span declared splits: {ex} - evaluation "
                    "would be measured on training data",
                    details={"pairs": [list(p) for p, _ in leaks], "kind": "leakage"},
                ))

        kinds = [f.details.get("kind") for f in res.findings]
        if not res.findings:
            res.summary = "no exact, partial or near duplicates" + (
                " across or within the declared splits" if multi_split else "")
        else:
            n_partial = sum(len(f.details.get("pairs", [None])) for f in res.findings
                            if f.details.get("kind") == "partial")
            res.summary = (
                f"{kinds.count('exact')} exact group(s), {n_partial} pair(s) sharing motion, "
                f"{kinds.count('near')} near-duplicate pair(s)"
                + ("; LEAKAGE across splits" if "leakage" in kinds else "")
            )
        return res

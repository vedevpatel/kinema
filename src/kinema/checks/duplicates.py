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
_NEAR_MAX_EPISODES = 10_000
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


def _windows(a: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Byte keys of every _WINDOW-frame window that contains motion."""
    if len(a) < _WINDOW:
        return np.empty(0, dtype=object)
    moving = np.r_[False, (np.abs(np.diff(a, axis=0)) > eps).any(axis=1)]
    keys = []
    raw = np.ascontiguousarray(a)
    for s in range(0, len(a) - _WINDOW + 1):
        if moving[s + 1: s + _WINDOW].sum() >= _WINDOW // 2:
            keys.append(raw[s: s + _WINDOW].tobytes())
    return np.asarray(keys, dtype=object)


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

        # 2. partial copies: shared windows of motion
        std = np.nanstd(action, axis=0)
        move_eps = np.maximum(1e-8, 0.01 * std)
        owner: dict[bytes, set[int]] = {}
        n_windows: dict[int, int] = {}
        for ep in eps:
            if ep in exact_eps:
                continue
            keys = set(_windows(trajectories[ep], move_eps).tolist())
            n_windows[ep] = len(keys)
            for k in keys:
                owner.setdefault(k, set()).add(ep)
        shared: dict[tuple[int, int], int] = {}
        for owners in owner.values():
            if 2 <= len(owners) <= _COMMON_WINDOW_EPISODES:
                o = sorted(owners)
                for i in range(len(o)):
                    for k in range(i + 1, len(o)):
                        shared[(o[i], o[k])] = shared.get((o[i], o[k]), 0) + 1
        for (e1, e2), n in sorted(shared.items()):
            smaller = max(1, min(n_windows.get(e1, 0), n_windows.get(e2, 0)))
            if n >= 5 and n >= 0.10 * smaller:
                pairs.setdefault((e1, e2), "partial")
                res.findings.append(Finding(
                    self.id, Severity.WARN,
                    f"episodes {e1} and {e2}: share {n} identical {_WINDOW}-frame stretches of "
                    f"motion ({n / smaller:.0%} of the shorter) - a partial copy (botched merge "
                    "or replayed segment)",
                    details={"episodes": [e1, e2], "kind": "partial", "shared_windows": n},
                ))

        # 3. near-duplicates, vectorized over all episodes
        live = std > 1e-8
        if len(eps) > _NEAR_MAX_EPISODES:
            res.findings.append(Finding(
                self.id, Severity.INFO,
                f"{len(eps)} episodes: near-duplicate scan skipped (> {_NEAR_MAX_EPISODES}); "
                "exact and partial-copy scans still ran",
            ))
        elif len(eps) > 1 and live.any():
            mu = np.nanmean(action, axis=0)[live]
            sd = std[live]
            feats = np.stack([
                ((trajectories[e][np.linspace(0, len(trajectories[e]) - 1, _NEAR_GRID).astype(int)][:, live]
                  - mu) / sd).ravel()
                for e in eps
            ])
            feats = np.nan_to_num(feats)
            thresh = (0.05 ** 2) * feats.shape[1]  # mean squared diff < 0.05^2
            sq = (feats ** 2).sum(axis=1)
            block = 1024
            for i0 in range(0, len(eps), block):
                fi = feats[i0:i0 + block]
                d2 = sq[i0:i0 + block, None] + sq[None, :] - 2 * fi @ feats.T
                ii, kk = np.nonzero(d2 < thresh)
                for a_, b_ in zip(ii + i0, kk):
                    if a_ >= b_:
                        continue
                    e1, e2 = eps[a_], eps[b_]
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
            res.summary = (
                f"{kinds.count('exact')} exact group(s), {kinds.count('partial')} partial "
                f"copies, {kinds.count('near')} near-duplicate pair(s)"
                + ("; LEAKAGE across splits" if "leakage" in kinds else "")
            )
        return res

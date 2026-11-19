"""Metamorphic probes: transformations of a loaded dataset that must not
change any check's verdict.

A check that is right about the data gives the same answer when the data
is re-expressed without changing its quality:

- affine: a unit change (degrees <-> radians, mm -> m) or re-zeroed joint
  encoders - the same per-dim scale and offset applied to state, action and
  every stored statistic, so a stale stats.json stays exactly as stale.
- permute_episodes: the same episodes, rows and episode list in another
  order.
- keep_episodes: a subset of the episodes (for the nightly robustness
  probe: the flagged episodes plus part of the rest).

A verdict that moves under these is an artifact of the representation - an
absolute threshold, an order dependence, a small-sample inference - not a
fact about the data. tests/test_invariance.py runs every check under them on
every push; the leaderboard re-runs new WARN/FAIL findings under them before
publishing.

These operate on an already-loaded DatasetContext and copy only what they
change; nothing is written to disk.
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from kinema.dataset import ACTION_KEY, STATE_KEY, DatasetContext

_LOCATION_STATS = re.compile(r"^(mean|min|max|q\d+)$")


def _affine_stats(stats: dict, feat: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Copy of a {feature: {stat: values}} map with one feature's location
    statistics mapped through x*a + b and its spread through |a|."""
    if feat not in stats or not isinstance(stats[feat], dict):
        return stats
    out = dict(stats)
    st = dict(stats[feat])
    for k, v in st.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.shape != a.shape:
            continue  # count, or a stat of another width: untouched
        if _LOCATION_STATS.match(k):
            new = arr * a + b
        elif k == "std":
            new = arr * np.abs(a)
        else:
            continue
        st[k] = new.tolist() if isinstance(v, list) else new
    out[feat] = st
    return out


def affine(ctx: DatasetContext, scale, offset=0.0) -> DatasetContext:
    """state and action mapped through x*scale + offset (per dim, or scalar),
    with every stored statistic mapped the same way. scale must be > 0: a
    sign flip swaps min and max, which is a different probe."""
    out = dataclasses.replace(ctx)
    for canon, attr in ((STATE_KEY, "state"), (ACTION_KEY, "action")):
        arr = getattr(ctx, attr)
        if arr is None:
            continue
        d = arr.shape[1]
        a = np.broadcast_to(np.asarray(scale, dtype=np.float64), (d,)).copy()
        b = np.broadcast_to(np.asarray(offset, dtype=np.float64), (d,)).copy()
        if (a <= 0).any():
            raise ValueError("affine probe needs a positive scale")
        setattr(out, attr, np.asarray(arr, dtype=np.float64) * a + b)
        feat = (ctx.feature_keys or {}).get(canon, canon)
        out.stats = _affine_stats(out.stats, feat, a, b)
        out.stats_json = _affine_stats(out.stats_json, feat, a, b)
        out.episode_stats = {ep: _affine_stats(per, feat, a, b)
                             for ep, per in out.episode_stats.items()}
    return out


_ROW_ARRAYS = ("episode_index", "frame_index", "timestamp", "state", "action",
               "task_index", "reward", "done")


def _take_rows(ctx: DatasetContext, rows: np.ndarray, episodes) -> DatasetContext:
    out = dataclasses.replace(ctx, episodes=list(episodes), _row_slices={}, _meta_index={})
    for name in _ROW_ARRAYS:
        arr = getattr(ctx, name)
        if arr is not None:
            setattr(out, name, np.asarray(arr)[rows])
    return out


def permute_episodes(ctx: DatasetContext, seed: int = 0) -> DatasetContext:
    """The same episodes with their row blocks and the episode list in a
    seeded random order (rows within an episode keep their order)."""
    rng = np.random.default_rng(seed)
    eps = ctx.episode_ids()
    order = [eps[i] for i in rng.permutation(len(eps))]
    rows = np.concatenate([ctx.rows_for(e) for e in order] or [np.empty(0, dtype=int)])
    meta = {e.index: e for e in ctx.episodes}
    return _take_rows(ctx, rows, [meta[e] for e in order])


def reverse_episodes(ctx: DatasetContext) -> DatasetContext:
    """The episode order reversed: deterministically swaps any prefix and
    suffix, which a random shuffle only does by chance."""
    order = list(reversed(ctx.episode_ids()))
    rows = np.concatenate([ctx.rows_for(e) for e in order] or [np.empty(0, dtype=int)])
    meta = {e.index: e for e in ctx.episodes}
    return _take_rows(ctx, rows, [meta[e] for e in order])


def keep_episodes(ctx: DatasetContext, keep) -> DatasetContext:
    """Only the given episodes (their rows and catalogue entries)."""
    keep = {int(e) for e in keep}
    eps = [e for e in ctx.episodes if e.index in keep]
    rows = np.concatenate([ctx.rows_for(e.index) for e in eps] or [np.empty(0, dtype=int)])
    out = _take_rows(ctx, np.sort(rows), eps)
    out.truncated = True
    return out


def signature(result) -> tuple[str, frozenset]:
    """What must not move under a probe: the check's verdict and the episodes
    its WARN/FAIL findings point at."""
    from kinema.report import flagged_episodes

    return result.verdict.name, frozenset(flagged_episodes(result))

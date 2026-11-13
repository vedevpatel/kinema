"""Normalization-stats helpers and feature semantics shared by the checks.

LeRobot records min/max/mean/std/count per episode (v2.1:
meta/episodes_stats.jsonl, v3.0: stats/<feature>/<stat> columns in
meta/episodes/*.parquet) and aggregates them into the dataset-level stats
a policy normalizes with. Re-aggregating the per-episode stats of exactly
the episodes we loaded gives an exact reference for a sampled subset -
comparing a 50-episode sample against whole-dataset stats does not.
"""

from __future__ import annotations

import re

import numpy as np

STAT_KEYS = ("mean", "std", "min", "max", "count")


def as_stat_dict(entry: dict | None) -> dict[str, np.ndarray] | None:
    """Coerce a stats entry ({stat: list}) to float arrays; None if unusable."""
    if not isinstance(entry, dict):
        return None
    out: dict[str, np.ndarray] = {}
    for k in STAT_KEYS:
        if k in entry and entry[k] is not None:
            try:
                out[k] = np.asarray(entry[k], dtype=np.float64).ravel()
            except (TypeError, ValueError):
                return None
    return out or None


def aggregate(per_episode: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray] | None:
    """Pool per-episode stats the way LeRobot's aggregate_stats does:
    count-weighted mean, pooled population variance, elementwise min/max."""
    eps = [e for e in per_episode if e and all(k in e for k in STAT_KEYS)]
    if not eps:
        return None
    dims = {e["mean"].shape for e in eps}
    if len(dims) != 1:
        return None
    counts = np.array([float(e["count"][0]) for e in eps])
    total = counts.sum()
    if total <= 0:
        return None
    w = counts[:, None]
    means = np.stack([e["mean"] for e in eps])
    stds = np.stack([e["std"] for e in eps])
    mean = (w * means).sum(axis=0) / total
    var = (w * (stds**2 + (means - mean) ** 2)).sum(axis=0) / total
    return {
        "mean": mean,
        "std": np.sqrt(np.maximum(var, 0.0)),
        "min": np.stack([e["min"] for e in eps]).min(axis=0),
        "max": np.stack([e["max"] for e in eps]).max(axis=0),
        "count": np.array([total]),
    }


def data_stats(arr: np.ndarray) -> dict[str, np.ndarray] | None:
    data = arr[np.isfinite(arr).all(axis=1)]
    if len(data) == 0:
        return None
    return {
        "mean": data.mean(axis=0),
        "std": data.std(axis=0),
        "min": data.min(axis=0),
        "max": data.max(axis=0),
        "count": np.array([float(len(data))]),
    }


# --- feature semantics from info.json names -------------------------------

_GRIPPER_RE = re.compile(r"grip|finger|claw|jaw|hand|gripper|pinch")
# names that carry no meaning ("motor_3", "dim_0", "4"): they cannot tell us
# whether state and action live in the same space
_GENERIC_RE = re.compile(r"^(motor|dim|axis|channel|state|action|act|obs)?[_.\- ]?\d+$")
_UNIT_SUFFIX_RE = re.compile(r"([._](pos|position|rad|deg|degrees|angle|m|cmd|command))+$")


def feature_names(info: dict, key: str) -> list[str] | None:
    """Per-dim names from info.json (list, or {"motors": [...]}-style dict)."""
    names = ((info.get("features") or {}).get(key) or {}).get("names")
    if isinstance(names, dict):
        lists = [v for v in names.values() if isinstance(v, list)]
        names = lists[0] if len(lists) == 1 else None
    if isinstance(names, list) and names and all(isinstance(n, str) for n in names):
        return [n.strip() for n in names]
    return None


def is_gripper(name: str | None) -> bool:
    return bool(name) and bool(_GRIPPER_RE.search(name.lower()))


def is_generic(names: list[str] | None) -> bool:
    return not names or all(_GENERIC_RE.match(n.lower()) for n in names)


def canonical(name: str) -> str:
    return _UNIT_SUFFIX_RE.sub("", name.lower().strip())


def gripper_dims(names: list[str] | None, dim: int) -> set[int]:
    if not names or len(names) != dim:
        return set()
    return {j for j, n in enumerate(names) if is_gripper(n)}

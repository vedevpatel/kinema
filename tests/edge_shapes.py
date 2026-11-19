"""A synthetic dataset carrying the data shapes real boards punished us on,
as extra dims beside the synthetic arm. Each one sits near a threshold a
rule must handle at ANY scale, so the invariance probes (tests/test_invariance.py)
actually cross the thresholds - plain synthetic data never comes near them.

Shapes (dim name -> where it came from):
  jitter      RoboCOIN Alphabot: a channel at ~0.9 jittering by std 4e-6, its
              stored stats carrying converter noise at that tiny scale
  parked      ucsd_pick_and_place dim 5: parked at 0.90, moves in one episode
  dropout     austin_buds / hggforget: a live value near 0.95 drops to 0 for
              a few frames (a true positive: must stay flagged)
  big_const   RoboCOIN: constant at 4096.7, stored std ~0.5 of f32 noise
  small_live  a live joint recorded in small units (std 2e-4)
  grip2       OpenArm J8: a two-level gripper at exactly 0 / 1.2121
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinema.fixtures.synth import make_dataset

EDGE_NAMES = ["jitter", "parked", "dropout", "big_const", "small_live", "grip2"]


def _edge_columns(ep: np.ndarray, rng) -> np.ndarray:
    n = len(ep)
    t = np.arange(n)
    cols = np.zeros((n, len(EDGE_NAMES)))
    cols[:, 0] = 0.9 + 4e-6 * rng.normal(size=n)
    cols[:, 1] = 0.9 + 1e-6 * rng.normal(size=n)
    cols[np.flatnonzero(ep == 5)[:13], 1] = 1.278
    cols[:, 2] = 0.95 + 0.005 * rng.normal(size=n)
    cols[np.flatnonzero(ep == 11)[[10, 20, 30]], 2] = 0.0
    cols[:, 3] = 4096.7
    cols[:, 4] = 0.3 + 2e-4 * np.sin(t / 5.0) + 2e-5 * rng.normal(size=n)
    cols[:, 5] = np.where(np.sin(t / 9.0) > 0, 1.2121, 0.0)
    flip = np.flatnonzero(np.diff(cols[:, 5]) != 0)
    for k in flip[::2]:  # a few in-between frames on some transitions
        cols[k + 1, 5] = 0.6
    return cols


def _stats(x: np.ndarray) -> dict:
    return {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}


def make_edge_dataset(path: str | Path, episodes: int = 24) -> Path:
    root = make_dataset(path, episodes=episodes)
    rng = np.random.default_rng(7)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    ep = np.asarray(t["episode_index"])
    edge = _edge_columns(ep, rng)
    s = np.hstack([np.asarray(t["observation.state"]), edge])
    a = np.hstack([np.asarray(t["action"]), edge])  # position commands equal measurement
    t["observation.state"], t["action"] = s.tolist(), a.tolist()
    pq.write_table(pa.table(t), f)

    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    for feat in ("observation.state", "action"):
        info["features"][feat]["shape"] = [s.shape[1]]
        info["features"][feat]["names"] = list(info["features"][feat]["names"]) + EDGE_NAMES
    info_p.write_text(json.dumps(info))

    # stored stats: truthful, except the converter noise RoboCOIN carries on
    # its near-constant channels (jitter: std 1.6x and mean off by 1e-6;
    # big_const: a fabricated std of 0.5)
    j0, jc = s.shape[1] - len(EDGE_NAMES), s.shape[1] - len(EDGE_NAMES) + 3

    def noisy(st: dict) -> dict:
        st = {k: np.array(v, dtype=float) for k, v in st.items()}
        st["std"][j0] *= 1.6
        st["mean"][j0] += 1e-6
        st["std"][jc] = 0.5
        return st

    stats = {}
    for feat, arr in (("observation.state", s), ("action", a)):
        st = noisy(_stats(arr))
        stats[feat] = {k: v.tolist() for k, v in st.items()} | {"count": [len(arr)]}
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    for feat, arr in (("observation.state", s), ("action", a)):
        for i, e in enumerate(m["episode_index"]):
            st = noisy(_stats(arr[ep == e]))
            for k in ("mean", "std", "min", "max"):
                m[f"stats/{feat}/{k}"][i] = st[k].tolist()
    pq.write_table(pa.table(m), mf)
    return root

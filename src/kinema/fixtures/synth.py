"""Synthetic LeRobot v3 dataset generator with injectable defects.

Used by the test suite (one test per check) and by `kinema synth` for
demos. Writes a real v3.0 layout: meta/info.json, meta/stats.json,
meta/tasks.parquet, meta/episodes/chunk-000/file-000.parquet and
data/chunk-000/file-000.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KNOWN_DEFECTS = [
    "stats-drift",
    "timestamp-grid",
    "dropped-frames",
    "calibration-drift",
    "idle-segments",
    "stuck-actuator",
    "action-saturation",
    "nan",
    "duplicates",
    "outlier-episodes",
    "deployment",
]

_FPS = 10
_STATE_DIM = 6
_LEN = 60


def _smooth(rng, n, dim, scale=1.0):
    t = np.linspace(0, 4 * np.pi, n)
    return scale * np.stack(
        [np.sin(t + rng.uniform(0, 2 * np.pi)) + 0.2 * rng.standard_normal(n) for _ in range(dim)],
        axis=1,
    )


def make_dataset(
    path: str | Path,
    episodes: int = 24,
    defects: list[str] | None = None,
    seed: int = 0,
) -> Path:
    defects = defects or []
    unknown = set(defects) - set(KNOWN_DEFECTS)
    if unknown:
        raise ValueError(f"unknown defects: {sorted(unknown)} (known: {KNOWN_DEFECTS})")
    rng = np.random.default_rng(seed)
    root = Path(path)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows_ep, rows_fi, rows_ts, rows_state, rows_action, rows_task = [], [], [], [], [], []
    ep_meta = []
    global_index = 0
    for ep in range(episodes):
        n = _LEN
        if "outlier-episodes" in defects and ep == episodes - 1:
            n = _LEN * 5
        t = np.arange(n) / _FPS
        state = _smooth(rng, n, _STATE_DIM)
        # action tracks state (a well-calibrated leader/follower pair)
        action = state + 0.005 * rng.standard_normal(state.shape)
        if ep == 0:
            first_state, first_action = state.copy(), action.copy()

        if "calibration-drift" in defects:
            action[:, 2] += 0.35 * (state.max() - state.min())
        if "idle-segments" in defects and ep == 2:
            idle = int(n * 0.6)
            action[n - idle:] = action[n - idle - 1]  # ends idle
        if "stuck-actuator" in defects and ep == 3:
            action[:, 4] = action[0, 4]
        if "nan" in defects and ep == 5:
            state[n // 2, 0] = np.nan
        if "timestamp-grid" in defects and ep == 6:
            t = t + rng.uniform(0, 0.5 / _FPS, n)  # off the grid
        if "timestamp-grid" in defects and ep == 7:
            t = t.copy()
            t[n // 2] = t[n // 2 - 1]  # non-monotonic
        if "duplicates" in defects and ep == 1:
            state, action = first_state.copy(), first_action.copy()
        if "deployment" in defects and ep == 8:
            n = 30  # shorter than any policy chunk
            t = np.arange(n) / _FPS
            state = _smooth(rng, n, _STATE_DIM)
            action = state + 0.005 * rng.standard_normal(state.shape)

        drop = "dropped-frames" in defects and ep == 9
        for i in range(n):
            if drop and i == n // 3:
                continue  # one dropped frame
            rows_ep.append(ep)
            rows_fi.append(i)
            rows_ts.append(float(t[i]))
            rows_state.append(state[i].tolist())
            rows_action.append(action[i].tolist())
            rows_task.append(0)
            global_index += 1
        ep_meta.append(
            {
                "episode_index": ep,
                "length": n if not drop else n,  # metadata lies: says n, data has n-1
                "tasks": ["pick up the cube"],
            }
        )

    if "action-saturation" in defects:
        # clip dim 1 at its own 90th percentile: 10% of commands pile up at
        # exactly the limit, the signature of actuator saturation
        _a = np.asarray(rows_action)
        _hi = np.percentile(_a[:, 1], 90)
        _a[:, 1] = np.minimum(_a[:, 1], _hi)
        rows_action = _a.tolist()

    table = pa.table(
        {
            "episode_index": pa.array(rows_ep, type=pa.int64()),
            "frame_index": pa.array(rows_fi, type=pa.int64()),
            "timestamp": pa.array(rows_ts, type=pa.float64()),
            "observation.state": pa.array(rows_state, type=pa.list_(pa.float64())),
            "action": pa.array(rows_action, type=pa.list_(pa.float64())),
            "task_index": pa.array(rows_task, type=pa.int64()),
        }
    )
    pq.write_table(table, root / "data" / "chunk-000" / "file-000.parquet")

    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([m["episode_index"] for m in ep_meta], type=pa.int64()),
                "length": pa.array([m["length"] for m in ep_meta], type=pa.int64()),
                "tasks": pa.array([m["tasks"] for m in ep_meta], type=pa.list_(pa.string())),
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick up the cube"]),
            }
        ),
        root / "meta" / "tasks.parquet",
    )

    state_arr = np.asarray(rows_state)
    action_arr = np.asarray(rows_action)

    def _stats(a):
        return {
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
            "count": [len(a)],
        }

    stats = {
        "observation.state": _stats(state_arr[np.isfinite(state_arr).all(axis=1)]),
        "action": _stats(action_arr),
    }
    if "stats-drift" in defects:
        for k in ("mean", "std"):
            stats["action"][k] = (np.asarray(stats["action"][k]) * 1.08).tolist()

    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": _FPS,
                "total_episodes": episodes,
                "total_frames": global_index,
                "features": {
                    "observation.state": {"dtype": "float64", "shape": [_STATE_DIM]},
                    "action": {"dtype": "float64", "shape": [_STATE_DIM]},
                    "timestamp": {"dtype": "float64", "shape": [1]},
                    "frame_index": {"dtype": "int64", "shape": [1]},
                    "episode_index": {"dtype": "int64", "shape": [1]},
                    "task_index": {"dtype": "int64", "shape": [1]},
                },
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            },
            indent=2,
        )
    )
    return root

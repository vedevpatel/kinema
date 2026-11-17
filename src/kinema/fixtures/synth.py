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
    "stale-file-pointer",
    "missing-video-file",
]

_FPS = 10
_STATE_DIM = 6
_LEN = 60
_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper"]


def _stats(a):
    a = a[np.isfinite(a).all(axis=1)]
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "count": [len(a)],
    }


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
    with_state: bool = True,
    velocity_dim: bool = False,
    action_mode: str = "position",
    delta_scale: float = 1.0,
    shards: int = 1,
    cameras: list[str] | None = None,
) -> Path:
    defects = defects or []
    unknown = set(defects) - set(KNOWN_DEFECTS)
    if unknown:
        raise ValueError(f"unknown defects: {sorted(unknown)} (known: {KNOWN_DEFECTS})")
    if "stale-file-pointer" in defects and shards < 2:
        raise ValueError(
            "stale-file-pointer needs shards >= 2 (the converter bug lives at the "
            "file-rollover boundary)"
        )
    if "missing-video-file" in defects and not cameras:
        raise ValueError("missing-video-file needs cameras= (there is no video to lose otherwise)")
    cameras = cameras or []
    if action_mode not in ("position", "delta"):
        raise ValueError(f"unknown action_mode: {action_mode!r}")
    rng = np.random.default_rng(seed)
    root = Path(path)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    a_dim = _STATE_DIM + (1 if velocity_dim else 0)
    a_names = _NAMES + ["base.velocity"] if velocity_dim else _NAMES

    rows_ep, rows_fi, rows_ts, rows_state, rows_action, rows_task = [], [], [], [], [], []
    ep_meta = []
    global_index = 0
    for ep in range(episodes):
        n = _LEN
        if "outlier-episodes" in defects and ep == episodes - 1:
            n = _LEN * 8  # far-out duration must clear the 6x-median bar
        t = np.arange(n) / _FPS
        state = _smooth(rng, n, _STATE_DIM)
        # action tracks state (a well-calibrated leader/follower pair)
        action = state + 0.005 * rng.standard_normal(state.shape)
        if action_mode == "delta":
            # delta commands: a[t] is the step the state takes next
            delta = np.diff(state, axis=0)
            action = np.vstack([delta, delta[-1:]]) * delta_scale
            action += 0.002 * rng.standard_normal(action.shape)
        if velocity_dim:
            # a named velocity command on top of the position arm: parked at
            # 0 everywhere except episode 4, whose back half drives the base
            # at a held constant velocity while the arm holds still (the
            # commands-only idle false positive: constant command, real motion)
            vel = np.zeros((n, 1))
            if ep == 4:
                hold = int(n * 0.6)
                vel[n - hold:] = 0.6
                action[n - hold:] = action[n - hold - 1]
                state[n - hold:] = state[n - hold - 1]
            action = np.hstack([action, vel])
        if ep == 0:
            first_state, first_action = state.copy(), action.copy()

        if "calibration-drift" in defects:
            action[:, 2] += 0.35 * (state.max() - state.min())
        if "idle-segments" in defects and ep == 2:
            idle = int(n * 0.6)
            action[n - idle:] = action[n - idle - 1]  # ends idle
            state[n - idle:] = state[n - idle - 1]  # and the follower stops with it
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

    cols = {
        "episode_index": pa.array(rows_ep, type=pa.int64()),
        "frame_index": pa.array(rows_fi, type=pa.int64()),
        "timestamp": pa.array(rows_ts, type=pa.float64()),
        "action": pa.array(rows_action, type=pa.list_(pa.float64())),
        "task_index": pa.array(rows_task, type=pa.int64()),
    }
    if with_state:
        cols["observation.state"] = pa.array(rows_state, type=pa.list_(pa.float64()))
    table = pa.table(cols)
    # episodes fill shard files contiguously, as the v2.1->v3.0 converter
    # packs them (file-000 until its size cap, then file-001, ...)
    ep_shard = {ep: min(ep * shards // episodes, shards - 1) for ep in range(episodes)}
    ep_col = np.asarray(rows_ep)
    for shard in range(shards):
        eps_in_shard = {e for e, s in ep_shard.items() if s == shard}
        if not eps_in_shard:
            continue
        sel = np.isin(ep_col, list(eps_in_shard))
        pq.write_table(table.filter(sel), root / "data" / "chunk-000" / f"file-{shard:03d}.parquet")
    # the converter rollover bug (huggingface/lerobot#2401): the first episode
    # of each new file is written there but its metadata keeps the previous
    # file's pointer
    stale = "stale-file-pointer" in defects
    file_idx = []
    # ep_meta is appended in episode order, so the position IS the episode index
    for ep, m in enumerate(ep_meta):
        s = ep_shard[ep]
        if stale and s > 0 and ep == min(e for e, sh in ep_shard.items() if sh == s):
            s -= 1
        file_idx.append(s)

    # per-episode stats, as LeRobot v3 writes them (stats/<feature>/<stat>)
    ep_arr = np.asarray(rows_ep)
    state_all, action_all = np.asarray(rows_state), np.asarray(rows_action)
    feats = [("action", action_all)] + ([("observation.state", state_all)] if with_state else [])
    ep_cols: dict[str, list] = {}
    for m in ep_meta:
        sel = ep_arr == m["episode_index"]
        for feat, arr in feats:
            for k, v in _stats(arr[sel]).items():
                ep_cols.setdefault(f"stats/{feat}/{k}", []).append(v)
    # video segments, as LeRobot v3 records them per camera: episodes packed
    # 5 to a shared MP4, [from, to) seconds inside it. ep_meta is appended in
    # episode order, so the position IS the episode index.
    video_cols: dict[str, pa.Array] = {}
    for cam in cameras:
        video_cols.update({
            f"videos/{cam}/chunk_index": pa.array([0] * len(ep_meta), type=pa.int64()),
            f"videos/{cam}/file_index": pa.array([i // 5 for i in range(len(ep_meta))], type=pa.int64()),
            f"videos/{cam}/from_timestamp": pa.array(
                [(i % 5) * _LEN / _FPS for i in range(len(ep_meta))], type=pa.float64()),
            f"videos/{cam}/to_timestamp": pa.array(
                [(i % 5 + 1) * _LEN / _FPS for i in range(len(ep_meta))], type=pa.float64()),
        })
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([m["episode_index"] for m in ep_meta], type=pa.int64()),
                "length": pa.array([m["length"] for m in ep_meta], type=pa.int64()),
                "tasks": pa.array([m["tasks"] for m in ep_meta], type=pa.list_(pa.string())),
                # every real v3 export records the shard each episode lives in
                "data/chunk_index": pa.array([0] * len(ep_meta), type=pa.int64()),
                "data/file_index": pa.array(file_idx, type=pa.int64()),
                **video_cols,
                **{
                    c: pa.array(v, type=pa.list_(pa.int64() if c.endswith("/count") else pa.float64()))
                    for c, v in ep_cols.items()
                },
            }
        ),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    # placeholder MP4s: existence checks only need the path to be there (the
    # visual check is not part of fixture runs). The missing-video-file
    # defect leaves the last camera's last shard on the floor: the metadata
    # points at it, the repo does not hold it.
    for cam in cameras:
        n_files = (episodes + 4) // 5
        for fi in range(n_files):
            if "missing-video-file" in defects and cam == cameras[-1] and fi == n_files - 1:
                continue
            p = root / "videos" / cam / "chunk-000" / f"file-{fi:03d}.mp4"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"")
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array(["pick up the cube"]),
            }
        ),
        root / "meta" / "tasks.parquet",
    )

    stats = {"action": _stats(action_all)}
    if with_state:
        stats["observation.state"] = _stats(state_all)
    if "stats-drift" in defects:
        for k in ("mean", "std"):
            stats["action"][k] = (np.asarray(stats["action"][k]) * 1.08).tolist()

    features = {
        "action": {"dtype": "float64", "shape": [a_dim], "names": a_names},
        "timestamp": {"dtype": "float64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    if with_state:
        features["observation.state"] = {"dtype": "float64", "shape": [_STATE_DIM], "names": _NAMES}

    for cam in cameras:
        features[cam] = {"dtype": "video", "shape": [480, 640, 3],
                         "names": ["height", "width", "channels"]}

    (root / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    info = {
        "codebase_version": "v3.0",
        "fps": _FPS,
        "total_episodes": episodes,
        "total_frames": global_index,
        "features": features,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    }
    if cameras:
        info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        info["chunks_size"] = 1000
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    return root

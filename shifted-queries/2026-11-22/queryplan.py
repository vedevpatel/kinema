"""Pure query derivation for the shifted-timestamp experiment.

Nothing here imports LeRobot, torch or a video library, so the rules can be
unit-tested offline. Each function mirrors one upstream code path. The cited
lines are in the installed releases whose file hashes are pinned in
`expectations.json` (`upstream_sources`):

* `delta_rows` follows `lerobot/datasets/factory.py::resolve_delta_timestamps`
  (`[i / ds_meta.fps for i in cfg.observation_delta_indices]`) and then
  `lerobot/datasets/utils.py::get_delta_indices`
  (`[round(d * fps) for d in delta_ts]`).
* `query_rows` follows `LeRobotDataset._get_query_indices`: it clamps
  `idx + delta` to `[ep_start, ep_end - 1]`, and `*_is_pad` is true outside
  that range. Rows are episode-relative here. The episode is contiguous, so
  this matches 0.3.3's `episode_data_index` and 0.4.0's
  `dataset_from_index`/`dataset_to_index`, an equality `plan.py` asserts.
* `loader_timestamp` follows `hf_transform_to_torch`. It applies
  `torch.tensor(x)` to the Python float that pyarrow hands the transform,
  which yields torch's default float32. `_get_query_timestamps` then calls
  `torch.stack(...).tolist()`. `probe.py` checks this equivalence against
  the installed upstream functions.
* `video_query` follows `_query_videos`. 0.4.0 (v3.0) adds the episode's
  `videos/<key>/from_timestamp`, and 0.3.3 adds nothing.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

TOLERANCE_S = 1e-4  # LeRobotDataset(tolerance_s=1e-4) default in 0.3.3 and 0.4.0
ANCHORS = ("first", "middle", "worst_grid", "last")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delta_rows(observation_delta_indices: list[int] | None, fps: float) -> list[int] | None:
    """Row offsets a policy config produces for an observation key, or None."""
    if observation_delta_indices is None:
        return None
    delta_ts = [i / fps for i in observation_delta_indices]
    return [round(d * fps) for d in delta_ts]


def select_anchors(timestamps: np.ndarray, fps: float) -> dict[str, int]:
    """First, middle, worst-grid and last rows.

    The worst-grid rule is the one phase 2's decode_fineart.py used: the
    earlier row of the consecutive pair whose spacing deviates most from 1/fps,
    computed on the stored dtype. np.argmax takes the first maximum on ties.
    """
    ts = np.asarray(timestamps)
    n = len(ts)
    if n < 2:
        raise ValueError("an anchor plan needs at least two rows")
    worst = int(np.argmax(np.abs(np.diff(ts) - 1 / fps)))
    return {"first": 0, "middle": n // 2, "worst_grid": worst, "last": n - 1}


def query_rows(anchor: int, length: int, deltas: list[int]) -> list[tuple[int, bool]]:
    """(clamped row, is_pad) for every delta, as _get_query_indices builds them."""
    if not 0 <= anchor < length:
        raise ValueError("anchor outside episode")
    out = []
    for delta in deltas:
        target = anchor + delta
        out.append((max(0, min(length - 1, target)), target < 0 or target >= length))
    return out


def loader_timestamp(stored: float) -> float:
    """Python float the loader holds after torch.tensor(stored).tolist() (float32)."""
    return float(np.float32(stored))


def video_query(loader_ts: float, from_timestamp: float | None) -> float:
    """Timestamp passed to decode_video_frames for one query."""
    return loader_ts if from_timestamp is None else from_timestamp + loader_ts


def decoder_distance(query: float, pts: float) -> float:
    """|query - pts| as decode_video_frames_torchvision computes it (float32 cdist)."""
    return float(abs(np.float32(query) - np.float32(pts)))


def nearest_pts(query: float, pts: list[float]) -> tuple[int, float] | None:
    """Index and float32 distance of the frame the decoder's argmin would pick."""
    if not pts:
        return None
    q = np.float32(query)
    loaded = np.asarray(pts, dtype=np.float32)
    dist = np.abs(q - loaded)
    index = int(np.argmin(dist))
    return index, float(dist[index])


def build_case_plan(
    *,
    timestamps: np.ndarray,
    fps: float,
    cameras: list[str],
    from_timestamps: dict[str, float] | None,
    configs: dict[str, list[int] | None],
    nominal_offsets: list[int],
) -> dict:
    """Enumerate every query before any decoder runs.

    Class `loader_derived`: one batch per (camera, anchor, config), holding
    exactly the timestamps `__getitem__(anchor)` would pass to the decoder for
    that config. A config with no observation deltas queries the current
    timestamp only.

    Class `nominal_grid`: the phase-2 hypothesis `raw + k/fps`. LeRobot 0.3.3
    and 0.4.0 `__getitem__` never produce these queries; they are kept as a
    labeled comparator. Each query is a single call, never clamped, and is
    flagged when it falls outside the episode's stored time range.
    """
    ts = np.asarray(timestamps)
    n = len(ts)
    anchors = select_anchors(ts, fps)
    first_ts, last_ts = loader_timestamp(ts[0]), loader_timestamp(ts[-1])
    batches = []
    for camera in cameras:
        start = None if from_timestamps is None else from_timestamps[camera]
        for position in ANCHORS:
            anchor = anchors[position]
            current = loader_timestamp(ts[anchor])
            for config, indices in configs.items():
                deltas = delta_rows(indices, fps)
                if deltas is None:
                    rows = [(anchor, False)]
                    deltas_used = [0]
                else:
                    rows = query_rows(anchor, n, deltas)
                    deltas_used = deltas
                queries = []
                for delta, (row, pad) in zip(deltas_used, rows, strict=True):
                    lts = loader_timestamp(ts[row])
                    queries.append({
                        "delta_rows": delta,
                        "row": row,
                        "is_pad": pad,
                        "stored_timestamp": float(ts[row]),
                        "loader_timestamp": lts,
                        "video_query_s": video_query(lts, start),
                        "intended_offset_s": delta / fps,
                        "effective_offset_s": lts - current,
                    })
                batches.append({
                    "query_class": "loader_derived",
                    "camera": camera,
                    "anchor": position,
                    "anchor_row": anchor,
                    "config": config,
                    "padded": any(q["is_pad"] for q in queries),
                    "queries": queries,
                })
            for k in nominal_offsets:
                q = current + k / fps
                batches.append({
                    "query_class": "nominal_grid",
                    "camera": camera,
                    "anchor": position,
                    "anchor_row": anchor,
                    "config": f"raw_plus_{k}_over_fps",
                    "padded": False,
                    "outside_episode_time_range": not (first_ts <= q <= last_ts),
                    "queries": [{
                        "delta_rows": None,
                        "row": None,
                        "is_pad": False,
                        "stored_timestamp": None,
                        "loader_timestamp": q,
                        "video_query_s": video_query(q, start),
                        "intended_offset_s": k / fps,
                        "effective_offset_s": k / fps,
                    }],
                })
    return {"rows": n, "fps": fps, "anchors": anchors, "batches": batches}


def batch_bucket(batch: dict) -> str:
    """Analysis bucket: in-range queries are kept apart from boundary behavior."""
    if batch["query_class"] == "loader_derived":
        return "loader_derived/padded" if batch["padded"] else "loader_derived/in_range"
    return "nominal_grid/outside_episode" if batch["outside_episode_time_range"] else "nominal_grid/in_range"


def summarize(batches: list[dict]) -> dict:
    """Per bucket counts of executed batches by outcome."""
    out: dict[str, dict[str, int]] = {}
    for batch in batches:
        bucket = out.setdefault(batch_bucket(batch), {"ok": 0, "tolerance_error": 0, "other_error": 0, "not_run": 0})
        outcome = batch.get("outcome", "not_run")
        bucket[outcome if outcome in bucket else "other_error"] += 1
    return out


def bucket_verdict(counts: dict[str, int] | None) -> str:
    """Preregistered rule for one bucket of one case.

    supports_claim: at least one batch failed the decoder tolerance.
    contradicts_claim: every batch ran and decoded within tolerance.
    unresolved: any batch hit a non-tolerance error or did not run, with no
    tolerance error. An empty bucket is not_applicable.
    """
    if not counts or sum(counts.values()) == 0:
        return "not_applicable"
    if counts["tolerance_error"]:
        return "supports_claim"
    if counts["other_error"] or counts["not_run"]:
        return "unresolved"
    return "contradicts_claim"


def is_finite_plan(plan: dict) -> bool:
    return all(math.isfinite(q["video_query_s"]) for b in plan["batches"] for q in b["queries"])

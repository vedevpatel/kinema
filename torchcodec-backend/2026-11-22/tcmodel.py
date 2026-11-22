"""Pure rules for the torchcodec backend comparison (no torch, LeRobot or video imports).

Each function mirrors one upstream code path so it can be unit-tested offline.
Sources (paths relative to `site-packages/`, hashes in `environment_spec.json`):

* LeRobot 0.3.3 and 0.4.0 `lerobot/datasets/video_utils.py::decode_video_frames_torchcodec`:
  `VideoDecoder(..., seek_mode="approximate")`,
  `frame_indices = [round(ts * metadata.average_fps) for ts in timestamps]`,
  `decoder.get_frames_at(indices=frame_indices)`, then
  `torch.cdist(torch.tensor(timestamps)[:, None], torch.tensor(loaded_ts)[:, None], p=1).min(1)`
  and `assert (min_ < tolerance_s).all()`. Each query takes the loaded frame at
  `argmin_`, which can belong to another query in the same batch.
* torchcodec 0.5 `torchcodec/decoders/_video_decoder.py::VideoDecoder.get_frames_at`:
  `index if index >= 0 else index + self._num_frames`. In approximate mode
  `_num_frames` is `metadata.num_frames`, which equals `num_frames_from_header`.
* torchcodec v0.5.0 `src/torchcodec/_core/_metadata.py::VideoStreamMetadata.average_fps`:
  without a scan it returns `average_fps_from_header`.
* torchcodec v0.5.0 `src/torchcodec/_core/SingleStreamDecoder.cpp` (tag v0.5.0,
  commit 93cc0ade58be5e424fc17489f1ba50b7b530d00b):
  - `averageFpsFromHeader = av_q2d(avStream->r_frame_rate)`;
  - `validateFrameIndex`: `frameIndex >= 0` and `frameIndex < numFramesFromHeader`
    (approximate mode), else `TORCH_CHECK` "Invalid frame index=...";
  - `getPts` (approximate): `secondsToClosestPts(frameIndex / averageFpsFromHeader, timeBase)`,
    with `secondsToClosestPts = std::round(seconds * den / num)`;
  - `getNextFrameInternal`: returns the first decoded frame with `pts >= cursor`;
    when the stream ends first it raises "Requested next frame while there are no
    more frames left to decode."
"""

from __future__ import annotations

import bisect
import math

import numpy as np

TOLERANCE_S = 1e-4

QUERY_REJECTIONS = ("tolerance_error", "index_error", "end_of_stream_error")
NON_EVIDENCE = ("dependency_error", "other_error", "harness_discrepancy", "not_run")
OUTCOMES = ("ok", *QUERY_REJECTIONS, *NON_EVIDENCE)


def lerobot_frame_index(query_s: float, average_fps: float) -> int:
    """`round(ts * average_fps)` as LeRobot writes it (Python round, half to even)."""
    return round(query_s * average_fps)


def wrap_negative_index(index: int, num_frames: int | None) -> int:
    """VideoDecoder.get_frames_at maps a negative index to index + num_frames."""
    if index >= 0 or num_frames is None:
        return index
    return index + num_frames


def index_is_valid(index: int, num_frames_from_header: int | None) -> bool:
    """validateFrameIndex in approximate mode."""
    return index >= 0 and (num_frames_from_header is None or index < num_frames_from_header)


def cpp_round(x: float) -> int:
    """std::round: halfway cases away from zero."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def seconds_to_closest_pts(seconds: float, tb_num: int, tb_den: int) -> int:
    """secondsToClosestPts: std::round(seconds * den / num), evaluated left to right in double."""
    return cpp_round(seconds * tb_den / tb_num)


def pts_to_seconds(pts: int, tb_num: int, tb_den: int) -> float:
    """ptsToSeconds: double(pts) * num / den."""
    return float(pts) * tb_num / tb_den


def approximate_cursor(index: int, average_fps_from_header: float, tb_num: int, tb_den: int) -> int:
    """getPts in approximate mode: the cursor pts for a frame index."""
    return seconds_to_closest_pts(index / average_fps_from_header, tb_num, tb_den)


def first_pts_at_or_after(sorted_pts: list[int], cursor: int) -> int | None:
    """getNextFrameInternal: first frame (presentation order) with pts >= cursor."""
    position = bisect.bisect_left(sorted_pts, cursor)
    return sorted_pts[position] if position < len(sorted_pts) else None


def f32_distance(a: float, b: float) -> float:
    """|float32(a) - float32(b)|, the elementwise value of LeRobot's float32 cdist(p=1)."""
    return float(abs(np.float32(a) - np.float32(b)))


def predict_query(query_s: float, stream: dict) -> dict:
    """Source-model prediction for one query, before any decoder runs.

    `stream` holds `average_fps_from_header` (r_frame_rate), `num_frames_from_header`
    (nb_frames), `time_base` as [num, den] and `sorted_pts` (packet pts in ticks).
    """
    fps = stream["average_fps_from_header"]
    nb = stream["num_frames_from_header"]
    num, den = stream["time_base"]
    raw_index = lerobot_frame_index(query_s, fps)
    index = wrap_negative_index(raw_index, nb)
    out = {"raw_index": raw_index, "index": index, "wrapped": index != raw_index}
    if not index_is_valid(index, nb):
        out["outcome"] = "index_error"
        return out
    cursor = approximate_cursor(index, fps, num, den)
    pts = first_pts_at_or_after(stream["sorted_pts"], cursor)
    out["cursor_pts"] = cursor
    if pts is None:
        out["outcome"] = "end_of_stream_error"
        return out
    out["returned_pts_ticks"] = pts
    out["returned_pts_s"] = pts_to_seconds(pts, num, den)
    out["own_frame_distance_s"] = f32_distance(query_s, out["returned_pts_s"])
    return out


def lerobot_selection(queries: list[float], loaded_pts_s: list[float]) -> list[tuple[int, float]]:
    """(argmin position, float32 distance) per query over every loaded frame in the batch.

    Ties keep the first position, as torch.min(dim) does on CPU for these inputs.
    """
    loaded = np.asarray(loaded_pts_s, dtype=np.float32)
    out = []
    for q in queries:
        dist = np.abs(np.float32(q) - loaded)
        position = int(np.argmin(dist))
        out.append((position, float(dist[position])))
    return out


def predict_batch(queries: list[float], stream: dict, tolerance_s: float = TOLERANCE_S) -> dict:
    """Batch prediction for decode_video_frames_torchcodec.

    getFramesAtIndices walks the indices in ascending order, validating and
    decoding each, so the first index-validation or end-of-stream failure in
    that order is raised before LeRobot's tolerance check runs.
    """
    per_query = [predict_query(q, stream) for q in queries]
    for p in sorted(per_query, key=lambda p: p["index"]):
        if "outcome" in p:
            return {"outcome": p["outcome"], "per_query": per_query}
    loaded = [p["returned_pts_s"] for p in per_query]
    selection = lerobot_selection(queries, loaded)
    for p, (position, distance) in zip(per_query, selection, strict=True):
        p["selected_position"] = position
        p["selected_pts_s"] = loaded[position]
        p["decoder_distance_s"] = distance
        p["within_tolerance"] = distance < tolerance_s
    outcome = "ok" if all(p["within_tolerance"] for p in per_query) else "tolerance_error"
    return {"outcome": outcome, "per_query": per_query}


def classify_exception(exc_type: str, message: str) -> str:
    """Outcome class for an exception raised by decode_video_frames."""
    if exc_type == "AssertionError" and "violate the tolerance" in message:
        return "tolerance_error"
    if "Could not load libtorchcodec" in message:
        return "dependency_error"
    if "Invalid frame index" in message:
        return "index_error"
    if "no more frames left to decode" in message:
        return "end_of_stream_error"
    return "other_error"


def bucket_counts(batches: list[dict], key: str) -> dict[str, int]:
    """Counts of batch[key] over the fixed outcome list."""
    counts = dict.fromkeys(OUTCOMES, 0)
    for batch in batches:
        outcome = batch.get(key, "not_run")
        counts[outcome if outcome in counts else "other_error"] += 1
    return counts


def bucket_verdict(counts: dict[str, int]) -> dict:
    """Preregistered verdict for one bucket of one case and backend.

    supports_claim: at least one batch was rejected for a query-caused reason
    (tolerance, index validation or end of stream); `via` names which.
    contradicts_claim: every batch ran and decoded within tolerance.
    unresolved: no rejection, and at least one batch hit a dependency error,
    another error, a harness discrepancy, or did not run.
    not_applicable: the bucket is empty.
    """
    total = sum(counts.values())
    if total == 0:
        return {"verdict": "not_applicable", "via": []}
    via = [k for k in QUERY_REJECTIONS if counts.get(k)]
    if via:
        return {"verdict": "supports_claim", "via": via}
    if any(counts.get(k) for k in NON_EVIDENCE):
        return {"verdict": "unresolved", "via": []}
    return {"verdict": "contradicts_claim", "via": []}


VERDICT_LABEL = {"supports_claim": "confirmed", "contradicts_claim": "refuted", "unresolved": "unresolved", "not_applicable": "not applicable"}


def frame_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixel statistics between two uint8 CHW frames of equal shape."""
    if a.shape != b.shape:
        return {"equal": False, "shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "equal": bool(not diff.any()),
        "max_abs": int(diff.max()),
        "mean_abs": float(diff.mean()),
        "fraction_values_differing": float((diff > 0).mean()),
    }

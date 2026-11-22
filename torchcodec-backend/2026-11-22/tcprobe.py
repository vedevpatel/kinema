"""Replay the frozen phase-3 query plans through LeRobot's torchcodec decoder path (direct decoder).

    LD_LIBRARY_PATH=/tmp/kinema-tc-ffmpeg-v3:/tmp/kinema-phase2-v3/lib/python3.11/site-packages/av.libs \
        /tmp/kinema-phase2-v3/bin/python -I -B torchcodec-backend/2026-11-22/tcprobe.py r1:c1:f0

For every batch in shifted-queries/2026-11-22/plan/<case>.json, unchanged:

1. upstream `decode_video_frames(path, queries, 1e-4, "torchcodec")`;
2. an independent torchcodec replay (fresh VideoDecoder, same approximate seek
   mode and index rule, one query at a time) that records per-query index,
   returned pts and frame hash, and must reproduce the upstream outcome and,
   on success, its output tensor;
3. a pyav control in the same process: upstream
   `decode_video_frames(path, queries, 1e-4, "pyav")` plus a replay of its
   VideoReader window, recorded separately from phase 3's frozen evidence;
4. a per-query backend comparison of returned pts and pixels.

LeRobotDataset is not constructed, patched or subclassed, and no LeRobot or
torchcodec code is modified. The script refuses to run when a preregistered
file, the phase-3 plan, an input, an upstream source or the package set has
changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PHASE3, case_key, case_root, check_preregistration, frozen_source_checks, load_environment_spec, load_expectations, load_phase3_plan, sha256_file  # noqa: E402

sys.path.insert(1, str(PHASE3))

import torch  # noqa: E402
import torchvision  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec  # noqa: E402

import plan as phase3_plan  # noqa: E402  (frozen phase-3 module, imported read-only)
import probe as phase3_probe  # noqa: E402  (frozen phase-3 module, imported read-only)
import tcmodel  # noqa: E402

TOLERANCE_S = tcmodel.TOLERANCE_S
NEIGHBOURS = 2


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def mapped_libraries() -> list[str]:
    """Shared objects of interest mapped into this process (from /proc/self/maps)."""
    keys = ("libav", "libsw", "torchcodec", "libstdc++", "libdav1d", "libx264", "libvpx")
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split()
            if len(parts) >= 6 and any(k in parts[5] for k in keys):
                paths.add(parts[5])
    return sorted(paths)


def torchcodec_environment(spec: dict, env: str) -> dict:
    environment = spec["environments"][env]
    site = Path(environment["av_libs"]).parent
    expected_ld = f"{environment['alias_dir']}:{environment['av_libs']}"
    installed = {rel: sha256_file(site / "torchcodec" / rel) for rel in spec["torchcodec"]["installed_sha256"]}
    out = {
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "ld_library_path_matches_spec": os.environ.get("LD_LIBRARY_PATH") == expected_ld,
        "torchcodec_distribution": importlib.metadata.version("torchcodec"),
        "installed_sha256_ok": installed == spec["torchcodec"]["installed_sha256"],
        "alias_targets": {link.name: os.readlink(link) for link in sorted(Path(environment["alias_dir"]).glob("*"))} if Path(environment["alias_dir"]).exists() else None,
    }
    try:
        import torchcodec
        from torchcodec._core import get_ffmpeg_library_versions

        out.update(import_ok=True, torchcodec_version=torchcodec.__version__, ffmpeg_library_versions=get_ffmpeg_library_versions())
    except Exception as exc:  # recorded, not retried
        out.update(import_ok=False, outcome=tcmodel.classify_exception(type(exc).__name__, str(exc)), exception=type(exc).__name__, message=str(exc)[:3000])
    import av

    out["pyav_library_versions"] = {k: ".".join(map(str, v)) for k, v in av.library_versions.items()}
    out["ok"] = bool(out["ld_library_path_matches_spec"] and out["installed_sha256_ok"] and out.get("import_ok"))
    return out


def selection_record() -> dict:
    return {"find_spec_torchcodec": importlib.util.find_spec("torchcodec") is not None, "get_safe_default_codec": get_safe_default_codec()}


# --- torchcodec --------------------------------------------------------------

def tc_upstream(path: Path, queries: list[float]) -> tuple[dict, torch.Tensor | None]:
    started = time.perf_counter()
    try:
        frames = decode_video_frames(path, queries, TOLERANCE_S, "torchcodec")
        out = {"outcome": "ok", "shape": list(frames.shape), "dtype": str(frames.dtype)}
    except Exception as exc:
        frames = None
        out = {"outcome": tcmodel.classify_exception(type(exc).__name__, str(exc)), "exception": type(exc).__name__, "message": str(exc)[:1500]}
        if out["outcome"] == "other_error":
            out["traceback"] = traceback.format_exc()[-3000:]
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out, frames


def tc_replay(path: Path, queries: list[float]) -> tuple[dict, list[torch.Tensor | None]]:
    """Per-query torchcodec decode with LeRobot's index rule and a fresh decoder."""
    from torchcodec.decoders import VideoDecoder

    decoder = VideoDecoder(str(path), seek_mode="approximate")
    metadata = decoder.metadata
    fps = metadata.average_fps
    per_query, frames = [], []
    for q in queries:
        raw = tcmodel.lerobot_frame_index(q, fps)
        entry = {"raw_index": raw, "index": tcmodel.wrap_negative_index(raw, metadata.num_frames)}
        try:
            batch = decoder.get_frames_at(indices=[raw])
            frame = batch.data[0]
            entry.update(outcome="decoded", returned_pts_s=float(batch.pts_seconds[0]), duration_s=float(batch.duration_seconds[0]), own_frame_distance_s=tcmodel.f32_distance(q, float(batch.pts_seconds[0])), frame_sha256=tensor_sha256(frame))
            frames.append(frame)
        except Exception as exc:
            entry.update(outcome=tcmodel.classify_exception(type(exc).__name__, str(exc)), exception=type(exc).__name__, message=str(exc)[:300])
            frames.append(None)
        per_query.append(entry)
    out = {
        "metadata": {
            "average_fps": metadata.average_fps,
            "average_fps_from_header": metadata.average_fps_from_header,
            "num_frames": metadata.num_frames,
            "num_frames_from_header": metadata.num_frames_from_header,
            "num_frames_from_content": metadata.num_frames_from_content,
            "begin_stream_seconds_from_content": metadata.begin_stream_seconds_from_content,
            "width": metadata.width,
            "height": metadata.height,
            "codec": metadata.codec,
        },
        "per_query": per_query,
    }
    failures = sorted((p["index"], p["outcome"]) for p in per_query if p["outcome"] != "decoded")
    if failures:
        out["expected_outcome"] = failures[0][1]
        return out, []
    loaded = [p["returned_pts_s"] for p in per_query]
    dist = torch.cdist(torch.tensor(queries)[:, None], torch.tensor(loaded)[:, None], p=1)
    min_, argmin_ = dist.min(1)
    for p, d, i in zip(per_query, min_, argmin_, strict=True):
        p.update(selected_position=int(i), selected_pts_s=loaded[int(i)], decoder_distance_s=float(d), within_tolerance=bool(d < TOLERANCE_S), selected_frame_sha256=per_query[int(i)]["frame_sha256"])
    out["expected_outcome"] = "ok" if bool((min_ < TOLERANCE_S).all()) else "tolerance_error"
    return out, [frames[int(i)] for i in argmin_]


# --- pyav control -------------------------------------------------------------

def pyav_window(path: Path, queries: list[float]) -> tuple[dict, list[torch.Tensor], list[torch.Tensor]]:
    """The VideoReader window of decode_video_frames_torchvision, keeping every loaded frame.

    Same steps as shifted-queries/2026-11-22/probe.py::replay_window.
    """
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(path), "video")
    reader.seek(min(queries), keyframes_only=True)
    loaded_frames, loaded_pts = [], []
    for frame in reader:
        loaded_frames.append(frame["data"])
        loaded_pts.append(frame["pts"])
        if frame["pts"] >= max(queries):
            break
    reader.container.close()
    out = {"loaded_count": len(loaded_pts), "loaded_first_pts": loaded_pts[0] if loaded_pts else None, "loaded_last_pts": loaded_pts[-1] if loaded_pts else None}
    if not loaded_pts:
        out.update(per_query=[None] * len(queries), expected_outcome="tolerance_error")
        return out, [], []
    dist = torch.cdist(torch.tensor(queries)[:, None], torch.tensor(loaded_pts)[:, None], p=1)
    min_, argmin_ = dist.min(1)
    out["per_query"] = [
        {"window_position": int(i), "returned_pts_s": loaded_pts[int(i)], "decoder_distance_s": float(d), "within_tolerance": bool(d < TOLERANCE_S), "frame_sha256": tensor_sha256(loaded_frames[int(i)])}
        for d, i in zip(min_, argmin_, strict=True)
    ]
    out["expected_outcome"] = "ok" if bool((min_ < TOLERANCE_S).all()) else "tolerance_error"
    out["loaded_pts"] = loaded_pts
    return out, [loaded_frames[int(i)] for i in argmin_], loaded_frames


def pyav_upstream(path: Path, queries: list[float]) -> tuple[dict, torch.Tensor | None]:
    started = time.perf_counter()
    try:
        frames = decode_video_frames(path, queries, TOLERANCE_S, "pyav")
        out = {"outcome": "ok", "shape": list(frames.shape)}
    except Exception as exc:
        frames = None
        out = {"outcome": tcmodel.classify_exception(type(exc).__name__, str(exc)), "exception": type(exc).__name__, "message": str(exc)[:600]}
    out["seconds"] = round(time.perf_counter() - started, 3)
    return out, frames


# --- comparison ---------------------------------------------------------------

def ticks(seconds: float, time_base: list[int]) -> int:
    return tcmodel.cpp_round(seconds * time_base[1] / time_base[0])


def compare_frames(tc_frame: torch.Tensor, pyav_frame: torch.Tensor, pyav_position: int, pyav_loaded: list[torch.Tensor], pyav_pts: list[float]) -> dict:
    out = tcmodel.frame_difference(tc_frame.numpy(), pyav_frame.numpy())
    if not out["equal"] and "shape_mismatch" not in out:
        neighbours = []
        for position in range(max(0, pyav_position - NEIGHBOURS), min(len(pyav_loaded), pyav_position + NEIGHBOURS + 1)):
            stats = tcmodel.frame_difference(tc_frame.numpy(), pyav_loaded[position].numpy())
            neighbours.append({"offset": position - pyav_position, "pts_s": pyav_pts[position], "mean_abs": stats["mean_abs"], "equal": stats["equal"]})
        best = min(neighbours, key=lambda n: n["mean_abs"])
        out["neighbours"] = neighbours
        out["closest_pyav_offset"] = best["offset"]
    return out


def compare_batch(tc: dict, tc_frames: list, pyav: dict, pyav_frames: list, pyav_loaded: list, time_base: list[int]) -> dict:
    tc_rows = tc["replay"]["per_query"]
    pyav_rows = pyav["replay"].get("per_query") or [None] * len(tc_rows)
    per_query = []
    for k, (t, p) in enumerate(zip(tc_rows, pyav_rows, strict=True)):
        entry: dict = {}
        if t.get("selected_pts_s") is not None and p is not None:
            entry["tc_ticks"] = ticks(t["selected_pts_s"], time_base)
            entry["pyav_ticks"] = ticks(p["returned_pts_s"], time_base)
            entry["same_pts"] = entry["tc_ticks"] == entry["pyav_ticks"]
            entry["pts_difference_s"] = t["selected_pts_s"] - p["returned_pts_s"]
        elif t.get("returned_pts_s") is not None and p is not None:
            entry["tc_own_frame_ticks"] = ticks(t["returned_pts_s"], time_base)
            entry["pyav_ticks"] = ticks(p["returned_pts_s"], time_base)
            entry["same_pts_own_frame"] = entry["tc_own_frame_ticks"] == entry["pyav_ticks"]
        both_ok = tc["upstream"]["outcome"] == "ok" and pyav["upstream"]["outcome"] == "ok" and tc["replay_consistent"] and pyav["replay_consistent"]
        if both_ok:
            entry["pixels"] = compare_frames(tc_frames[k], pyav_frames[k], p["window_position"], pyav_loaded, pyav["replay"]["loaded_pts"])
        per_query.append(entry)
    return {
        "outcomes": {"torchcodec": tc["outcome"], "pyav_control": pyav["outcome"]},
        "same_outcome": tc["outcome"] == pyav["outcome"],
        "per_query": per_query,
    }


def run_batch(batch: dict, root: Path, time_base: list[int]) -> dict:
    path = root / batch["video_path"]
    queries = [q["video_query_s"] for q in batch["queries"]]
    tc_up, tc_out = tc_upstream(path, queries)
    try:
        replay, tc_frames = tc_replay(path, queries)
        consistent = replay["expected_outcome"] == tc_up["outcome"]
        if consistent and tc_out is not None:
            consistent = torch.equal(tc_out, (torch.stack(tc_frames) / 255.0).type(torch.float32))
            replay["frames_equal_upstream"] = consistent
    except Exception as exc:
        replay, tc_frames, consistent = {"error": type(exc).__name__, "message": str(exc)[:500]}, [], False
    tc = {"upstream": tc_up, "replay": replay, "replay_consistent": consistent, "outcome": tc_up["outcome"] if consistent else "harness_discrepancy"}

    py_up, py_out = pyav_upstream(path, queries)
    try:
        window, pyav_frames, pyav_loaded = pyav_window(path, queries)
        consistent = window["expected_outcome"] == py_up["outcome"]
        if consistent and py_out is not None:
            consistent = torch.equal(py_out, torch.stack(pyav_frames).type(torch.float32) / 255)
            window["frames_equal_upstream"] = consistent
    except Exception as exc:
        window, pyav_frames, pyav_loaded, consistent = {"error": type(exc).__name__, "message": str(exc)[:500]}, [], [], False
    pyav = {"upstream": py_up, "replay": window, "replay_consistent": consistent, "outcome": py_up["outcome"] if consistent else "harness_discrepancy"}

    comparison = compare_batch(tc, tc_frames, pyav, pyav_frames, pyav_loaded, time_base) if "per_query" in replay else {"outcomes": {"torchcodec": tc["outcome"], "pyav_control": pyav["outcome"]}, "same_outcome": tc["outcome"] == pyav["outcome"], "per_query": []}
    window.pop("loaded_pts", None)
    return {"torchcodec": tc, "pyav_control": pyav, "comparison": comparison}


def default_backend_attempt(path: Path, query: float) -> dict:
    """decode_video_frames(..., backend=None) after all batches, compared with an explicit torchcodec call."""
    out = selection_record()
    try:
        default = decode_video_frames(path, [query], TOLERANCE_S, None)
        explicit = decode_video_frames(path, [query], TOLERANCE_S, "torchcodec")
        out.update(outcome="ok", equals_explicit_torchcodec=torch.equal(default, explicit), shape=list(default.shape))
    except Exception as exc:
        out.update(outcome=tcmodel.classify_exception(type(exc).__name__, str(exc)), exception=type(exc).__name__, message=str(exc)[:1500])
    return out


def identity_checks(case: dict, record: dict, root: Path, phase3_expectations: dict, spec: dict, env: str, expectations: dict) -> dict:
    frozen = frozen_source_checks(expectations)
    rehash = {e["path"]: sha256_file(root / e["path"]) == e["sha256"] for e in record["inputs"]["entries"]}
    sources = phase3_plan.verify_sources(case["env"], phase3_expectations, phase3_plan.lerobot_root())
    environment = phase3_probe.environment_record(case["env"], phase3_expectations)
    torchcodec_env = torchcodec_environment(spec, env)
    checks = {
        "frozen_sources_ok": all(frozen.values()),
        "inputs_rehashed_ok": all(rehash.values()),
        "upstream_sources_ok": sources["ok"],
        "packages_match_phase2_freeze": not environment["packages_differing_from_phase2_freeze"] and not environment["packages_not_in_phase2_freeze"],
        "torchcodec_environment_ok": torchcodec_env["ok"],
    }
    return {"checks": checks, "ok": all(checks.values()), "frozen_sources": frozen, "rehash": rehash, "sources": sources, "environment": environment, "torchcodec_environment": torchcodec_env}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--identity-only", action="store_true", help="run identity and environment checks only; decode nothing, write nothing")
    args = parser.parse_args()
    prereg = {} if args.identity_only else check_preregistration()
    phase3_probe.check_preregistration()
    expectations = load_expectations()
    spec = load_environment_spec()
    phase3_expectations = phase3_plan.load_expectations()
    case = {c["case_id"]: c for c in phase3_expectations["cases"]}[args.case_id]
    env = {"v2": "tc-v2", "v3": "tc-v3"}[case["env"]]
    if args.case_id not in spec["environments"][env]["cases"]:
        raise SystemExit("case not assigned to this environment in environment_spec.json")
    record, plan_sha = load_phase3_plan(args.case_id)
    root = case_root(args.case_id)
    identity = identity_checks(case, record, root, phase3_expectations, spec, env, expectations)
    if args.identity_only:
        print(json.dumps({"case_id": args.case_id, "checks": identity["checks"], "torchcodec_environment": identity["torchcodec_environment"], "packages_differing": identity["environment"]["packages_differing_from_phase2_freeze"], "packages_extra": identity["environment"]["packages_not_in_phase2_freeze"]}, indent=1, default=str))
        return
    result = {
        "case_id": args.case_id,
        "experiment": expectations["experiment"],
        "environment_name": env,
        "preregistration_note": prereg.get("note"),
        "phase3_plan_sha256": plan_sha,
        "python": platform.python_version(),
        "executable": sys.executable,
        "identity": identity,
        "selection_before_batches": selection_record(),
        "execution_order": "per batch in plan order: torchcodec upstream, torchcodec replay, pyav upstream, pyav window replay; default-backend attempt after all batches",
    }
    if not identity["ok"]:
        result["status"] = "not_run_identity"
    else:
        import av

        time_bases: dict[str, list[int]] = {}
        batches = []
        for batch in record["plan"]["batches"]:
            if batch["video_path"] not in time_bases:
                with av.open(str(root / batch["video_path"])) as container:
                    tb = container.streams.video[0].time_base
                    time_bases[batch["video_path"]] = [tb.numerator, tb.denominator]
            batches.append({**batch, **run_batch(batch, root, time_bases[batch["video_path"]])})
        result["time_bases"] = time_bases
        result["batches"] = batches
        first = record["plan"]["batches"][0]
        result["default_backend_attempt"] = default_backend_attempt(root / first["video_path"], first["queries"][0]["video_query_s"])
        result["status"] = "executed"
    result["mapped_libraries_at_end"] = mapped_libraries()
    out = HERE / "evidence" / f"{case_key(args.case_id)}_torchcodec.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for batch in result.get("batches", []):
        key = f"{batch['query_class']}: tc={batch['torchcodec']['outcome']} pyav={batch['pyav_control']['outcome']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "checks": identity["checks"], "outcomes": counts, "evidence": str(out)}, indent=1))


if __name__ == "__main__":
    main()

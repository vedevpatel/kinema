"""Run the frozen query plan through LeRobot's video decoder (direct decoder).

This is a direct-decoder experiment. It does not construct, patch or subclass
LeRobotDataset, and it does not change any source file or schema. It calls the
installed upstream `decode_video_frames(..., backend="pyav")` with the exact
query lists in plan/<case>.json and records the outcome of each batch.

Run with the case's isolated environment Python after preregistration:

    /tmp/kinema-phase2-v3/bin/python -I shifted-queries/2026-11-22/probe.py r1:c1:f0

The script refuses to run if any preregistered file hash has changed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402
from lerobot.datasets.utils import check_delta_timestamps, get_delta_indices, hf_transform_to_torch  # noqa: E402
from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec  # noqa: E402

from plan import REPO, case_root, load_expectations  # noqa: E402
from queryplan import TOLERANCE_S, sha256_file  # noqa: E402


def check_preregistration() -> dict:
    prereg = json.loads((HERE / "preregistration.json").read_text())
    changed = [p for p, digest in prereg["sha256"].items() if sha256_file(HERE / p) != digest]
    if changed:
        raise SystemExit(f"preregistered files changed after freeze: {changed}")
    return prereg


def replay_window(path: Path, queries: list[float]) -> dict:
    """Same VideoReader window as decode_video_frames_torchvision, keeping full pts."""
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
        out.update(per_query=[None] * len(queries), within_tolerance=False, frames=None)
        return out
    dist = torch.cdist(torch.tensor(queries)[:, None], torch.tensor(loaded_pts)[:, None], p=1)
    min_, argmin_ = dist.min(1)
    out["per_query"] = [
        {"returned_pts": loaded_pts[int(i)], "decoder_distance_s": float(d), "exact_difference_s": queries[k] - loaded_pts[int(i)], "within_tolerance": bool(d < TOLERANCE_S)}
        for k, (d, i) in enumerate(zip(min_, argmin_, strict=True))
    ]
    out["within_tolerance"] = bool((min_ < TOLERANCE_S).all())
    out["frames"] = torch.stack([loaded_frames[int(i)] for i in argmin_]).type(torch.float32) / 255
    return out


def run_batch(batch: dict, root: Path) -> dict:
    path = root / batch["video_path"]
    queries = [q["video_query_s"] for q in batch["queries"]]
    result = {"upstream": {}, "replay": {}}
    started = time.perf_counter()
    frames = None
    try:
        frames = decode_video_frames(path, queries, TOLERANCE_S, "pyav")
        result["upstream"] = {"outcome": "ok", "shape": list(frames.shape)}
    except AssertionError as exc:
        kind = "tolerance_error" if "violate the tolerance" in str(exc) else "other_error"
        result["upstream"] = {"outcome": kind, "exception": "AssertionError", "message": str(exc)[:2000]}
    except Exception as exc:  # recorded, never retried
        result["upstream"] = {"outcome": "other_error", "exception": type(exc).__name__, "message": str(exc)[:2000], "traceback": traceback.format_exc()[-4000:]}
    result["upstream"]["seconds"] = round(time.perf_counter() - started, 3)
    try:
        replay = replay_window(path, queries)
        replay_frames = replay.pop("frames")
        result["replay"] = replay
        expected = "ok" if replay["within_tolerance"] else "tolerance_error"
        consistent = result["upstream"]["outcome"] == expected
        if consistent and frames is not None:
            consistent = replay_frames is not None and torch.equal(frames, replay_frames)
            result["replay"]["frames_equal_upstream"] = consistent
    except Exception as exc:
        result["replay"] = {"error": type(exc).__name__, "message": str(exc)[:500]}
        consistent = False
    result["replay_consistent"] = consistent
    result["outcome"] = result["upstream"]["outcome"] if consistent else "harness_discrepancy"
    return result


def loader_timestamp_conformance(case: dict, plan: dict, root: Path) -> dict:
    """Check queryplan.loader_timestamp against upstream hf_transform_to_torch."""
    table = pq.read_table(root / case["data_path"], columns=["episode_index", "timestamp"])
    values = [t for e, t in zip(table["episode_index"].to_pylist(), table["timestamp"].to_pylist(), strict=True) if e == case["episode"]]
    tensors = hf_transform_to_torch({"timestamp": values})["timestamp"]
    mismatches = 0
    checked = 0
    for batch in plan["batches"]:
        if batch["query_class"] != "loader_derived":
            continue
        rows = [q["row"] for q in batch["queries"]]
        upstream = torch.stack([tensors[r] for r in rows]).tolist()
        planned = [q["loader_timestamp"] for q in batch["queries"]]
        checked += 1
        mismatches += upstream != planned
    out = {"method": "upstream hf_transform_to_torch on the pyarrow Python values of the episode timestamp column", "tensor_dtype": str(tensors[0].dtype), "batches_checked": checked, "mismatches": mismatches}
    if case["env"] == "v2":
        # 0.3.3 load_hf_dataset(episodes=[...]) is load_dataset("parquet", data_files=[...]) + set_transform.
        from datasets import load_dataset

        with tempfile.TemporaryDirectory() as cache:
            hf = load_dataset("parquet", data_files=[str(root / case["data_path"])], split="train", cache_dir=cache)
            hf.set_transform(hf_transform_to_torch)
            replay_mismatch = 0
            for batch in plan["batches"]:
                if batch["query_class"] == "loader_derived":
                    rows = [q["row"] for q in batch["queries"]]
                    replay_mismatch += torch.stack(hf.select(rows)["timestamp"]).tolist() != [q["loader_timestamp"] for q in batch["queries"]]
        out["component_replay"] = {"method": "datasets.load_dataset('parquet', data_files=[episode file]) + set_transform(hf_transform_to_torch) + select(rows)['timestamp'], as 0.3.3 _get_query_timestamps", "mismatches": replay_mismatch}
    else:
        out["component_replay"] = {"method": "not run: 0.4.0 load_hf_dataset casts to the info.json features, which phase 2 showed fail on dtype 'language'; not bypassed"}
    return out


def delta_conformance(expectations: dict, fps: float) -> dict:
    """Upstream check_delta_timestamps/get_delta_indices on each config's delta_timestamps."""
    out = {}
    for name, config in expectations["offset_selection"]["configs"].items():
        indices = config["observation_delta_indices"]
        if indices is None:
            out[name] = {"delta_timestamps": None}
            continue
        delta_ts = {"observation.images.x": [i / fps for i in indices]}
        out[name] = {
            "delta_timestamps": delta_ts["observation.images.x"],
            "check_delta_timestamps": check_delta_timestamps(delta_ts, fps, TOLERANCE_S, raise_value_error=False),
            "get_delta_indices": get_delta_indices(delta_ts, fps)["observation.images.x"],
        }
    return out


def environment_record(env: str, expectations: dict) -> dict:
    packages_file = REPO / expectations["environments"][env]["packages_file"]
    pinned = dict(line.strip().split("==", 1) for line in packages_file.read_text().splitlines() if "==" in line)
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    norm = {k.lower().replace("_", "-"): v for k, v in pinned.items()}
    differ = {k: {"pinned": v, "installed": installed.get(k)} for k, v in norm.items() if installed.get(k) != v}
    extra = sorted(set(installed) - set(norm))
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "lerobot": importlib.metadata.version("lerobot"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "av": importlib.metadata.version("av"),
        "torch_default_dtype": str(torch.get_default_dtype()),
        "packages_file": str(packages_file.relative_to(REPO)),
        "packages_differing_from_phase2_freeze": differ,
        "packages_not_in_phase2_freeze": extra,
    }


def default_backend_attempt(path: Path, query: float) -> dict:
    out = {"get_safe_default_codec": get_safe_default_codec()}
    try:
        frames = decode_video_frames(path, [query], TOLERANCE_S, None)
        out.update(outcome="ok", shape=list(frames.shape))
    except Exception as exc:
        out.update(outcome="dependency_error" if "libtorchcodec" in str(exc) or "FFmpeg" in str(exc) else "other_error", exception=type(exc).__name__, message=str(exc)[:1500])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    prereg = check_preregistration()
    expectations = load_expectations()
    case = {c["case_id"]: c for c in expectations["cases"]}[args.case_id]
    record = json.loads((HERE / "plan" / f"{args.case_id.replace(':', '_')}.json").read_text())
    root = case_root(args.case_id)
    result = {
        "case_id": args.case_id,
        "experiment": "direct decoder: upstream decode_video_frames(path, plan queries, tolerance_s=1e-4, backend='pyav'); LeRobotDataset not constructed",
        "preregistration_commit_note": prereg.get("note"),
        "environment": environment_record(case["env"], expectations),
    }
    rehash = {e["path"]: sha256_file(root / e["path"]) == e["sha256"] for e in record["inputs"]["entries"]}
    result["inputs_rehashed_ok"] = all(rehash.values())
    if not record["identity_ok"] or not result["inputs_rehashed_ok"]:
        result.update(status="not_run_identity", rehash=rehash)
    else:
        plan = record["plan"]
        result["delta_conformance"] = delta_conformance(expectations, record["data_facts"]["declared_fps"])
        result["loader_timestamp_conformance"] = loader_timestamp_conformance(case, plan, root)
        first = plan["batches"][0]
        result["default_backend_attempt"] = default_backend_attempt(root / first["video_path"], first["queries"][0]["video_query_s"])
        batches = []
        for batch in plan["batches"]:
            batches.append({**batch, **run_batch(batch, root)})
        result["batches"] = batches
        result["status"] = "executed"
    out = HERE / "evidence" / f"{args.case_id.replace(':', '_')}_probe.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for batch in result.get("batches", []):
        counts[batch["outcome"]] = counts.get(batch["outcome"], 0) + 1
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "outcomes": counts, "evidence": str(out)}))


if __name__ == "__main__":
    main()

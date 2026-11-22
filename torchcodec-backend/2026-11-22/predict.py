"""Source-model predictions for every frozen phase-3 batch, written before execution.

    /tmp/kinema-phase2-v2/bin/python -I -B torchcodec-backend/2026-11-22/predict.py r71:c1:f1

The stream facts come from PyAV demuxing (packet pts, r_frame_rate, nb_frames,
time base); no frame is decoded and torchcodec is not imported. The rule
applied is tcmodel.predict_batch, which restates LeRobot's
decode_video_frames_torchcodec and torchcodec v0.5.0's approximate-mode index
handling. The predictions are not decision criteria. The report compares them
with what the installed binary did.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASE3 = REPO / "shifted-queries" / "2026-11-22"
sys.path.insert(0, str(HERE))

import av  # noqa: E402

import tcmodel  # noqa: E402
from common import case_root, load_phase3_plan, sha256_file  # noqa: E402


def stream_facts(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = sorted(int(p.pts) for p in container.demux(stream) if p.pts is not None)
        return {
            "average_fps_from_header": float(stream.base_rate),
            "r_frame_rate": str(stream.base_rate),
            "num_frames_from_header": int(stream.frames) if stream.frames else None,
            "time_base": [stream.time_base.numerator, stream.time_base.denominator],
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "codec": stream.codec_context.name,
            "packets_with_pts": len(pts),
            "first_pts_ticks": pts[0] if pts else None,
            "last_pts_ticks": pts[-1] if pts else None,
            "sorted_pts": pts,
        }


def predict_case(case_id: str) -> dict:
    record, plan_sha = load_phase3_plan(case_id)
    root = case_root(case_id)
    rehash = {e["path"]: sha256_file(root / e["path"]) == e["sha256"] for e in record["inputs"]["entries"]}
    if not all(rehash.values()):
        raise SystemExit(f"input hash mismatch: {rehash}")
    streams = {}
    batches = []
    for batch in record["plan"]["batches"]:
        path = batch["video_path"]
        if path not in streams:
            streams[path] = stream_facts(root / path)
        queries = [q["video_query_s"] for q in batch["queries"]]
        prediction = tcmodel.predict_batch(queries, streams[path])
        batches.append({
            "query_class": batch["query_class"],
            "camera": batch["camera"],
            "anchor": batch["anchor"],
            "config": batch["config"],
            "predicted_outcome": prediction["outcome"],
            "per_query": prediction["per_query"],
        })
    public_streams = {p: {k: v for k, v in s.items() if k != "sorted_pts"} for p, s in streams.items()}
    return {
        "case_id": case_id,
        "note": "Predicted from source reading and demuxed packet timestamps before any torchcodec decode. Not a decision criterion.",
        "phase3_plan_sha256": plan_sha,
        "inputs_rehashed_ok": True,
        "streams": public_streams,
        "batches": batches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    out = HERE / "predictions" / f"{args.case_id.replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    result = predict_case(args.case_id)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for b in result["batches"]:
        key = f"{b['query_class']}:{b['predicted_outcome']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"case_id": args.case_id, "predicted": counts, "out": str(out)}))


if __name__ == "__main__":
    main()

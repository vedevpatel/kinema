"""Probe pinned episode videos at raw timestamps with LeRobot's decoder."""

import argparse
import json
import traceback

import pyarrow.parquet as pq
from lerobot.datasets.video_utils import decode_video_frames

from prepare import HERE, case_dir


CAMERAS = {
    "r94:c1:f1": ["observation.images.front", "observation.images.high_opposite", "observation.images.wrist_left", "observation.images.wrist_right"],
    "r71:c1:f1": ["observation.images.primary", "observation.images.secondary", "observation.images.wrist"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", choices=CAMERAS)
    args = parser.parse_args()
    root = case_dir(args.case_id)
    timestamps = pq.read_table(root / "data/chunk-000/episode_000000.parquet", columns=["timestamp"]).column("timestamp").to_pylist()
    result = {"case_id": args.case_id, "operation": "lerobot.datasets.video_utils.decode_video_frames(path, [raw_timestamp], tolerance_s=1e-4, backend='pyav')", "rows": len(timestamps), "probes": []}
    for camera in CAMERAS[args.case_id]:
        path = root / f"videos/chunk-000/{camera}/episode_000000.mp4"
        for label, index in (("first", 0), ("middle", len(timestamps) // 2), ("last", len(timestamps) - 1)):
            timestamp = timestamps[index]
            probe = {"camera": camera, "position": label, "row_index": index, "timestamp": timestamp, "path": str(path)}
            try:
                frames = decode_video_frames(path, [timestamp], 1e-4, "pyav")
                probe.update(outcome="ok", shape=list(frames.shape))
            except Exception as exc:
                probe.update(outcome="error", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            result["probes"].append(probe)
    output = HERE / "evidence" / f"{args.case_id.replace(':', '_')}_decoder.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": args.case_id, "probes": [{k: v for k, v in p.items() if k not in ("traceback", "message", "path")} for p in result["probes"]], "evidence": str(output)}))


if __name__ == "__main__":
    main()

"""Isolate FineART v3 video lookup after its custom dtype blocks loading."""

import json
import traceback

import numpy as np
import pyarrow.parquet as pq
from lerobot.datasets.video_utils import decode_video_frames

from prepare import HERE, case_dir


def main() -> None:
    root = case_dir("r1:c1:f0")
    table = pq.read_table(root / "data/chunk-000/file-210.parquet", columns=["episode_index", "timestamp"])
    selected = np.asarray(table["episode_index"]) == 40455
    timestamps = np.asarray(table["timestamp"])[selected]
    worst = int(np.argmax(np.abs(np.diff(timestamps) - 1 / 30)))
    probes = []
    for camera in ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"):
        path = root / f"videos/observation.images.{camera}/chunk-040/file-455.mp4"
        for label, index in (("first", 0), ("middle", len(timestamps) // 2), ("worst_grid", worst), ("last", len(timestamps) - 1)):
            timestamp = float(timestamps[index])
            probe = {"camera": camera, "position": label, "row_index": index, "timestamp": timestamp, "path": str(path)}
            try:
                frames = decode_video_frames(path, [timestamp], 1e-4, "pyav")
                probe.update(outcome="ok", shape=list(frames.shape))
            except Exception as exc:
                probe.update(outcome="error", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            probes.append(probe)
    result = {"case_id": "r1:c1:f0", "episode_rows": len(timestamps), "worst_grid_deviation_s": float(np.max(np.abs(np.diff(timestamps) - 1 / 30))), "operation": "LeRobot 0.4.0 decode_video_frames(... tolerance_s=1e-4, backend='pyav')", "probes": probes}
    output = HERE / "evidence/r1_c1_f0_decoder.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": result["case_id"], "episode_rows": result["episode_rows"], "worst_grid_deviation_s": result["worst_grid_deviation_s"], "probes": [{k: v for k, v in p.items() if k not in ("message", "traceback", "path")} for p in probes]}))


if __name__ == "__main__":
    main()

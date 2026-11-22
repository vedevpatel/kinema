"""Isolate hand_right lookup after an earlier camera fails in __getitem__."""

import json
import traceback

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from prepare import HERE, case_dir


def main() -> None:
    root = case_dir("r261:c2:f4")
    dataset = LeRobotDataset("Traly/AgiBotWorld-Alpha-lerobot", root=root, episodes=[59], revision="b38fd11acfe54f06b8b73b217f1b7c5c89b2c552", download_videos=False, video_backend="pyav")
    rows = [i for i, value in enumerate(dataset.hf_dataset["episode_index"]) if int(value) == 59]
    result = {"case_id": "r261:c2:f4", "episode": 59, "matching_rows": len(rows), "operation": "LeRobotDataset._query_videos({'observation.images.hand_right': [timestamp]}, 59) at first, middle and last matching row", "probes": []}
    for label, index in (("first", rows[0]), ("middle", rows[len(rows) // 2]), ("last", rows[-1])):
        timestamp = float(dataset.hf_dataset[index]["timestamp"])
        probe = {"position": label, "index": index, "timestamp": timestamp}
        try:
            frame = dataset._query_videos({"observation.images.hand_right": [timestamp]}, 59)["observation.images.hand_right"]
            probe.update(outcome="ok", shape=list(frame.shape))
        except Exception as exc:
            probe.update(outcome="error", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        result["probes"].append(probe)
    output = HERE / "evidence/r261_c2_f4_hand_right.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": result["case_id"], "matching_rows": result["matching_rows"], "probes": [{k: v for k, v in p.items() if k not in ("message", "traceback")} for p in result["probes"]]}))


if __name__ == "__main__":
    main()

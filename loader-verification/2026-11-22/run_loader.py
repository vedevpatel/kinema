"""Exercise an affected episode with a pinned upstream LeRobot release.

No kinema modules are imported. The caller supplies a local tree populated by
prepare.py; failures to obtain dependencies or inputs are recorded separately.
"""

import argparse
import importlib.metadata
import json
import platform
import traceback
from prepare import HERE, case_dir, source_parts


EPISODES = {
    "r75:c12:f4": 2035,
    "r36:c12:f1": 129,
    "r1:c1:f0": 40455,
    "r261:c2:f4": 59,
}


def run(case: dict) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    case_id = case["case_id"]
    episode = EPISODES.get(case_id, 0)
    repo, _ = source_parts(case["dataset"])
    root = case_dir(case_id)
    result = {
        "case_id": case_id,
        "repo": repo,
        "revision": case["revision"],
        "episode": episode,
        "root": str(root),
        "operation": "LeRobotDataset(... episodes=[episode], revision=commit, video_backend='pyav'); select matching rows in hf_dataset; __getitem__ at first, middle, last",
        "environment": {
            "python": platform.python_version(),
            "lerobot": importlib.metadata.version("lerobot"),
            "datasets": importlib.metadata.version("datasets"),
            "torch": importlib.metadata.version("torch"),
            "av": importlib.metadata.version("av"),
        },
        "steps": [],
    }
    try:
        dataset = LeRobotDataset(repo, root=root, episodes=[episode], revision=case["revision"], download_videos=False, video_backend="pyav")
        result["steps"].append({"step": "construct", "outcome": "ok", "length": len(dataset), "video_keys": list(dataset.meta.video_keys), "feature_keys": list(dataset.features)})
    except Exception as exc:
        result["steps"].append({"step": "construct", "outcome": "error", "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        return result
    try:
        # This reads the loader's selected frame table; it is not a kinema check.
        matches = [i for i, value in enumerate(dataset.hf_dataset["episode_index"]) if int(value) == episode]
        result["steps"].append({"step": "episode_index_scan", "outcome": "ok", "matching_rows": len(matches), "selected_table_rows": len(dataset.hf_dataset)})
    except Exception as exc:
        result["steps"].append({"step": "episode_index_scan", "outcome": "error", "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        return result
    if matches:
        for label, index in (("first", matches[0]), ("middle", matches[len(matches) // 2]), ("last", matches[-1])):
            try:
                item = dataset[index]
                result["steps"].append({"step": "item_" + label, "outcome": "ok", "index": index, "keys": list(item), "timestamp": float(item["timestamp"]) if "timestamp" in item else None})
            except Exception as exc:
                result["steps"].append({"step": "item_" + label, "outcome": "error", "index": index, "exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    cases = {case["case_id"]: case for case in json.loads((HERE / "expectations.json").read_text())["cases"]}
    result = run(cases[args.case_id])
    output = HERE / "evidence" / f"{args.case_id.replace(':', '_')}_loader{args.suffix}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": args.case_id, "steps": [{k: v for k, v in step.items() if k not in ("keys", "traceback")} for step in result["steps"]], "evidence": str(output)}))


if __name__ == "__main__":
    main()

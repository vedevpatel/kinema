"""Describe each probed video stream (post-hoc and descriptive, not preregistered).

    /tmp/kinema-phase2-v2/bin/python -I shifted-queries/2026-11-22/video_facts.py r71:c1:f1

Demuxes packets with PyAV to record the stream rate, packet count and first/last
pts. Nothing here decides an outcome; the report uses it only to describe
timelines next to the decoder results.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av  # noqa: E402

from plan import case_root, load_expectations  # noqa: E402


def facts(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        pts = sorted(float(p.pts * stream.time_base) for p in container.demux(stream) if p.pts is not None)
        return {
            "average_rate": float(stream.average_rate) if stream.average_rate else None,
            "time_base": str(stream.time_base),
            "frames": len(pts),
            "first_pts": pts[0] if pts else None,
            "last_pts": pts[-1] if pts else None,
            "median_spacing_s": sorted(b - a for a, b in zip(pts, pts[1:]))[len(pts) // 2] if len(pts) > 2 else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    case = {c["case_id"]: c for c in load_expectations()["cases"]}[args.case_id]
    root = case_root(args.case_id)
    out = {camera: facts(root / case["video_path_template"].format(camera=camera)) for camera in case["cameras"]}
    path = HERE / "evidence" / f"{args.case_id.replace(':', '_')}_video_facts.json"
    path.write_text(json.dumps({"case_id": args.case_id, "note": "post-hoc descriptive record, not preregistered", "videos": out}, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()

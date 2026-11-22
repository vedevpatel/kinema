"""Post-hoc, descriptive: codec profile and B-frame reordering of each probed stream.

    /tmp/kinema-phase2-v2/bin/python -I -B torchcodec-backend/2026-11-22/stream_facts_posthoc.py

Not preregistered and never used to decide an outcome. It records whether
the probed streams exercise decode-order/presentation-order reordering, which the
report's adversarial review discusses. Demux only; no frame is decoded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av  # noqa: E402

from common import case_root, load_expectations, load_phase3_plan  # noqa: E402


def facts(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        packets = [p.pts for p in container.demux(stream) if p.pts is not None]
        return {
            "codec": stream.codec_context.name,
            "profile": stream.codec_context.profile,
            "pix_fmt": stream.codec_context.pix_fmt,
            "has_b_frames": bool(stream.codec_context.has_b_frames),
            "packets": len(packets),
            "packets_with_pts_below_previous": sum(1 for a, b in zip(packets, packets[1:]) if b < a),
            "width": stream.codec_context.width,
            "width_multiple_of_32": stream.codec_context.width % 32 == 0,
        }


def main() -> None:
    out = {"note": "post-hoc descriptive record, not preregistered", "cases": {}}
    for case in load_expectations()["cases"]:
        record, _ = load_phase3_plan(case["case_id"])
        root = case_root(case["case_id"])
        paths = sorted({b["video_path"] for b in record["plan"]["batches"]})
        out["cases"][case["case_id"]] = {p: facts(root / p) for p in paths}
    target = HERE / "evidence" / "stream_facts_posthoc.json"
    target.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: {"streams": len(v), "all_have_b_frames": all(x["has_b_frames"] for x in v.values()), "all_widths_multiple_of_32": all(x["width_multiple_of_32"] for x in v.values())} for c, v in out["cases"].items()}))


if __name__ == "__main__":
    main()

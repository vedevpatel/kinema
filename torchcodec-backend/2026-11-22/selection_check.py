"""Record default-backend selection and torchcodec import behaviour on a synthetic video.

    /tmp/kinema-phase2-v2/bin/python -I -B torchcodec-backend/2026-11-22/selection_check.py tc-v2 no_ffmpeg
    LD_LIBRARY_PATH=... /tmp/kinema-phase2-v2/bin/python -I -B torchcodec-backend/2026-11-22/selection_check.py tc-v2 ffmpeg_aliases

The video is generated in a temporary directory, so no pinned input is read.
Also records the installed LeRobot source lines that make get_safe_default_codec()
the dataset and config default. Each run is scoped to the release installed in
the interpreter it runs in.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import tcmodel  # noqa: E402

SOURCE_LINES = {
    "datasets/video_utils.py": ["def get_safe_default_codec():", 'if importlib.util.find_spec("torchcodec"):', 'return "torchcodec"', "backend = get_safe_default_codec()"],
    "datasets/lerobot_dataset.py": ["self.video_backend = video_backend if video_backend else get_safe_default_codec()"],
    "configs/default.py": ["video_backend: str = field(default_factory=get_safe_default_codec)"],
}


def write_video(path: Path, fps: int = 10, frames: int = 20) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
    for i in range(frames):
        image = np.full((32, 64, 3), i * 10, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def attempt(fn) -> dict:
    try:
        value = fn()
        return {"outcome": "ok", "value": value}
    except Exception as exc:
        return {"outcome": tcmodel.classify_exception(type(exc).__name__, str(exc)), "exception": type(exc).__name__, "message": str(exc)[:1200]}


def source_lines(root: Path) -> dict:
    out = {}
    for relative, needles in SOURCE_LINES.items():
        text = (root / relative).read_text()
        out[relative] = {needle: needle in text for needle in needles}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=["tc-v2", "tc-v3"])
    parser.add_argument("variant", choices=["no_ffmpeg", "ffmpeg_aliases"])
    args = parser.parse_args()
    if (args.variant == "no_ffmpeg") == bool(os.environ.get("LD_LIBRARY_PATH")):
        raise SystemExit("variant does not match LD_LIBRARY_PATH")
    from lerobot.datasets.video_utils import decode_video_frames, get_safe_default_codec

    lerobot_root = Path(importlib.util.find_spec("lerobot").submodule_search_locations[0])
    result = {
        "env": args.env,
        "variant": args.variant,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "lerobot": importlib.metadata.version("lerobot"),
        "torchcodec_distribution": importlib.metadata.version("torchcodec"),
        "find_spec_torchcodec": importlib.util.find_spec("torchcodec") is not None,
        "get_safe_default_codec": get_safe_default_codec(),
        "source_lines_present": source_lines(lerobot_root),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "synthetic.mp4"
        write_video(path)
        result["import_torchcodec_decoders"] = attempt(lambda: __import__("torchcodec.decoders", fromlist=["VideoDecoder"]).__name__)
        result["decode_default_backend"] = attempt(lambda: list(decode_video_frames(path, [0.5], tcmodel.TOLERANCE_S, None).shape))
        result["decode_explicit_pyav"] = attempt(lambda: list(decode_video_frames(path, [0.5], tcmodel.TOLERANCE_S, "pyav").shape))
        if result["decode_default_backend"]["outcome"] == "ok":
            result["default_equals_explicit_torchcodec"] = attempt(lambda: bool(torch.equal(decode_video_frames(path, [0.5], tcmodel.TOLERANCE_S, None), decode_video_frames(path, [0.5], tcmodel.TOLERANCE_S, "torchcodec"))))
    out = HERE / "evidence" / f"selection_{args.env}_{args.variant}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("env", "variant", "get_safe_default_codec")} | {k: result[k]["outcome"] for k in ("import_torchcodec_decoders", "decode_default_backend", "decode_explicit_pyav")}))


if __name__ == "__main__":
    main()

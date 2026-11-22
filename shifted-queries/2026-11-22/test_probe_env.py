"""Synthetic-video checks for probe.py; run inside a phase-2 environment.

    /tmp/kinema-phase2-v2/bin/python -I -m unittest discover -s shifted-queries/2026-11-22 -p 'test_probe_env.py'

The video is generated here, so these checks never touch the pinned inputs.
"""

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av  # noqa: E402
import numpy as np  # noqa: E402

from probe import replay_window, run_batch  # noqa: E402


def write_video(path: Path, fps: int = 10, frames: int = 20) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
    for i in range(frames):
        image = np.full((32, 32, 3), i * 10, dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


class ProbeHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_video(cls.root / "v.mp4")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def batch(self, queries):
        return {"video_path": "v.mp4", "queries": [{"video_query_s": q} for q in queries]}

    def test_on_grid_batch_decodes_and_replay_matches(self):
        result = run_batch(self.batch([0.5, 0.6, 0.7]), self.root)
        self.assertEqual(result["outcome"], "ok")
        self.assertTrue(result["replay"]["frames_equal_upstream"])
        self.assertEqual([round(q["returned_pts"], 6) for q in result["replay"]["per_query"]], [0.5, 0.6, 0.7])

    def test_half_period_offset_is_a_tolerance_error(self):
        result = run_batch(self.batch([0.55]), self.root)
        self.assertEqual(result["upstream"]["outcome"], "tolerance_error")
        self.assertEqual(result["outcome"], "tolerance_error")
        self.assertTrue(result["replay_consistent"])

    def test_one_bad_query_fails_the_whole_batch(self):
        result = run_batch(self.batch([0.5, 0.55]), self.root)
        self.assertEqual(result["outcome"], "tolerance_error")
        self.assertEqual([q["within_tolerance"] for q in result["replay"]["per_query"]], [True, False])

    def test_query_far_past_the_end_is_recorded_not_raised(self):
        result = run_batch(self.batch([1.784e9]), self.root)
        self.assertIn(result["outcome"], {"tolerance_error", "other_error", "harness_discrepancy"})
        self.assertNotEqual(result["outcome"], "ok")
        window = replay_window(self.root / "v.mp4", [1.784e9])
        self.assertFalse(window["within_tolerance"])


if __name__ == "__main__":
    unittest.main()

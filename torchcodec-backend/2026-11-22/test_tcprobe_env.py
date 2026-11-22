"""Synthetic-video checks for tcprobe.py; run inside a phase-2 venv with the FFmpeg aliases.

    LD_LIBRARY_PATH=/tmp/kinema-tc-ffmpeg-v2:/tmp/kinema-phase2-v2/lib/python3.11/site-packages/av.libs \
        /tmp/kinema-phase2-v2/bin/python -I -B -m unittest discover -s torchcodec-backend/2026-11-22 -p 'test_tcprobe_env.py'

Every video is generated here, so these checks never read the pinned inputs.
"""

import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import av  # noqa: E402
import numpy as np  # noqa: E402

import tcmodel  # noqa: E402
from predict import stream_facts  # noqa: E402
from tcprobe import run_batch, tc_replay  # noqa: E402

TB = [1, 1000]


def write_video(path: Path, pts_ms: list[int], rate: int = 10) -> None:
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=rate)
    stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
    stream.time_base = Fraction(1, 1000)
    stream.codec_context.time_base = Fraction(1, 1000)
    for i, pts in enumerate(pts_ms):
        image = np.full((32, 64, 3), (i * 11) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = pts
        frame.time_base = Fraction(1, 1000)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


class TorchcodecHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        write_video(cls.root / "grid.mp4", [i * 100 for i in range(20)])
        cls.facts = {"grid.mp4": stream_facts(cls.root / "grid.mp4")}
        for shift in (330, 350):
            pts = [i * 100 for i in range(20)]
            pts[3] = shift  # one frame moved off the 10 fps grid
            name = f"irregular_{shift}.mp4"
            write_video(cls.root / name, pts)
            cls.facts[name] = stream_facts(cls.root / name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def batch(self, name, queries):
        return {"video_path": name, "queries": [{"video_query_s": q} for q in queries]}

    def run_and_predict(self, name, queries):
        result = run_batch(self.batch(name, queries), self.root, TB)
        prediction = tcmodel.predict_batch(queries, self.facts[name])
        return result, prediction

    def test_on_grid_batch_agrees_across_backends_and_with_model(self):
        result, prediction = self.run_and_predict("grid.mp4", [0.5, 0.6, 0.7])
        self.assertEqual(result["torchcodec"]["outcome"], "ok")
        self.assertEqual(result["pyav_control"]["outcome"], "ok")
        self.assertTrue(result["torchcodec"]["replay"]["frames_equal_upstream"])
        self.assertTrue(result["pyav_control"]["replay"]["frames_equal_upstream"])
        self.assertEqual(prediction["outcome"], "ok")
        self.assertTrue(all(q["same_pts"] for q in result["comparison"]["per_query"]))
        self.assertEqual([round(q["selected_pts_s"], 6) for q in result["torchcodec"]["replay"]["per_query"]], [0.5, 0.6, 0.7])
        self.assertIn("pixels", result["comparison"]["per_query"][0])

    def test_half_period_query_rejected_by_both(self):
        result, prediction = self.run_and_predict("grid.mp4", [0.55])
        self.assertEqual(result["torchcodec"]["outcome"], "tolerance_error")
        self.assertEqual(result["pyav_control"]["outcome"], "tolerance_error")
        self.assertEqual(prediction["outcome"], "tolerance_error")

    def test_negative_query_wraps_to_the_end_then_fails_tolerance(self):
        result, prediction = self.run_and_predict("grid.mp4", [-0.3])
        row = result["torchcodec"]["replay"]["per_query"][0]
        self.assertEqual((row["raw_index"], row["index"]), (-3, 17))
        self.assertAlmostEqual(row["returned_pts_s"], 1.7, places=6)
        self.assertEqual(result["torchcodec"]["outcome"], "tolerance_error")
        self.assertEqual(prediction["outcome"], "tolerance_error")
        self.assertTrue(prediction["per_query"][0]["wrapped"])

    def test_far_query_is_an_index_error_not_a_tolerance_error(self):
        result, prediction = self.run_and_predict("grid.mp4", [1.784e9])
        self.assertEqual(result["torchcodec"]["outcome"], "index_error")
        self.assertEqual(prediction["outcome"], "index_error")
        self.assertNotEqual(result["pyav_control"]["outcome"], "ok")

    def test_irregular_streams_follow_the_model(self):
        # FFmpeg guesses r_frame_rate from the timestamps; torchcodec indexes with it.
        for name, query in (("irregular_330.mp4", 0.33), ("irregular_350.mp4", 0.35)):
            result, prediction = self.run_and_predict(name, [query])
            self.assertEqual(result["pyav_control"]["outcome"], "ok", name)
            self.assertEqual(result["torchcodec"]["outcome"], prediction["outcome"], name)
            row = result["torchcodec"]["replay"]["per_query"][0]
            self.assertEqual(row["raw_index"], prediction["per_query"][0]["raw_index"], name)
            if prediction["outcome"] == "ok":
                self.assertEqual(row["returned_pts_s"], prediction["per_query"][0]["returned_pts_s"], name)

    def test_off_grid_frame_found_by_pyav_is_rejected_by_index_rule(self):
        facts = self.facts["irregular_330.mp4"]
        self.assertGreater(round(0.33 * facts["average_fps_from_header"]), facts["num_frames_from_header"] - 1)
        result, prediction = self.run_and_predict("irregular_330.mp4", [0.33])
        self.assertEqual(result["pyav_control"]["outcome"], "ok")
        self.assertEqual(result["torchcodec"]["outcome"], "index_error")
        self.assertEqual(prediction["outcome"], "index_error")

    def test_replay_metadata_is_header_only(self):
        replay, _ = tc_replay(self.root / "grid.mp4", [0.5])
        self.assertIsNone(replay["metadata"]["num_frames_from_content"])
        self.assertEqual(replay["metadata"]["average_fps"], self.facts["grid.mp4"]["average_fps_from_header"])
        self.assertEqual(replay["metadata"]["num_frames_from_header"], self.facts["grid.mp4"]["num_frames_from_header"])


if __name__ == "__main__":
    unittest.main()

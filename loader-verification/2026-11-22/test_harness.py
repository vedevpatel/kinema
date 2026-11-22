"""Offline checks for selection, pin verification and report consistency."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_report import build_report, evidence_command  # noqa: E402
from prepare import sha256, source_parts, verify_frozen  # noqa: E402


HERE = Path(__file__).resolve().parent


class HarnessTests(unittest.TestCase):
    def test_nested_source_path_and_hash(self):
        self.assertEqual(source_parts("repo/name//task_392"), ("repo/name", "task_392/"))
        self.assertEqual(source_parts("repo/name"), ("repo/name", ""))
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "input"
            file.write_bytes(b"pinned bytes")
            self.assertEqual(sha256(file), hashlib.sha256(b"pinned bytes").hexdigest())

    def test_frozen_hash_mismatch_is_rejected(self):
        case = {"dataset": "repo/name", "revision": "a" * 40, "pinned_inputs": {"info_sha256": "expected", "stats_sha256": None}}
        with self.assertRaisesRegex(ValueError, "Frozen info hash mismatch"):
            verify_frozen(case, [{"path": "meta/info.json", "remote_path": "meta/info.json", "sha256": "changed"}], [])

    def test_report_keeps_all_eleven_original_unresolved_cases(self):
        report = build_report()
        source = json.loads((HERE.parents[1] / "audits/2026-11-22/audit.json").read_text())
        original = {case["key"] for case in source["cases"] if case["label"] == "unresolved" and case["check"] in {"episode-file-pointers", "dropped-frames", "timestamp-grid", "deployment-readiness"}}
        self.assertEqual(len(original), 11)
        self.assertEqual({case["case_id"] for case in report["cases"]}, original)
        self.assertEqual({case["original_audit_classification"] for case in report["cases"]}, {"unresolved"})
        self.assertEqual(sum(case["classification"] == "confirmed" for case in report["cases"]), 4)
        self.assertEqual(sum(case["classification"] == "refuted" for case in report["cases"]), 3)
        self.assertEqual(sum(case["classification"] == "unresolved" for case in report["cases"]), 4)
        self.assertNotIn("r413:c2:f2", original)
        for case in report["cases"]:
            self.assertEqual(len(case["revision"]), 40)
            self.assertTrue(case["artifact_hashes"])
            self.assertTrue(case["calls"])

    def test_command_records_loader_version_and_probe(self):
        self.assertIn("kinema-phase2-v3-051", evidence_command("r1:c1:f0", "r1_c1_f0_loader_0.5.1.json", "0.5.1"))
        self.assertIn("decode_probe.py r71:c1:f1", evidence_command("r71:c1:f1", "r71_c1_f1_decoder.json", "0.3.3"))


if __name__ == "__main__":
    unittest.main()

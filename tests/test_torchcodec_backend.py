"""Offline checks for the torchcodec backend comparison (torchcodec-backend/2026-11-22).

No LeRobot, torch, torchcodec or video library is imported. These tests pin the
selection model to the cited upstream rules, and the committed artifacts to the
frozen phase-1/2/3 inputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "torchcodec-backend" / "2026-11-22"
PHASE3 = ROOT / "shifted-queries" / "2026-11-22"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PHASE3))

import common  # noqa: E402
import queryplan as qp  # noqa: E402
import tcmodel as tm  # noqa: E402

EXPECTATIONS = json.loads((HERE / "expectations.json").read_text())
SPEC = json.loads((HERE / "environment_spec.json").read_text())
CASES = [c["case_id"] for c in EXPECTATIONS["cases"]]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid_stream(n=20, fps=10.0, tb=(1, 1000)):
    return {"average_fps_from_header": fps, "num_frames_from_header": n, "time_base": list(tb), "sorted_pts": [round(i * tb[1] / (fps * tb[0])) for i in range(n)]}


# --- selection model ------------------------------------------------------------

def test_lerobot_index_uses_python_round_half_to_even():
    assert tm.lerobot_frame_index(0.25, 10.0) == 2
    assert tm.lerobot_frame_index(0.35, 10.0) == 4
    assert tm.lerobot_frame_index(1784331520.0, 30.0) == 53529945600


def test_cpp_round_is_half_away_from_zero():
    assert [tm.cpp_round(x) for x in (0.5, 1.5, 2.5, -0.5, -2.5, 2.4999)] == [1, 2, 3, -1, -3, 2]


def test_negative_index_wraps_once_then_validation_applies():
    assert tm.wrap_negative_index(-3, 20) == 17
    assert tm.wrap_negative_index(-25, 20) == -5
    assert tm.wrap_negative_index(4, 20) == 4
    assert tm.index_is_valid(19, 20) and not tm.index_is_valid(20, 20) and not tm.index_is_valid(-1, 20)
    assert tm.index_is_valid(10**12, None)


def test_cursor_and_first_frame_at_or_after():
    num, den = 1, 11520
    assert tm.approximate_cursor(3, 90.0, num, den) == 384
    assert tm.first_pts_at_or_after([0, 128, 256, 384], 300) == 384
    assert tm.first_pts_at_or_after([0, 128, 256, 384], 385) is None
    assert tm.pts_to_seconds(384, num, den) == pytest.approx(3 / 90)


def test_f32_distance_matches_phase3_decoder_distance():
    for q, p in ((0.7333333492279053, 0.7333333333), (299.38888, 299.3888888), (1784331520.0, 8.3)):
        assert tm.f32_distance(q, p) == qp.decoder_distance(q, p)


def test_predict_query_grid_wrap_index_and_end_of_stream():
    stream = grid_stream()
    assert tm.predict_query(0.5, stream)["returned_pts_s"] == pytest.approx(0.5)
    wrapped = tm.predict_query(-0.3, stream)
    assert (wrapped["raw_index"], wrapped["index"], wrapped["wrapped"]) == (-3, 17, True)
    assert wrapped["own_frame_distance_s"] == pytest.approx(2.0, abs=1e-6)
    assert tm.predict_query(5.0, stream)["outcome"] == "index_error"
    short = dict(stream, sorted_pts=stream["sorted_pts"][:10])
    assert tm.predict_query(1.5, short)["outcome"] == "end_of_stream_error"


def test_predict_batch_rejects_before_tolerance_in_ascending_index_order():
    stream = dict(grid_stream(), sorted_pts=grid_stream()["sorted_pts"][:10])
    assert tm.predict_batch([1.5, 5.0], stream)["outcome"] == "end_of_stream_error"
    assert tm.predict_batch([1.5, -5.0], stream)["outcome"] == "index_error"
    assert tm.predict_batch([0.5, 0.55], grid_stream())["outcome"] == "tolerance_error"
    assert tm.predict_batch([0.5, 0.6], grid_stream())["outcome"] == "ok"


def test_selection_is_argmin_over_every_frame_in_the_batch():
    # A query may take another query's frame when that one is closer.
    picks = tm.lerobot_selection([0.30, 0.34], [0.40, 0.33])
    assert [p for p, _ in picks] == [1, 1]


def test_exception_classes():
    assert tm.classify_exception("AssertionError", "One or several query timestamps unexpectedly violate the tolerance (...)") == "tolerance_error"
    assert tm.classify_exception("RuntimeError", "Invalid frame index=53529945600 for streamIndex=0; must be less than 437") == "index_error"
    assert tm.classify_exception("RuntimeError", "Requested next frame while there are no more frames left to decode.") == "end_of_stream_error"
    assert tm.classify_exception("RuntimeError", "Could not load libtorchcodec. Likely causes: ...") == "dependency_error"
    assert tm.classify_exception("ValueError", "something else") == "other_error"
    assert tm.classify_exception("RuntimeError", "violate the tolerance") == "other_error"


@pytest.mark.parametrize(
    ("counts", "verdict", "via"),
    [
        ({"ok": 3}, "contradicts_claim", []),
        ({"ok": 2, "tolerance_error": 1}, "supports_claim", ["tolerance_error"]),
        ({"index_error": 2, "other_error": 1}, "supports_claim", ["index_error"]),
        ({"ok": 2, "dependency_error": 1}, "unresolved", []),
        ({"ok": 2, "harness_discrepancy": 1}, "unresolved", []),
        ({}, "not_applicable", []),
    ],
)
def test_bucket_verdict(counts, verdict, via):
    full = dict.fromkeys(tm.OUTCOMES, 0) | counts
    assert tm.bucket_verdict(full) == {"verdict": verdict, "via": via}


def test_frame_difference_reports_equality_and_magnitude():
    a = np.zeros((3, 2, 2), dtype=np.uint8)
    b = a.copy()
    assert tm.frame_difference(a, b)["equal"]
    b[0, 0, 0] = 5
    diff = tm.frame_difference(a, b)
    assert (diff["equal"], diff["max_abs"]) == (False, 5)
    assert diff["fraction_values_differing"] == pytest.approx(1 / 12)


# --- frozen inputs and environment spec ------------------------------------------

def test_frozen_phase1_to_3_artifacts_unchanged():
    checks = common.frozen_source_checks(EXPECTATIONS)
    assert checks and all(checks.values()), checks


def test_plans_reused_are_the_phase3_preregistered_plans():
    prereg = json.loads((PHASE3 / "preregistration.json").read_text())["sha256"]
    for case_id, entry in EXPECTATIONS["frozen_sources"]["phase3_plans"].items():
        assert entry["sha256"] == prereg[f"plan/{common.case_key(case_id)}.json"]


def test_expected_bucket_sizes_match_the_plans():
    for case in EXPECTATIONS["cases"]:
        plan = json.loads((PHASE3 / "plan" / f"{common.case_key(case['case_id'])}.json").read_text())["plan"]
        batches, queries = {}, {}
        for b in plan["batches"]:
            bucket = qp.batch_bucket(b)
            batches[bucket] = batches.get(bucket, 0) + 1
            queries[bucket] = queries.get(bucket, 0) + len(b["queries"])
        assert batches == case["batches"] and queries == case["queries"]
        assert plan["anchors"] == case["anchors"]
        assert sorted({b["camera"] for b in plan["batches"]}) == sorted(case["cameras"])


def test_predictions_cover_every_plan_batch_in_order():
    for case_id in CASES:
        key = common.case_key(case_id)
        predictions = json.loads((HERE / "predictions" / f"{key}.json").read_text())
        plan = json.loads((PHASE3 / "plan" / f"{key}.json").read_text())["plan"]
        assert predictions["phase3_plan_sha256"] == EXPECTATIONS["frozen_sources"]["phase3_plans"][case_id]["sha256"]
        assert len(predictions["batches"]) == len(plan["batches"])
        for p, b in zip(predictions["batches"], plan["batches"], strict=True):
            assert (p["query_class"], p["camera"], p["anchor"], p["config"]) == (b["query_class"], b["camera"], b["anchor"], b["config"])
            assert len(p["per_query"]) == len(b["queries"])


def test_alias_sonames_are_the_ffmpeg7_names_torchcodec_requests():
    # fetch_and_expose_non_gpl_ffmpeg_libs.cmake at v0.5.0, f7_library_file_names
    expected = {"libavutil.so.59", "libavcodec.so.61", "libavformat.so.61", "libavdevice.so.61", "libavfilter.so.10", "libswscale.so.8", "libswresample.so.5"}
    assert set(SPEC["ffmpeg_provisioning"]["target_sha256"]) == expected
    for soname, entry in SPEC["ffmpeg_provisioning"]["target_sha256"].items():
        stem, major = soname.split(".so.")
        assert entry["target"].startswith(stem + "-") and f".so.{major}." in entry["target"]


def test_environment_cases_cover_each_case_once():
    assigned = [c for env in SPEC["environments"].values() for c in env["cases"]]
    assert sorted(assigned) == sorted(CASES)
    for case in EXPECTATIONS["cases"]:
        assert case["case_id"] in SPEC["environments"][case["environment"]]["cases"]


def test_preregistration_hashes_match_when_present():
    path = HERE / "preregistration.json"
    if not path.exists():
        pytest.skip("not yet preregistered")
    for relative, digest in json.loads(path.read_text())["sha256"].items():
        assert sha(HERE / relative) == digest, relative


# --- executed evidence (skipped until present) -------------------------------------

def evidence(case_id: str) -> dict:
    path = HERE / "evidence" / f"{common.case_key(case_id)}_torchcodec.json"
    if not path.exists():
        pytest.skip("not yet executed")
    return json.loads(path.read_text())


@pytest.mark.parametrize("case_id", CASES)
def test_evidence_replays_exactly_the_frozen_plan(case_id):
    ev = evidence(case_id)
    plan = json.loads((PHASE3 / "plan" / f"{common.case_key(case_id)}.json").read_text())["plan"]
    assert ev["status"] == "executed" and all(ev["identity"]["checks"].values())
    assert ev["phase3_plan_sha256"] == EXPECTATIONS["frozen_sources"]["phase3_plans"][case_id]["sha256"]
    assert len(ev["batches"]) == len(plan["batches"])
    for b, p in zip(ev["batches"], plan["batches"], strict=True):
        for field in ("query_class", "camera", "anchor", "anchor_row", "config", "padded", "video_path", "queries"):
            assert b[field] == p[field], field


@pytest.mark.parametrize("case_id", CASES)
def test_every_batch_has_a_classified_outcome_for_both_backends(case_id):
    for b in evidence(case_id)["batches"]:
        for backend in ("torchcodec", "pyav_control"):
            assert b[backend]["outcome"] in tm.OUTCOMES
            assert b[backend]["outcome"] == "harness_discrepancy" or b[backend]["replay_consistent"]


def test_report_is_regenerated_from_evidence():
    path = HERE / "report.json"
    if not path.exists():
        pytest.skip("report not yet built")
    import importlib.util

    spec = importlib.util.spec_from_file_location("torchcodec_backend_make_report", HERE / "make_report.py")
    make_report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_report)
    assert make_report.build_report() == json.loads(path.read_text())
    assert (HERE / "report.md").read_text() == make_report.render(make_report.build_report())

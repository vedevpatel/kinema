"""Offline checks for the shifted-timestamp query experiment (shifted-queries/2026-11-22).

These tests import no LeRobot, torch or video code. They pin the query
derivation rules to upstream behavior and the committed artifacts to the
frozen inputs.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "shifted-queries" / "2026-11-22"
sys.path.insert(0, str(HERE))

import queryplan as qp  # noqa: E402
from plan import policy_observation_deltas  # noqa: E402

EXPECTATIONS = json.loads((HERE / "expectations.json").read_text())
CASES = {c["case_id"]: c for c in EXPECTATIONS["cases"]}
CONFIGS = {k: v["observation_delta_indices"] for k, v in EXPECTATIONS["offset_selection"]["configs"].items()}
AUDIT_CASES = {c["key"]: c for c in json.loads((ROOT / "audits/2026-11-22/audit.json").read_text())["cases"]}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- derivation rules -------------------------------------------------------

@pytest.mark.parametrize("fps", [15.0, 30.0, 90.0, 30])
def test_delta_rows_round_trip_like_resolve_delta_timestamps(fps):
    for indices in (v for v in CONFIGS.values() if v is not None):
        assert qp.delta_rows(indices, fps) == indices
    assert qp.delta_rows(None, fps) is None


def test_query_rows_clamp_and_pad_at_both_episode_ends():
    assert qp.query_rows(0, 10, [-2, -1, 0]) == [(0, True), (0, True), (0, False)]
    assert qp.query_rows(9, 10, [0, 1, 2]) == [(9, False), (9, True), (9, True)]
    assert qp.query_rows(5, 10, [-1, 0, 1]) == [(4, False), (5, False), (6, False)]
    with pytest.raises(ValueError):
        qp.query_rows(10, 10, [0])


def test_worst_grid_anchor_uses_first_maximum_on_stored_dtype():
    ts = np.asarray([0.0, 0.1, 0.25, 0.35, 0.5], dtype=np.float32)
    anchors = qp.select_anchors(ts, 10.0)
    assert anchors == {"first": 0, "middle": 2, "worst_grid": 1, "last": 4}


def test_loader_timestamp_is_float32_like_torch_tensor_default():
    assert qp.loader_timestamp(1784331549.834715) == 1784331520.0
    assert qp.loader_timestamp(0.7333333492279053) == 0.7333333492279053
    assert qp.loader_timestamp(0.7333333333333333) != 0.7333333333333333


def test_video_query_adds_v3_from_timestamp_only():
    assert qp.video_query(1.5, None) == 1.5
    assert qp.video_query(1.5, 10.0) == 11.5


def test_decoder_distance_is_float32():
    assert qp.decoder_distance(1784331549.834715, 0.0) == pytest.approx(1784331520.0)
    assert qp.nearest_pts(0.55, [0.5, 0.6]) == (0, pytest.approx(0.05, abs=1e-6))
    assert qp.nearest_pts(0.5, []) is None


def _plan(n=20, fps=10.0, start=0.0):
    ts = (start + np.arange(n) / fps).astype(np.float32)
    return qp.build_case_plan(timestamps=ts, fps=fps, cameras=["cam"], from_timestamps=None, configs=CONFIGS, nominal_offsets=EXPECTATIONS["offset_selection"]["nominal_offsets"])


def test_plan_enumerates_every_anchor_config_and_keeps_classes_apart():
    plan = _plan()
    loader = [b for b in plan["batches"] if b["query_class"] == "loader_derived"]
    nominal = [b for b in plan["batches"] if b["query_class"] == "nominal_grid"]
    assert len(loader) == len(qp.ANCHORS) * len(CONFIGS)
    assert len(nominal) == len(qp.ANCHORS) * len(EXPECTATIONS["offset_selection"]["nominal_offsets"])
    first_vqbet = next(b for b in loader if b["anchor"] == "first" and b["config"] == "vqbet")
    assert first_vqbet["padded"] and [q["is_pad"] for q in first_vqbet["queries"]] == [True] * 4 + [False]
    assert {q["row"] for q in first_vqbet["queries"]} == {0}
    last_tdmpc = next(b for b in loader if b["anchor"] == "last" and b["config"] == "tdmpc")
    assert last_tdmpc["padded"] and {q["row"] for q in last_tdmpc["queries"]} == {19}
    first_nominal_back = next(b for b in nominal if b["anchor"] == "first" and b["config"] == "raw_plus_-1_over_fps")
    assert first_nominal_back["outside_episode_time_range"]
    assert qp.batch_bucket(first_nominal_back) == "nominal_grid/outside_episode"
    no_delta = next(b for b in loader if b["anchor"] == "middle" and b["config"] == "no_delta")
    assert [q["row"] for q in no_delta["queries"]] == [10] and not no_delta["padded"]


def test_loader_derived_queries_read_row_timestamps_not_nominal_grid():
    ts = np.asarray([0.0, 0.011, 0.022, 0.033, 0.044, 0.055], dtype=np.float32)
    plan = qp.build_case_plan(timestamps=ts, fps=30.0, cameras=["c"], from_timestamps={"c": 2.0}, configs={"diffusion": [-1, 0]}, nominal_offsets=[-1])
    b = next(b for b in plan["batches"] if b["query_class"] == "loader_derived" and b["anchor"] == "middle")
    assert [q["video_query_s"] for q in b["queries"]] == [2.0 + float(ts[2]), 2.0 + float(ts[3])]
    assert b["queries"][0]["intended_offset_s"] == pytest.approx(-1 / 30)
    assert b["queries"][0]["effective_offset_s"] == pytest.approx(-0.011, abs=1e-6)


@pytest.mark.parametrize(("counts", "verdict"), [
    (None, "not_applicable"),
    ({"ok": 3, "tolerance_error": 0, "other_error": 0, "not_run": 0}, "contradicts_claim"),
    ({"ok": 3, "tolerance_error": 1, "other_error": 1, "not_run": 0}, "supports_claim"),
    ({"ok": 3, "tolerance_error": 0, "other_error": 1, "not_run": 0}, "unresolved"),
    ({"ok": 0, "tolerance_error": 0, "other_error": 0, "not_run": 2}, "unresolved"),
])
def test_bucket_verdict_rules(counts, verdict):
    assert qp.bucket_verdict(counts) == verdict


def test_harness_discrepancy_is_never_counted_as_a_pass():
    batches = [{"query_class": "loader_derived", "padded": False, "outcome": "harness_discrepancy"}, {"query_class": "loader_derived", "padded": False, "outcome": "ok"}]
    counts = qp.summarize(batches)["loader_derived/in_range"]
    assert counts["other_error"] == 1 and qp.bucket_verdict(counts) == "unresolved"


def test_policy_default_evaluator_reads_dataclass_defaults(tmp_path):
    src = tmp_path / "configuration_x.py"
    src.write_text("class XConfig:\n    n_obs_steps: int = 3\n    @property\n    def observation_delta_indices(self):\n        return list(range(1 - self.n_obs_steps, 1))\n")
    assert policy_observation_deltas(src) == [-2, -1, 0]
    src.write_text("class YConfig:\n    @property\n    def observation_delta_indices(self):\n        return None\n")
    assert policy_observation_deltas(src) is None


# --- committed artifacts -----------------------------------------------------

def test_frozen_sources_are_unchanged_and_assertions_match_the_audit():
    for key in ("audit_json", "phase2_expectations"):
        entry = EXPECTATIONS["frozen_sources"][key]
        assert sha(ROOT / entry["path"]) == entry["sha256"]
    phase2 = {c["case_id"]: c for c in json.loads((ROOT / "loader-verification/2026-11-22/report.json").read_text())["cases"]}
    for case_id, case in CASES.items():
        assert AUDIT_CASES[case_id]["revision"] == case["revision"]
        assert phase2[case_id]["classification"] == case["phase2_classification"]


def test_preregistration_hashes_still_match():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    for relative, digest in prereg["sha256"].items():
        assert sha(HERE / relative) == digest, relative
    assert {"expectations.json", "queryplan.py", "plan.py", "probe.py"} <= set(prereg["sha256"])
    assert {f"plan/{c.replace(':', '_')}.json" for c in CASES} <= set(prereg["sha256"])


def test_committed_plans_cover_the_whole_preregistered_grid():
    for case_id, case in CASES.items():
        record = json.loads((HERE / "plan" / f"{case_id.replace(':', '_')}.json").read_text())
        assert record["identity_ok"] and record["sources"]["ok"] and record["inputs"]["ok"]
        assert record["original_assertion"] == AUDIT_CASES[case_id]["message"]
        batches = record["plan"]["batches"]
        cells = {(b["camera"], b["anchor"], b["config"]) for b in batches}
        expected_configs = set(CONFIGS) | {f"raw_plus_{k}_over_fps" for k in EXPECTATIONS["offset_selection"]["nominal_offsets"]}
        assert cells == {(c, a, k) for c in case["cameras"] for a in qp.ANCHORS for k in expected_configs}
        assert len(batches) == len(cells)


def test_fineart_worst_grid_anchor_matches_phase2_decoder_probe():
    record = json.loads((HERE / "plan" / "r1_c1_f0.json").read_text())
    phase2 = json.loads((ROOT / "loader-verification/2026-11-22/evidence/r1_c1_f0_decoder.json").read_text())
    rows = {p["position"]: p["row_index"] for p in phase2["probes"]}
    assert record["plan"]["anchors"] == {"first": rows["first"], "middle": rows["middle"], "worst_grid": rows["worst_grid"], "last": rows["last"]}

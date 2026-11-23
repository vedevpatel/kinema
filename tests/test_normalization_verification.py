"""Offline checks for normalization-verification/2026-11-23.

No openpi, LeRobot, GR00T, torch or network access. These tests pin the frozen
expectations to the phase-1 and phase-2 records they cite, exercise the frozen
selection rules the probes use (common.py), and, once results exist, check that
the report and adjudications follow from the committed evidence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "normalization-verification" / "2026-11-23"
sys.path.insert(0, str(HERE))

import common  # noqa: E402

EXP = json.loads((HERE / "expectations.json").read_text())
CASES = {c["case_id"]: c for c in EXP["cases"]}
SAMPLE = {x["key"]: x for x in json.loads((ROOT / "audits/2026-11-22/sample.json").read_text())["selected"]}
TARGETS = ["r15:c17:f0", "r51:c17:f1", "r67:c17:f0", "r79:c17:f0", "r390:c7:f8", "r393:c7:f14"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the frozen plan cites the frozen records exactly ---------------------------

def test_cases_are_exactly_the_six_targets():
    assert sorted(CASES) == sorted(TARGETS)


@pytest.mark.parametrize("cid", TARGETS)
def test_findings_are_copied_from_the_frozen_sample(cid):
    f, s = CASES[cid]["finding"], SAMPLE[cid]
    for k in ("check", "dataset", "revision", "engine", "format_version", "severity", "message", "details"):
        assert f[k] == s[k], k
    assert f["engine"] == EXP["engine"]["board_engine"]


@pytest.mark.parametrize("cid", TARGETS)
def test_source_revision_matches_finding(cid):
    case = CASES[cid]
    assert case["source"]["revision"] == case["finding"]["revision"]
    repo = case["finding"]["dataset"].split("//")[0]
    assert case["source"]["repo"] == repo


@pytest.mark.parametrize("cid", TARGETS)
def test_prior_labels_match_frozen_adjudications(cid):
    p1 = json.loads((ROOT / "audits/2026-11-22/adjudications.json").read_text())[cid]
    assert CASES[cid]["prior"]["phase1"] == p1
    p2 = json.loads((ROOT / "loader-verification/2026-11-22/adjudications.json").read_text())
    if cid in p2:
        assert CASES[cid]["prior"]["phase2"]["classification"] == p2[cid]["classification"]


def test_frozen_source_artifacts_are_unchanged():
    for rel, digest in EXP["frozen_sources"].items():
        assert sha(ROOT / rel) == digest, rel


def test_every_case_has_one_primary_configuration():
    for cid, case in CASES.items():
        primaries = [k for k, v in case["configurations"].items() if v.get("role") == "primary"]
        assert len(primaries) == 1, (cid, primaries)


def test_missing_stats_dataset_keys_match_phase2_feature_lists():
    for cid in ("r15:c17:f0", "r51:c17:f1", "r67:c17:f0"):
        loader = json.loads((ROOT / "loader-verification/2026-11-22/evidence" / f"{cid.replace(':', '_')}_loader.json").read_text())
        keys = loader["steps"][0]["feature_keys"]
        assert CASES[cid]["dataset_key"] in keys
        assert CASES[cid]["finding"]["details"]["feature"] not in keys


# --- frozen selection rules ------------------------------------------------------

def _listing(n, size=10):
    return [{"path": f"data/chunk-000/episode_{i:08d}.parquet", "bytes": size} for i in range(n)]


def test_select_files_takes_first_then_rank_order():
    rule = {"seed": "s", "count": 4, "byte_cap": 10**9}
    chosen = common.select_files(rule, "c", _listing(20))
    assert chosen[0]["path"].endswith("episode_00000000.parquet")
    rest = sorted(_listing(20)[1:], key=lambda r: common.rank("s", "c", r["path"]))[:3]
    assert [r["path"] for r in chosen[1:]] == [r["path"] for r in rest]
    assert common.select_files(rule, "c", list(reversed(_listing(20)))) == chosen


def test_select_files_byte_cap_drops_last_added():
    rule = {"seed": "s", "count": 5, "byte_cap": 30}
    chosen = common.select_files(rule, "c", _listing(10, size=10))
    assert len(chosen) == 3 and chosen[0]["path"].endswith("episode_00000000.parquet")
    assert chosen == common.select_files({**rule, "byte_cap": 10**9}, "c", _listing(10, size=10))[:3]


def test_control_dim_is_lower_median_excluding_flagged():
    stds = {j: float(j) for j in range(10)}
    assert common.control_dim(stds, 5, [0, 10]) == 4  # siblings 0..9 without 5: lower median is 4
    stds = {j: 1.0 for j in range(4)}
    assert common.control_dim(stds, 0, [0, 4]) == 2  # ties broken by index: [1, 2, 3] -> 2


def test_checkpoint_model_type_prefixes():
    assert common.checkpoint_model_type("VLABench/pi0-fast-primitive-10task") == "PI0_FAST"
    assert common.checkpoint_model_type("VLABench/pifast-pretrain-vlabench-primitive-aligned") == "PI0_FAST"
    assert common.checkpoint_model_type("VLABench/pi05-primitive-10task") == "PI05"
    assert common.checkpoint_model_type("VLABench/pi0-primitive-10task") == "PI0"


def test_vlabench_mapping_follows_the_frozen_rule():
    inv = EXP["fork_config_inventory"]["rows"]
    quant = {r[0]: r[3] for r in inv}
    for cid in ("r15:c17:f0", "r51:c17:f1"):
        conf = CASES[cid]["configurations"]["openpi_fork"]
        for rel in conf["released"]:
            cands = common.vlabench_candidates(rel["source"], rel["asset_id"], inv)
            assert rel["candidates"] == cands, rel["source"]
            unanimous = len({quant[c] for c in cands}) == 1
            if cands and unanimous:
                assert rel["config"] in cands
            else:
                assert rel["config"] is None
            assert rel["asset_id"] == conf["fork_repo_id"]
        assert set(conf["config_names"]) == {r[0] for r in inv if r[2] == conf["fork_repo_id"]}


def test_upstream_openpi_quantile_rule_is_recorded_not_assumed():
    # the case lists all three LIBERO configs so PI0 vs PI05/PI0_FAST is observed, not inferred
    assert CASES["r67:c17:f0"]["configurations"]["openpi_upstream"]["config_names"] == ["pi0_libero", "pi05_libero", "pi0_fast_libero"]


# --- preregistration --------------------------------------------------------------

def test_preregistered_files_unchanged():
    prereg = json.loads((HERE / "preregistration.json").read_text())
    assert set(prereg["sha256"]) == set(EXP["frozen_files"])
    for name, digest in prereg["sha256"].items():
        assert sha(HERE / name) == digest, name

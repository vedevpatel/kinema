"""Sampling and reporting guard for the frozen phase 1 audit."""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import assurance_report as report  # noqa: E402
import assurance_sample as sample  # noqa: E402

AUDIT = ROOT / "audits" / "2026-11-22"


def test_frozen_frame_and_seeded_sample_are_reproducible():
    raw = gzip.decompress((AUDIT / "board.json.gz").read_bytes())
    board = json.loads(raw)
    saved = json.loads((AUDIT / "sample.json").read_text())
    assert len(board["datasets"]) == 414
    assert len(sample.frame(board)) == 1839
    assert sample.make(AUDIT / "board.json.gz", AUDIT) == saved
    picked = saved["selected"]
    assert len(picked) == 40 == len({c["key"] for c in picked})
    assert len({c["dataset"] for c in picked}) == 39
    assert Counter(c["severity"] for c in picked) == {"WARN": 29, "FAIL": 11}


def test_missing_dataset_pages_cannot_change_eligibility(tmp_path, monkeypatch):
    board = {"datasets": [{"id": "a/b", "revision": "a" * 40, "engine": "v1",
                           "report": {"checks": [{"id": "check", "findings": [
                               {"severity": "WARN", "message": "one", "details": {}},
                               {"severity": "INFO", "message": "note", "details": {}},
                               {"severity": "FAIL", "message": "two", "details": {}}]}]}}]}
    monkeypatch.chdir(tmp_path)  # no docs/dataset directory exists
    entries = sample.frame(board)
    assert len(entries) == 2
    assert {e["severity"] for e in sample.select(entries, 2)} == {"WARN", "FAIL"}


def test_report_has_all_cases_and_consistent_counts():
    doc = report.assemble(AUDIT)
    assert doc["results"] == {"selected": 40, "confirmed": 9, "false alarm": 1,
                              "unresolved": 30, "unresolved_over_total": "30/40",
                              "datasets_selected": 39}
    assert sum(v["total"] for v in doc["per_check"].values()) == 40
    assert sum(v["total"] for v in doc["per_dataset"].values()) == 40
    assert {c["key"] for c in doc["cases"]} == {
        c["key"] for c in json.loads((AUDIT / "sample.json").read_text())["selected"]}
    rendered = report.markdown(doc)
    assert "## False alarms" in rendered and "## Unresolved cases" in rendered
    for c in doc["cases"]:
        assert c["key"] in rendered and c["revision"] and c["commands"]


def test_report_rejects_revision_drift(tmp_path):
    for name in ("board.json.gz", "sample.json", "provenance.json", "observations.json", "adjudications.json", "misses.json", "artifact_hashes.json"):
        shutil.copyfile(AUDIT / name, tmp_path / name)
    p = json.loads((tmp_path / "provenance.json").read_text())
    p[0]["revision"] = "0" * 40
    (tmp_path / "provenance.json").write_text(json.dumps(p))
    with pytest.raises(ValueError, match="revision drift"):
        report.assemble(tmp_path)

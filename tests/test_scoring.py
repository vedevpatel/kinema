"""Health-score model tiered-v1: pins the tier table, the per-check weight
table with its basis labels, the per-check (never per-finding) rule, and the
recompute-from-stored-verdicts path the leaderboard migrates through."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from kinema.checks import ALL_CHECKS
from kinema.checks.base import CheckResult, Finding, Severity
from kinema.report import (
    SCORE_MODEL,
    WARN_TIERS,
    DatasetReport,
    harm_tier,
    render_markdown,
    report_from_dict,
    report_to_dict,
    rescore_report_dict,
    score_from_verdicts,
    warn_weight,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "leaderboard.py"

# legacy-v0, the hand-tuned table tiered-v1 replaced (default 8 for unlisted)
LEGACY_V0 = {
    "stats-drift": 10,
    "nan-and-normalization": 10,
    "deployment-readiness": 10,
    "timestamp-grid": 10,
    "dropped-frames": 8,
    "calibration-drift": 6,
    "idle-segments": 6,
    "stuck-actuator": 6,
    "action-saturation": 6,
    "visual-quality": 6,
    "duplicate-and-leakage": 4,
    "outlier-episodes": 3,
}

TIERED_V1 = {
    "idle-segments": (10, "measured"),
    "calibration-drift": (10, "measured"),
    "duplicate-and-leakage": (2, "measured"),
    "stats-drift": (10, "placeholder"),
    "nan-and-normalization": (10, "placeholder"),
    "deployment-readiness": (10, "placeholder"),
    "timestamp-grid": (10, "placeholder"),
    "dropped-frames": (8, "placeholder"),
    "stuck-actuator": (6, "placeholder"),
    "action-saturation": (6, "placeholder"),
    "visual-quality": (6, "placeholder"),
    "outlier-episodes": (3, "placeholder"),
    "task-strings": (8, "placeholder"),
    "reward-done": (8, "placeholder"),
}


def test_model_label():
    assert SCORE_MODEL == "tiered-v1"


def test_tier_table_is_pinned():
    assert WARN_TIERS == (
        ("A", "training killer", 100.0, 10),
        ("B", "skews learning", 10.0, 6),
        ("C", "review list", 0.0, 2),
    )


@pytest.mark.parametrize(
    "pct, tier",
    [(881.0, "A"), (347.0, "A"), (100.0, "A"), (99.9, "B"), (10.0, "B"),
     (9.99, "C"), (0.1, "C"), (0.0, "C"), (-3.0, "C")],
)
def test_band_edges(pct, tier):
    assert harm_tier(pct)[0] == tier
    assert harm_tier(pct)[2] == {"A": 10, "B": 6, "C": 2}[tier]


def test_weight_table_and_basis_are_pinned():
    for cid, expected in TIERED_V1.items():
        assert warn_weight(cid) == expected, cid
    assert warn_weight("some-future-check") == (8, "placeholder")


def test_only_the_three_measured_weights_moved():
    moved = {cid for cid, w in LEGACY_V0.items() if warn_weight(cid)[0] != w}
    assert moved == {"idle-segments", "calibration-drift", "duplicate-and-leakage"}
    measured = {c.id for c in ALL_CHECKS if warn_weight(c.id)[1] == "measured"}
    assert measured == moved


def test_fail_cost_and_floor():
    assert score_from_verdicts([("idle-segments", "FAIL")]) == 75
    assert score_from_verdicts([("duplicate-and-leakage", "FAIL")]) == 75
    assert score_from_verdicts([(f"c{i}", "FAIL") for i in range(5)]) == 0
    assert score_from_verdicts([("x", v) for v in ("PASS", "INFO", "SKIP")]) == 100


def _report(check_id: str, n_findings: int, sev: Severity = Severity.WARN) -> DatasetReport:
    r = CheckResult(check_id=check_id, title=check_id)
    for i in range(n_findings):
        r.findings.append(Finding(check_id=check_id, severity=sev, message=f"ep {i}",
                                  episode_index=i))
    rep = DatasetReport(source="t", format_version="v3.0", episodes=40, frames=400, fps=10.0)
    rep.results.append(r)
    return rep


def test_scoring_is_per_check_not_per_finding():
    assert _report("idle-segments", 1).score == _report("idle-segments", 40).score == 90
    assert _report("duplicate-and-leakage", 40).score == 98
    assert _report("timestamp-grid", 40, Severity.FAIL).score == 75


def test_report_json_labels_model_and_weights():
    d = report_to_dict(_report("duplicate-and-leakage", 3))
    assert d["score_model"] == "tiered-v1"
    assert d["score"] == 98
    (c,) = d["checks"]
    assert (c["weight"], c["weight_basis"]) == (2, "measured")
    assert "tiered-v1" in render_markdown(_report("idle-segments", 1))


def _legacy_dict(verdicts: dict[str, str], score: int) -> dict:
    """A stored legacy-v0 report: no score_model, no per-check weights."""
    return {
        "source": "org/ds", "format_version": "v3.0", "episodes": 10, "frames": 100,
        "fps": 10.0, "truncated": False, "score": score, "verdict": "WARN",
        "checks": [
            {"id": cid, "title": cid, "verdict": v, "summary": "", "skipped": v == "SKIP",
             "skip_reason": "", "details": {},
             "findings": [] if v in ("PASS", "SKIP") else
             [{"severity": v, "message": "m", "episode_index": i, "details": {}}
              for i in range(7)]}
            for cid, v in verdicts.items()
        ],
    }


def test_rescore_uses_stored_verdicts_and_is_idempotent():
    d = _legacy_dict({"idle-segments": "WARN", "duplicate-and-leakage": "WARN",
                      "stats-drift": "PASS", "visual-quality": "SKIP"}, score=90)
    rescore_report_dict(d)
    assert d["score"] == 100 - 10 - 2
    assert d["score_model"] == "tiered-v1"
    assert [(c["weight"], c["weight_basis"]) for c in d["checks"]] == [
        (10, "measured"), (2, "measured"), (10, "placeholder"), (6, "placeholder"),
    ]
    again = rescore_report_dict(copy.deepcopy(d))
    assert again == d
    # the round-tripped report (what dataset pages render) agrees with it
    assert report_from_dict(d).score == d["score"]


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_leaderboard_rescore_site(tmp_path):
    lb = _load_leaderboard()
    out = tmp_path / "docs"
    (out / "dataset").mkdir(parents=True)
    idle = _legacy_dict({"idle-segments": "WARN"}, score=94)
    dup = _legacy_dict({"duplicate-and-leakage": "WARN"}, score=96)
    dup["source"] = "org/dup"
    entries = [
        {"id": "org/ds", "slug": "org--ds", "downloads": 10, "engine": "e0", "report": idle},
        {"id": "org/dup", "slug": "org--dup", "downloads": 5, "engine": "e0", "report": dup},
        {"id": "org/gated", "downloads": 1, "status": "gated", "reason": "GatedRepoError"},
    ]
    (out / "data.json").write_text(json.dumps({"generated": "2026-11-14 00:01 UTC",
                                               "datasets": entries}))
    stamp = "audited 2026-11-13 22:10 UTC by"
    (out / "dataset" / "org--ds.html").write_text(f"<h2>Health score: 94/100</h2><p>{stamp}</p>")

    assert lb.rescore_site(out) == 0

    board = json.loads((out / "data.json").read_text())
    assert board["generated"] == "2026-11-14 00:01 UTC"  # nothing was re-audited
    assert board["score_model"] == "tiered-v1"
    by_id = {e["id"]: e for e in board["datasets"]}
    assert by_id["org/ds"]["report"]["score"] == 90
    assert by_id["org/dup"]["report"]["score"] == 98
    assert all(e["report"]["score_model"] == "tiered-v1" for e in by_id.values() if "report" in e)
    assert "report" not in by_id["org/gated"]

    page = (out / "dataset" / "org--ds.html").read_text()
    assert "Health score: 90/100" in page and stamp in page and "tiered-v1" in page
    assert json.loads((out / "dataset" / "org--dup.json").read_text())["score"] == 98
    index = (out / "index.html").read_text()
    assert "tiered-v1" in index and "legacy-v0" not in index


def test_index_labels_unrescored_legacy_rows():
    lb = _load_leaderboard()
    legacy = _legacy_dict({"idle-segments": "WARN"}, score=94)
    html = lb.render_index([{"id": "org/ds", "slug": "org--ds", "downloads": 1,
                             "engine": "e0", "report": legacy}], "now")
    assert "(legacy-v0)" in html

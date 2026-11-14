"""The diagnosis layer: routing, cause matching, the negative answer.

Every answer - with or without causes - carries the coverage block, and
the words 'clean' and 'ruled out' never appear: a 'no cause found' answer
is an enumeration of what was and was not examined, not an all-clear."""

from __future__ import annotations

import pytest

from kinema.diagnose import (
    CancelledError,
    checks_for_symptom,
    diagnose,
    list_checks,
    run_checks,
    summarize_report,
)
from kinema.fixtures.synth import make_dataset


def test_symptom_routing():
    ids, clusters = checks_for_symptom("loss converged but the robot doesn't move")
    assert clusters == ["idle"]
    assert set(ids) == {"idle-segments", "stuck-actuator"}
    ids, clusters = checks_for_symptom("actions look scaled wrong; maybe a normalization issue")
    assert clusters == ["normalization"]
    assert set(ids) == {"stats-drift", "nan-and-normalization", "deployment-readiness"}
    ids, clusters = checks_for_symptom("florp gibberish unrouteable")
    assert clusters == [] and len(ids) > 5  # fallback: all checks


def test_stats_drift_ranks_the_stats_cause(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12, defects=["stats-drift"])
    out = diagnose(str(root), "pi0", "the stats look off")
    ids = [c["cause_id"] for c in out["causes"]]
    assert "stored-stats-drift" in ids
    # severity ranks first; every cause carries evidence + docs + next step
    for c in out["causes"]:
        assert c["evidence"] and c["docs"].startswith("https://github.com/vedevpatel/kinema")
        assert "#" in c["docs"] and c["next_step"] and c["confidence"] in ("high", "medium")


def test_idle_symptom_finds_idle_cause(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12, defects=["idle-segments"])
    out = diagnose(str(root), "pi0.5", "robot doesn't move at inference")
    assert out["policy"] == "pi05"  # alias resolved
    ids = [c["cause_id"] for c in out["causes"]]
    assert "idle-heavy-episodes" in ids


def test_negative_answer_enumerates_coverage(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12)
    out = diagnose(str(root), "pi0", "the robot drifts left")
    cov = out["coverage"]
    assert out["causes"] == [] or all("cause" in c for c in out["causes"])
    assert cov["checks_run"] and isinstance(cov["checks_skipped"], list)
    assert cov["sampled"] is False
    assert cov["blind_spots"]  # what v0 cannot see is always stated
    # unmapped WARN/FAIL findings are surfaced verbatim, never dropped
    from kinema.checks import ALL_CHECKS, CheckConfig
    from kinema.dataset import load_dataset
    ctx = load_dataset(str(root))
    from kinema.diagnose import run_check_subset
    report = run_check_subset(ctx, CheckConfig(policy="pi0"), [c.id for c in ALL_CHECKS], str(root))
    n_warn_fail = sum(
        1 for r in report.results if not r.skipped
        for f in r.findings if f.severity.name in ("WARN", "FAIL")
    )
    mapped = sum(len(c["evidence"]) for c in out["causes"])
    assert mapped + len(cov["findings_outside_coverage"]) >= n_warn_fail
    # never an all-clear
    import json
    text = json.dumps(out).lower()
    assert "clean on all" not in text and "ruled out" not in text


def test_policy_overrides_change_the_diagnosis(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12)
    out = diagnose(str(root), "pi0", "stats problem", policy_overrides={"quantile_norm": False})
    assert "missing-quantile-stats" not in [c["cause_id"] for c in out["causes"]]
    out = diagnose(str(root), "pi0", "stats problem", policy_overrides={"chunk_size": 100})
    assert "chunk-too-short" in [c["cause_id"] for c in out["causes"]]
    with pytest.raises(ValueError, match="unknown policy override"):
        diagnose(str(root), "pi0", "s", policy_overrides={"chunk": 10})
    with pytest.raises(ValueError, match="unknown policy"):
        diagnose(str(root), "hal-9000", "s")


def test_cancellation_between_checks(tmp_path):
    import threading
    root = make_dataset(tmp_path / "ds", episodes=12)
    cancel = threading.Event()
    calls = {"n": 0}

    def progress(done, total, message):
        calls["n"] += 1
        cancel.set()  # cancel after the first check completes

    with pytest.raises(CancelledError):
        diagnose(str(root), "pi0", "unrouteable symptom runs all checks",
                 cancelled=cancel, progress=progress)
    assert calls["n"] >= 1


def test_run_checks_summary_is_capped_and_writes_report(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12, defects=["idle-segments", "stats-drift"])
    report, ctx = run_checks(str(root), policy="pi0")
    payload = summarize_report(report, top=2)
    assert payload["verdict"] in ("WARN", "FAIL")
    assert payload["findings_omitted"] >= 0
    assert len(payload["top_findings"]) <= 2
    assert payload["checks"] and all("id" in c for c in payload["checks"])
    with pytest.raises(ValueError, match="unknown checks"):
        run_checks(str(root), checks=["not-a-check"])


def test_list_checks_from_the_registry():
    checks = list_checks()
    assert len(checks) >= 15
    ids = {c["id"] for c in checks}
    assert {"stats-drift", "deployment-readiness", "idle-segments"} <= ids
    for c in checks:
        assert c["docs"].endswith(f"#{c['id']}")
        assert c["title"] and c["description"]

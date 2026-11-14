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


def test_symptom_routing_never_filters_checks():
    """Review finding 1: routing used to pick a check subset, so the
    headline symptom ('robot doesn't move') skipped every normalization
    check and answered 'no cause' over an unexamined stats-drift defect.
    Routing now returns ALL checks and matched clusters only order causes."""
    from kinema.checks import ALL_CHECKS
    all_ids = [c.id for c in ALL_CHECKS]
    ids, clusters = checks_for_symptom("loss converged but the robot doesn't move")
    assert clusters == ["idle"]
    assert ids == all_ids  # every check runs, whatever the symptom
    ids, clusters = checks_for_symptom("actions look scaled wrong; maybe a normalization issue")
    assert clusters == ["normalization"]
    assert ids == all_ids
    ids, clusters = checks_for_symptom("florp gibberish unrouteable")
    assert clusters == [] and ids == all_ids


def test_keyword_matching_respects_word_boundaries():
    """Substrings must not fire clusters: 'large-scale' is not 'scale',
    'static camera' is not an idle robot (review finding 1)."""
    _, clusters = checks_for_symptom("the dataset is large-scale, collected in the wild")
    assert "normalization" not in clusters
    _, clusters = checks_for_symptom("a static camera viewed from the side")
    assert "idle" not in clusters
    # real phrases still route
    _, clusters = checks_for_symptom("the robot stays still at inference")
    assert clusters == ["idle"]


def test_headline_symptom_still_finds_the_stats_defect(tmp_path):
    """Regression for the review's headline case: 'the robot does not move'
    on a stats-drift dataset must surface stored-stats-drift - before the
    fix it returned zero causes because the normalization checks never ran."""
    root = make_dataset(tmp_path / "ds", episodes=12, defects=["stats-drift"])
    out = diagnose(str(root), "pi05", "the robot does not move")
    ids = [c["cause_id"] for c in out["causes"]]
    assert "stored-stats-drift" in ids
    # coverage is complete: every check appears in the answer
    ran = {c["check_id"] for c in out["coverage"]["checks_run"]}
    skipped = {c["check_id"] for c in out["coverage"]["checks_skipped"]}
    from kinema.checks import ALL_CHECKS
    assert ran | skipped == {c.id for c in ALL_CHECKS}


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


def test_zip_source_tempdir_is_cleaned_up(tmp_path):
    """Review finding 13: every .zip audit extracted to a fresh temp dir
    that was never deleted."""
    import zipfile
    from pathlib import Path

    import kinema.diagnose as d

    root = make_dataset(tmp_path / "ds", episodes=8)
    zpath = tmp_path / "ds.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root))
    seen = {}
    orig = d.load_dataset

    def spy(*a, **k):
        ctx = orig(*a, **k)
        seen["dir"] = ctx.temp_extract_dir
        return ctx

    d.load_dataset = spy
    try:
        run_checks(str(zpath), policy="pi0")
    finally:
        d.load_dataset = orig
    assert seen["dir"], "zip extract dir was never tracked"
    assert not Path(seen["dir"]).exists(), "zip extract dir was left behind"


def test_cancellation_lands_during_the_dataset_load(tmp_path):
    """Review finding 10: cancellation used to be noticed only after the
    load - the long part. A cancelled event now aborts the load itself."""
    import threading
    root = make_dataset(tmp_path / "ds", episodes=8)
    cancel = threading.Event()
    cancel.set()  # cancelled before the run even starts
    with pytest.raises(CancelledError):
        diagnose(str(root), "pi0", "x", cancelled=cancel)

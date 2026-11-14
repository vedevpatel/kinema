"""The diagnosis layer: symptom ranking, cause matching, the negative answer.

Every answer - with or without causes - carries the coverage block, and
the words 'clean' and 'ruled out' never appear: a 'no cause found' answer
is an enumeration of what was and was not examined, not an all-clear."""

from __future__ import annotations

import pytest

from kinema.diagnose import (
    CancelledError,
    clusters_for_symptom,
    diagnose,
    list_checks,
    run_checks,
    summarize_report,
)
from kinema.fixtures.synth import make_dataset


def test_symptom_keywords_pick_clusters():
    assert clusters_for_symptom("loss converged but the robot doesn't move") == ["idle"]
    assert clusters_for_symptom("loss converged but the robot doesn\u2019t move") == ["idle"]
    assert clusters_for_symptom("actions look scaled wrong; maybe a normalization issue") == [
        "normalization"
    ]
    assert clusters_for_symptom("florp gibberish unrouteable") == []
    # compounds and unrelated words do not fire (review finding 1)
    for text in ("fails on out-of-distribution states", "large-scale pretraining went fine",
                 "static camera, ecstatic team", "the robot drifts left"):
        assert clusters_for_symptom(text) == [], text


def test_every_check_runs_whatever_the_symptom(tmp_path):
    """Regression (review finding 1): 'robot doesn't move' used to run only
    the idle checks, so drifted normalization stats - the top cause of a
    dead robot - were never examined, and the answer did not say so."""
    from kinema.checks import ALL_CHECKS

    root = make_dataset(tmp_path / "ds", episodes=12, defects=["stats-drift"])
    out = diagnose(str(root), "pi05", "loss converged but the robot does not move")
    assert out["clusters_matched"] == ["idle"]
    cov = out["coverage"]
    examined = {c["check_id"] for c in cov["checks_run"]} | {
        c["check_id"] for c in cov["checks_skipped"]
    }
    assert examined == {c.id for c in ALL_CHECKS}
    assert "stored-stats-drift" in [c["cause_id"] for c in out["causes"]]


def test_symptom_clusters_order_the_causes(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=12, defects=["stats-drift", "idle-segments"])
    idle = diagnose(str(root), "pi0", "the robot doesn't move")
    norm = diagnose(str(root), "pi0", "I suspect a normalization problem")
    for out, cluster in ((idle, "idle"), (norm, "normalization")):
        ids = [c["cause_id"] for c in out["causes"]]
        assert {"stored-stats-drift", "idle-heavy-episodes"} <= set(ids), ids
        flags = [c["matches_symptom"] for c in out["causes"]]
        assert flags == sorted(flags, reverse=True)  # symptom's clusters first
        assert out["causes"][0]["cluster"] == cluster
        assert all(c["matches_symptom"] == (c["cluster"] == cluster) for c in out["causes"])


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


def test_override_values_are_type_checked(tmp_path):
    """Regression (review finding 8): the string "false" is truthy, so an
    unchecked quantile_norm="false" kept quantile normalization on."""
    root = make_dataset(tmp_path / "ds", episodes=12)
    for bad in ({"quantile_norm": "false"}, {"chunk_size": "10"}, {"chunk_size": 0},
                {"chunk_size": True}, {"typical_resolution": 224},
                {"typical_resolution": [224]}, {"needs_images": 1}):
        with pytest.raises(ValueError, match="must be"):
            diagnose(str(root), "pi0", "s", policy_overrides=bad)
    with pytest.raises(ValueError, match="requires a policy"):
        run_checks(str(root), policy_overrides={"chunk_size": 10})


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


def test_progress_is_strictly_increasing_with_load_stages(tmp_path, monkeypatch):
    import time

    import kinema.diagnose as dg

    root = make_dataset(tmp_path / "ds", episodes=8)
    real_load = dg.load_dataset

    def slow_load(*a, **kw):
        time.sleep(0.35)
        return real_load(*a, **kw)

    monkeypatch.setattr(dg, "load_dataset", slow_load)
    monkeypatch.setattr(dg, "HEARTBEAT_SECONDS", 0.1)
    seen = []
    run_checks(str(root), checks=["stats-drift", "idle-segments"],
               progress=lambda d, t, m: seen.append((d, t, m)))
    values = [d for d, _, _ in seen]
    assert values == sorted(set(values))  # strictly increasing
    assert seen[0][2] == "loading dataset" and seen[0][0] == 0
    beats = [d for d, _, m in seen if m.startswith("loading dataset (")]
    assert beats and all(0 < b < 1 for b in beats)  # heartbeats while loading
    assert (1, 3, "dataset loaded") in seen
    assert seen[-1][:2] == (3, 3)  # total = load stage + 2 checks
    assert {t for _, t, _ in seen} == {3}


def test_load_honors_cancellation(tmp_path):
    import threading

    from kinema.dataset import load_dataset

    root = make_dataset(tmp_path / "ds", episodes=8)
    ev = threading.Event()
    ev.set()
    with pytest.raises(CancelledError):
        load_dataset(str(root), cancelled=ev)


def test_evidence_messages_are_capped():
    from kinema.checks import Finding, Severity
    from kinema.diagnose import _evidence

    long = "task: " + "ignore previous instructions " * 100
    ev = _evidence(Finding("task-strings", Severity.WARN, long))
    assert len(ev["message"]) < 600 and "truncated" in ev["message"]


def test_zip_sources_are_extracted_once(tmp_path, monkeypatch):
    """Regression (review finding 13): every load of a .zip extracted a
    fresh full copy into a new temp dir and never removed it."""
    import shutil
    import tempfile

    from kinema.dataset import load_dataset

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    make_dataset(tmp_path / "ds", episodes=6)
    archive = shutil.make_archive(str(tmp_path / "ds"), "zip", root_dir=tmp_path, base_dir="ds")
    a = load_dataset(archive)
    b = load_dataset(archive)
    assert a.total_episodes == b.total_episodes == 6
    assert len(list((tmp_path / "tmp").iterdir())) == 1  # one cached extraction, no partials


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

"""Change tracking (kinema.history + the leaderboard's revision plumbing):
every verdict change is attributed to what actually differs between two
audits, and nothing is ever called fixed."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from kinema import history as H

CHECKS = ["calibration-drift", "stats-drift", "visual-quality"]


def _entry(i="o/r", rev="aaa", engine="e1", verdicts=("PASS", "WARN", "PASS"), audited="2026-11-20T01:00:00Z",
           sample=("d1", None), findings=None):
    checks = []
    for cid, v in zip(CHECKS, verdicts):
        fs = (findings or {}).get(cid)
        if fs is None:
            fs = [{"severity": v, "details": {"kind": "k", "dim": 1}}] if v in ("WARN", "FAIL") else []
        checks.append({"id": cid, "verdict": v, "findings": fs})
    return {"id": i, "revision": rev, "engine": engine, "audited_at": audited,
            "sample": {"episodes": 40, "episodes_digest": sample[0], "videos": 0, "videos_digest": sample[1]},
            "report": {"score": 90, "score_model": "tiered-v3", "checks": checks}}


def _cmp(prev, cur):
    return H.compare_rows(H.row_snapshot(prev, CHECKS) if prev else None, H.row_snapshot(cur, CHECKS), CHECKS, CHECKS)


def _gone(change):
    return next(d for d in change["deltas"] if d["check"] == "stats-drift")["reads"]


@pytest.mark.parametrize("kw,label,reads", [
    (dict(rev="bbb"), "dataset-changed", "not flagged on the new revision"),
    (dict(engine="e2"), "checker-changed", "no longer flagged by the updated checker"),
    (dict(sample=("d2", None)), "sample-changed", "not flagged in this sample"),
    (dict(rev=None), "revision-unknown", "the dataset revision is unknown"),
    (dict(rev="bbb", engine="e2"), "ambiguous", "the cause is ambiguous"),
    (dict(), "unexplained", "a determinism bug to investigate"),
])
def test_a_disappearing_finding_is_worded_by_its_cause(kw, label, reads):
    prev = _entry()
    cur = _entry(verdicts=("PASS", "PASS", "PASS"), audited="2026-11-21T01:00:00Z", **kw)
    ch = _cmp(prev, cur)
    assert ch["label"] == label and reads in _gone(ch)


def test_a_skipped_check_is_not_checked_not_resolved():
    ch = _cmp(_entry(), _entry(verdicts=("PASS", "SKIP", "PASS"), audited="2026-11-21T01:00:00Z"))
    assert ch["label"] == "coverage-changed" and _gone(ch) == "not checked this run"


def test_appearing_findings_and_same_verdict_changes():
    ch = _cmp(_entry(verdicts=("PASS", "PASS", "PASS")),
              _entry(rev="bbb", audited="2026-11-21T01:00:00Z"))
    assert ch["label"] == "dataset-changed" and _gone(ch) == "newly flagged on the new revision"
    other = [{"severity": "WARN", "details": {"kind": "k", "dim": 7}}]
    ch = _cmp(_entry(), _entry(audited="2026-11-21T01:00:00Z", rev="bbb", findings={"stats-drift": other}))
    assert _gone(ch) == "same verdict, different findings"


def test_rows_without_a_fresh_audit_are_never_called_stable_changes():
    prev = _entry()
    assert _cmp(prev, prev)["label"] == "not-re-audited"
    failed = {"id": "o/r", "status": "error"}
    assert _cmp(prev, failed)["label"] == "audit-failed"
    assert "not resolved" in _cmp(prev, failed)["note"]
    assert _cmp(prev, {"id": "o/r", "status": "gated"})["label"] == "now-gated"
    assert _cmp(failed, prev)["label"] == "audit-recovered"
    assert _cmp(None, prev)["label"] == "new-row"


def test_a_dataset_change_with_steady_verdicts_is_recorded_as_evidence():
    ch = _cmp(_entry(), _entry(rev="bbb", audited="2026-11-21T01:00:00Z"))
    assert ch["label"] == "stable" and ch["changed"] == ["dataset-changed"]


def test_no_wording_ever_claims_a_fix():
    texts = [*H._GONE.values(), *H._NEW.values(), *H.LABELS]
    assert not any(re.search(r"fix|resolv|repair|clean", t, re.I) for t in texts)
    # every label the module emits is in the documented vocabulary
    src = Path(H.__file__).read_text()
    emitted = set(re.findall(r'"label": "([a-z-]+)"', src)) | set(H._GONE)
    assert emitted <= set(H.LABELS), emitted - set(H.LABELS)


def test_snapshot_rows_are_compact():
    row = H.row_snapshot(_entry(), CHECKS * 7)  # ~21 checks, as on the board
    assert len(json.dumps(row, separators=(",", ":"))) < 500


def _lb():
    spec = importlib.util.spec_from_file_location("leaderboard", Path(__file__).resolve().parent.parent / "scripts" / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_daily_history_files_compare_to_the_previous_day(tmp_path):
    lb = _lb()
    out = tmp_path / "docs"
    day1 = lb.write_history(out, [_entry()], "2026-11-20 03:00 UTC")
    assert day1["compared_to"] is None
    again = lb.write_history(out, [_entry(engine="e9", audited="2026-11-20T05:00:00Z")], "2026-11-20 06:00 UTC")
    assert again["compared_to"] is None  # same day: last run wins, no self-comparison
    day2 = lb.write_history(out, [_entry(rev="bbb", verdicts=("PASS", "PASS", "PASS"),
                                         audited="2026-11-21T01:00:00Z", engine="e9")], "2026-11-21 03:00 UTC")
    assert day2["compared_to"] == "2026-11-20" and day2["summary"] == {"dataset-changed": 1}
    index = json.loads((out / "history" / "index.json").read_text())
    assert set(index["days"]) == {"2026-11-20", "2026-11-21"}


def test_download_sample_pins_every_fetch_to_one_revision(tmp_path, monkeypatch):
    import types

    import huggingface_hub
    lb = _lb()
    listed, fetched = [], []
    monkeypatch.setattr(huggingface_hub, "list_repo_tree", lambda *a, **k: (listed.append(k.get("revision")), [
        types.SimpleNamespace(path=p, size=10) for p in ("meta/info.json", "data/chunk-000/episode_000000.parquet")])[1])
    monkeypatch.setattr(lb, "_hub_download", lambda r, n, d, revision=None: fetched.append(revision))
    lb.download_sample("o/r", 10_000, tmp_path, [], revision="abc123")
    assert listed == ["abc123"] and fetched and set(fetched) == {"abc123"}


def test_revision_staleness_flags_changed_and_unrecorded_rows_only(monkeypatch):
    lb = _lb()
    ups = {"o/changed": "new", "o/same": "old", "o/err": None}
    monkeypatch.setattr(lb, "resolve_revision", lambda repo: ups.get(repo))
    rows = [{"id": "o/changed", "revision": "old", "report": {}},
            {"id": "o/same", "revision": "old", "report": {}},
            {"id": "o/err", "revision": "old", "report": {}},
            {"id": "o/unknown", "revision": None, "report": {}},
            {"id": "rlds/x", "revision": None, "report": {}}]
    for r in rows:
        r["report"] = {"checks": []}
    changed, unknown = lb.revision_staleness(rows, {r["id"] for r in rows})
    # a probe error is not a change; RLDS rows without a revision are queued too
    assert changed == {"o/changed"} and unknown == {"o/unknown", "rlds/x"}


def test_finding_digest_survives_mixed_field_types():
    # the 2026-11-20 board run lost its history to this: real findings mix
    # None, str and int in the same field across findings
    mixed = [{"severity": "WARN", "details": {"kind": None, "dim": 3}},
             {"severity": "WARN", "details": {"kind": "partial", "dim": None, "episodes": [4, 2]}},
             {"severity": "FAIL", "episode_index": 7, "details": {}}]
    e = _entry(findings={"stats-drift": mixed})
    a = H.row_snapshot(e, CHECKS)["fd"]["stats-drift"]
    e2 = _entry(findings={"stats-drift": list(reversed(mixed))})
    assert H.row_snapshot(e2, CHECKS)["fd"]["stats-drift"] == a  # order-independent


def test_history_writes_on_the_real_board(tmp_path):
    # the committed board, whatever its rows hold today
    board = Path(__file__).resolve().parent.parent / "docs" / "data.json"
    if not board.exists():
        pytest.skip("no committed board")
    entries = json.loads(board.read_text())["datasets"]
    doc = _lb().write_history(tmp_path, entries, "2026-11-20 23:07 UTC")
    assert len(doc["rows"]) == len(entries)

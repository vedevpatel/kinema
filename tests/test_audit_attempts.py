"""Audit-attempt provenance: every actual attempt is recorded apart from the
last successful audit, a failed retry never touches that success, recovery
clears the failure marker, and failure reasons are sanitized before they are
stored. Every secret below is fake and assembled at runtime, so no literal
credential shape exists in this file."""

from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ENGINE = "e0e0e0e0e0e0"
NEW_ENGINE = "b2b2b2b2b2b2"
SHA_A = "0123456789abcdef0123456789abcdef01234567"
SHA_B = "fedcba9876543210fedcba9876543210fedcba98"


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPTS / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


lb = _load_leaderboard()

# fake, low-entropy credentials, built at runtime (secret scanners read source)
FAKE = {
    "hf": "hf_" + "FAKE" * 8,
    "gh": "ghp_" + "Fake" * 9,
    "aws": "AKIA" + "FAKE" * 4,
    "jwt": ".".join(["eyJ" + "fakefake" * 2, "fakefake" * 3, "fakefake" * 3]),
    "bearer": "Fake" * 6,
    "cookie": "fakesession" + "0" * 12,
    "sig": "fakesig" + "0" * 20,
    "pw": "fakepass" + "1" * 6,
}


def _clock(*stamps):
    """A fixed clock: each call returns the next stamp."""
    it = iter(dt.datetime.fromisoformat(s.replace("Z", "+00:00")) for s in stamps)
    return lambda: next(it)


def _config(**kw):
    cfg = {"kind": "hub-sample", "max_episodes": 250, "size_cap_mb": 200, "sampled": True,
           "probes": True, "transition_cap": 10}
    cfg.update(kw)
    return cfg


def _report(source="org/a", score_findings=1):
    from kinema.checks import Severity
    from kinema.checks.base import CheckResult, Finding
    from kinema.report import DatasetReport

    rep = DatasetReport(source, "v3.0", 40, 4000, 30.0, truncated=True,
                        video_audited=lb.VIDEO_AUDIT_MARK)
    rep.results.append(CheckResult("stuck-actuator", "stuck actuator", findings=[
        Finding("stuck-actuator", Severity.WARN, f"joint {i} does not move")
        for i in range(score_findings)]))
    rep.audit_meta = {"revision": SHA_A, "sample": {"episodes": 40, "episodes_digest": "abc",
                                                    "videos": 2, "videos_digest": "def"}}
    return rep


def _ok(rev=SHA_A):
    def audit(d, pin):
        pin["revision"] = rev
        rep = _report(d["id"])
        rep.audit_meta["revision"] = rev
        return rep
    return audit


def _fail(exc, pin_rev=None):
    def audit(d, pin):
        if pin_rev:
            pin["revision"] = pin_rev
        raise exc
    return audit


ROW = {"id": "org/a", "downloads": 10, "likes": 1, "data_bytes": 300 * 2 ** 20, "lane": "top"}
SUCCESS_KEYS = ("report", "audited_at", "audit_started_at", "revision", "engine", "sample",
                "audit_config", "audit_max_episodes", "sampled", "slug")


def _success(engine=ENGINE, t0="2026-11-20T22:30:00Z", t1="2026-11-20T22:36:16Z"):
    return lb.run_attempt(dict(ROW), engine, _config(), _ok(), clock=_clock(t0, t1))


def _failure(exc, engine=NEW_ENGINE, pin_rev=None, t0="2026-11-22T06:10:00Z",
             t1="2026-11-22T06:12:09Z", row=ROW):
    return lb.run_attempt(dict(row), engine, _config(), _fail(exc, pin_rev), clock=_clock(t0, t1))


# --- what an attempt records ------------------------------------------------

def test_success_records_its_attempt_and_its_own_times():
    e = _success()
    att = e["latest_attempt"]
    assert att == {"scheme": lb.ATTEMPT_SCHEME, "started_at": "2026-11-20T22:30:00Z",
                   "finished_at": "2026-11-20T22:36:16Z", "outcome": "success", "engine": ENGINE,
                   "revision": SHA_A, "revision_status": "pinned", "config": _config()}
    assert e["audit_started_at"] == "2026-11-20T22:30:00Z"
    assert e["audited_at"] == "2026-11-20T22:36:16Z"  # completion time, as before
    assert e["audit_config"] == _config() and e["revision"] == SHA_A
    assert e["report"]["score"] < 100 and "status" not in e and "reason" not in e


def test_failure_after_revision_resolved_records_the_pinned_revision():
    e = _failure(lb.DatasetError("download failed mid-shard"), pin_rev=SHA_B)
    att = e["latest_attempt"]
    assert att["outcome"] == "error" and att["revision"] == SHA_B and att["revision_status"] == "pinned"
    assert att["started_at"] == "2026-11-22T06:10:00Z" and att["finished_at"] == "2026-11-22T06:12:09Z"
    assert att["engine"] == NEW_ENGINE and att["config"] == _config()
    # a failure stores no audit result: no top-level revision, score or audit time
    assert not {"report", "revision", "sample", "audited_at", "audit_started_at"} & set(e)
    assert e["status"] == "error" and e["reason"] == att["reason"]


def test_failure_before_revision_resolved_marks_it_unknown():
    e = _failure(TimeoutError("resolve timed out"))
    att = e["latest_attempt"]
    assert att["revision"] is None and att["revision_status"] == "unknown"
    assert att["reason"] == "audit error: TimeoutError: resolve timed out"


@pytest.mark.parametrize("exc,outcome", [
    (lb.DatasetError("GatedRepoError: 403 Cannot access gated repo"), "gated"),
    (lb.DatasetError("not a LeRobot dataset: no meta/info.json"), "unsupported"),
    (RuntimeError("boom"), "error"),
])
def test_failure_classes(exc, outcome):
    e = _failure(exc)
    assert e["status"] == e["latest_attempt"]["outcome"] == outcome


def test_classification_reads_past_the_length_cap():
    # the gated marker sits after 600 characters of noise: still classified gated
    e = _failure(lb.DatasetError("x" * 600 + " GatedRepoError"))
    assert e["status"] == "gated" and len(e["reason"]) <= lb.REASON_MAX_CHARS


def test_a_report_write_failure_fails_the_attempt():
    def boom(entry):
        raise OSError("disk full")
    e = lb.run_attempt(dict(ROW), ENGINE, _config(), _ok(), clock=_clock(
        "2026-11-20T22:30:00Z", "2026-11-20T22:36:16Z"), on_success=boom)
    assert "report" not in e and e["latest_attempt"]["outcome"] == "error"
    assert e["latest_attempt"]["revision"] == SHA_A  # it did read at the pinned revision


def test_attempt_config_reflects_what_the_run_did():
    assert lb.attempt_config({"data_bytes": 10}, 50, 200, True, 10) == {
        "kind": "hub-sample", "max_episodes": 50, "size_cap_mb": 200, "sampled": False,
        "probes": True, "transition_cap": 10}
    rlds = lb.attempt_config({"rlds": True, "data_bytes": 1}, 50, 200, True, 0)
    assert rlds["kind"] == "rlds-metadata" and rlds["sampled"] and not rlds["probes"]


# --- merging an attempt into the stored row ---------------------------------

def _assert_success_untouched(kept, success):
    for k in SUCCESS_KEYS:
        assert kept.get(k) == success.get(k), k
    # score, verdicts, findings and coverage live in the report: byte-identical
    assert json.dumps(kept["report"], sort_keys=True) == json.dumps(success["report"], sort_keys=True)


def test_success_then_failure_keeps_every_success_field():
    success = _success()
    frozen = copy.deepcopy(success)
    fail = _failure(lb.DatasetError("HTTP 503 temporarily unavailable"), pin_rev=SHA_B)
    kept = lb.preserve_failed_retry(success, fail)
    _assert_success_untouched(kept, frozen)
    assert kept["engine"] == ENGINE != fail["engine"]  # never the failed attempt's engine
    assert kept["revision"] == SHA_A  # never the failed attempt's revision
    assert kept["latest_attempt"] == fail["latest_attempt"]
    assert kept["last_audit_error"] == fail["reason"]
    assert "status" not in kept and "reason" not in kept
    assert success == frozen  # the stored row object is not mutated


def test_repeated_failures_keep_the_success_and_the_newest_attempt():
    success = _success()
    frozen = copy.deepcopy(success)
    row = success
    for i, exc in enumerate((TimeoutError("first"), lb.DatasetError("second"),
                             lb.DatasetError("GatedRepoError: now gated"))):
        fail = _failure(exc, t0=f"2026-11-2{2 + i}T06:10:00Z", t1=f"2026-11-2{2 + i}T06:12:09Z")
        row = lb.preserve_failed_retry(row, fail)
        _assert_success_untouched(row, frozen)
        assert row["latest_attempt"]["finished_at"] == f"2026-11-2{2 + i}T06:12:09Z"
    assert row["latest_attempt"]["outcome"] == "gated"  # a gated retry keeps the success too
    assert "now gated" in row["last_audit_error"]


def test_first_attempt_failure_is_an_honest_no_score_row():
    fail = _failure(lb.DatasetError("GatedRepoError: 403"))
    row = lb.preserve_failed_retry({}, fail)
    assert row is fail and "report" not in row and row["status"] == "gated"
    assert "last_audit_error" not in row and row["latest_attempt"]["outcome"] == "gated"


def test_recovery_clears_the_failure_marker():
    row = lb.preserve_failed_retry(_success(), _failure(TimeoutError("x")))
    assert row["last_audit_error"]
    rec = _success(engine=NEW_ENGINE, t0="2026-11-23T01:00:00Z", t1="2026-11-23T01:05:00Z")
    rec["last_audit_error"] = "leaked marker"  # even if a caller carried one over
    row = lb.preserve_failed_retry(row, rec)
    assert not {"last_audit_error", "status", "reason"} & set(row)
    assert row["latest_attempt"]["outcome"] == "success" and row["engine"] == NEW_ENGINE
    assert row["audited_at"] == "2026-11-23T01:05:00Z"
    assert not lb.audit_stale({**row, "engine": NEW_ENGINE}, NEW_ENGINE, 0)
    page = lb.render_audit_page(row, NEW_ENGINE, 0)
    assert "Latest attempt: failed" not in page and "re-audit failed" not in page
    assert "class='state s-current'" in page


def test_recovery_of_a_never_audited_row():
    row = lb.preserve_failed_retry({}, _failure(RuntimeError("x")))
    row = lb.preserve_failed_retry(row, _success())
    assert row["report"] and not {"status", "reason", "last_audit_error"} & set(row)


def test_legacy_rows_merge_without_invented_metadata():
    legacy_ok = {"id": "org/a", "engine": "old", "report": {"score": 80}, "last_audit_error": "429"}
    fail = _failure(TimeoutError("again"))
    kept = lb.preserve_failed_retry(legacy_ok, fail)
    assert kept["report"] == {"score": 80} and kept["engine"] == "old"
    assert "audited_at" not in kept and "audit_started_at" not in kept  # nothing invented
    # a caller-built failure without an attempt record never leaves an old success as latest
    kept = lb.preserve_failed_retry({**_success()}, {"id": "org/a", "status": "error", "reason": "x"})
    assert "latest_attempt" not in kept and kept["last_audit_error"] == "x"


def test_failed_retry_leaves_the_history_snapshot_input_unchanged():
    from kinema.history import row_snapshot

    success = _success()
    kept = lb.preserve_failed_retry(success, _failure(TimeoutError("x")))
    checks = ["stuck-actuator"]
    assert row_snapshot(kept, checks) == row_snapshot(success, checks)


# --- sanitizing failure reasons ---------------------------------------------

def _leaks(text: str) -> list[str]:
    return [k for k, v in FAKE.items() if v in text]


@pytest.mark.parametrize("raw", [
    f"401 for url https://huggingface.co/api/datasets/x?token={FAKE['hf']}&expand=1",
    f"headers={{'Authorization': 'Bearer {FAKE['bearer']}', 'user-agent': 'x'}}",
    f"Authorization: Bearer {FAKE['bearer']}\nnext line",
    f"Authorization: Basic {FAKE['bearer']}",
    f"Cookie: session={FAKE['cookie']}; theme=dark",
    f"Set-Cookie: sid={FAKE['cookie']}; Path=/; HttpOnly",
    f"GET https://user:{FAKE['pw']}@mirror.example.org/file failed",
    f"GET https://{FAKE['hf']}@huggingface.co/file failed",
    f"https://cas.example/x?X-Amz-Credential={FAKE['aws']}%2F20261122&X-Amz-Signature={FAKE['sig']}",
    f"https://storage.googleapis.com/b/o?X-Goog-Signature={FAKE['sig']}&alt=media",
    f"config {{'password': '{FAKE['pw']}', 'hf_token': \"{FAKE['hf']}\"}}",
    f"api_key={FAKE['pw']} rejected",
    f"token {FAKE['hf']} is invalid",
    f"push with {FAKE['gh']} denied",
    f"aws {FAKE['aws']} expired",
    f"jwt {FAKE['jwt']} expired",
    "key: -----BEGIN RSA PRIVATE KEY-----\nMIIfake\nfake\n-----END RSA PRIVATE KEY----- end",
])
def test_credentials_are_redacted(raw):
    out = lb.sanitize_reason(raw)
    assert not _leaks(out), out
    assert "MIIfake" not in out
    assert lb._REDACTED in out
    assert lb.sanitize_reason(out) == out  # idempotent: re-sanitizing never drifts


def test_redaction_keeps_the_useful_text():
    raw = ("audit error: HTTPError: 503 Server Error for url "
           f"https://huggingface.co/datasets/org/x/resolve/main/data/f.parquet?download=true&token={FAKE['hf']}")
    out = lb.sanitize_reason(raw)
    assert out == ("audit error: HTTPError: 503 Server Error for url "
                   "https://huggingface.co/datasets/org/x/resolve/main/data/f.parquet?download=true&token=[redacted]")
    # ordinary key-ish words and quoted values are not redacted
    plain = "KeyError: key='observation.state' missing; monkey=1; basic checks failed"
    assert lb.sanitize_reason(plain) == plain


def test_local_paths_whitespace_and_controls():
    raw = "  failed\n\treading /tmp/kinema-leaderboard-abc123/data/x.parquet\x00 \u202eevil  "
    out = lb.sanitize_reason(raw)
    assert out == "failed reading <sample> \ufffdevil"  # the sample path (and its NUL) is gone
    assert lb.sanitize_reason("a\x00b\x1bc\u2066d") == "a\ufffdb\ufffdc\ufffdd"
    assert "\n" not in out and "kinema-leaderboard-" not in out


def test_truncation_caps_length_and_never_splits_a_marker():
    out = lb.sanitize_reason("y" * 2000)
    assert len(out) == lb.REASON_MAX_CHARS and out.endswith(lb._TRUNCATED)
    assert lb.sanitize_reason(out) == out
    # a cut that would land inside "[redacted]" drops the partial marker
    for n in range(0, 12):
        raw = "z" * (lb.REASON_MAX_CHARS - len(lb._TRUNCATED) - n) + f" token={FAKE['hf']} tail"
        out = lb.sanitize_reason(raw)
        assert len(out) <= lb.REASON_MAX_CHARS and not _leaks(out)
        if out.endswith(lb._TRUNCATED):
            kept = out[:-len(lb._TRUNCATED)]
            assert not re.search(r"\[r?e?d?a?c?t?e?d?$", kept), kept[-20:]
        assert lb.sanitize_reason(out) == out
    assert lb.sanitize_reason("short", cap=None) == "short"


def test_secrets_never_reach_the_stored_entry():
    exc = lb.DatasetError(f"401 Unauthorized https://huggingface.co/x?token={FAKE['hf']} "
                          f"Authorization: Bearer {FAKE['bearer']} " + "pad " * 300)
    e = _failure(exc, pin_rev=SHA_B)
    blob = json.dumps(e)
    assert not _leaks(blob) and len(e["reason"]) <= lb.REASON_MAX_CHARS
    kept = lb.preserve_failed_retry(_success(), e)
    assert not _leaks(json.dumps(kept))


def test_check_crash_reasons_are_sanitized(monkeypatch, tmp_path):
    # a check that crashes publishes its exception text as the skip reason
    import types

    from kinema.checks.base import CheckResult

    class Crashing:
        id, title = "crashy", "crashy"

        def run(self, ctx, cfg):
            raise ValueError(f"bad response for https://x.example/?token={FAKE['hf']}\n" + "z" * 900)

        def skip(self, reason):
            return CheckResult(self.id, self.title, skipped=True, skip_reason=reason)

    ctx = types.SimpleNamespace(
        format_version="v3.0", total_episodes=1, row_count=10, fps=10.0, truncated=False,
        unreadable_files=[], episode_ids=lambda: [0])
    monkeypatch.setattr(lb, "resolve_revision", lambda repo: None)
    monkeypatch.setattr(lb, "download_sample", lambda *a, **kw: (tmp_path, {}))
    monkeypatch.setattr(lb, "_target_video_files", lambda *a: [])
    monkeypatch.setattr(lb, "load_dataset", lambda *a, **kw: ctx)
    monkeypatch.setattr(lb, "ALL_CHECKS", [Crashing()])
    rep = lb.audit_one("org/a", 10, 1 << 20)
    reason = rep.results[0].skip_reason
    assert reason.startswith("check crashed: ValueError: bad response")
    assert not _leaks(reason) and len(reason) <= len("check crashed: ") + lb.REASON_MAX_CHARS


def test_unservable_file_reasons_are_sanitized(monkeypatch, tmp_path):
    # an unservable data file's error text becomes dropped-frames evidence
    import huggingface_hub

    files = [types_file("meta/info.json"), types_file("data/a.parquet"), types_file("data/b.parquet")]
    monkeypatch.setattr(huggingface_hub, "list_repo_tree", lambda *a, **kw: files)

    def fetch(repo, name, dest, revision=None):
        if name == "data/a.parquet":
            raise lb.DatasetError(f"the Hub cannot serve {name}: https://cas.example/x?"
                                  f"X-Amz-Signature={FAKE['sig']}&X-Amz-Credential={FAKE['aws']}")
    monkeypatch.setattr(lb, "_hub_download", fetch)
    unservable: list = []
    lb.download_sample("org/a", 1 << 30, tmp_path, unservable)
    assert [n for n, _ in unservable] == ["data/a.parquet"]
    assert not _leaks(unservable[0][1]) and len(unservable[0][1]) <= 200


def types_file(path, size=10):
    import types

    return types.SimpleNamespace(path=path, size=size)


def test_legacy_reasons_are_sanitized_when_the_board_is_written(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    legacy = {"id": "org/legacy", "downloads": 1, "lane": "top", "engine": ENGINE,
              "report": _success()["report"],
              "last_audit_error": f"audit error: 401 token={FAKE['hf']}\n\nsecond line"}
    gated = {"id": "org/gated", "downloads": 1, "lane": "new", "engine": ENGINE, "status": "gated",
             "reason": "GatedRepoError: 403 " + "long " * 200}
    lb._write(out, [legacy, gated], "2026-11-22 06:00 UTC", max_episodes=0)
    data = (out / "data.json").read_text()
    assert not _leaks(data)
    rows = {r["id"]: r for r in json.loads(data)["datasets"]}
    assert rows["org/legacy"]["last_audit_error"] == "audit error: 401 token=[redacted] second line"
    assert len(rows["org/gated"]["reason"]) == lb.REASON_MAX_CHARS
    first = {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    lb._write(out, json.loads(data)["datasets"], "2026-11-22 06:00 UTC", max_episodes=0)
    assert {p.name: p.read_bytes() for p in out.rglob("*") if p.is_file()} == first  # idempotent
    for p in out.rglob("*.html"):
        assert not _leaks(p.read_text())


def test_render_sanitizes_reasons_read_from_an_unsanitized_board(tmp_path):
    # --render-pages (checkpoints) renders data.json as stored, without writing it
    out = tmp_path / "docs"
    out.mkdir()
    rows = [{"id": "org/raw", "downloads": 1, "lane": "top", "engine": ENGINE, "status": "error",
             "reason": f"audit error: token={FAKE['hf']}"},
            {**_success(), "id": "org/raw2", "last_audit_error": f"Bearer {FAKE['bearer']}"}]
    (out / "data.json").write_text(json.dumps({"generated": "g", "datasets": rows}))
    assert lb.render_site(out, max_episodes=0) == 0
    for p in out.rglob("*.html"):
        assert not _leaks(p.read_text()), p


# --- rendering ---------------------------------------------------------------

def _failed_retry_row():
    return lb.preserve_failed_retry(_success(), _failure(
        lb.DatasetError("HTTP 503 temporarily unavailable"), pin_rev=SHA_B))


def _body(page):
    return page.split("<body>", 1)[1]


def test_failed_retry_page_separates_the_two_records():
    page = lb.render_audit_page(_failed_retry_row(), NEW_ENGINE, 250)
    body = _body(page)
    a = body.index("<h2>Latest attempt: failed</h2>")
    s = body.index("<h2>Last successful audit</h2>")
    assert a < s
    attempt, success = body[a:s], body[s:]
    # the failed attempt: its own times, engine, revision, config, reason
    assert "<th>attempt started</th><td><time datetime='2026-11-22T06:10:00Z'>2026-11-22 06:10:00 UTC" in attempt
    assert "<th>attempt ended</th><td><time datetime='2026-11-22T06:12:09Z'>2026-11-22 06:12:09 UTC" in attempt
    assert f"<code>{NEW_ENGINE}</code>" in attempt and SHA_B[:12] in attempt
    assert "pinned and read by this attempt" in attempt
    assert "HTTP 503 temporarily unavailable" in attempt and "max 250 episodes" in attempt
    assert "class=score" not in attempt and "did not replace or change" in attempt
    # the success: its own times, engine and revision, none of the attempt's
    assert "<th>audit started</th><td><time datetime='2026-11-20T22:30:00Z'>" in success
    assert "<th>audit completed</th><td><time datetime='2026-11-20T22:36:16Z'>" in success
    assert f"<th>check engine</th><td><code>{ENGINE}</code> <span class=dim>(the board now runs" in success
    assert SHA_A[:12] in success
    assert SHA_B[:12] not in success and "2026-11-22" not in success
    assert "(stored, stale)" in success
    # the banner never calls the old result current or dates it by the attempt
    banner = body[:a]
    assert "s-error" in banner and "Latest re-audit failed; stale result." in banner
    assert "last successful audit, completed <time datetime='2026-11-20T22:36:16Z'>" in banner
    assert "the latest re-audit attempt, which ended <time datetime='2026-11-22T06:12:09Z'>" in banner
    # the stored-result reasons come first, so their "it" never reads as the attempt
    assert banner.index("older check logic") < banner.index("the latest re-audit attempt")
    assert "state s-current" not in page and "Latest stored result" not in page


def test_never_audited_error_page_with_attempt_record():
    row = _failure(TimeoutError("read timed out"), row={**ROW, "id": "org/never"})
    page = lb.render_audit_page(row, NEW_ENGINE)
    assert "recorded status: <b>error</b>" in page and "There is no score" in page
    assert "<h2>Latest attempt</h2>" in page and "failed: <b>error</b>" in page
    assert "2026-11-22 06:12:09 UTC" in page and "unknown <span class=dim>(not pinned" in page
    assert "<h2>Last successful audit</h2><p>None recorded" in page and "class=score" not in page


def test_gated_page_and_legacy_rows_read_not_recorded():
    gated = {"id": "org/gated", "downloads": 3, "lane": "new", "engine": ENGINE, "status": "gated",
             "reason": "audit error: GatedRepoError: 403 Client Error."}
    page = lb.render_audit_page(gated, ENGINE)
    assert "recorded status: <b>gated</b>" in page and "class=score" not in page
    for label in ("attempt started", "attempt ended", "dataset revision", "run configuration"):
        assert re.search(rf"<th>{label}</th><td><span class=dim>not recorded</span>", page), label
    assert "unknown" not in _body(page).lower()  # legacy: not recorded, never "unknown"
    legacy = {**_success(), "last_audit_error": "429 Too Many Requests"}
    for k in ("latest_attempt", "audit_started_at", "audit_config"):
        legacy.pop(k)
    page = lb.render_audit_page(legacy, ENGINE, 250)
    attempt = _body(page).split("<h2>Latest attempt: failed</h2>", 1)[1].split("<h2>Last successful", 1)[0]
    for label in ("attempt started", "attempt ended", "check engine", "dataset revision", "run configuration"):
        assert re.search(rf"<th>{label}</th><td><span class=dim>not recorded</span>", attempt), label
    assert "429 Too Many Requests" in attempt and "its time was not recorded" in page
    assert "<th>audit started</th><td><span class=dim>not recorded</span>" in page
    assert "max 250 episodes <span class=dim>(the run's other settings were not recorded)" in page


def test_board_flags_a_failed_retry_with_the_results_own_date():
    row = _failed_retry_row()
    html = lb.render_index([row], "2026-11-22 06:15 UTC", page_ids=lb.audit_page_ids([row]))
    assert "(retry failed; result from 2026-11-20)" in html
    assert "latest re-audit attempt failed (ended 2026-11-22 06:12:09 UTC): HTTP 503" in html
    assert "the latest re-audit attempt failed; this is the last successful audit" in html
    assert "deepening" not in html
    legacy = {k: v for k, v in row.items() if k not in ("latest_attempt", "audited_at")}
    html = lb.render_index([legacy], "g", page_ids=lb.audit_page_ids([legacy]))
    assert "(retry failed; result date not recorded)" in html and "time not recorded" in html


def test_html_in_reasons_is_inert():
    evil = "<script>alert(1)</script><img src=x onerror=alert(2)>\"'&"
    row = lb.preserve_failed_retry(_success(), _failure(RuntimeError(evil)))
    never = _failure(lb.DatasetError(evil), row={**ROW, "id": "org/never"})
    for page in (lb.render_audit_page(row, ENGINE, 0), lb.render_audit_page(never, ENGINE, 0),
                 lb.render_index([row, never], "g", page_ids=lb.audit_page_ids([row, never]))):
        assert "<script>alert" not in page and "<img" not in page
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_rendering_is_deterministic_with_fixed_timestamps(tmp_path):
    rows = [_failed_retry_row(), _failure(TimeoutError("x"), row={**ROW, "id": "org/never"}),
            {"id": "org/gated", "downloads": 3, "lane": "new", "engine": ENGINE, "status": "gated",
             "reason": "GatedRepoError: 403"}]
    pages = []
    for name in ("a", "b"):
        out = tmp_path / name
        out.mkdir()
        lb._write(out, copy.deepcopy(rows), "2026-11-22 06:15 UTC", max_episodes=250)
        pages.append({p.name: p.read_bytes() for p in (out / "d").iterdir()})
    assert pages[0] == pages[1] and len(pages[0]) == 3
    # no render clock: the only dates on the pages are the recorded ones
    dates = {d for text in pages[0].values() for d in re.findall(r"\d{4}-\d\d-\d\d", text.decode())}
    assert dates == {"2026-11-20", "2026-11-22"}


# --- the nightly flow, end to end (no network) -------------------------------

def _run_main(lb, monkeypatch, tmp_path, engine, audit, now, extra=()):
    monkeypatch.setattr(lb, "engine_version", lambda: engine)
    monkeypatch.setattr(lb, "find_lerobot_datasets", lambda *a, **kw: [dict(ROW)])
    monkeypatch.setattr(lb, "audit_one", audit)
    monkeypatch.setattr(lb, "_utcnow", _clock(*now))
    monkeypatch.setattr(sys, "argv", ["leaderboard", "--out", str(tmp_path), "--max-episodes", "250",
                                      "--no-revision-probe", *extra])
    monkeypatch.chdir(tmp_path)  # held-transitions.json, if any, stays in the sandbox
    assert lb.main() == 0
    return json.loads((tmp_path / "data.json").read_text())["datasets"][0]


def _one_audit(fn):
    def audit_one(repo_id, max_episodes, cap_bytes, prior=None, gate=None, probe=True, pin=None):
        return fn({"id": repo_id}, pin)
    return audit_one


def test_nightly_success_failure_failure_recovery(monkeypatch, tmp_path):
    lb = _load_leaderboard()
    first = _run_main(lb, monkeypatch, tmp_path, ENGINE, _one_audit(_ok()),
                      ["2026-11-20T22:30:00Z", "2026-11-20T22:36:16Z"])
    assert first["latest_attempt"]["outcome"] == "success" and first["engine"] == ENGINE
    frozen = copy.deepcopy(first)
    # engine rotation queues a re-audit; it fails after pinning a new commit
    second = _run_main(lb, monkeypatch, tmp_path, NEW_ENGINE,
                       _one_audit(_fail(lb.DatasetError("HTTP 503"), pin_rev=SHA_B)),
                       ["2026-11-22T06:10:00Z", "2026-11-22T06:12:09Z"])
    _assert_success_untouched(second, frozen)
    assert second["engine"] == ENGINE and second["latest_attempt"]["engine"] == NEW_ENGINE
    assert second["latest_attempt"]["revision"] == SHA_B and second["last_audit_error"] == "HTTP 503"
    # the failure marker keeps it queued; a second failure, revision unknown this time
    third = _run_main(lb, monkeypatch, tmp_path, NEW_ENGINE,
                      _one_audit(_fail(TimeoutError("resolve timed out"))),
                      ["2026-11-23T06:10:00Z", "2026-11-23T06:11:00Z"])
    _assert_success_untouched(third, frozen)
    assert third["latest_attempt"]["finished_at"] == "2026-11-23T06:11:00Z"
    assert third["latest_attempt"]["revision_status"] == "unknown"
    page = next((tmp_path / "d").iterdir()).read_text()
    assert "2026-11-23 06:11:00 UTC" in page and "2026-11-20 22:36:16 UTC" in page
    # recovery
    fourth = _run_main(lb, monkeypatch, tmp_path, NEW_ENGINE, _one_audit(_ok(SHA_B)),
                       ["2026-11-24T06:10:00Z", "2026-11-24T06:15:00Z"])
    assert not {"last_audit_error", "status", "reason"} & set(fourth)
    assert fourth["engine"] == NEW_ENGINE and fourth["revision"] == SHA_B
    assert fourth["audited_at"] == "2026-11-24T06:15:00Z"
    assert fourth["latest_attempt"]["outcome"] == "success"
    page = next((tmp_path / "d").iterdir()).read_text()
    assert "class='state s-current'" in page and "re-audit failed" not in page and "Latest attempt: failed" not in page
    index = (tmp_path / "index.html").read_text()
    assert "retry failed" not in index


def test_skipped_queue_items_and_revision_probes_are_not_attempts(monkeypatch, tmp_path):
    lb = _load_leaderboard()
    first = _run_main(lb, monkeypatch, tmp_path, ENGINE, _one_audit(_ok()),
                      ["2026-11-20T22:30:00Z", "2026-11-20T22:36:16Z"])
    calls = []
    # the revision probe reports an upstream move; the time budget then starts nothing
    monkeypatch.setattr(lb, "resolve_revision", lambda repo: calls.append(repo) or SHA_B)
    monkeypatch.setattr(lb, "engine_version", lambda: ENGINE)
    monkeypatch.setattr(lb, "find_lerobot_datasets", lambda *a, **kw: [dict(ROW)])
    monkeypatch.setattr(lb, "audit_one", lambda *a, **kw: pytest.fail("no audit may start"))
    monkeypatch.setattr(lb, "_utcnow", lambda: pytest.fail("no attempt may be stamped"))
    monkeypatch.setattr(sys, "argv", ["leaderboard", "--out", str(tmp_path), "--max-episodes", "250",
                                      "--time-budget", "1e-9"])
    assert lb.main() == 0
    assert calls == ["org/a"]  # the probe ran
    row = json.loads((tmp_path / "data.json").read_text())["datasets"][0]

    def audit_state(r):  # discovery bookkeeping aside
        return {k: v for k, v in r.items() if not k.startswith("discovery_")}
    assert audit_state(row) == audit_state(first)  # ...and recorded nothing: not an attempt

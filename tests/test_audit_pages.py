"""Shareable per-dataset audit pages (docs/d/): identifiers, escaping, links,
honest states, safe pruning, deterministic output, full board link coverage."""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import re
import shutil
import sys
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ENGINE = "e0e0e0e0e0e0"
OLD_ENGINE = "a1a1a1a1a1a1"
PAGES_BASE = "https://vedevpatel.github.io/kinema/"


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPTS / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


lb = _load_leaderboard()


def _check(cid="stats-drift", verdict="PASS", findings=(), skipped=False, **kw):
    return {"id": cid, "title": f"{cid} title", "verdict": "SKIP" if skipped else verdict,
            "weight": 6, "weight_basis": "measured", "summary": kw.get("summary", f"{cid} ok"),
            "skipped": skipped, "skip_reason": kw.get("skip_reason", "not run" if skipped else ""),
            "details": kw.get("details", {}), "findings": list(findings), "charge": kw.get("charge", 0)}


def _hub_row(rid="org/ds", **kw):
    rep = {"source": rid, "format_version": "v3.0", "episodes": 50, "frames": 5000, "fps": 30.0,
           "truncated": True, "video_audited": 5, "score": 90, "score_model": lb.SCORE_MODEL,
           "verdict": "WARN",
           "checks": [_check(), _check("stuck-actuator", "WARN", charge=10, findings=[
               {"severity": "WARN", "message": "joint 0 does not move", "episode_index": None, "details": {}},
               {"severity": "INFO", "message": "joint 3 rarely moves", "episode_index": 7, "details": {}}]),
               _check("reward-done", skipped=True, skip_reason="no reward/done fields")]}
    row = {"id": rid, "downloads": 1234, "likes": 1, "data_bytes": 3 * 2 ** 30, "lane": "top",
           "slug": rid.replace("/", "--"), "engine": ENGINE, "audit_max_episodes": 50, "sampled": True,
           "revision": "0123456789abcdef0123456789abcdef01234567",
           "sample": {"episodes": 50, "episodes_digest": "abc", "videos": 3, "videos_digest": "def"},
           "audited_at": "2026-11-20T22:36:16Z", "discovery_current": True, "report": rep}
    row.update(kw)
    return row


def _rlds_row(name="rlds/toy"):
    return {"id": name, "rlds": True, "rlds_prefix": "robotics/toy/", "rlds_bucket": "gresearch",
            "version": "1.0.0", "downloads": 0, "likes": 0, "data_bytes": 10, "lane": "rlds",
            "slug": name.replace("/", "--"), "engine": ENGINE, "audit_max_episodes": 50, "sampled": True,
            "revision": "gcs:1da032540176", "sample": None, "audited_at": "2026-11-21T05:41:39Z",
            "discovery_current": True,
            "report": {"source": name, "format_version": "rlds-1.0.0", "episodes": 118, "frames": 0,
                       "fps": None, "truncated": False, "video_audited": 5, "score": 100,
                       "score_model": lb.SCORE_MODEL, "verdict": "PASS",
                       "checks": [_check("rlds-episode-counts"),
                                  _check("rlds-step-structure", skipped=True, skip_reason="sample mode off")]}}


def _status_row(rid, status, reason="audit error: GatedRepoError: 403", **kw):
    row = {"id": rid, "downloads": 5, "likes": 0, "data_bytes": 1, "lane": "new",
           "slug": rid.replace("/", "--"), "engine": OLD_ENGINE, "status": status, "reason": reason}
    row.update(kw)
    return row


def _board():
    return [
        _hub_row(),
        _hub_row("nvidia/Big-Repo//LeRobot/task.A", subpath="LeRobot/task.A", repo_scope=True),
        _hub_row("org/stale", engine=OLD_ENGINE),
        _rlds_row(),
        _status_row("org/gated", "gated"),
        _status_row("org/broken", "error", reason="audit error: Timeout"),
        _status_row("org/odd", "unsupported", reason="not a LeRobot dataset"),
    ]


# --- identifiers ------------------------------------------------------------

def test_page_id_is_stable_and_pinned():
    # a page URL is a permalink: the scheme must not drift silently
    assert lb.audit_page_id("BitRobot/HIW-500-LeRobot") == "bitrobot-hiw-500-lerobot-e34d7b7b61ad"
    assert lb.audit_page_id("org/ds") == lb.audit_page_id("org/ds")


@pytest.mark.parametrize("rid", [
    "../../etc/passwd", "/abs/path", "org/..", "..", ".", "org/ds\x00.html", "a\\b\\..\\c",
    "_config/x", ".git/config", "~user/x", "#frag/x", "%2e%2e/%2f", "org/ds?x=1#y",
    "org/ds/../../index", "org/ds‮gpj.html", "\ud800/lonely", "ÄÖÜ/日本語", "x" * 500,
    "nvidia/Repo//LeRobot/../../../etc", "CON/aux", "org/ds.html",
])
def test_page_ids_never_become_paths(rid, tmp_path):
    pid = lb.audit_page_id(rid)
    assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,47}-[0-9a-f]{12}", pid)
    assert not pid.startswith(("_", ".", "-"))  # Jekyll drops _ and . files
    assert urllib.parse.quote(pid, safe="") == pid  # URL-safe as is
    out = tmp_path / "docs"
    out.mkdir()
    lb.write_audit_pages(out, [_hub_row(rid)], engine=ENGINE)
    files = [p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()]
    assert files == [Path("docs") / "d" / f"{pid}.html"]


def test_nested_ids_get_distinct_pages():
    ids = lb.audit_page_ids([{"id": "org/repo"}, {"id": "org/repo//sub/a"},
                             {"id": "org/repo//sub-a"}, {"id": "org/repo/sub/a"}])
    assert len(set(ids.values())) == 4


def test_case_variants_get_distinct_pages():
    ids = lb.audit_page_ids([{"id": "Org/DS"}, {"id": "org/ds"}])
    assert len({p.casefold() for p in ids.values()}) == 2


def test_duplicate_row_ids_fail_instead_of_overwriting():
    with pytest.raises(lb.PageIdCollision):
        lb.audit_page_ids([_hub_row(), _hub_row()])


def test_digest_collisions_fail(monkeypatch):
    monkeypatch.setattr(lb, "_page_digest", lambda rid: "0" * 12)
    with pytest.raises(lb.PageIdCollision, match="refusing to overwrite"):
        lb.audit_page_ids([{"id": "org/ds"}, {"id": "org//ds"}])  # same readable prefix
    with pytest.raises(lb.PageIdCollision):
        lb.audit_page_ids([{"id": "Org/DS"}, {"id": "org/ds"}])  # case-folded clash
    lb.audit_page_ids([{"id": "org/a"}, {"id": "org/b"}])  # distinct prefixes still fine


@pytest.mark.parametrize("bad", [None, "", 7, ["org/ds"]])
def test_rows_without_a_string_id_fail(bad):
    with pytest.raises(ValueError):
        lb.audit_page_ids([{"id": bad}])


def test_write_on_collision_keeps_evidence_but_writes_no_page_or_link(tmp_path, monkeypatch):
    out = tmp_path / "docs"
    out.mkdir()
    lb._write(out, [_hub_row("org/a")], "2026-11-20 00:00 UTC")
    index_before = (out / "index.html").read_text()
    pages_before = sorted(p.name for p in (out / "d").iterdir())
    monkeypatch.setattr(lb, "_page_digest", lambda rid: "0" * 12)
    with pytest.raises(lb.PageIdCollision):
        lb._write(out, [_hub_row("org/a"), _hub_row("org//a")], "2026-11-21 00:00 UTC")
    assert len(json.loads((out / "data.json").read_text())["datasets"]) == 2  # evidence kept
    assert (out / "index.html").read_text() == index_before
    assert sorted(p.name for p in (out / "d").iterdir()) == pages_before


# --- untrusted text and links -----------------------------------------------

class _Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags, self.attrs, self.hrefs = set(), [], []

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        for k, v in attrs:
            self.attrs.append(k)
            if k == "href":
                self.hrefs.append(v)


_ALLOWED_TAGS = {"html", "head", "meta", "title", "style", "body", "p", "a", "h1", "h2", "h3", "div",
                 "table", "thead", "tbody", "tr", "th", "td", "span", "code", "b", "br", "ul", "li",
                 "details", "summary", "time", "footer"}
_ALLOWED_ATTRS = {"charset", "name", "content", "http-equiv", "lang", "href", "rel", "class", "title",
                  "datetime"}
EVIL = "<script>alert(1)</script>\"'><img src=x onerror=alert(2)>&amp;‮\x00\x1b"


def _assert_inert(page: str):
    p = _Tags()
    p.feed(page)
    assert p.tags <= _ALLOWED_TAGS, p.tags - _ALLOWED_TAGS
    assert set(p.attrs) <= _ALLOWED_ATTRS, set(p.attrs) - _ALLOWED_ATTRS
    for h in p.hrefs:
        if h.startswith("https://"):
            assert lb.safe_outbound_url(h) == h
        else:  # internal links: relative, inside the site
            assert h in ("../index.html", "../data.json"), h
    assert "<script" not in page.lower() and "‮" not in page and "\x00" not in page


def test_dataset_text_is_escaped_everywhere():
    row = _hub_row("org/" + EVIL, engine=EVIL, revision=EVIL, audited_at=EVIL, lane=EVIL,
                   last_audit_error=EVIL, subpath=EVIL, discovery_current=False,
                   video_pointer_correction={"at": EVIL, "status": EVIL, "scope": EVIL})
    rep = row["report"]
    rep.update(score_model=EVIL, format_version=EVIL, verdict=EVIL, fps=EVIL, episodes=EVIL)
    rep["checks"] = [_check(EVIL, EVIL, summary=EVIL, charge=EVIL,
                            details={EVIL + "_total": 5},
                            findings=[{"severity": EVIL, "message": EVIL, "episode_index": EVIL}]),
                     dict(_check("x", skipped=True, skip_reason=EVIL), title=EVIL, waived_by=EVIL)]
    page = lb.render_audit_page(row, ENGINE, 250)
    _assert_inert(page)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page  # shown, not run
    for status in ("gated", "error", EVIL, None):
        _assert_inert(lb.render_audit_page(_status_row("org/" + EVIL, status, reason=EVIL, engine=EVIL), ENGINE))


def test_board_link_text_is_escaped(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    lb._write(out, [_hub_row("org/" + EVIL), _status_row("org/x" + EVIL, "gated")], "now")
    index = (out / "index.html").read_text()
    assert "<img" not in index and "‮" not in index


@pytest.mark.parametrize("url", [
    "javascript:alert(1)", "JAVASCRIPT:alert(1)", "data:text/html,<script>", "vbscript:x",
    "http://huggingface.co/datasets/a/b", "//huggingface.co/datasets/a/b", "https://evil.example/",
    "https://huggingface.co.evil.example/", "https://user@huggingface.co/", "https://huggingface.co:444/",
    "https://huggingface.co\\@evil.example/", "https://huggingface.co/ x", "https://huggingface.co/\x00",
    "https://HuggingFace.co/datasets/a/b", "https://evil.example@huggingface.co/", " https://github.com/",
    None, 5,
])
def test_unsafe_outbound_urls_are_refused(url):
    assert lb.safe_outbound_url(url) is None


def test_safe_outbound_urls_pass():
    for url in ("https://huggingface.co/datasets/a/b", "https://github.com/vedevpatel/kinema",
                "https://console.cloud.google.com/storage/browser/gresearch/robotics/toy"):
        assert lb.safe_outbound_url(url) == url


def test_hub_links_are_rebuilt_from_validated_parts():
    sha = "0123456789abcdef0123456789abcdef01234567"
    assert lb._hub_dataset_url("org/ds") == "https://huggingface.co/datasets/org/ds"
    assert lb._hub_dataset_url("org/ds", sha) == f"https://huggingface.co/datasets/org/ds/tree/{sha}"
    assert (lb._hub_dataset_url("org/r//LeRobot/t.A", sha)
            == f"https://huggingface.co/datasets/org/r/tree/{sha}/LeRobot/t.A")
    assert lb._hub_dataset_url("org/ds", "javascript:alert(1)") == "https://huggingface.co/datasets/org/ds"
    for bad in ("org/../x", "../org", "org/ds/extra", "org/<b>", "org/r//../../x", "org/r//a//b",
                "javascript:alert(1)//x", "org/ds?x", "/org/ds"):
        assert lb._hub_dataset_url(bad) is None, bad
    page = lb.render_audit_page(_hub_row("org/<b>", revision="javascript:alert(1)"), ENGINE)
    assert "javascript:" not in re.sub(r"<code>[^<]*</code>", "", page.split("<body>")[1]).split("as recorded")[0]


def test_rlds_link_needs_a_recorded_valid_bucket():
    row = _rlds_row()
    assert lb._rlds_url(row) == "https://console.cloud.google.com/storage/browser/gresearch/robotics/toy"
    for bucket, prefix in ((None, "robotics/toy/"), ("Evil..", "x/"), ("gresearch", "../x"),
                           ("gresearch", None), ("gresearch", "a//b")):
        assert lb._rlds_url({**row, "rlds_bucket": bucket, "rlds_prefix": prefix}) is None


# --- honest states ------------------------------------------------------------

def _body(page: str) -> str:
    return page.split("<body>", 1)[1]


def test_current_result():
    page = lb.render_audit_page(_hub_row(), ENGINE, 50)
    assert "s-current" in page and "Latest stored result" in page and "Stale" not in page
    assert "2026-11-20 22:36:16 UTC" in page  # the recorded audit time
    assert "<span class=score>90</span>/100" in page and "(stored, stale)" not in page
    assert "0123456789ab" in page and f"<code>{lb.SCORE_MODEL}</code>" in page
    assert f"<code>{ENGINE}</code> <span class=dim>(the board's current engine)" in page
    assert "Sampled Hub audit" in page and "50 episodes read" in page
    assert "Episode sample: the audit loaded 50 episodes" in page
    assert "3.0 GiB" in page and "3 video file(s) were fetched" in page
    assert "1 of 3 checks were skipped" in page and "no reward/done fields" in page
    assert "not a guarantee about the whole dataset" in page
    assert "&minus;10" in page  # the score charge
    assert "joint 3 rarely moves" in page.split("informational note(s)")[1]  # INFO folded


def test_stale_by_engine_is_explicit():
    page = lb.render_audit_page(_hub_row(engine=OLD_ENGINE), ENGINE, 50)
    assert "s-stale" in page and "<b>Stale result.</b>" in page and "older check logic" in page
    assert OLD_ENGINE in page and "not a current one" in page and "(stored, stale)" in page


def test_stale_by_depth_and_failed_retry():
    page = lb.render_audit_page(_hub_row(), ENGINE, 250)
    assert "deeper audit (250) is queued" in page
    page = lb.render_audit_page(_hub_row(last_audit_error="audit error: Timeout"), ENGINE, 50)
    assert "s-error" in page and "latest re-audit attempt failed (audit error: Timeout)" in page


def test_old_score_model_is_not_presented_as_comparable():
    row = _hub_row()
    row["report"]["score_model"] = "legacy-v0"
    assert "not comparable with" in lb.render_audit_page(row, ENGINE, 50)


@pytest.mark.parametrize("status,needle", [
    ("gated", "recorded status: <b>gated</b>"), ("error", "recorded status: <b>error</b>"),
    ("unsupported", "recorded status: <b>unsupported</b>"), (None, "its status is not recorded"),
    ("mystery", "its status is <b>mystery</b>")])
def test_unaudited_rows(status, needle):
    page = lb.render_audit_page(_status_row("org/x", status), ENGINE)
    assert needle in page and "There is no score" in page
    assert "class=score" not in page and "Coverage limits" not in page
    assert "audit error: GatedRepoError: 403" in page and OLD_ENGINE in page
    assert "the board stores no time for failed attempts" in page


def test_rlds_page_is_metadata_only():
    page = lb.render_audit_page(_rlds_row(), ENGINE, 50)
    assert "RLDS metadata/listing audit (no data download)" in page
    assert "118 episodes declared in the TFDS metadata (none read)" in page
    assert "frame count not declared" in page and "0 frames" not in page
    assert "downloaded no episode data" in page and "shard-inventory digest <code>1da032540176</code>" in page
    assert "does not detect later changes" in page and "Sampled Hub audit" not in page
    assert "console.cloud.google.com/storage/browser/gresearch/robotics/toy" in page


def test_nested_and_discovery_notes():
    row = _hub_row("nvidia/R//LeRobot/t.A", subpath="LeRobot/t.A", discovery_current=False,
                   discovery_misses=2, discovery_uncertain=True)
    page = lb.render_audit_page(row, ENGINE, 50)
    assert "Nested dataset" in page and "(repo-level)" in page
    assert "Not listed by the latest discovery run" in page and "Missing from 2 fresh discovery" in page
    assert "probe for this row errored" in page


def test_capped_evidence_lists_are_disclosed():
    row = _hub_row()
    row["report"]["checks"][0]["details"] = {"episodes": list(range(100)), "episodes_total": 731}
    assert "<code>stats-drift</code> episodes (731 in total)" in lb.render_audit_page(row, ENGINE)


def test_missing_fields_say_not_recorded_and_nothing_is_invented():
    today = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}"
    page = lb.render_audit_page({"id": "org/bare", "report": {"checks": [{"id": "x"}]}}, ENGINE)
    body = _body(page)
    for label in ("health score", "score model", "audited at", "dataset revision", "check engine"):
        assert re.search(rf"<th>{label}</th><td><span class=dim>not recorded</span>", body), label
    assert "sample signature" in body and "was not recorded" in body
    assert today not in body and "unknown" not in body.lower()
    assert "Stale result" in body  # no engine stamp: not current
    page = lb.render_audit_page({"id": "org/bare2", "report": {"score": 80}}, ENGINE)
    assert "No check results recorded" in page and "No findings recorded" in page


def test_garbage_types_do_not_crash():
    row = _hub_row()
    row["report"].update(checks={"not": "a list"}, episodes="many", truncated="yes", score="90")
    row.update(sample=[1, 2], audit_max_episodes="x", data_bytes=-1, downloads="lots")
    page = lb.render_audit_page(row, ENGINE, 50)
    assert "health score</th><td><span class=dim>not recorded" in page


def test_audit_time_is_never_the_render_time():
    row = _hub_row()
    del row["audited_at"]
    page = lb.render_audit_page(row, ENGINE, 50)
    assert "<th>audited at</th><td><span class=dim>not recorded</span>" in page
    assert f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}" not in page
    page = lb.render_audit_page(_hub_row(audited_at="last tuesday"), ENGINE, 50)
    assert "last tuesday <span class=dim>(as recorded; not a recognized timestamp)" in page


def test_pages_are_self_contained():
    for row in _board():
        page = lb.render_audit_page(row, ENGINE, 50)
        assert "<script" not in page and "src=" not in page and "@import" not in page
        assert "default-src 'none'" in page and lb._AUDIT_PAGE_MARKER in page[:512]


def test_long_findings_fold():
    msgs = [{"severity": "WARN", "message": f"w{i}", "episode_index": i, "details": {}} for i in range(45)]
    msgs.insert(30, {"severity": "FAIL", "message": "the fail", "episode_index": None, "details": {}})
    row = _hub_row()
    row["report"]["checks"][1]["findings"] = msgs
    page = lb.render_audit_page(row, ENGINE)
    shown, folded = page.split("26 more warnings and failures")
    assert "the fail" in shown.split("<h3>")[-1]  # FAIL first, never folded
    assert "episode 44</span> w44</li>" in folded and "1 FAIL &middot; 45 WARN" in page


# --- writing, pruning, determinism, link coverage -----------------------------

def test_generation_is_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for out, gen in ((a, "2026-11-20 00:00 UTC"), (b, "2026-11-21 09:09 UTC")):
        out.mkdir()
        lb._write(out, json.loads(json.dumps(_board())), gen, max_episodes=50)
    pages_a = {p.name: p.read_bytes() for p in (a / "d").iterdir()}
    pages_b = {p.name: p.read_bytes() for p in (b / "d").iterdir()}
    assert pages_a == pages_b and len(pages_a) == len(_board())  # board snapshot time not in pages
    stats = lb.write_audit_pages(a, json.loads((a / "data.json").read_text())["datasets"], 50)
    assert stats["written"] == 0 and stats["pruned"] == []  # rerun: nothing rewritten
    assert {p.name: p.read_bytes() for p in (a / "d").iterdir()} == pages_a


def test_render_site_matches_write(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    lb._write(out, _board(), "2026-11-20 00:00 UTC", max_episodes=50)
    before = {p.name: p.read_bytes() for p in (out / "d").iterdir()}
    index = (out / "index.html").read_bytes()
    data = (out / "data.json").read_bytes()
    shutil.rmtree(out / "d")
    assert lb.render_site(out, max_episodes=50) == 0
    assert {p.name: p.read_bytes() for p in (out / "d").iterdir()} == before
    assert (out / "index.html").read_bytes() == index and (out / "data.json").read_bytes() == data


def _board_links(index_html: str) -> list[str]:
    rows = re.findall(r"<tr><td>\d+</td><td>(.*?)</td>", index_html)
    return [re.findall(r"href='([^']+)'", r) for r in rows]


def _assert_full_link_coverage(out: Path, n_rows: int):
    index = (out / "index.html").read_text(encoding="utf-8")
    links = _board_links(index)
    assert len(links) == n_rows and all(len(ls) == 1 for ls in links)
    page_files = sorted(p.name for p in (out / "d").glob("*.html"))
    assert sorted(ls[0].removeprefix("d/") for ls in links) == page_files  # 1:1, no orphans
    for (href,) in links:
        for base in (PAGES_BASE + "index.html", PAGES_BASE):  # /kinema/ and /kinema/index.html
            url = urllib.parse.urlsplit(urllib.parse.urljoin(base, href))
            assert url.netloc == "vedevpatel.github.io" and url.path.startswith("/kinema/d/")
            assert (out / url.path.removeprefix("/kinema/")).is_file()
        page_url = urllib.parse.urljoin(PAGES_BASE, href)
        page = (out / href).read_text(encoding="utf-8")
        for inner in re.findall(r"href='([^']+)'", page):
            target = urllib.parse.urlsplit(urllib.parse.urljoin(page_url, inner)).path
            assert target.startswith("/kinema/") and (out / target.removeprefix("/kinema/")).is_file()


def test_every_board_row_links_to_its_page(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    rows = _board() + [_hub_row("org/" + EVIL)]
    lb._write(out, rows, "2026-11-20 00:00 UTC", max_episodes=50)
    _assert_full_link_coverage(out, len(rows))


def test_real_board_link_coverage(tmp_path):
    """The committed board: every row, nested and RLDS included, renders and links."""
    out = tmp_path / "docs"
    out.mkdir()
    shutil.copy(ROOT / "docs" / "data.json", out / "data.json")
    assert lb.render_site(out, max_episodes=250) == 0
    n = len(json.loads((out / "data.json").read_text())["datasets"])
    _assert_full_link_coverage(out, n)
    for p in (out / "d").iterdir():
        _assert_inert(p.read_text(encoding="utf-8"))


def test_prune_removes_only_orphaned_generated_pages(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    lb._write(out, _board(), "now")
    d = out / "d"
    gone_id = lb.audit_page_id("org/gated")
    hand_named = lb.audit_page_id("org/hand-written")
    (d / f"{hand_named}.html").write_text("<p>hand written, no generator marker</p>")
    (d / "README.md").write_text("keep")
    (d / "notes.html").write_text(lb._AUDIT_PAGE_MARKER)  # marker, but not a page-id name
    (d / "sub").mkdir()
    (d / "sub" / f"{gone_id}.html").write_text(lb._AUDIT_PAGE_MARKER)
    outside = tmp_path / "outside.html"
    outside.write_text(lb._AUDIT_PAGE_MARKER)
    link_id = lb.audit_page_id("org/linked")
    (d / f"{link_id}.html").symlink_to(outside)
    (d / f".{lb.audit_page_id('org/crashed')}.html.tmp").write_text("partial")
    (out / "index.html.bak").write_text("untouched")

    lb._write(out, [r for r in _board() if r["id"] != "org/gated"], "now")
    names = {p.name for p in d.iterdir()}
    assert f"{gone_id}.html" not in names  # the orphaned generated page went
    assert {f"{hand_named}.html", "README.md", "notes.html", "sub", f"{link_id}.html"} <= names
    assert not any(n.endswith(".tmp") for n in names)
    assert outside.read_text() == lb._AUDIT_PAGE_MARKER and (d / "sub" / f"{gone_id}.html").exists()
    assert (out / "index.html.bak").read_text() == "untouched"
    assert "org/gated" not in (out / "index.html").read_text()


def test_writes_never_follow_symlinks(tmp_path):
    out = tmp_path / "docs"
    out.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("original")
    (out / "d").mkdir()
    pid = lb.audit_page_id("org/ds")
    (out / "d" / f".{pid}.html.tmp").symlink_to(victim)  # planted temp-file symlink
    lb.write_audit_pages(out, [_hub_row()], engine=ENGINE)
    assert victim.read_text() == "original" and not (out / "d" / f"{pid}.html").is_symlink()

    other = tmp_path / "other"
    other.mkdir()
    shutil.rmtree(out / "d")
    (out / "d").symlink_to(other, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        lb.write_audit_pages(out, [_hub_row()], engine=ENGINE)
    assert list(other.iterdir()) == []

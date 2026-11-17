"""Leaderboard board-listing lanes: top downloads + newest-uploads probation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_leaderboard():
    spec = importlib.util.spec_from_file_location("leaderboard", SCRIPTS / "leaderboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["leaderboard"] = mod
    spec.loader.exec_module(mod)
    return mod


def _d(i, dl=0):
    return {"id": f"org/ds{i}", "downloads": dl, "likes": 0, "data_bytes": 1}


def test_merge_lanes_unions_without_duplicates():
    lb = _load_leaderboard()
    top = [_d(i, 100 - i) for i in range(5)]
    recent = [_d(3), _d(100), _d(101), _d(102)]
    out = lb._merge_lanes(top, recent, limit=4, new_lane=3)
    assert [d["id"] for d in out] == ["org/ds0", "org/ds1", "org/ds2", "org/ds3",
                                      "org/ds100", "org/ds101", "org/ds102"]
    assert [d["lane"] for d in out] == ["top"] * 4 + ["new"] * 3


def test_merge_lanes_caps_and_empty_lane():
    lb = _load_leaderboard()
    top = [_d(i) for i in range(10)]
    assert [d["lane"] for d in lb._merge_lanes(top, [], 3, 0)] == ["top"] * 3
    out = lb._merge_lanes(top, [_d(50)], 3, 5)  # lane bigger than supply
    assert len(out) == 4


def test_index_marks_new_lane_rows():
    lb = _load_leaderboard()
    entries = [{"id": "org/ds", "slug": "org--ds", "downloads": 1, "lane": "new",
                "report": {"verdict": "PASS", "score": 100, "score_model": "tiered-v2",
                           "episodes": 10, "truncated": False, "checks": []}}]
    html = lb.render_index(entries, "now")
    assert "(new)" in html
    entries[0]["lane"] = "top"
    assert "(new)" not in lb.render_index(entries, "now")


def test_video_subsample_picks_from_targets_under_cap():
    lb = _load_leaderboard()
    sizes = {f"videos/chunk-{i:03d}/f.mp4": 10 for i in range(100)}
    pick = lb._video_subsample(sorted(sizes), sizes, 24, 400)
    assert pick is not None and len(pick) <= 24
    assert set(pick) <= set(sizes)  # only files the targeting returned
    # spread across the range, not a prefix of early chunks
    assert pick[0].endswith("chunk-000/f.mp4") and pick[-1].endswith("chunk-099/f.mp4")


def test_video_subsample_refuses_over_cap_and_unknown_sizes():
    lb = _load_leaderboard()
    sizes = {"videos/a.mp4": 300, "videos/b.mp4": 200}
    assert lb._video_subsample(sorted(sizes), sizes, 24, 400) is None  # 500 > 400
    assert lb._video_subsample(sorted(sizes), sizes, 24, 500) == ["videos/a.mp4", "videos/b.mp4"]
    assert lb._video_subsample(["videos/missing.mp4"], sizes, 24, 500) is None
    assert lb._video_subsample([], sizes, 24, 500) is None


def test_video_subsample_expands_whole_tree_sentinel():
    lb = _load_leaderboard()
    sizes = {"videos/a.mp4": 10, "videos/b/c.mp4": 10, "data/x.parquet": 10, "meta/y.json": 1}
    pick = lb._video_subsample(["videos/**"], sizes, 24, 400)
    assert pick == ["videos/a.mp4", "videos/b/c.mp4"]  # only videos/ paths
    assert lb._video_subsample(["videos/**"], sizes, 24, 15) is None  # 20 > 15


def test_hub_listing_kwargs_match_installed_api(monkeypatch):
    """Listing calls must only use kwargs the installed huggingface_hub accepts.

    hub 2.x removed `direction`; the nightly then died inside
    find_lerobot_datasets before auditing anything and `|| true` kept the
    run green. Capture every kwarg both lanes pass and validate against the
    real signature of the installed HfApi.list_datasets.
    """
    import inspect

    from huggingface_hub import HfApi

    lb = _load_leaderboard()
    calls = []

    class _FakeApi:
        def list_datasets(self, **kwargs):
            calls.append(kwargs)
            return iter([])

    monkeypatch.setattr("huggingface_hub.HfApi", lambda: _FakeApi())
    lb.find_lerobot_datasets(2, 2)

    assert len(calls) == 2  # top lane + newest lane
    params = inspect.signature(HfApi.list_datasets).parameters
    for kwargs in calls:
        unknown = set(kwargs) - set(params)
        assert not unknown, f"list_datasets kwargs not in installed hub API: {unknown}"


def test_audit_stale_reaudits_rows_without_video_audit():
    lb = _load_leaderboard()
    engine = "e" * 12
    base = {"engine": engine, "report": {"source": "org/ds", "episodes": 10, "truncated": False}}
    assert lb.audit_stale(base, engine, 250) is True  # no video_audited key
    buggy = {"engine": engine, "report": {**base["report"], "video_audited": True}}
    assert lb.audit_stale(buggy, engine, 250) is True  # v1 bug mark (True == 1 < 2)
    settled = {"engine": engine, "report": {**base["report"], "video_audited": 2}}
    assert lb.audit_stale(settled, engine, 250) is False
    # depth-staleness still applies on top once video is settled
    deep = {"engine": engine, "report": {**base["report"], "video_audited": 2, "truncated": True}}
    assert lb.audit_stale(deep, engine, 250) is True


def test_report_video_audited_roundtrip():
    from kinema.report import DatasetReport, report_from_dict, report_to_dict

    rep = DatasetReport(source="org/ds", format_version="v3", episodes=1,
                        frames=10, fps=30.0, video_audited=2)
    assert report_from_dict(report_to_dict(rep)).video_audited == 2
    # rows stored before the field existed read as mark 0 (re-audit once)
    d = report_to_dict(rep)
    del d["video_audited"]
    assert report_from_dict(d).video_audited == 0
    # rows stored by the buggy v1 targeting (true) read as mark 1
    d["video_audited"] = True
    assert report_from_dict(d).video_audited == 1

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

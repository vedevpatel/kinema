"""similar_datasets: retrieval over the leaderboard corpus (kinema.similar)."""

from __future__ import annotations

import json

import pytest

from kinema import mcp_server as mcp
from kinema.similar import distance, fetch_board, load_query_report, rank


def _report(check_verdicts, score=80, fps=30, fmt="v3.0"):
    return {
        "dataset": "me/mine",
        "score": score,
        "verdict": "WARN",
        "format_version": fmt,
        "fps": fps,
        "checks": [{"id": cid, "verdict": v} for cid, v in check_verdicts.items()],
    }


def _entry(i, check_verdicts, score=80, downloads=100):
    return {
        "id": f"org/ds{i}",
        "slug": f"org--ds{i}",
        "downloads": downloads,
        "report": _report(check_verdicts, score=score),
    }


def test_distance_grows_with_warning_mismatch():
    q = _report({"action-scale": "WARN", "idle": "WARN"})
    same = _entry(1, {"action-scale": "WARN", "idle": "WARN"}, score=80)
    partial = _entry(2, {"action-scale": "WARN"}, score=80)
    clean = _entry(3, {"action-scale": "PASS", "idle": "PASS"}, score=95)
    assert distance(q, same) < distance(q, partial) < distance(q, clean)


def test_rank_orders_by_similarity_and_excludes_self():
    q = _report({"calibration-drift": "WARN"})
    entries = [
        _entry(0, {"calibration-drift": "WARN"}),
        _entry(1, {"idle-segments": "WARN"}),
        _entry(2, {"idle": "PASS"}),
        {**_entry(99, {"calibration-drift": "WARN"}), "id": "me/mine"},
    ]
    matches = rank(q, entries, k=3, exclude_id="me/mine")
    assert [m["id"] for m in matches] == ["org/ds0", "org/ds1", "org/ds2"]
    assert matches[0]["shared_warnings"][0]["check"] == "calibration-drift"
    assert matches[0]["shared_warnings"][0]["measured_mse_harm_pct"] == 1475.3


def test_rank_caps_k():
    q = _report({})
    assert len(rank(q, [_entry(i, {"idle": "PASS"}) for i in range(10)], k=4)) == 4


def test_fetch_board_caches_then_serves_stale(tmp_path):
    payload = json.dumps({"generated": "t0", "datasets": [{"id": "org/a", "report": {}}]}).encode()
    entries, meta = fetch_board(cache_dir=tmp_path, fetcher=lambda url: payload)
    assert entries[0]["id"] == "org/a" and meta["cached"] is False

    def boom(url):
        raise OSError("offline")

    entries, meta = fetch_board(cache_dir=tmp_path, fetcher=boom)
    assert entries[0]["id"] == "org/a" and meta["cached"] is True
    assert entries, "a stale cache beats no corpus"


def test_fetch_board_no_cache_raises(tmp_path):
    def boom(url):
        raise OSError("offline")

    with pytest.raises(RuntimeError, match="no cached copy"):
        fetch_board(cache_dir=tmp_path, fetcher=boom)


def test_load_query_report(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(_report({"idle": "WARN"})))
    assert load_query_report(p)["verdict"] == "WARN"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": 1}))
    with pytest.raises(ValueError, match="not a kinema JSON report"):
        load_query_report(bad)


def test_similar_datasets_tool_over_protocol(tmp_path, monkeypatch):
    report = tmp_path / "kinema-report.json"
    report.write_text(json.dumps(_report({"calibration-drift": "WARN"})))
    monkeypatch.setattr(
        mcp, "fetch_board",
        lambda: ([_entry(1, {"calibration-drift": "WARN"}), _entry(2, {})],
                 {"url": "test", "datasets": 2, "cached": False}),
    )
    s = mcp.McpServer()
    out = []
    s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18"}}, emit=out.append)
    out.clear()
    s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "similar_datasets",
                         "arguments": {"report_path": str(report), "k": 2}}}, emit=out.append)
    (reply,) = out
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert payload["matches"][0]["id"] == "org/ds1"
    assert payload["corpus"]["datasets"] == 2


def test_similar_datasets_requires_a_query(monkeypatch):
    s = mcp.McpServer()
    out = []
    s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18"}}, emit=out.append)
    out.clear()
    s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "similar_datasets", "arguments": {"k": 3}}}, emit=out.append)
    (reply,) = out
    assert "one of dataset or report_path" in reply["result"]["content"][0]["text"]


def test_fail_is_a_problem_and_unaudited_is_not_a_perfect_match():
    q = _report({'stats-drift': 'FAIL'})
    assert distance(q, _entry(1, {'stats-drift': 'FAIL'})) < distance(q, _entry(2, {'idle': 'FAIL'}))
    entries = [_entry(0, {}), _entry(1, {'stats-drift': 'FAIL'}), {'id': 'empty'}]
    assert [m['id'] for m in rank(q, entries)] == ['org/ds1']


def test_native_source_excludes_self():
    q = _report({'idle': 'WARN'})
    q['source'] = 'hf://org/ds1@main'
    entries = [_entry(1, {'idle': 'WARN'}), _entry(2, {'idle': 'WARN'})]
    assert [m['id'] for m in rank(q, entries)] == ['org/ds2']


def test_dataset_route_serializes_report(tmp_path, monkeypatch):
    from kinema.fixtures.synth import make_dataset
    root = make_dataset(tmp_path / 'd', episodes=3)
    monkeypatch.setattr(mcp, 'fetch_board', lambda: ([_entry(1, {'idle': 'WARN'})], {}))
    s, out = mcp.McpServer(), []
    s.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
              'params': {'protocolVersion': '2025-06-18'}}, emit=out.append)
    out.clear()
    s.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
              'params': {'name': 'similar_datasets', 'arguments': {'dataset': str(root)}}}, emit=out.append)
    assert not out[0]['result'].get('isError'), out
    payload = json.loads(out[0]['result']['content'][0]['text'])
    assert payload['query']['dataset'] == str(root)

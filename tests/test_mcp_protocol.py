"""MCP protocol conformance: every item on the checklist in mcp_server.py's
docstring is asserted here, over an in-process stdio pair.

1. newline-delimited JSON on stdout, nothing else
2. initialize negotiates protocolVersion
3. notifications never get replies
4. ping answered
5. tool failures are isError results, not protocol errors
6. cancelled honored between checks
7. progress notifications at stage boundaries
"""

from __future__ import annotations

import io
import json

from kinema.fixtures.synth import make_dataset
from kinema.mcp_server import SUPPORTED_PROTOCOL_VERSIONS, McpServer


class Session:
    """Feed messages, collect raw output lines."""

    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.server = McpServer(out=self.out, err=self.err)

    def send(self, **msg):
        self.server.handle(msg)

    def feed(self, text: str):
        self.server.serve(io.StringIO(text))

    def lines(self):
        return [ln for ln in self.out.getvalue().splitlines() if ln.strip()]

    def replies(self):
        return [json.loads(ln) for ln in self.lines()]

    def take(self):
        r = self.replies()
        self.out.seek(0)
        self.out.truncate(0)
        return r


def test_stdout_is_pure_newline_delimited_json(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.feed(
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18"}})
        + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                      "params": {"name": "run_checks",
                                 "arguments": {"dataset": str(root), "max_episodes": 4,
                                               "report_path": str(tmp_path / "report.json")}}})
        + "\n"
    )
    for line in s.lines():
        assert "\n" not in line  # one message per line, no embedded newlines
        msg = json.loads(line)  # every line parses as protocol JSON
        assert msg.get("jsonrpc") == "2.0"
    # stderr may carry logs; stdout carried nothing but protocol messages
    assert s.err.getvalue() == ""


def test_initialize_version_negotiation_matrix():
    for v in SUPPORTED_PROTOCOL_VERSIONS:
        s = Session()
        s.send(jsonrpc="2.0", id=1, method="initialize", params={"protocolVersion": v})
        (reply,) = s.take()
        assert reply["result"]["protocolVersion"] == v  # supported: echoed
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="initialize", params={"protocolVersion": "1999-01-01"})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="initialize", params={})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    assert reply["result"]["serverInfo"]["name"] == "kinema"
    assert "tools" in reply["result"]["capabilities"]


def test_notifications_get_no_replies():
    s = Session()
    s.send(jsonrpc="2.0", method="notifications/initialized")
    s.send(jsonrpc="2.0", method="notifications/cancelled", params={"requestId": "nope"})
    s.send(jsonrpc="2.0", method="notifications/whatever", params={})
    assert s.take() == []


def test_ping_answered():
    s = Session()
    s.send(jsonrpc="2.0", id="p1", method="ping")
    (reply,) = s.take()
    assert reply == {"jsonrpc": "2.0", "id": "p1", "result": {}}


def test_unknown_method_is_a_protocol_error():
    s = Session()
    s.send(jsonrpc="2.0", id=9, method="resources/list")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32601
    s.send(jsonrpc="2.0", id=10, method="tools/call", params={"name": "nope"})
    (reply,) = s.take()
    assert reply["error"]["code"] == -32602  # unknown tool: invalid params


def test_malformed_json_is_a_parse_error():
    s = Session()
    s.feed("this is not json\n")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32700 and reply["id"] is None


def test_tool_failures_are_iserror_results_not_protocol_errors(tmp_path):
    s = Session()
    # bad dataset source
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(tmp_path / "missing"), "policy": "pi0",
                                 "symptom": "x"}})
    (reply,) = s.take()
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    # unknown policy
    root = make_dataset(tmp_path / "ds", episodes=8)
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "hal-9000", "symptom": "x"}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "unknown policy" in reply["result"]["content"][0]["text"]
    # unknown policy override keys
    s.send(jsonrpc="2.0", id=3, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "x",
                                 "policy_overrides": {"chunk": 10}}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "unknown policy override" in reply["result"]["content"][0]["text"]


def test_tools_list_shape():
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="tools/list")
    (reply,) = s.take()
    tools = {t["name"]: t for t in reply["result"]["tools"]}
    assert set(tools) == {"diagnose_finetune", "list_checks", "run_checks"}
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
    req = tools["diagnose_finetune"]["inputSchema"]["required"]
    assert set(req) == {"dataset", "policy", "symptom"}


def test_progress_notifications_at_stage_boundaries(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "x"},
                   "_meta": {"progressToken": "tok-1"}})
    replies = s.take()
    progress = [r for r in replies if r.get("method") == "notifications/progress"]
    assert progress  # stage boundaries emitted
    for p in progress:
        assert p["params"]["progressToken"] == "tok-1"
    (result,) = [r for r in replies if r.get("id") == 1]
    assert "result" in result


def test_cancellation_between_checks(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    # an unrouteable symptom runs all checks; cancel the request mid-run by
    # having progress observe the notification first
    import threading
    done = threading.Event()

    orig_handle = s.server.handle

    def cancelling_handle(msg):
        if msg.get("method") == "tools/call" and not done.is_set():
            s.server._notify("notifications/cancelled", {"requestId": 7})
            done.set()
        orig_handle(msg)

    s.server.handle = cancelling_handle
    s.send(jsonrpc="2.0", id=7, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "xyzzy"}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "cancelled" in reply["result"]["content"][0]["text"]

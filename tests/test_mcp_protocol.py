"""MCP protocol conformance: every item on the checklist in mcp_server.py's
docstring is asserted here, over an in-process stdio pair.

1. newline-delimited JSON on stdout, nothing else
2. initialize negotiates protocolVersion; requests before initialize are
   rejected (ping excepted)
3. notifications never get replies - malformed ones neither reply nor hang
4. ping answered - on the reader thread, never queued behind a tool call
5. tool failures are isError results, not protocol errors; malformed
   params/_meta, unknown arguments and wrongly-typed arguments are -32602
6. cancelled honored during load and between checks; a cancelled request
   gets NO response (MCP SHOULD NOT)
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

    def init(self):
        """The MCP handshake every client performs; most tests need the
        server past the initialize-first gate."""
        self.send(jsonrpc="2.0", id="init", method="initialize",
                  params={"protocolVersion": "2025-06-18"})
        self.take()

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
    # 2025-03-26 is NOT negotiated: it obliges JSON-RPC batch support, which
    # this server rejects with -32600 - agreeing would promise what we lack
    s.send(jsonrpc="2.0", id=1, method="initialize", params={"protocolVersion": "2025-03-26"})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    assert "2025-03-26" not in SUPPORTED_PROTOCOL_VERSIONS
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="initialize", params={})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    assert reply["result"]["serverInfo"]["name"] == "kinema"
    assert "tools" in reply["result"]["capabilities"]


def test_json_rpc_batches_are_rejected():
    s = Session()
    s.feed(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]) + "\n")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32600


def test_requests_before_initialize_are_rejected():
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="tools/list")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32600
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "list_checks", "arguments": {}})
    (reply,) = s.take()
    assert reply["error"]["code"] == -32600
    # ping is the one exception: clients probe liveness before initializing
    s.send(jsonrpc="2.0", id=3, method="ping")
    (reply,) = s.take()
    assert reply["result"] == {}
    # and after the handshake the same requests work
    s.send(jsonrpc="2.0", id=4, method="initialize", params={"protocolVersion": "2025-06-18"})
    s.take()
    s.send(jsonrpc="2.0", id=5, method="tools/list")
    (reply,) = s.take()
    assert "tools" in reply["result"]


def test_notifications_get_no_replies():
    s = Session()
    s.send(jsonrpc="2.0", method="notifications/initialized")
    s.send(jsonrpc="2.0", method="notifications/cancelled", params={"requestId": "nope"})
    s.send(jsonrpc="2.0", method="notifications/whatever", params={})
    assert s.take() == []


def test_malformed_notification_neither_replies_nor_hangs():
    """Regression (review finding 2): {"method":"notifications/cancelled",
    "params":[1]} used to kill the reader thread before it emitted the
    end-of-stream sentinel, hanging serve() on pending.get() forever."""
    s = Session()
    s.feed(json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                       "params": [1]}) + "\n")
    assert s.take() == []  # no reply - and feed() returned at all
    # the server is still alive afterwards
    s.send(jsonrpc="2.0", id=1, method="ping")
    (reply,) = s.take()
    assert reply["result"] == {}


def test_malformed_request_params_are_invalid_params_errors():
    """Review finding 3: non-object params/_meta used to crash the process
    with AttributeError; they are -32602 now."""
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call", params=["x"])
    (reply,) = s.take()
    assert reply["error"]["code"] == -32602
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "list_checks", "arguments": {}, "_meta": ["x"]})
    (reply,) = s.take()
    assert reply["error"]["code"] == -32602
    # the server survived both
    s.send(jsonrpc="2.0", id=3, method="ping")
    (reply,) = s.take()
    assert reply["result"] == {}


def test_unknown_and_mistyped_arguments_are_rejected():
    """additionalProperties:false is enforced (a misspelled argument used
    to tune nothing), and a bare string 'checks' no longer iterates
    character by character (review finding 9)."""
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": "/tmp/whatever", "policy_override": {}}})
    (reply,) = s.take()
    assert reply["error"]["code"] == -32602
    assert "policy_override" in reply["error"]["message"]
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": "/tmp/whatever", "checks": "stats-drift"}})
    (reply,) = s.take()
    assert reply["error"]["code"] == -32602
    assert "array" in reply["error"]["message"]


def test_missing_required_argument_is_a_tool_error_not_a_keyerror():
    """A missing argument is reported by name; an internal KeyError from a
    malformed dataset can no longer masquerade as one (review finding 9)."""
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"policy": "pi0", "symptom": "x"}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "dataset" in reply["result"]["content"][0]["text"]


def test_ping_answered():
    s = Session()
    s.send(jsonrpc="2.0", id="p1", method="ping")
    (reply,) = s.take()
    assert reply == {"jsonrpc": "2.0", "id": "p1", "result": {}}


def test_unknown_method_is_a_protocol_error():
    s = Session()
    s.init()
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
    s.init()
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
    # wrongly-typed override values fail loudly (review finding 8): a
    # truthy string "false" used to keep the quantile cause firing
    s.send(jsonrpc="2.0", id=4, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "x",
                                 "policy_overrides": {"quantile_norm": "false"}}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "boolean" in reply["result"]["content"][0]["text"]


def test_relative_dataset_paths_are_rejected(tmp_path, monkeypatch):
    """Review finding 7: a relative path resolves against the server
    process's cwd and, on a miss, silently falls through to a Hub download
    of a same-named repo. The tools demand absolute paths."""
    monkeypatch.chdir(tmp_path)
    make_dataset(tmp_path / "relds", episodes=8)
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": "relds", "max_episodes": 4,
                                 "report_path": str(tmp_path / "r.json")}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "absolute path" in reply["result"]["content"][0]["text"]
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": "deep/relative/path", "max_episodes": 4}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "absolute path" in reply["result"]["content"][0]["text"]


def test_report_path_defaults_next_to_dataset_and_never_overwrites(tmp_path):
    """Review finding 6: the old default wrote kinema-report.json into the
    server's cwd (PermissionError after the whole run when cwd=/) and
    silently overwrote it on every call."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": str(root), "max_episodes": 4}})
    (reply,) = s.take()
    assert "error" not in reply and "isError" not in reply["result"]
    payload = json.loads(reply["result"]["content"][0]["text"])
    from pathlib import Path
    path = Path(payload["full_report"])
    assert path.parent == root  # next to the dataset, not the server's cwd
    assert path.name.startswith("kinema-report-") and path.exists()
    original = path.read_text()
    # a relative report_path is refused
    s.send(jsonrpc="2.0", id=2, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": str(root), "max_episodes": 4,
                                 "report_path": "rel.json"}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "absolute" in reply["result"]["content"][0]["text"]
    # an existing file is refused, not overwritten
    s.send(jsonrpc="2.0", id=3, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": str(root), "max_episodes": 4,
                                 "report_path": str(path)}})
    (reply,) = s.take()
    assert reply["result"]["isError"] is True
    assert "already exists" in reply["result"]["content"][0]["text"]
    assert path.read_text() == original


def test_tools_list_shape():
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/list")
    (reply,) = s.take()
    tools = {t["name"]: t for t in reply["result"]["tools"]}
    assert set(tools) == {"diagnose_finetune", "list_checks", "run_checks"}
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
    req = tools["diagnose_finetune"]["inputSchema"]["required"]
    assert set(req) == {"dataset", "policy", "symptom"}
    # policy_overrides carries real types so client models get hints
    props = tools["diagnose_finetune"]["inputSchema"]["properties"]
    ov = props["policy_overrides"]
    assert ov["properties"]["quantile_norm"]["type"] == "boolean"
    assert ov["properties"]["chunk_size"]["type"] == "integer"
    assert ov["additionalProperties"] is False


def test_tool_payloads_label_dataset_text_as_untrusted(tmp_path):
    """Review finding 12: task strings and other dataset text reach the
    client model; payloads must label it as data, never instructions."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "x"}})
    (reply,) = s.take()
    payload = json.loads(reply["result"]["content"][0]["text"])
    assert "untrusted" in payload["_untrusted_content"]


def test_progress_notifications_at_stage_boundaries(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "diagnose_finetune",
                   "arguments": {"dataset": str(root), "policy": "pi0", "symptom": "x"},
                   "_meta": {"progressToken": "tok-1"}})
    replies = s.take()
    progress = [r for r in replies if r.get("method") == "notifications/progress"]
    assert progress  # stage boundaries emitted
    for p in progress:
        assert p["params"]["progressToken"] == "tok-1"
    messages = [p["params"]["message"] for p in progress]
    assert "dataset loaded" in messages
    # totals agree across the check stages (review finding 13: the first
    # progress message and the later ones used to disagree)
    check_totals = {p["params"]["total"] for p in progress
                    if p["params"]["message"] != "dataset loaded"
                    and not p["params"]["message"].startswith(("downloading", "reading shard"))}
    assert len(check_totals) == 1
    (result,) = [r for r in replies if r.get("id") == 1]
    assert "result" in result


def test_run_checks_sends_the_dataset_loaded_stage(tmp_path):
    """Review finding 10: run_checks never sent the 'dataset loaded'
    progress stage the spec promises."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.init()
    s.send(jsonrpc="2.0", id=1, method="tools/call",
           params={"name": "run_checks",
                   "arguments": {"dataset": str(root), "max_episodes": 4,
                                 "report_path": str(tmp_path / "r.json")},
                   "_meta": {"progressToken": "tok-2"}})
    replies = s.take()
    messages = [r["params"]["message"] for r in replies
                if r.get("method") == "notifications/progress"]
    assert "dataset loaded" in messages


def test_cancellation_between_checks_gets_no_response(tmp_path):
    """A cancelled request gets NO response (the MCP spec's SHOULD NOT) -
    the old isError reply is gone (review finding 10)."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    s.init()
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
    assert [r for r in s.take() if r.get("id") == 7] == []  # no response at all
    # the id was pruned: the server does not accumulate dead request ids
    assert 7 not in s.server.cancelled


def test_cancellation_arriving_mid_run_over_the_wire(tmp_path):
    """A real client sends notifications/cancelled AFTER the run starts.
    Regression: the read loop used to block inside the running call, so the
    cancel sat in the pipe until the run finished and never landed. And per
    the MCP spec the cancelled request receives no response."""
    import os
    import threading
    import time

    root = make_dataset(tmp_path / "ds", episodes=8)
    out = io.StringIO()
    srv = McpServer(out=out, err=io.StringIO())

    # slow each progress emission a touch so the cancel has a wide window
    orig_progress = srv._progress

    def slow_progress(token, done, total, message):
        time.sleep(0.1)
        orig_progress(token, done, total, message)

    srv._progress = slow_progress

    r, w = os.pipe()
    t = threading.Thread(target=srv.serve, kwargs={"inp": os.fdopen(r)}, daemon=True)
    t.start()
    wf = os.fdopen(w, "w")
    wf.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18"}}) + "\n")
    wf.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                         "params": {"name": "run_checks",
                                    "arguments": {"dataset": str(root), "max_episodes": 4,
                                                  "report_path": str(tmp_path / "r.json")},
                                    "_meta": {"progressToken": "p"}}}) + "\n")
    wf.flush()
    deadline = time.time() + 15
    while "notifications/progress" not in out.getvalue():
        assert time.time() < deadline, "run never started"
        time.sleep(0.01)
    wf.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/cancelled",
                         "params": {"requestId": 1}}) + "\n")
    wf.flush()
    wf.close()
    t.join(timeout=30)
    assert not t.is_alive(), "server hung after the cancel"
    replies = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
    answered = [m for m in replies if m.get("id") == 1]
    assert answered == []  # a cancelled request gets no response (SHOULD NOT)
    assert 1 not in srv.cancelled  # pruned once the request settled


def test_ping_answered_on_the_reader_thread_while_a_tool_runs(tmp_path):
    """Review finding 4: a ping queued behind a multi-second (or
    minutes-long, on Hub loads) tool call makes liveness-checking clients
    drop the server. Pings are answered on the reader thread."""
    import os
    import threading
    import time

    import kinema.mcp_server as m

    root = make_dataset(tmp_path / "ds", episodes=8)
    out = io.StringIO()
    srv = McpServer(out=out, err=io.StringIO())
    started = threading.Event()
    release = threading.Event()
    orig = m.run_checks

    def slow_run_checks(**kwargs):
        started.set()
        release.wait(15)
        return orig(**kwargs)

    m.run_checks = slow_run_checks
    try:
        r, w = os.pipe()
        t = threading.Thread(target=srv.serve, kwargs={"inp": os.fdopen(r)}, daemon=True)
        t.start()
        wf = os.fdopen(w, "w")
        wf.write(json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize",
                             "params": {"protocolVersion": "2025-06-18"}}) + "\n")
        wf.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                             "params": {"name": "run_checks",
                                        "arguments": {"dataset": str(root), "max_episodes": 4,
                                                      "report_path": str(tmp_path / "r.json")}}}) + "\n")
        wf.flush()
        assert started.wait(15), "tool call never started"
        wf.write(json.dumps({"jsonrpc": "2.0", "id": "ping-1", "method": "ping"}) + "\n")
        wf.flush()
        deadline = time.time() + 10
        while "ping-1" not in out.getvalue():
            assert time.time() < deadline, "ping queued behind the running tool call"
            time.sleep(0.01)
        release.set()
        wf.close()
        t.join(timeout=30)
    finally:
        m.run_checks = orig
    replies = [json.loads(ln) for ln in out.getvalue().splitlines() if ln.strip()]
    (ping_reply,) = [m_ for m_ in replies if m_.get("id") == "ping-1"]
    assert ping_reply["result"] == {}
    (tool_reply,) = [m_ for m_ in replies if m_.get("id") == 1]
    assert "result" in tool_reply  # the run still completed normally

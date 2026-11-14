"""MCP protocol conformance: every item on the checklist in mcp_server.py's
docstring is asserted here, in-process (Session: synchronous handle()) and
over a real pipe (Wire: the reader/worker threads of serve()).

1. newline-delimited UTF-8 JSON on stdout, nothing else
2. initialize negotiates protocolVersion and must come first
3. notifications (and stray responses) never get replies
4. ping answered - including while a tool call runs
5. tool failures and invalid arguments are isError results, not protocol errors
6. cancellation: honored queued / mid-run, and a cancelled request gets no response
7. progress notifications at stage boundaries, strictly increasing
8. batches only under 2025-03-26
9. malformed input never kills or hangs the session
"""

from __future__ import annotations

import io
import json
import os
import threading
import time

import pytest

import kinema.mcp_server as mcp
from kinema.checks.deployment import OVERRIDEABLE_POLICY_FIELDS
from kinema.fixtures.synth import make_dataset
from kinema.mcp_server import (
    ERR_NOT_INITIALIZED,
    SUPPORTED_PROTOCOL_VERSIONS,
    McpServer,
    _dataset_problem,
)


def _init(version="2025-06-18") -> dict:
    return {"jsonrpc": "2.0", "id": "init", "method": "initialize",
            "params": {"protocolVersion": version}}


def _call(req_id, name, arguments, meta=None) -> dict:
    params = {"name": name, "arguments": arguments}
    if meta is not None:
        params["_meta"] = meta
    return {"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": params}


class Session:
    """In-process: feed messages to handle(), collect raw output lines."""

    def __init__(self, init=True, version="2025-06-18"):
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.server = McpServer(out=self.out, err=self.err)
        if init:
            self.server.handle(_init(version))
            self.take()

    def send(self, **msg):
        self.server.handle(msg)

    def call(self, req_id, name, arguments, meta=None) -> dict:
        self.server.handle(_call(req_id, name, arguments, meta))
        (reply,) = [m for m in self.take() if m.get("id") == req_id]
        return reply

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


class Wire:
    """Over a real pipe, through serve()'s reader and worker threads."""

    def __init__(self):
        self.out = io.StringIO()
        self.err = io.StringIO()
        self.server = McpServer(out=self.out, err=self.err)
        r, w = os.pipe()
        self.thread = threading.Thread(
            target=self.server.serve, kwargs={"inp": os.fdopen(r, "rb")}, daemon=True
        )
        self.thread.start()
        self.w = os.fdopen(w, "wb")

    def send(self, msg):
        self.w.write((json.dumps(msg) + "\n").encode())
        self.w.flush()

    def messages(self):
        return [json.loads(ln) for ln in self.out.getvalue().splitlines() if ln.strip()]

    def wait_for(self, pred, timeout=15.0):
        deadline = time.time() + timeout
        while True:
            for m in self.messages():
                if pred(m):
                    return m
            assert time.time() < deadline, f"timed out; got {self.messages()}"
            time.sleep(0.01)

    def close(self, timeout=30.0):
        self.w.close()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "serve() did not return after end of input"


def _text(reply) -> str:
    return reply["result"]["content"][0]["text"]


# ---- 1. the stream ----

def test_stdout_is_pure_newline_delimited_json(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session(init=False)
    s.feed(
        json.dumps(_init()) + "\n"
        + json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        + json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}) + "\n"
        + json.dumps(_call(3, "run_checks", {"dataset": str(root), "max_episodes": 4,
                                             "report_path": str(tmp_path / "report.json")}))
        + "\n"
    )
    for line in s.lines():
        assert "\n" not in line  # one message per line, no embedded newlines
        msg = json.loads(line)  # every line parses as protocol JSON
        assert msg.get("jsonrpc") == "2.0"
    assert {m.get("id") for m in s.replies()} == {"init", 2, 3}
    # stderr may carry logs; stdout carried nothing but protocol messages
    assert s.err.getvalue() == ""


# ---- 2. initialize ----

def test_initialize_version_negotiation_matrix():
    for v in SUPPORTED_PROTOCOL_VERSIONS:
        s = Session(init=False)
        s.send(jsonrpc="2.0", id=1, method="initialize", params={"protocolVersion": v})
        (reply,) = s.take()
        assert reply["result"]["protocolVersion"] == v  # supported: echoed
    s = Session(init=False)
    s.send(jsonrpc="2.0", id=1, method="initialize", params={"protocolVersion": "1999-01-01"})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    s = Session(init=False)
    s.send(jsonrpc="2.0", id=1, method="initialize", params={})
    (reply,) = s.take()
    assert reply["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]
    assert reply["result"]["serverInfo"]["name"] == "kinema"
    assert "tools" in reply["result"]["capabilities"]


def test_requests_before_initialize_are_rejected():
    s = Session(init=False)
    s.send(jsonrpc="2.0", id=1, method="tools/list")
    s.send(jsonrpc="2.0", id=2, method="tools/call", params={"name": "list_checks"})
    s.send(jsonrpc="2.0", id=3, method="ping")  # ping is always allowed
    r1, r2, r3 = s.take()
    assert r1["error"]["code"] == ERR_NOT_INITIALIZED
    assert r2["error"]["code"] == ERR_NOT_INITIALIZED
    assert r3 == {"jsonrpc": "2.0", "id": 3, "result": {}}
    s.send(**_init())
    s.send(jsonrpc="2.0", id=4, method="tools/list")
    assert "tools" in s.take()[-1]["result"]


# ---- 3. notifications ----

def test_notifications_and_stray_responses_get_no_replies():
    s = Session()
    s.send(jsonrpc="2.0", method="notifications/initialized")
    s.send(jsonrpc="2.0", method="notifications/cancelled", params={"requestId": "nope"})
    s.send(jsonrpc="2.0", method="notifications/cancelled", params=[1])  # malformed: ignored
    s.send(jsonrpc="2.0", method="notifications/whatever", params={})
    s.send(jsonrpc="2.0", id=5, result={})  # a response; the server sent no request
    assert s.take() == []


# ---- 4. ping ----

def test_ping_answered():
    s = Session()
    s.send(jsonrpc="2.0", id="p1", method="ping")
    (reply,) = s.take()
    assert reply == {"jsonrpc": "2.0", "id": "p1", "result": {}}


def test_ping_is_answered_while_a_tool_call_runs(tmp_path, monkeypatch):
    """Regression (review finding 4): requests queued behind a running tool
    call, so a liveness ping waited minutes on a Hub load."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    gate = threading.Event()
    real = mcp.run_checks

    def gated(**kw):
        assert gate.wait(15)
        return real(**kw)

    monkeypatch.setattr(mcp, "run_checks", gated)
    w = Wire()
    w.send(_init())
    w.send(_call(1, "run_checks", {"dataset": str(root),
                                   "report_path": str(tmp_path / "r.json")}))
    w.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    w.wait_for(lambda m: m.get("id") == 2)
    assert not any(m.get("id") == 1 for m in w.messages())  # the call is still running
    gate.set()
    reply = w.wait_for(lambda m: m.get("id") == 1)
    assert "result" in reply and not reply["result"].get("isError")
    w.close()


# ---- 5. errors ----

def test_protocol_errors():
    s = Session()
    s.send(jsonrpc="2.0", id=9, method="resources/list")
    assert s.take()[0]["error"]["code"] == -32601
    s.send(jsonrpc="2.0", id=10, method="tools/call", params={"name": "nope"})
    assert s.take()[0]["error"]["code"] == -32602  # unknown tool: invalid params
    s.send(jsonrpc="2.0", id=11, method="tools/call", params=["x"])
    assert s.take()[0]["error"]["code"] == -32602
    s.send(jsonrpc="2.0", id=12, method="tools/call",
           params={"name": "list_checks", "_meta": "tok"})
    assert s.take()[0]["error"]["code"] == -32602
    s.send(jsonrpc="2.0", id=13, method="tools/call",
           params={"name": "list_checks", "arguments": [1]})
    assert s.take()[0]["error"]["code"] == -32602
    s.send(jsonrpc="2.0", id={"bad": 1}, method="ping")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32600 and reply["id"] is None
    s.send(jsonrpc="1.0", id=14, method="ping")
    assert s.take()[0]["error"]["code"] == -32600


def test_malformed_json_is_a_parse_error():
    s = Session()
    s.feed("this is not json\n")
    (reply,) = s.take()
    assert reply["error"]["code"] == -32700 and reply["id"] is None


def test_tool_failures_are_iserror_results_not_protocol_errors(tmp_path):
    s = Session()
    # bad dataset source
    reply = s.call(1, "diagnose_finetune",
                   {"dataset": str(tmp_path / "missing"), "policy": "pi0", "symptom": "x"})
    assert "error" not in reply
    assert reply["result"]["isError"] is True
    # unknown policy
    root = make_dataset(tmp_path / "ds", episodes=8)
    reply = s.call(2, "diagnose_finetune",
                   {"dataset": str(root), "policy": "hal-9000", "symptom": "x"})
    assert reply["result"]["isError"] is True
    assert "unknown policy" in _text(reply)
    # unknown policy override keys
    reply = s.call(3, "diagnose_finetune",
                   {"dataset": str(root), "policy": "pi0", "symptom": "x",
                    "policy_overrides": {"chunk": 10}})
    assert reply["result"]["isError"] is True
    assert "unknown argument policy_overrides.chunk" in _text(reply)


def test_arguments_are_validated_against_the_input_schema(tmp_path):
    """Regression (review finding 9): misspelled arguments were silently
    ignored and a string `checks` was iterated per character."""
    root = str(make_dataset(tmp_path / "ds", episodes=8))
    s = Session()
    cases = [
        ({"dataset": root, "policy": "pi0"}, "missing required argument symptom"),
        ({"dataset": root, "policy": "pi0", "symptom": "x", "policy_override": {}},
         "unknown argument policy_override"),
        ({"dataset": root, "policy": "pi0", "symptom": "x",
          "policy_overrides": {"quantile_norm": "false"}},
         "policy_overrides.quantile_norm: expected boolean, got string"),
        ({"dataset": root, "policy": "pi0", "symptom": "x", "max_episodes": 0},
         "max_episodes: must be >= 1"),
        ({"dataset": root, "policy": "pi0", "symptom": "x", "max_episodes": True},
         "max_episodes: expected integer, got boolean"),
        ({"dataset": 5, "policy": "pi0", "symptom": "x"}, "dataset: expected string"),
    ]
    for i, (args, expected) in enumerate(cases):
        reply = s.call(i, "diagnose_finetune", args)
        assert reply["result"]["isError"] is True, args
        assert expected in _text(reply), (_text(reply), expected)
    reply = s.call(50, "run_checks", {"dataset": root, "checks": "stats-drift"})
    assert "checks: expected array, got string" in _text(reply)
    # null optional arguments are treated as omitted
    reply = s.call(51, "run_checks", {"dataset": root, "checks": ["stats-drift"],
                                      "policy": None, "report_path": str(tmp_path / "r.json")})
    assert not reply["result"].get("isError"), _text(reply)


def test_internal_keyerror_is_not_reported_as_a_missing_argument(tmp_path, monkeypatch):
    def broken(**kw):
        raise KeyError("features")

    monkeypatch.setattr(mcp, "diagnose", broken)
    s = Session()
    reply = s.call(1, "diagnose_finetune",
                   {"dataset": str(tmp_path), "policy": "pi0", "symptom": "x"})
    assert reply["result"]["isError"] is True
    assert _text(reply).startswith("internal error: KeyError")


def test_relative_dataset_paths_are_rejected(tmp_path, monkeypatch):
    """Regression (review finding 7): relative paths resolved against the
    server's cwd, and a missing relative a/b fell through to a Hub download."""
    monkeypatch.chdir(tmp_path)
    assert _dataset_problem("lerobot/pusht") is None  # Hub id
    assert _dataset_problem("lerobot/pusht@v2.1") is None
    assert _dataset_problem("hf://lerobot/pusht") is None
    assert _dataset_problem(str(tmp_path / "anything")) is None  # absolute
    for rel in ("ds", "./ds", "../ds", "data/sets/ds", "ds.zip"):
        assert "relative path" in _dataset_problem(rel), rel
    (tmp_path / "lerobot" / "pusht").mkdir(parents=True)
    assert "ambiguous" in _dataset_problem("lerobot/pusht")
    s = Session()
    reply = s.call(1, "diagnose_finetune", {"dataset": "ds", "policy": "pi0", "symptom": "x"})
    assert reply["result"]["isError"] and "absolute path" in _text(reply)


# ---- report files ----

def test_default_report_lands_next_to_the_dataset(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    reply = s.call(1, "run_checks", {"dataset": str(root), "checks": ["stats-drift"]})
    payload = json.loads(_text(reply))
    path = payload["full_report"]
    assert path.startswith(str(tmp_path / "ds.kinema-report-")) and path.endswith(".json")
    assert json.loads(open(path).read())["source"] == str(root)
    reply = s.call(2, "run_checks", {"dataset": str(root), "checks": ["stats-drift"]})
    assert json.loads(_text(reply))["full_report"] != path  # never overwrites


def test_report_path_is_restricted(tmp_path):
    """Regression (review findings 6, 12): report_path is model-chosen, so
    it may only create a .json file or replace an earlier kinema report."""
    root = str(make_dataset(tmp_path / "ds", episodes=8))
    s = Session()
    precious = tmp_path / "notes.json"
    precious.write_text('{"mine": true}')
    for bad, expected in (("r.json", "absolute path"), (str(tmp_path / "r.txt"), "end in .json"),
                          (str(precious), "refusing to overwrite"),
                          (str(tmp_path / "nope" / "r.json"), "does not exist")):
        reply = s.call(1, "run_checks", {"dataset": root, "report_path": bad})
        assert reply["result"]["isError"] and expected in _text(reply), bad
    assert precious.read_text() == '{"mine": true}'
    target = tmp_path / "r.json"
    for i in (2, 3):  # first creates, second replaces its own earlier report
        reply = s.call(i, "run_checks", {"dataset": root, "checks": ["stats-drift"],
                                         "report_path": str(target)})
        assert json.loads(_text(reply))["full_report"] == str(target)


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions")
def test_unwritable_report_keeps_the_summary(tmp_path, monkeypatch):
    """Regression (review finding 6): a report that could not be written
    turned a finished run into a bare error."""
    ro = tmp_path / "ro"
    root = make_dataset(ro / "ds", episodes=8)
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(cache))
    ro.chmod(0o555)
    cache.chmod(0o555)
    try:
        s = Session()
        reply = s.call(1, "run_checks", {"dataset": str(root), "checks": ["stats-drift"]})
    finally:
        ro.chmod(0o755)
        cache.chmod(0o755)
    assert not reply["result"].get("isError")
    payload = json.loads(_text(reply))
    assert payload["verdict"] and payload["full_report"] is None
    assert "could not write" in payload["full_report_error"]


# ---- tools/list ----

def test_tools_list_shape():
    s = Session()
    s.send(jsonrpc="2.0", id=1, method="tools/list")
    (reply,) = s.take()
    tools = {t["name"]: t for t in reply["result"]["tools"]}
    assert set(tools) == {"diagnose_finetune", "list_checks", "run_checks"}
    for t in tools.values():
        assert t["inputSchema"]["type"] == "object"
    schema = tools["diagnose_finetune"]["inputSchema"]
    assert set(schema["required"]) == {"dataset", "policy", "symptom"}
    overrides = schema["properties"]["policy_overrides"]
    assert set(overrides["properties"]) == set(OVERRIDEABLE_POLICY_FIELDS)


# ---- 7. progress ----

def test_progress_notifications_at_stage_boundaries(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    for name, args in (("diagnose_finetune", {"policy": "pi0", "symptom": "x"}),
                       ("run_checks", {"report_path": str(tmp_path / "r.json")})):
        s.server.handle(_call(1, name, {"dataset": str(root), **args},
                              meta={"progressToken": "tok-1"}))
        replies = s.take()
        progress = [r["params"] for r in replies if r.get("method") == "notifications/progress"]
        assert all(p["progressToken"] == "tok-1" for p in progress)
        values = [p["progress"] for p in progress]
        assert values == sorted(set(values)), values  # strictly increasing
        messages = [p["message"] for p in progress]
        assert messages[:2] == ["loading dataset", "dataset loaded"], name
        assert progress[-1]["progress"] == progress[-1]["total"]
        (result,) = [r for r in replies if r.get("id") == 1]
        assert "result" in result


# ---- 6. cancellation ----

def test_cancelled_request_gets_no_response(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=8)
    s = Session()
    orig = s.server._progress

    def cancel_on_first_progress(token, done, total, message):
        s.server._notify("notifications/cancelled", {"requestId": 7})
        orig(token, done, total, message)

    s.server._progress = cancel_on_first_progress
    s.server.handle(_call(7, "diagnose_finetune",
                          {"dataset": str(root), "policy": "pi0", "symptom": "x"},
                          meta={"progressToken": "t"}))
    assert not [m for m in s.take() if m.get("id") == 7]  # MCP: SHOULD NOT respond
    assert "cancelled" in s.err.getvalue()
    assert s.server._inflight == {}  # no per-request state left behind
    s.send(jsonrpc="2.0", id=8, method="ping")
    assert s.take() == [{"jsonrpc": "2.0", "id": 8, "result": {}}]


def test_cancellation_arriving_mid_run_over_the_wire(tmp_path):
    """A real client sends notifications/cancelled AFTER the run starts; the
    reader thread must apply it while the worker is inside the call."""
    root = make_dataset(tmp_path / "ds", episodes=8)
    w = Wire()
    orig = w.server._progress

    def slow_progress(token, done, total, message):
        time.sleep(0.1)  # widen the window so the cancel lands mid-run
        orig(token, done, total, message)

    w.server._progress = slow_progress
    w.send(_init())
    w.send(_call(1, "run_checks", {"dataset": str(root), "max_episodes": 4,
                                   "report_path": str(tmp_path / "r.json")},
                 meta={"progressToken": "p"}))
    w.wait_for(lambda m: m.get("method") == "notifications/progress")
    w.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}})
    w.send({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    w.wait_for(lambda m: m.get("id") == 2)
    w.close()
    assert not [m for m in w.messages() if m.get("id") == 1]
    assert "cancelled" in w.err.getvalue()


def test_cancelling_a_queued_call_means_it_never_runs(tmp_path, monkeypatch):
    root = make_dataset(tmp_path / "ds", episodes=8)
    gate = threading.Event()
    calls = []
    real = mcp.run_checks

    def gated(**kw):
        calls.append(kw)
        assert gate.wait(15)
        return real(**kw)

    monkeypatch.setattr(mcp, "run_checks", gated)
    w = Wire()
    w.send(_init())
    args = {"dataset": str(root), "checks": ["stats-drift"]}
    w.send(_call(1, "run_checks", {**args, "report_path": str(tmp_path / "a.json")}))
    w.send(_call(2, "run_checks", {**args, "report_path": str(tmp_path / "b.json")}))
    deadline = time.time() + 15
    while not calls:
        assert time.time() < deadline
        time.sleep(0.01)
    w.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 2}})
    w.send({"jsonrpc": "2.0", "id": 3, "method": "ping"})
    w.wait_for(lambda m: m.get("id") == 3)
    gate.set()
    w.wait_for(lambda m: m.get("id") == 1)
    w.close()
    assert len(calls) == 1  # request 2 never started
    assert not [m for m in w.messages() if m.get("id") == 2]
    assert w.server._inflight == {}


# ---- 8. batches ----

def test_batches_only_under_2025_03_26():
    ping = lambda i: {"jsonrpc": "2.0", "id": i, "method": "ping"}  # noqa: E731
    s = Session(version="2025-03-26")
    s.server.handle([ping(1), {"jsonrpc": "2.0", "method": "notifications/initialized"}, ping(2)])
    (batch_reply,) = s.take()
    assert [r["id"] for r in batch_reply] == [1, 2]
    s.server.handle([{"jsonrpc": "2.0", "method": "notifications/initialized"}])
    assert s.take() == []  # all notifications: no response at all
    s.server.handle([])
    assert s.take()[0]["error"]["code"] == -32600
    s = Session(version="2025-06-18")
    s.server.handle([ping(1), ping(2)])
    (reply,) = s.take()
    assert reply["error"]["code"] == -32600 and "batch" in reply["error"]["message"]


# ---- 9. malformed input ----

def test_malformed_input_never_kills_or_hangs_the_session():
    """Regression (review findings 2, 3): a list `params` on a notification
    killed the reader thread and hung the server forever; on a request it
    crashed the process."""
    bad = [
        {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": [1]},
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": ["x"]},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": ["x"]},
        {"jsonrpc": "2.0", "id": 3, "method": 42},
        [],
        "just a string",
    ]
    s = Session()
    text = "".join(json.dumps(m) + "\n" for m in bad)
    text += "\n" + json.dumps({"jsonrpc": "2.0", "id": "alive", "method": "ping"}) + "\n"
    t = threading.Thread(target=s.feed, args=(text,), daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive(), "serve() hung on malformed input"
    replies = s.replies()
    assert {"jsonrpc": "2.0", "id": "alive", "result": {}} in replies
    by_id = {r.get("id"): r for r in replies}
    assert by_id[1]["error"]["code"] == -32602
    assert by_id[2]["error"]["code"] == -32602


def test_undecodable_bytes_are_a_parse_error_not_a_dead_reader():
    s = Session()
    ping = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}).encode()
    s.server.serve(io.BytesIO(b"\xff\xfe\x81 not utf-8\n" + ping + b"\n"))
    first, second = s.replies()
    assert first["error"]["code"] == -32700
    assert second == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_a_failing_input_stream_still_ends_the_session():
    class Exploding:
        def __iter__(self):
            yield json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}) + "\n"
            raise OSError("stdin went away")

    s = Session()
    t = threading.Thread(target=s.server.serve, args=(Exploding(),), daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive()
    assert s.replies() == [{"jsonrpc": "2.0", "id": 1, "result": {}}]

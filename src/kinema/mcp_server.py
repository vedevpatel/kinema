"""Hand-rolled MCP server: JSON-RPC 2.0 over stdio, zero new dependencies.

The tradeoff is deliberate: the repo stays dependency-light, and in return
protocol churn is our problem - four MCP spec versions shipped in about a
year. That is why the supported versions are pinned here and the
conformance checklist below is asserted by tests/test_mcp_protocol.py
rather than left as prose. If HTTP transport or newer protocol features
are ever needed, revisit the official SDK then.

Conformance checklist (each item has a test):

1. Stdio is newline-delimited UTF-8 JSON: one message per line, no embedded
   newlines, and nothing but protocol messages ever goes to stdout. main()
   points sys.stdout at stderr, so a stray print anywhere cannot corrupt
   the stream; logs and tracebacks go to stderr.
2. initialize comes first: protocolVersion is negotiated (echo the client's
   version when supported, else the newest supported), and any other
   request before it gets ERR_NOT_INITIALIZED (ping is always allowed).
3. Notifications (no id) never get replies; neither do client responses
   (this server sends no requests, so there is nothing to answer).
4. ping is answered promptly - including while a tool call runs. Only
   tools/call goes to the worker; the reader answers everything else as it
   arrives.
5. Tool-level failures - including arguments that fail the tool's
   inputSchema - come back as tools/call results with isError: true, never
   as protocol errors. Protocol errors: parse (-32700), malformed JSON-RPC
   (-32600), unknown methods (-32601), invalid tools/call params or unknown
   tools (-32602), not initialized (-32002).
6. notifications/cancelled is honored while the call is queued, during the
   dataset load (between row groups) and between checks. A cancelled
   request gets NO response (MCP: receivers SHOULD NOT respond). A Hub
   download already in flight and a running check are not interrupted.
7. _meta.progressToken gets notifications/progress: 0 "loading dataset",
   heartbeats while loading, 1 "dataset loaded", one per check - strictly
   increasing.
8. JSON-RPC batches are accepted only when 2025-03-26 is negotiated (the
   one supported version that requires receiving them); otherwise -32600.
9. Malformed input never kills or hangs the session: every message is
   handled under a guard, undecodable bytes are a parse error, and the
   reader always signals end-of-input to the worker.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from kinema import __version__
from kinema.checks.deployment import OVERRIDEABLE_POLICY_FIELDS
from kinema.dataset import DatasetError
from kinema.diagnose import (
    CancelledError,
    diagnose,
    known_policies,
    list_checks,
    run_checks,
    summarize_report,
)
from kinema.report import render_json

SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"]
# 2025-03-26 added JSON-RPC batching (receivers MUST accept batches);
# 2025-06-18 removed it again
_BATCH_VERSIONS = {"2025-03-26"}

SERVER_INFO = {"name": "kinema", "version": __version__}

ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_NOT_INITIALIZED = -32002  # implementation-defined server error range

# run_checks returns a summary sized for MCP client output limits (Claude
# Code warns at 10k output tokens); the full JSON report goes to a file
_TOP_FINDINGS = 25


def _schema(types: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": types, "required": required, "additionalProperties": False}


_DATASET_ARG: dict[str, Any] = {
    "type": "string",
    "description": "LeRobot dataset: an ABSOLUTE local path (directory or .zip), or a Hub "
    "repo id owner/name (optionally @revision; prefix hf:// to force the Hub). Relative "
    "paths are rejected - the server's working directory is not yours.",
}
_POLICY_OVERRIDES_ARG: dict[str, Any] = {
    "type": "object",
    "description": "Per-call overrides merged over the named policy's declared defaults "
    "for this run only",
    "properties": {
        "chunk_size": {"type": "integer", "minimum": 1, "description": "action horizon in frames"},
        "quantile_norm": {"type": "boolean", "description": "the normalizer consumes q01/q99"},
        "needs_images": {"type": "boolean", "description": "the policy is vision-conditioned"},
        "needs_language": {"type": "boolean", "description": "the policy is language-conditioned"},
        "typical_resolution": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "minItems": 2,
            "maxItems": 2,
            "description": "[width, height] the policy is usually trained at",
        },
    },
    "additionalProperties": False,
}
assert set(_POLICY_OVERRIDES_ARG["properties"]) == set(OVERRIDEABLE_POLICY_FIELDS)
_MAX_EPISODES_ARG: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "description": "Audit a sample of N episodes spread across the dataset",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "diagnose_finetune",
        "description": "Why did my fine-tune fail: rank the data causes that explain a "
        "symptom, with check evidence. Every check runs; the symptom only orders the "
        "causes. Answers always enumerate what ran, what skipped and what the v0 cause "
        "map cannot see - never a blanket 'data is clean'.",
        "inputSchema": _schema(
            {
                "dataset": _DATASET_ARG,
                "policy": {
                    "type": "string",
                    "description": f"Target policy ({', '.join(known_policies())}; "
                    "aliases like pi0.5 resolve to their family entry)",
                },
                "symptom": {
                    "type": "string",
                    "description": "Free text, e.g. 'loss converged but the robot does not move'",
                },
                "policy_overrides": _POLICY_OVERRIDES_ARG,
                "max_episodes": _MAX_EPISODES_ARG,
            },
            ["dataset", "policy", "symptom"],
        ),
    },
    {
        "name": "list_checks",
        "description": "The kinema check catalog: id, title, one-line description, docs link.",
        "inputSchema": _schema({}, []),
    },
    {
        "name": "run_checks",
        "description": "Run kinema checks and get a capped summary (verdict, score, per-check "
        "results, top findings); the full JSON report is written to a file whose path is returned.",
        "inputSchema": _schema(
            {
                "dataset": _DATASET_ARG,
                "checks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Check ids to run (default: all)",
                },
                "policy": {"type": "string", "description": "Target policy (optional)"},
                "policy_overrides": _POLICY_OVERRIDES_ARG,
                "max_episodes": _MAX_EPISODES_ARG,
                "report_path": {
                    "type": "string",
                    "description": "Absolute path ending in .json for the full JSON report; an "
                    "existing file is only replaced if it is a kinema report. Default: "
                    "<dataset>.kinema-report-<timestamp>.json next to a local dataset, "
                    "else ~/.cache/kinema/reports/",
                },
            },
            ["dataset"],
        ),
    },
]
_TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def _negotiate_version(client) -> str:
    if client in SUPPORTED_PROTOCOL_VERSIONS:
        return client
    return SUPPORTED_PROTOCOL_VERSIONS[-1]


def _valid_id(x) -> bool:
    """MCP request ids (and progress tokens): a string or an integer."""
    return isinstance(x, str) or (isinstance(x, int) and not isinstance(x, bool))


# ---- argument validation (the subset of JSON Schema the tools use) ----

_JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "object": dict, "array": list}


def _json_type(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "boolean"
    for name, py in (("integer", int), ("number", float), ("string", str),
                     ("array", list), ("object", dict)):
        if isinstance(v, py):
            return name
    return type(v).__name__


def _validate(schema: dict, value, path: str = "") -> list[str]:
    """Problems with value against schema; [] when it conforms. Clients
    are not required to validate tool arguments, so the server does - a
    misspelled argument must fail, not silently run with defaults."""
    where = path or "arguments"
    t = schema.get("type")
    py = _JSON_TYPES.get(t) if isinstance(t, str) else None
    if py is not None and (
        not isinstance(value, py) or (t == "integer" and isinstance(value, bool))
    ):
        return [f"{where}: expected {t}, got {_json_type(value)}"]
    problems: list[str] = []
    if t == "integer" and "minimum" in schema and value < schema["minimum"]:
        problems.append(f"{where}: must be >= {schema['minimum']}, got {value}")
    if t == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{where}: needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{where}: allows at most {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                problems += _validate(schema["items"], v, f"{where}[{i}]")
    if t == "object":
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                problems.append(f"missing required argument {(path + '.' if path else '') + key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    problems.append(
                        f"unknown argument {(path + '.' if path else '') + key} "
                        f"(valid: {', '.join(props) or 'none'})"
                    )
        for key, sub in props.items():
            if key in value:
                problems += _validate(sub, value[key], (path + "." if path else "") + key)
    return problems


# ---- dataset and report paths (relative to the CLIENT, not this process) ----

_HUB_ID = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+(@\S+)?")


def _dataset_problem(dataset: str) -> str | None:
    """Why a dataset argument cannot be resolved unambiguously, or None.

    The server's working directory is whatever the client launched it in
    (often / or the client's install dir), so a relative path would resolve
    against the wrong place - and a missing relative a/b would fall through
    to a Hub download of repo a/b, auditing a different dataset."""
    src = dataset.strip()
    if src.startswith("hf://"):
        return None
    p = Path(src).expanduser()
    if p.is_absolute():
        return None
    cwd = os.getcwd()
    if _HUB_ID.fullmatch(src) and not src.startswith("."):
        if not p.exists():
            return None  # owner/name: a Hub repo id
        return (
            f"dataset {dataset!r} is ambiguous: it exists relative to the server's working "
            f"directory ({cwd}) and is also a Hub repo id. Pass an absolute path for the "
            f"local copy, or hf://{src} for the Hub."
        )
    return (
        f"dataset {dataset!r} is a relative path, but the server's working directory "
        f"({cwd}) is not yours. Pass an absolute path, or a Hub repo id as owner/name."
    )


def _is_kinema_report(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return isinstance(d, dict) and "source" in d and isinstance(d.get("checks"), list)


def _report_path_problem(raw: str) -> str | None:
    """Checked BEFORE the run: report_path is model-chosen, so it may only
    create a .json file or replace an earlier kinema report."""
    p = Path(raw).expanduser()
    if not p.is_absolute():
        return (
            f"report_path {raw!r} must be an absolute path (the server's working "
            f"directory, {os.getcwd()}, is not yours)"
        )
    if p.suffix.lower() != ".json":
        return f"report_path {raw!r} must end in .json"
    if p.exists() and not (p.is_file() and _is_kinema_report(p)):
        return f"refusing to overwrite {p}: it exists and is not a kinema report"
    if not p.parent.is_dir():
        return f"report_path directory {p.parent} does not exist"
    return None


def _reports_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(base) / "kinema" / "reports"


def _default_report_paths(dataset: str) -> list[Path]:
    """Where an unrequested report goes, in order: next to a local
    dataset, then the user cache dir. Timestamped, never overwriting."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    src = dataset.strip()
    if src.startswith("hf://"):
        src = src[len("hf://"):]
    local = Path(src).expanduser()
    out: list[Path] = []
    if local.is_absolute() and local.exists():
        name = local.stem if local.suffix == ".zip" else local.name
        out.append(local.parent / f"{name}.kinema-report-{stamp}.json")
    else:
        name = src
    slug = re.sub(r"[^\w.-]+", "_", name).strip("_") or "dataset"
    out.append(_reports_cache_dir() / f"{slug}-{stamp}.json")
    return out


def _write_report(text: str, requested: str | None, dataset: str) -> tuple[str | None, str | None]:
    """(path written, error). Never raises: the run already finished, so a
    report that cannot be written must not throw its summary away."""
    if requested:
        p = Path(requested).expanduser()
        try:
            p.write_text(text)
            return str(p), None
        except OSError as e:
            return None, f"could not write {p}: {e}"
    errors = []
    for p in _default_report_paths(dataset):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "x") as f:
                f.write(text)
            return str(p), None
        except OSError as e:
            errors.append(f"{p}: {e}")
    return None, "could not write the full report (" + "; ".join(errors) + ")"


class McpServer:
    """One stdio session. Every message goes out through _send, which keeps
    stdout pure: one compact-JSON message per line, nothing else."""

    def __init__(self, out=None, err=None):
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        # tools/call request id -> cancel Event, from queueing until done;
        # cancels for ids not in here (finished, unknown) are dropped
        self._inflight: dict = {}
        self.protocol_version: str | None = None  # set by initialize

    # ---- wire ----

    def _send(self, msg) -> None:
        line = json.dumps(msg, separators=(",", ":"), default=str)
        if "\n" in line:  # json.dumps escapes newlines; this guards the invariant
            raise ValueError("protocol messages must not embed newlines")
        with self._write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def _log(self, text: str) -> None:
        print(text, file=self.err, flush=True)

    @staticmethod
    def _result(req_id, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_error(req_id, text: str) -> dict:
        """A tool-level failure: a result with isError, not a protocol error."""
        return McpServer._result(
            req_id, {"content": [{"type": "text", "text": text}], "isError": True}
        )

    @staticmethod
    def _tool_json(req_id, payload: dict) -> dict:
        return McpServer._result(
            req_id, {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]}
        )

    def _progress(self, token, done, total, message: str) -> None:
        self._send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {
                    "progressToken": token,
                    "progress": done,
                    "total": total,
                    "message": message,
                },
            }
        )

    # ---- dispatch ----

    def handle(self, msg, emit=None) -> None:
        """Process one parsed message - or a batch - to completion.

        Responses go to emit (default: stdout). Never raises: an internal
        failure becomes an error response (or a stderr log for messages
        that cannot carry one), and the session stays up."""
        emit = emit or self._send
        if isinstance(msg, list):
            self._handle_batch(msg)
            return
        try:
            self._dispatch(msg, emit)
        except Exception as e:  # noqa: BLE001 - malformed input must not end the session
            traceback.print_exc(file=self.err)
            rid = msg.get("id") if isinstance(msg, dict) else None
            if _valid_id(rid) and isinstance(msg, dict) and "method" in msg:
                emit(self._error(rid, ERR_INTERNAL, f"internal error: {type(e).__name__}: {e}"))

    def _handle_batch(self, batch: list) -> None:
        if self.protocol_version not in _BATCH_VERSIONS:
            self._send(self._error(
                None, ERR_INVALID_REQUEST,
                "JSON-RPC batches are not supported in protocol version "
                f"{self.protocol_version or '(not initialized)'}",
            ))
            return
        if not batch:
            self._send(self._error(None, ERR_INVALID_REQUEST, "empty batch"))
            return
        replies: list[dict] = []
        for m in batch:
            if isinstance(m, list):
                replies.append(self._error(None, ERR_INVALID_REQUEST, "nested batch"))
            else:
                self.handle(m, emit=replies.append)
        if replies:  # an all-notification batch gets no response
            self._send(replies)

    def _dispatch(self, msg, emit) -> None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            rid = msg.get("id") if isinstance(msg, dict) else None
            emit(self._error(rid if _valid_id(rid) else None, ERR_INVALID_REQUEST,
                             "invalid JSON-RPC message"))
            return
        if "method" not in msg:
            if "result" in msg or "error" in msg:
                # a response to a server->client request; this server sends none
                self._log(f"ignoring unexpected response message (id={msg.get('id')!r})")
                return
            rid = msg.get("id")
            emit(self._error(rid if _valid_id(rid) else None, ERR_INVALID_REQUEST,
                             "message has no method"))
            return
        method = msg["method"]
        if "id" not in msg:  # a notification: never reply (checklist item 3)
            if isinstance(method, str):
                self._notify(method, msg.get("params"))
            return
        req_id = msg["id"]
        if not _valid_id(req_id) or not isinstance(method, str):
            emit(self._error(req_id if _valid_id(req_id) else None, ERR_INVALID_REQUEST,
                             "request needs a string method and a string or integer id"))
            return
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            emit(self._error(req_id, ERR_INVALID_PARAMS, "params must be an object"))
            return
        if method == "ping":
            emit(self._result(req_id, {}))
            return
        if method == "initialize":
            self.protocol_version = _negotiate_version(params.get("protocolVersion"))
            emit(self._result(
                req_id,
                {
                    "protocolVersion": self.protocol_version,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            ))
            return
        if self.protocol_version is None:
            emit(self._error(req_id, ERR_NOT_INITIALIZED,
                             "server not initialized: send initialize first"))
            return
        if method == "tools/list":
            emit(self._result(req_id, {"tools": TOOLS}))
        elif method == "tools/call":
            self._call_tool(req_id, params, emit)
        else:
            emit(self._error(req_id, ERR_METHOD_NOT_FOUND, f"unknown method {method!r}"))

    def _notify(self, method: str, params) -> None:
        if method == "notifications/cancelled" and isinstance(params, dict):
            rid = params.get("requestId")
            if _valid_id(rid):
                with self._state_lock:
                    ev = self._inflight.get(rid)
                if ev is not None:
                    ev.set()
        # notifications/initialized and anything else: nothing to do

    def _register(self, req_id) -> threading.Event:
        with self._state_lock:
            return self._inflight.setdefault(req_id, threading.Event())

    def _unregister(self, req_id) -> None:
        with self._state_lock:
            self._inflight.pop(req_id, None)

    # ---- tools ----

    def _call_tool(self, req_id, params: dict, emit) -> None:
        name = params.get("name")
        if not isinstance(name, str) or name not in _TOOLS_BY_NAME:
            emit(self._error(req_id, ERR_INVALID_PARAMS, f"unknown tool {name!r}"))
            return
        args = params.get("arguments")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            emit(self._error(req_id, ERR_INVALID_PARAMS, "tools/call arguments must be an object"))
            return
        meta = params.get("_meta")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            emit(self._error(req_id, ERR_INVALID_PARAMS, "_meta must be an object"))
            return
        token = meta.get("progressToken")
        if token is not None and not _valid_id(token):
            emit(self._error(req_id, ERR_INVALID_PARAMS, "progressToken must be a string or integer"))
            return
        cancel = self._register(req_id)
        try:
            self._run_tool(req_id, name, args, token, cancel, emit)
        finally:
            self._unregister(req_id)

    def _run_tool(self, req_id, name: str, args: dict, token, cancel: threading.Event, emit) -> None:
        # model clients often send null for an optional argument they mean
        # to omit; treat it as absent (a null REQUIRED argument still fails)
        args = {k: v for k, v in args.items() if v is not None}
        problems = _validate(_TOOLS_BY_NAME[name]["inputSchema"], args)
        if not problems and "dataset" in args:
            problem = _dataset_problem(args["dataset"])
            if problem is None and args.get("report_path"):
                problem = _report_path_problem(args["report_path"])
            problems = [problem] if problem else []
        if problems:
            emit(self._tool_error(req_id, "invalid arguments: " + "; ".join(problems)))
            return
        if cancel.is_set():
            self._log(f"request {req_id!r} cancelled before it started; no response sent")
            return

        def progress(done, total, message: str) -> None:
            if token is not None and not cancel.is_set():
                self._progress(token, done, total, message)

        try:
            payload: dict[str, Any]
            if name == "list_checks":
                payload = {"checks": list_checks()}
            elif name == "diagnose_finetune":
                payload = diagnose(
                    dataset=args["dataset"],
                    policy=args["policy"],
                    symptom=args["symptom"],
                    policy_overrides=args.get("policy_overrides"),
                    max_episodes=args.get("max_episodes"),
                    cancelled=cancel,
                    progress=progress,
                )
            else:  # run_checks
                report, _ctx = run_checks(
                    dataset=args["dataset"],
                    checks=args.get("checks"),
                    policy=args.get("policy"),
                    policy_overrides=args.get("policy_overrides"),
                    max_episodes=args.get("max_episodes"),
                    cancelled=cancel,
                    progress=progress,
                )
                payload = summarize_report(report, top=_TOP_FINDINGS)
                path, error = _write_report(
                    render_json(report) + "\n", args.get("report_path"), args["dataset"]
                )
                payload["full_report"] = path
                if error:
                    payload["full_report_error"] = error
        except CancelledError as e:
            self._log(f"request {req_id!r} {e}; no response sent")
            return
        except (DatasetError, ValueError, OSError) as e:
            emit(self._tool_error(req_id, f"{type(e).__name__}: {e}"))
            return
        except Exception as e:  # noqa: BLE001 - a crashed tool is still a tool error
            traceback.print_exc(file=self.err)
            emit(self._tool_error(req_id, f"internal error: {type(e).__name__}: {e}"))
            return
        if cancel.is_set():  # cancelled after the last checkpoint: still no response
            self._log(f"request {req_id!r} cancelled after finishing; no response sent")
            return
        emit(self._tool_json(req_id, payload))

    # ---- main loop ----

    def _ingest(self, raw, pending: queue.Queue) -> None:
        """One input line, on the reader thread. tools/call is queued for
        the worker; everything else is cheap and answered right here, so a
        ping never waits behind a long tool call."""
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                self._send(self._error(None, ERR_PARSE, "parse error: input is not valid UTF-8"))
                return
        line = raw.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except (ValueError, RecursionError):
            self._send(self._error(None, ERR_PARSE, "parse error"))
            return
        if isinstance(msg, list):
            pending.put(msg)  # batches run on the worker, in order
        elif isinstance(msg, dict) and msg.get("method") == "tools/call" and _valid_id(msg.get("id")):
            self._register(msg["id"])  # a cancel can land while it is still queued
            pending.put(msg)
        else:
            self.handle(msg)

    def serve(self, inp=None) -> None:
        """Read on a reader thread, run tool calls on this one.

        Notifications and cheap requests are handled by the reader as they
        arrive, so a cancel lands DURING a run and a ping is answered while
        one is in progress. The reader always enqueues the end-of-input
        sentinel, whatever the input did, so the session can never hang."""
        stream = inp if inp is not None else sys.stdin.buffer
        pending: queue.Queue = queue.Queue()

        def reader() -> None:
            try:
                for raw in stream:
                    try:
                        self._ingest(raw, pending)
                    except Exception:  # noqa: BLE001 - keep reading
                        traceback.print_exc(file=self.err)
            except Exception:  # noqa: BLE001 - the stream itself failed
                traceback.print_exc(file=self.err)
            finally:
                pending.put(None)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        while True:
            msg = pending.get()
            if msg is None:
                break
            try:
                self.handle(msg)
            finally:
                if isinstance(msg, dict) and _valid_id(msg.get("id")):
                    self._unregister(msg["id"])


def main() -> int:
    proto_out = sys.stdout
    reconfigure = getattr(proto_out, "reconfigure", None)
    if reconfigure is not None:  # a real TextIOWrapper (not a test double)
        try:
            reconfigure(encoding="utf-8", newline="\n")
        except ValueError:
            pass
    # anything that prints - this code or a library - lands on stderr; the
    # protocol stream is written only through McpServer._send
    sys.stdout = sys.stderr
    print(f"kinema mcp server {__version__} on stdio", file=sys.stderr)
    McpServer(out=proto_out, err=sys.stderr).serve()
    return 0

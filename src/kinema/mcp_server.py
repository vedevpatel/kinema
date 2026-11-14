"""Hand-rolled MCP server: JSON-RPC 2.0 over stdio, zero new dependencies.

The tradeoff is deliberate: the repo stays dependency-light, and in return
protocol churn is our problem - four MCP spec versions shipped in about a
year. That is why the supported versions are pinned here and the
conformance checklist below is asserted by tests/test_mcp_protocol.py
rather than left as prose. If HTTP transport or newer protocol features
are ever needed, revisit the official SDK then.

Conformance checklist (each item has a test):

1. Stdio is newline-delimited JSON: one message per line, no embedded
   newlines, and nothing but protocol messages ever goes to stdout (logs
   and tracebacks go to stderr).
2. initialize is answered first; protocolVersion is negotiated (echo the
   client's version when supported, else the newest supported).
3. Notifications (no id) never get replies.
4. ping is answered with an empty result.
5. Tool-level failures come back as tools/call results with
   isError: true - never as JSON-RPC protocol errors. Protocol errors are
   reserved for malformed JSON-RPC (-32600), unknown methods (-32601) and
   invalid params (-32602).
6. notifications/cancelled is honored between checks.
7. _meta.progressToken gets notifications/progress at stage boundaries.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback

from kinema import __version__
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

SERVER_INFO = {"name": "kinema", "version": __version__}

# run_checks returns a summary sized for MCP client output limits (Claude
# Code warns at 10k output tokens); the full JSON report goes to a file
_TOP_FINDINGS = 25


def _schema(types: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": types, "required": required, "additionalProperties": False}


_DATASET_ARG = {
    "type": "string",
    "description": "LeRobot dataset: local directory, .zip, or Hub repo id (owner/name)",
}
_POLICY_OVERRIDES_ARG = {
    "type": "object",
    "description": "Per-call policies.json overrides merged over the named policy's defaults "
    "(chunk_size, quantile_norm, needs_images, needs_language, typical_resolution)",
}
_MAX_EPISODES_ARG = {
    "type": "integer",
    "description": "Audit a sample of N episodes spread across the dataset",
}

TOOLS = [
    {
        "name": "diagnose_finetune",
        "description": "Why did my fine-tune fail: rank the data causes that explain a "
        "symptom, with check evidence. Answers always enumerate what ran, what skipped "
        "and what the v0 cause map cannot see - never a blanket 'data is clean'.",
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
                    "description": "Where to write the full JSON report "
                    "(default: kinema-report.json next to the dataset)",
                },
            },
            ["dataset"],
        ),
    },
]


def _negotiate_version(client: str | None) -> str:
    if client in SUPPORTED_PROTOCOL_VERSIONS:
        return client  # type: ignore[return-value]
    return SUPPORTED_PROTOCOL_VERSIONS[-1]


class McpServer:
    """One stdio session. Replies are written by _reply only, which keeps
    stdout pure: one compact-JSON message per line, nothing else."""

    def __init__(self, out=None, err=None):
        self.out = out or sys.stdout
        self.err = err or sys.stderr
        self.cancelled: set = set()
        self._write_lock = threading.Lock()

    # ---- wire ----

    def _reply(self, msg: dict) -> None:
        line = json.dumps(msg, separators=(",", ":"), default=str)
        assert "\n" not in line, "protocol messages must not embed newlines"
        with self._write_lock:
            self.out.write(line + "\n")
            self.out.flush()

    def _result(self, req_id, result: dict) -> None:
        self._reply({"jsonrpc": "2.0", "id": req_id, "result": result})

    def _error(self, req_id, code: int, message: str) -> None:
        self._reply({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})

    def _progress(self, token, done: int, total: int, message: str) -> None:
        self._reply(
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

    def handle(self, msg: dict) -> None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            self._error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid JSON-RPC request")
            return
        method = msg.get("method")
        req_id = msg.get("id")
        if req_id is None:  # a notification: never reply (checklist item 3)
            self._notify(method, msg.get("params") or {})
            return
        if method == "initialize":
            params = msg.get("params") or {}
            self._result(
                req_id,
                {
                    "protocolVersion": _negotiate_version(params.get("protocolVersion")),
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            )
        elif method == "ping":
            self._result(req_id, {})
        elif method == "tools/list":
            self._result(req_id, {"tools": TOOLS})
        elif method == "tools/call":
            self._call_tool(req_id, msg.get("params") or {})
        else:
            self._error(req_id, -32601, f"unknown method {method!r}")

    def _notify(self, method: str | None, params: dict) -> None:
        if method == "notifications/cancelled":
            rid = params.get("requestId")
            if rid is not None:
                self.cancelled.add(rid)
        # notifications/initialized and anything else: nothing to do

    # ---- tools ----

    def _tool_error(self, req_id, text: str) -> None:
        """A tool-level failure: a result with isError, not a protocol error."""
        self._result(
            req_id,
            {"content": [{"type": "text", "text": text}], "isError": True},
        )

    def _tool_json(self, req_id, payload: dict) -> None:
        self._result(
            req_id,
            {"content": [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]},
        )

    def _call_tool(self, req_id, params: dict) -> None:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            self._error(req_id, -32602, "tools/call arguments must be an object")
            return
        if name not in {t["name"] for t in TOOLS}:
            self._error(req_id, -32602, f"unknown tool {name!r}")
            return
        token = (params.get("_meta") or {}).get("progressToken")
        cancel = threading.Event()

        def progress(done: int, total: int, message: str) -> None:
            if req_id in self.cancelled:
                cancel.set()
            if token is not None:
                self._progress(token, done, total, message)

        try:
            if name == "list_checks":
                self._tool_json(req_id, {"checks": list_checks()})
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
                payload["_cancel_supported"] = "cancellation is honored between checks, not mid-check"
                self._tool_json(req_id, payload)
            else:  # run_checks
                report, ctx = run_checks(
                    dataset=args["dataset"],
                    checks=args.get("checks"),
                    policy=args.get("policy"),
                    policy_overrides=args.get("policy_overrides"),
                    max_episodes=args.get("max_episodes"),
                    cancelled=cancel,
                    progress=progress,
                )
                path = args.get("report_path") or "kinema-report.json"
                with open(path, "w") as f:
                    f.write(render_json(report) + "\n")
                payload = summarize_report(report, top=_TOP_FINDINGS)
                payload["full_report"] = path
                self._tool_json(req_id, payload)
        except KeyError as e:
            self._tool_error(req_id, f"missing required argument: {e}")
        except CancelledError as e:
            self._tool_error(req_id, f"cancelled: {e}")
        except (DatasetError, ValueError, OSError) as e:
            self._tool_error(req_id, f"{type(e).__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - a crashed tool is still a tool error
            traceback.print_exc(file=self.err)
            self._tool_error(req_id, f"internal error: {type(e).__name__}: {e}")

    # ---- main loop ----

    def serve(self, inp=None) -> None:
        stream = inp or sys.stdin
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._error(None, -32700, "parse error")
                continue
            self.handle(msg)


def main() -> int:
    print(f"kinema mcp server {__version__} on stdio", file=sys.stderr)
    McpServer().serve()
    return 0

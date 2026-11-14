"""Real-client shakedown of `kinema mcp` using the official MCP Python SDK.

Covers: initialize + version negotiation, tools/list, list_checks,
run_checks on a small real dataset with a progress token, a mid-run cancel,
diagnose_finetune on a real failing scenario, and deliberate error paths.
"""

import anyio
import json
import sys

import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.shared.exceptions import MCPError
from mcp.client.stdio import stdio_client

DATASET = "lerobot/svla_so100_sorting@13870ca969084d8ce4d8a0391b06cead78e804ba"
BIG_DATASET = "lerobot/libero_10@551d7d86f25edd0ffeda8b60053c15438cfd1d6a"
RESULTS = []


def _cb(store):
    async def cb(p, t, m):
        store.append((p, t, m))
    return cb


def report(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)


class _IdRecorder:
    """Proxy on the write stream that records the last tools/call request id."""

    def __init__(self, inner):
        self._inner = inner
        self.last_call_id = None

    async def send(self, msg):
        root = getattr(msg, "message", msg)
        root = getattr(root, "root", root)
        if isinstance(root, types.JSONRPCRequest) and root.method == "tools/call":
            self.last_call_id = root.id
        await self._inner.send(msg)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


async def main():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kinema.cli", "mcp"],  # run from an env with kinema + mcp installed
        env=None,
    )
    async with stdio_client(params) as (read, write):
        rec = _IdRecorder(write)
        async with ClientSession(read, rec) as session:
            init = await session.initialize()
            report("initialize", True,
                   f"server={init.server_info.name} {init.server_info.version} "
                   f"protocol={init.protocol_version} caps={list(init.capabilities.model_dump(exclude_none=True))}")

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            report("tools/list", names == ["diagnose_finetune", "list_checks", "run_checks"], str(names))

            r = await session.call_tool("list_checks", {})
            txt = r.content[0].text if r.content else ""
            report("list_checks", not r.is_error and "calibration" in txt.lower() or "check" in txt.lower(),
                   f"{len(txt)} chars, isError={r.is_error}")

            # run_checks with a progress token on a small real dataset
            progress = []
            r = await session.call_tool(
                "run_checks",
                {"dataset": DATASET, "max_episodes": 8, "report_path": "/tmp/kinema-mcp-report.json"},
                progress_callback=_cb(progress),
                read_timeout_seconds=300.0,
            )
            txt = r.content[0].text if r.content else ""
            ok = not r.is_error and "verdict" in txt.lower()
            report("run_checks real dataset", ok,
                   f"progress events={len(progress)}, report={txt[:200]!r}")
            if progress:
                report("progress notifications sane",
                       all(t is None or p <= t for p, t, _ in progress),
                       f"first={progress[0]} last={progress[-1]}")
            try:
                full = json.load(open("/tmp/kinema-mcp-report.json"))
                report("full report file written", "checks" in full or "results" in full or "verdict" in full,
                       f"keys={list(full)[:8]}")
            except Exception as e:
                report("full report file written", False, str(e))

            # mid-run cancel: start a bigger run, cancel after first progress
            cancel_progress = []

            async def run_big():
                return await session.call_tool(
                    "run_checks",
                    {"dataset": BIG_DATASET, "max_episodes": 200,
                     "report_path": "/tmp/kinema-mcp-cancel.json"},
                    progress_callback=_cb(cancel_progress),
                    read_timeout_seconds=300.0,
                )

            # MCP: a cancelled request gets NO response, so the check is that
            # nothing arrives for it after the cancel - and the session lives
            quiet_s = 20.0
            holder = {}
            async with anyio.create_task_group() as tg:

                async def runner():
                    holder["res"] = await run_big()
                    tg.cancel_scope.cancel()  # answered: nothing left to wait for

                async def cancel_then_wait():
                    while not cancel_progress:
                        await anyio.sleep(0.05)
                    await session.send_notification(types.CancelledNotification(
                        method="notifications/cancelled",
                        params=types.CancelledNotificationParams(
                            requestId=rec.last_call_id,
                            reason="shakedown cancel")))
                    await anyio.sleep(quiet_s)
                    tg.cancel_scope.cancel()  # stays unanswered: abandon the call

                tg.start_soon(runner)
                tg.start_soon(cancel_then_wait)
            res = holder.get("res")
            if res is None:
                report("mid-run cancel", True,
                       f"no response within {quiet_s:.0f}s of the cancel; progress={len(cancel_progress)}")
            else:
                rtxt = res.content[0].text if res.content else ""
                report("mid-run cancel", False,
                       f"got a response: isError={res.is_error} text={rtxt[:120]!r} (either the run "
                       "finished before the cancel landed - a race, not a hang - or a cancelled "
                       "call was answered)")
            try:
                await session.send_ping()
                report("session alive after cancel", True)
            except Exception as e:  # noqa: BLE001
                report("session alive after cancel", False, str(e))

            # diagnose_finetune on a real failing scenario
            r = await session.call_tool(
                "diagnose_finetune",
                {"dataset": DATASET, "policy": "pi0.5",
                 "symptom": "loss converged but the robot does not move",
                 "max_episodes": 8},
                read_timeout_seconds=300.0,
            )
            txt = r.content[0].text if r.content else ""
            report("diagnose_finetune real scenario", not r.is_error and len(txt) > 100,
                   f"{len(txt)} chars: {txt[:160]!r}")

            # error paths: unknown tool / bad params are -32602 protocol errors per
            # the server's conformance checklist; the SDK surfaces them as MCPError.
            # What matters to a real client: the error is catchable and the session lives.
            async def expect_error(name, coro, allow_tool_error=True):
                try:
                    r = await coro
                except MCPError as e:
                    report(name, e.error.code == -32602, f"MCPError code={e.error.code}: {e.error.message[:100]}")
                    return
                detail = (r.content[0].text if r.content else "")[:120]
                report(name, allow_tool_error and r.is_error, f"tool result isError={r.is_error}: {detail}")

            await expect_error("bad tool name -> -32602, catchable",
                               session.call_tool("no_such_tool", {}))
            t = await session.list_tools()
            report("session alive after protocol error", len(t.tools) == 3)

            await expect_error("missing required arg -> error",
                               session.call_tool("run_checks", {}))
            await expect_error("nonexistent policy -> error",
                               session.call_tool("diagnose_finetune",
                                                 {"dataset": DATASET, "policy": "not_a_policy",
                                                  "symptom": "x", "max_episodes": 2},
                                                 read_timeout_seconds=120.0))
            await expect_error("nonexistent dataset -> error",
                               session.call_tool("run_checks",
                                                 {"dataset": "no_such_owner/no_such_dataset_xyz",
                                                  "max_episodes": 2},
                                                 read_timeout_seconds=120.0))
            t = await session.list_tools()
            report("session alive after all error paths", len(t.tools) == 3)

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" +
          (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(anyio.run(main))

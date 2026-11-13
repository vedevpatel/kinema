"""Report model and renderers: rich console, markdown, JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from kinema.checks.base import CheckResult, Severity

# health score: start at 100, pay per failing/warning check (not per finding)
_FAIL_COST = 25
_WARN_COST = 8


@dataclass
class DatasetReport:
    source: str
    format_version: str
    episodes: int
    frames: int
    fps: float | None
    results: list[CheckResult] = field(default_factory=list)
    truncated: bool = False

    @property
    def verdict(self) -> Severity:
        vs = [r.verdict for r in self.results if r.verdict != Severity.SKIP]
        return max(vs, default=Severity.OK)

    @property
    def score(self) -> int:
        score = 100
        for r in self.results:
            if r.verdict == Severity.FAIL:
                score -= _FAIL_COST
            elif r.verdict == Severity.WARN:
                score -= _WARN_COST
        return max(0, score)

    def counts(self) -> dict[str, int]:
        out = {"fail": 0, "warn": 0, "info": 0}
        for r in self.results:
            for f in r.findings:
                if f.severity == Severity.FAIL:
                    out["fail"] += 1
                elif f.severity == Severity.WARN:
                    out["warn"] += 1
                else:
                    out["info"] += 1
        return out


_STYLE = {
    Severity.OK: "green",
    Severity.INFO: "cyan",
    Severity.WARN: "yellow",
    Severity.FAIL: "bold red",
    Severity.SKIP: "dim",
}
_MARK = {
    Severity.OK: "PASS",
    Severity.INFO: "INFO",
    Severity.WARN: "WARN",
    Severity.FAIL: "FAIL",
    Severity.SKIP: "SKIP",
}


def render_console(report: DatasetReport, console=None, max_findings: int = 12) -> None:
    from rich.console import Console
    from rich.table import Table

    console = console or Console()
    c = report.counts()
    console.print()
    console.print(f"[bold]{report.source}[/bold]  health score [bold]{report.score}/100[/bold]")
    console.print(
        f"format {report.format_version} | {report.episodes} episodes | "
        f"{report.frames} frames | fps {report.fps or '?'}"
        + (" | [dim]loaded subset (--max-episodes)[/dim]" if report.truncated else "")
    )
    console.print()

    table = Table(show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("verdict")
    table.add_column("summary", overflow="fold")
    for r in report.results:
        v = r.verdict
        label = f"[{_STYLE[v]}]{_MARK[v]}[/{_STYLE[v]}]"
        summary = r.skip_reason if r.skipped else (r.summary or "")
        table.add_row(r.check_id, label, summary)
    console.print(table)

    shown = 0
    for r in report.results:
        if r.skipped or not r.findings:
            continue
        console.print(f"\n[bold]{r.check_id}[/bold] - {r.title}")
        for f in r.findings:
            if shown >= max_findings:
                remaining = sum(
                    len(x.findings) for x in report.results if not x.skipped
                ) - shown
                console.print(f"  [dim]... {remaining} more findings (use --json)[/dim]")
                break
            console.print(f"  [{_STYLE[f.severity]}]{_MARK[f.severity]}[/{_STYLE[f.severity]}] {f.message}")
            shown += 1

    console.print(
        f"\n[bold]verdict: [{_STYLE[report.verdict]}]{_MARK[report.verdict]}[/{_STYLE[report.verdict]}][/bold] "
        f"({c['fail']} fail, {c['warn']} warn, {c['info']} info)"
    )


def render_markdown(report: DatasetReport) -> str:
    c = report.counts()
    lines = [
        f"### Dataset QA: `{report.source}`",
        "",
        f"**Health score: {report.score}/100** - "
        f"{c['fail']} fail, {c['warn']} warn, {c['info']} info",
        "",
        f"format `{report.format_version}` | {report.episodes} episodes | "
        f"{report.frames} frames | fps {report.fps or '?'}",
        "",
        "| check | verdict | summary |",
        "|---|---|---|",
    ]
    for r in report.results:
        v = _MARK[r.verdict]
        summary = (r.skip_reason if r.skipped else r.summary) or ""
        lines.append(f"| `{r.check_id}` | {v} | {summary} |")
    lines.append("")
    for r in report.results:
        if r.skipped or not r.findings:
            continue
        lines += ["", f"**{r.check_id}** - {r.title}", ""]
        for f in r.findings[:20]:
            lines.append(f"- `{_MARK[f.severity]}` {f.message}")
        if len(r.findings) > 20:
            lines.append(f"- ... {len(r.findings) - 20} more")
    lines.append("")
    return "\n".join(lines)


def report_to_dict(report: DatasetReport) -> dict[str, Any]:
    return {
        "source": report.source,
        "format_version": report.format_version,
        "episodes": report.episodes,
        "frames": report.frames,
        "fps": report.fps,
        "truncated": report.truncated,
        "score": report.score,
        "verdict": _MARK[report.verdict],
        "checks": [
            {
                "id": r.check_id,
                "title": r.title,
                "verdict": _MARK[r.verdict],
                "summary": r.summary,
                "skipped": r.skipped,
                "skip_reason": r.skip_reason,
                "details": r.details,
                "findings": [
                    {
                        "severity": _MARK[f.severity],
                        "message": f.message,
                        "episode_index": f.episode_index,
                        "details": f.details,
                    }
                    for f in r.findings
                ],
            }
            for r in report.results
        ],
    }


def render_json(report: DatasetReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, default=str)


def report_from_dict(d: dict[str, Any]) -> DatasetReport:
    from kinema.checks.base import Finding

    rep = DatasetReport(
        source=d["source"],
        format_version=d.get("format_version", "?"),
        episodes=d.get("episodes", 0),
        frames=d.get("frames", 0),
        fps=d.get("fps"),
        truncated=d.get("truncated", False),
    )
    inv = {v: k for k, v in _MARK.items()}
    for cdict in d.get("checks", []):
        r = CheckResult(
            check_id=cdict["id"],
            title=cdict.get("title", ""),
            summary=cdict.get("summary", ""),
            skipped=cdict.get("skipped", False),
            skip_reason=cdict.get("skip_reason", ""),
            details=cdict.get("details", {}),
        )
        for fd in cdict.get("findings", []):
            r.findings.append(
                Finding(
                    check_id=cdict["id"],
                    severity=inv.get(fd.get("severity", "INFO"), Severity.INFO),
                    message=fd.get("message", ""),
                    episode_index=fd.get("episode_index"),
                    details=fd.get("details", {}),
                )
            )
        rep.results.append(r)
    return rep

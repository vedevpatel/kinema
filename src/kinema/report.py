"""Report model and renderers: rich console, markdown, JSON."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from kinema.checks.base import CheckResult, Severity

# Health score, model "tiered-v1" (docs/checks.md, "Health score"): start at
# 100, pay per failing/warning CHECK - never per finding, so correlated
# findings and verbose checks do not double-bill one defect. Floor at 0.
# Stored results without a "score_model" field were scored by legacy-v0
# (hand-tuned weights); the score is derived from stored per-check verdicts,
# so any stored result can be rescored without re-running checks.
SCORE_MODEL = "tiered-v1"

# A FAIL is a "data is not what it claims" blocker: its cost is trust, not
# measured training damage, so it is not harm-scaled.
_FAIL_COST = 25

# WARN tiers, anchored on the harm study v0 (scripts/harm_study.py: ridge BC,
# state-only, held-out action MSE delta vs clean). Band edges are provisional
# - three data points define them - and get re-anchored once when the full
# harm table lands. (tier, label, lower edge in % MSE delta, weight),
# highest band first.
WARN_TIERS: tuple[tuple[str, str, float, int], ...] = (
    ("A", "training killer", 100.0, 10),
    ("B", "skews learning", 10.0, 6),
    ("C", "review list", 0.0, 2),
)

# measured harm per check: held-out MSE delta (%) of the defect it targets
_MEASURED_HARM_PCT = {
    "idle-segments": 881.0,  # 30 idle frames prepended
    "calibration-drift": 347.0,  # action scale 0.5x in 25% of episodes
    "duplicate-and-leakage": 0.1,  # 20% duplicate episodes
}

# hand-tuned until the harm study measures them: wrong normalization or
# broken sampling corrupts every batch; behaviour issues skew what the
# policy learns; outlier lists are "review these episodes"
_PLACEHOLDER_COST = {
    "stats-drift": 10,
    "nan-and-normalization": 10,
    "deployment-readiness": 10,
    "timestamp-grid": 10,
    "dropped-frames": 8,
    "stuck-actuator": 6,
    "action-saturation": 6,
    "visual-quality": 6,
    "outlier-episodes": 3,
}
_WARN_COST_DEFAULT = 8  # any check without an entry (task-strings, reward-done, ...)


def harm_tier(mse_delta_pct: float) -> tuple[str, str, int]:
    """(tier, label, weight) for a measured held-out MSE delta in percent."""
    for tier, label, lower, weight in WARN_TIERS:
        if mse_delta_pct >= lower:
            return tier, label, weight
    return WARN_TIERS[-1][0], WARN_TIERS[-1][1], WARN_TIERS[-1][3]  # harmless/negative


_WARN_COST = {
    **_PLACEHOLDER_COST,
    **{cid: harm_tier(pct)[2] for cid, pct in _MEASURED_HARM_PCT.items()},
}


def warn_weight(check_id: str) -> tuple[int, str]:
    """(weight, basis) a WARN on this check costs; basis is measured/placeholder."""
    basis = "measured" if check_id in _MEASURED_HARM_PCT else "placeholder"
    return _WARN_COST.get(check_id, _WARN_COST_DEFAULT), basis


def score_from_verdicts(verdicts: Iterable[tuple[str, str]]) -> int:
    """Health score from (check_id, verdict) pairs - verdict as stored in report
    JSON ("PASS", "INFO", "WARN", "FAIL", "SKIP"). One entry per check."""
    score = 100
    for check_id, verdict in verdicts:
        if verdict == "FAIL":
            score -= _FAIL_COST
        elif verdict == "WARN":
            score -= warn_weight(check_id)[0]
    return max(0, score)


def rescore_report_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Re-derive score, score_model and per-check weights of a stored report
    dict in place from its stored per-check verdicts; no check re-runs."""
    d["score"] = score_from_verdicts((c["id"], c["verdict"]) for c in d.get("checks", []))
    d["score_model"] = SCORE_MODEL
    for c in d.get("checks", []):
        c["weight"], c["weight_basis"] = warn_weight(c["id"])
    return d


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
        return score_from_verdicts((r.check_id, _MARK[r.verdict]) for r in self.results)

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
    console.print(f"[dim]score model {SCORE_MODEL}[/dim]")


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
    lines += ["", f"<sub>score model `{SCORE_MODEL}`</sub>", ""]
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
        "score_model": SCORE_MODEL,
        "verdict": _MARK[report.verdict],
        "checks": [
            {
                "id": r.check_id,
                "title": r.title,
                "verdict": _MARK[r.verdict],
                "weight": warn_weight(r.check_id)[0],
                "weight_basis": warn_weight(r.check_id)[1],
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

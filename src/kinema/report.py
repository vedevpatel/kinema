"""Report model and renderers: rich console, markdown, JSON."""

from __future__ import annotations

import json
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from kinema.checks.base import CheckResult, Severity

# Health score, model "tiered-v2" (docs/checks.md, "Health score"): start
# at 100, pay per failing/warning CHECK - never per finding, so correlated
# findings and verbose checks do not double-bill one defect. Floor at 0.
# tiered-v2 = tiered-v1.1 (tiers, FAIL cost, correlated-check waiver below)
# with WARN weights re-anchored on harm table v1.1. Stored results without a
# "score_model" field were scored by legacy-v0 (hand-tuned weights); the
# score is derived from stored per-check verdicts and stored findings, so
# any stored result can be rescored without re-running checks.
SCORE_MODEL = "tiered-v2"

# Correlated-check waiver edges, (upstream, downstream). When the upstream
# check is charged (WARN or FAIL) and the downstream check is WARN, the
# downstream WARN weight is waived iff both checks expose non-empty flagged
# episode sets and their Jaccard overlap is >= WAIVER_MIN_OVERLAP. FAILs are
# never waived; missing episode evidence on either side never waives.
WAIVER_EDGES: tuple[tuple[str, str], ...] = (
    # mechanism: an episode that never moves is one idle run end to end AND a
    # "no joint moves at all" dead recording - one dead-episode defect trips
    # both checks. Idle padding on an episode whose arm then works, or a dead
    # joint elsewhere, are separate defects; the episode gate tells them apart.
    ("idle-segments", "stuck-actuator"),
)
WAIVER_MIN_OVERLAP = Fraction(1, 2)  # inclusive

# A FAIL is a "data is not what it claims" blocker: its cost is trust, not
# measured training damage, so it is not harm-scaled.
_FAIL_COST = 25

# WARN tiers, anchored on the harm table v1.1 (scripts/harm_table.py:
# diffusion-policy pusht, 20k steps, 3 seeds, held-out action MSE delta vs
# clean). Band edges were re-anchored once for tiered-v2 and stay 100/10/0.
# (tier, label, lower edge in % MSE delta, weight), highest band first.
WARN_TIERS: tuple[tuple[str, str, float, int], ...] = (
    ("A", "training killer", 100.0, 10),
    ("B", "skews learning", 10.0, 6),
    ("C", "review list", 0.0, 2),
)

# measured harm per check: held-out MSE delta (%) of the defect it targets,
# harm table v1.1 (diffusion pusht, 20k steps, 3 seeds, 50 eval episodes;
# docs/harm-table.json)
_MEASURED_HARM_PCT = {
    "calibration-drift": 1475.3,  # action scale 0.5x in 25% of episodes
    "stuck-actuator": 697.8,  # one joint frozen in 30% of episodes
    "idle-segments": 75.1,  # 30 idle frames prepended
    "duplicate-and-leakage": -3.6,  # 20% duplicate episodes
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


def flagged_episodes(result: CheckResult) -> set[int]:
    """Episodes a check's findings point at: each Finding.episode_index plus
    the int entries of Finding.details["episodes"]. The one reader of a
    check's episode evidence (the waiver gate); a check that exposes neither
    yields an empty set and so can never be waived. Lives here, not on
    CheckResult in checks/base.py, because checks/*.py feed the engine hash
    (kinema.engine) - a scoring helper there would mark every stored result
    stale and force re-audits for a billing-only change."""
    out: set[int] = set()
    for f in result.findings:
        if f.episode_index is not None:
            out.add(int(f.episode_index))
        eps = f.details.get("episodes") if isinstance(f.details, dict) else None
        if isinstance(eps, (list, tuple, set)):
            out.update(int(e) for e in eps
                       if isinstance(e, numbers.Integral) and not isinstance(e, bool))
    return out


def check_charges(
    verdicts: Iterable[tuple[str, str]],
    episodes: Mapping[str, set[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per check id: {"charge": int} - the WARN weight after any waiver
    (a waived downstream pays 0 and the upstream carries the LARGER of the
    two weights, so a re-anchor can never make the pair pay less than the
    heavier defect alone) - plus "waived_by" and
    "waiver_evidence" ({"shared_episodes", "overlap"}) when waived.

    `episodes` maps check id -> flagged episode set
    (flagged_episodes(result)); without it no waiver can fire (every WARN pays
    its full weight). A pure function of its inputs: input order never matters."""
    verdict = dict(verdicts)
    eps = episodes or {}
    out: dict[str, dict[str, Any]] = {cid: {"charge": warn_weight(cid)[0]} for cid in verdict}
    for up, down in WAIVER_EDGES:
        if verdict.get(up) not in ("WARN", "FAIL") or verdict.get(down) != "WARN":
            continue
        a, b = eps.get(up) or set(), eps.get(down) or set()
        if not a or not b:
            continue  # no episode evidence on one side: never waive
        shared, union = len(a & b), len(a | b)
        if Fraction(shared, union) >= WAIVER_MIN_OVERLAP:
            # the pair pays the heavier weight once, on the upstream check:
            # waiving must never make one defect cheaper than either reading
            out[up]["charge"] = max(out[up]["charge"], out[down]["charge"])
            out[down] = {
                "charge": 0,
                "waived_by": up,
                "waiver_evidence": {"shared_episodes": shared,
                                    "overlap": round(shared / union, 2)},
            }
    return out


def score_from_verdicts(
    verdicts: Iterable[tuple[str, str]],
    episodes: Mapping[str, set[int]] | None = None,
) -> int:
    """Health score from (check_id, verdict) pairs - verdict as stored in report
    JSON ("PASS", "INFO", "WARN", "FAIL", "SKIP"). One entry per check.
    `episodes` (check id -> flagged episode set) enables the correlated-check
    waiver; verdicts alone pay every WARN its full weight."""
    pairs = list(verdicts)
    charges = check_charges(pairs, episodes)
    score = 100
    for check_id, verdict in pairs:
        if verdict == "FAIL":
            score -= _FAIL_COST
        elif verdict == "WARN":
            score -= charges[check_id]["charge"]
    return max(0, score)


def _result_from_dict(cdict: dict[str, Any]) -> CheckResult:
    from kinema.checks.base import Finding

    inv = {v: k for k, v in _MARK.items()}
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
    return r


def _apply_charge_fields(c: dict[str, Any], charge: dict[str, Any]) -> None:
    c["charge"] = charge["charge"]
    for key in ("waived_by", "waiver_evidence"):
        if key in charge:
            c[key] = charge[key]
        else:
            c.pop(key, None)


def rescore_report_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Re-derive score, score_model, per-check weights and charges of a
    stored report dict in place from its stored per-check verdicts and
    stored findings (the waiver's episode evidence); no check re-runs.
    Idempotent."""
    checks = d.get("checks", [])
    verdicts = [(c["id"], c["verdict"]) for c in checks]
    episodes = {c["id"]: flagged_episodes(_result_from_dict(c)) for c in checks}
    d["score"] = score_from_verdicts(verdicts, episodes)
    d["score_model"] = SCORE_MODEL
    charges = check_charges(verdicts, episodes)
    for c in checks:
        c["weight"], c["weight_basis"] = warn_weight(c["id"])
        _apply_charge_fields(c, charges[c["id"]])
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
    # video-audit generation marker (see leaderboard.VIDEO_AUDIT_MARK):
    # 0 = pre-video board, 1 = v1 targeting bug (audited nothing), 2+ =
    # current. Rows below the current mark re-audit once, then settle.
    video_audited: int = 0

    @property
    def verdict(self) -> Severity:
        vs = [r.verdict for r in self.results if r.verdict != Severity.SKIP]
        return max(vs, default=Severity.OK)

    def _verdicts_and_episodes(self) -> tuple[list[tuple[str, str]], dict[str, set[int]]]:
        verdicts = [(r.check_id, _MARK[r.verdict]) for r in self.results]
        episodes = {r.check_id: flagged_episodes(r) for r in self.results}
        return verdicts, episodes

    def charges(self) -> dict[str, dict[str, Any]]:
        """Per check id: charge after the correlated-check waiver (see
        check_charges), read from the in-memory findings."""
        return check_charges(*self._verdicts_and_episodes())

    @property
    def score(self) -> int:
        return score_from_verdicts(*self._verdicts_and_episodes())

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
        if shown >= max_findings:
            break
        if r.skipped or not r.findings:
            continue
        console.print(f"\n[bold]{r.check_id}[/bold] - {r.title}")
        for f in r.findings:
            if shown >= max_findings:
                break
            console.print(f"  [{_STYLE[f.severity]}]{_MARK[f.severity]}[/{_STYLE[f.severity]}] {f.message}")
            shown += 1

    remaining = sum(len(x.findings) for x in report.results if not x.skipped) - shown
    if remaining > 0:
        console.print(f"  [dim]... {remaining} more findings (use --json)[/dim]")
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
    charges = report.charges()
    d: dict[str, Any] = {
        "source": report.source,
        "format_version": report.format_version,
        "episodes": report.episodes,
        "frames": report.frames,
        "fps": report.fps,
        "truncated": report.truncated,
        "video_audited": report.video_audited,
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
    for c in d["checks"]:
        _apply_charge_fields(c, charges[c["id"]])
    return d


def render_json(report: DatasetReport) -> str:
    return json.dumps(report_to_dict(report), indent=2, default=str)


def report_from_dict(d: dict[str, Any]) -> DatasetReport:
    rep = DatasetReport(
        source=d["source"],
        format_version=d.get("format_version", "?"),
        episodes=d.get("episodes", 0),
        frames=d.get("frames", 0),
        fps=d.get("fps"),
        truncated=d.get("truncated", False),
        video_audited=int(d.get("video_audited") or 0),
    )
    for cdict in d.get("checks", []):
        rep.results.append(_result_from_dict(cdict))
    return rep

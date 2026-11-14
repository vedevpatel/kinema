"""The diagnosis layer behind the MCP server: which data problem explains
this failed fine-tune.

This module never re-implements check logic. It runs the same checks the
CLI runs, reads the same CheckResult objects, and maps findings to causes
declared in src/kinema/causes.json (data, not code - same pattern as
policies.json). Adding a failure-mode cluster or a policy is a JSON edit.

Two honesty rules the answers follow:

- SKIP is not a verdict (docs/checks.md says so). The answer always
  enumerates what ran, what skipped and why, whether the load was sampled,
  and which dims sat out - never "your data is clean".
- A finding that matches no cause entry is surfaced verbatim under
  findings_outside_coverage, never dropped.
"""

from __future__ import annotations

import json
import threading
from importlib import resources

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.checks.deployment import (
    POLICY_DEFAULTS,
    resolve_policy,
)
from kinema.dataset import load_dataset
from kinema.report import DatasetReport

DOCS_BASE = "https://github.com/vedevpatel/kinema/blob/main/docs/checks.md"


def _load_causes() -> dict:
    with resources.files("kinema").joinpath("causes.json").open() as f:
        return json.load(f)


CAUSES = _load_causes()


def known_policies() -> list[str]:
    return sorted(POLICY_DEFAULTS)


def checks_for_symptom(symptom: str) -> tuple[list[str], list[str]]:
    """The check ids a symptom routes to, and the clusters matched.

    Keyword routing only picks a check subset - the client LLM does the
    real language understanding. A symptom matching no cluster runs all
    checks, so an unrouted symptom never hides a finding."""
    text = (symptom or "").lower()
    matched = [
        name
        for name, cluster in CAUSES["clusters"].items()
        if any(kw in text for kw in cluster["keywords"])
    ]
    if not matched:
        return [c.id for c in ALL_CHECKS], []
    ids: list[str] = []
    for name in matched:
        ids.extend(CAUSES["clusters"][name]["checks"])
    return list(dict.fromkeys(ids)), matched


def _evidence(f) -> dict:
    out = {
        "check_id": f.check_id,
        "severity": f.severity.name,
        "message": f.message,
    }
    if f.episode_index is not None:
        out["episode_index"] = f.episode_index
    if f.details:
        out["details"] = f.details
    return out


def _match_causes(results, policy_name: str) -> tuple[list[dict], list[dict]]:
    """Findings matched to cause entries, and WARN/FAIL findings no entry
    claims (surfaced verbatim, never dropped)."""
    causes: list[dict] = []
    claimed: set[int] = set()
    findings = [
        (r, f)
        for r in results
        for f in r.findings
        if f.severity >= Severity.WARN and not r.skipped
    ]
    for order, cause in enumerate(CAUSES["causes"]):
        ind = cause["indicates"]
        floor = Severity[ind["min_severity"]]
        hits = []
        for r, f in findings:
            if r.check_id != ind["check_id"] or f.severity < floor:
                continue
            needle = ind.get("message_contains")
            if needle and needle not in f.message:
                continue
            hits.append((r, f))
        if not hits:
            continue
        for r, f in hits:
            claimed.add(id(f))
        worst = max(f.severity for _, f in hits)
        causes.append(
            {
                "cause": cause["title"],
                "cause_id": cause["id"],
                "cluster": cause["cluster"],
                "confidence": cause["confidence"].get(worst.name, "low"),
                "evidence": [_evidence(f) for _, f in hits],
                "explanation": cause["explanation"].format(policy=policy_name or "the policy"),
                "docs": f"{DOCS_BASE}#{cause['docs_section']}",
                "next_step": cause["next_step"],
                "_rank": (worst, -order),  # severity first, then map order
            }
        )
    causes.sort(key=lambda c: c["_rank"], reverse=True)
    for c in causes:
        del c["_rank"]
    unmapped = [_evidence(f) for _, f in findings if id(f) not in claimed]
    return causes, unmapped


def _coverage(report: DatasetReport, ctx, unmapped: list[dict]) -> dict:
    """The negative-answer block: exactly what was and was not examined."""
    skipped = [
        {"check_id": r.check_id, "reason": r.skip_reason}
        for r in report.results
        if r.skipped
    ]
    dims_sat_out: dict[str, str] = {}
    for r in report.results:
        excl = (r.details or {}).get("excluded_joints")
        if excl:
            dims_sat_out.update(excl)
    notes: list[str] = []
    if ctx.truncated:
        notes.append(
            f"sampled load: {ctx.total_episodes} episodes examined "
            "(--max-episodes); a sampled load without per-episode stats can "
            "only prove gross corruption"
        )
    if ctx.stats_source != "meta/stats.json":
        notes.append(f"stats compared against {ctx.stats_source}, not meta/stats.json")
    out = {
        "checks_run": [
            {"check_id": r.check_id, "verdict": r.verdict.name, "summary": r.summary}
            for r in report.results
            if not r.skipped
        ],
        "checks_skipped": skipped,
        "sampled": ctx.truncated,
        "dims_sat_out": dims_sat_out,
        "notes": notes,
        "findings_outside_coverage": unmapped,
        "blind_spots": CAUSES["blind_spots"],
    }
    return out


def run_check_subset(
    ctx,
    cfg: CheckConfig,
    check_ids: list[str],
    source: str,
    cancelled: threading.Event | None = None,
    progress=None,
) -> DatasetReport:
    """Run a subset of ALL_CHECKS against an already-loaded context.

    Cancellation is cooperative between checks (mid-check interruption is
    a documented limit); progress, when given, is called with
    (completed, total, check_id)."""
    report = DatasetReport(
        source=source,
        format_version=ctx.format_version,
        episodes=ctx.total_episodes,
        frames=ctx.row_count,
        fps=ctx.fps,
        truncated=ctx.truncated,
    )
    wanted = [c for c in ALL_CHECKS if c.id in set(check_ids)]
    for i, check in enumerate(wanted):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError(f"cancelled before check {check.id}")
        try:
            report.results.append(check.run(ctx, cfg))
        except CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - one broken check must not sink the audit
            report.results.append(check.skip(f"check crashed: {type(e).__name__}: {e}"))
        if progress is not None:
            progress(i + 1, len(wanted), check.id)
    return report


class CancelledError(Exception):
    """The client sent notifications/cancelled for this request."""


def diagnose(
    dataset: str,
    policy: str,
    symptom: str,
    policy_overrides: dict | None = None,
    max_episodes: int | None = None,
    video: bool = False,
    cancelled: threading.Event | None = None,
    progress=None,
) -> dict:
    """Rank the data causes that explain a failed fine-tune.

    Raises DatasetError on an unloadable dataset, ValueError on an unknown
    policy or bad override keys, CancelledError on client cancellation -
    the MCP layer turns all three into isError results."""
    canon, entry = resolve_policy(policy, policy_overrides)
    if entry is None:
        raise ValueError(
            f"unknown policy {policy!r} (known: {', '.join(known_policies())}; "
            "aliases like pi0.5 resolve to their family entry)"
        )
    check_ids, clusters = checks_for_symptom(symptom)
    cfg = CheckConfig(
        video=video, policy=canon, policy_overrides=dict(policy_overrides or {})
    )
    ctx = load_dataset(dataset, max_episodes=max_episodes, video=video)
    if progress is not None:
        progress(0, len(check_ids), "dataset loaded")
    report = run_check_subset(ctx, cfg, check_ids, dataset, cancelled, progress)
    causes, unmapped = _match_causes(report.results, canon)
    return {
        "dataset": dataset,
        "policy": canon,
        "policy_assumptions": entry,
        "symptom": symptom,
        "clusters_matched": clusters,
        "causes": causes,
        "coverage": _coverage(report, ctx, unmapped),
    }


def run_checks(
    dataset: str,
    checks: list[str] | None = None,
    policy: str | None = None,
    policy_overrides: dict | None = None,
    max_episodes: int | None = None,
    video: bool = False,
    cancelled: threading.Event | None = None,
    progress=None,
):
    """Raw engine access for the run_checks tool: (report, context).

    Unknown check ids and unknown policies fail loudly here too."""
    known = {c.id for c in ALL_CHECKS}
    bad = sorted(set(checks or []) - known)
    if bad:
        raise ValueError(f"unknown checks: {', '.join(bad)} (known: {', '.join(sorted(known))})")
    if policy is not None:
        canon, entry = resolve_policy(policy, policy_overrides)
        if entry is None:
            raise ValueError(
                f"unknown policy {policy!r} (known: {', '.join(known_policies())})"
            )
        policy = canon
    cfg = CheckConfig(
        video=video, policy=policy, policy_overrides=dict(policy_overrides or {})
    )
    ctx = load_dataset(dataset, max_episodes=max_episodes, video=video)
    ids = checks or [c.id for c in ALL_CHECKS]
    report = run_check_subset(ctx, cfg, ids, dataset, cancelled, progress)
    return report, ctx


def summarize_report(report: DatasetReport, top: int = 25) -> dict:
    """The capped run_checks payload: sized for MCP client output limits
    (Claude Code warns at 10k output tokens). FAIL/WARN findings first."""
    counts = report.counts()
    findings = []
    for r in report.results:
        if r.skipped:
            continue
        for f in r.findings:
            if f.severity >= Severity.WARN:
                findings.append((r.check_id, f))
    findings.sort(key=lambda rf: rf[1].severity, reverse=True)
    top_findings = [
        {"check_id": cid, **_evidence(f)} for cid, f in findings[:top]
    ]
    return {
        "source": report.source,
        "format_version": report.format_version,
        "episodes": report.episodes,
        "frames": report.frames,
        "truncated": report.truncated,
        "score": report.score,
        "verdict": report.verdict.name,
        "counts": counts,
        "checks": [
            {
                "id": r.check_id,
                "verdict": r.verdict.name,
                "summary": r.skip_reason if r.skipped else r.summary,
                "skipped": r.skipped,
                "finding_count": len(r.findings),
            }
            for r in report.results
        ],
        "top_findings": top_findings,
        "findings_omitted": max(0, len(findings) - top),
    }


def list_checks() -> list[dict]:
    """The check catalog, built from ALL_CHECKS - not from docs/checks.md,
    which is not shipped in the package."""
    out = []
    for c in ALL_CHECKS:
        doc = (type(c).__doc__ or "").strip().splitlines()
        out.append(
            {
                "id": c.id,
                "title": c.title,
                "description": doc[0] if doc else c.title,
                "requires_video": c.requires_video,
                "docs": f"{DOCS_BASE}#{c.id}",
            }
        )
    return out

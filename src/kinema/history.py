"""Board history: one compact snapshot per UTC day, and honest comparisons
between snapshots (docs/checks.md, "Change tracking").

A finding that disappears is not a fix until the cause is known. Between
two audits of a dataset, any of these can move a verdict:

- the dataset (its Hub revision, or an RLDS shard-inventory digest),
- the checker (the engine hash),
- the sample (which episodes and video files the audit read),
- coverage (a check ran in one audit and was skipped in the other),
- or the audit itself failed.

compare() attributes each verdict change to whatever differs between the
two audits, and words it by that cause. When nothing differs and a verdict
still moved, it says so ("unexplained"): the checker should be
deterministic, so that label is a bug report, not a dataset fact.

Snapshots store no finding messages - per check a verdict letter, the
WARN/FAIL count and a digest of the gated findings' identities - so a day
of the board is a few hundred bytes per row.
"""

from __future__ import annotations

import hashlib
import json

# every label compare_rows/compare can emit (docs/checks.md, "Change tracking")
LABELS = ("new-row", "row-removed", "audit-failed", "now-gated", "still-not-audited",
          "audit-recovered", "not-re-audited", "stable", "dataset-changed", "checker-changed",
          "sample-changed", "coverage-changed", "revision-unknown", "ambiguous", "unexplained")

_LETTER = {"PASS": "P", "OK": "P", "INFO": "I", "WARN": "W", "FAIL": "F", "SKIP": "S"}
_RANK = {"P": 0, "I": 1, "W": 2, "F": 3}
_NAME = {"P": "PASS", "I": "INFO", "W": "WARN", "F": "FAIL", "S": "SKIP", "-": "absent"}


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _finding_key(f: dict) -> list:
    d = f.get("details") or {}
    eps = d.get("episodes")
    return [f.get("severity"), d.get("kind"), d.get("feature"), d.get("dim"), d.get("stat"),
            f.get("episode_index"), sorted(eps)[:50] if isinstance(eps, list) else None]


def row_snapshot(entry: dict, checks: list[str]) -> dict:
    rep = entry.get("report") or {}
    by_id = {c.get("id"): c for c in rep.get("checks", [])}
    letters, counts, digests = [], {}, {}
    for cid in checks:
        c = by_id.get(cid)
        if c is None:
            letters.append("-")
            continue
        letters.append(_LETTER.get(c.get("verdict"), "?"))
        gated = [f for f in c.get("findings", []) if f.get("severity") in ("WARN", "FAIL")]
        if gated:
            counts[cid] = len(gated)
            digests[cid] = _digest(sorted(_finding_key(f) for f in gated))
    row = {
        "id": entry["id"],
        "status": "audited" if rep else entry.get("status", "missing"),
        "revision": entry.get("revision"),
        "engine": entry.get("engine"),
        "audited_at": entry.get("audited_at"),
        "sample": entry.get("sample"),
        "v": "".join(letters) if rep else None,
    }
    if rep:
        row["score"] = rep.get("score")
        row["score_model"] = rep.get("score_model")
    if counts:
        row["n"] = counts
        row["fd"] = digests
    return row


def snapshot(entries: list[dict], checks: list[str], date: str, generated: str) -> dict:
    return {"schema": "kinema.board-history/v1", "date": date, "generated": generated,
            "checks": checks, "rows": [row_snapshot(e, checks) for e in entries]}


def _sample_key(s):
    if not s:
        return None
    return (s.get("episodes_digest"), s.get("videos_digest"))


# how a verdict change reads, by its cause - never "fixed"
_GONE = {
    "dataset-changed": "not flagged on the new revision",
    "checker-changed": "no longer flagged by the updated checker",
    "sample-changed": "not flagged in this sample",
    "coverage-changed": "not checked this run",
    "revision-unknown": "no longer flagged; the dataset revision is unknown, so a dataset change cannot be ruled out",
    "ambiguous": "no longer flagged; more than one thing changed, so the cause is ambiguous",
    "unexplained": "no longer flagged with nothing changed - a determinism bug to investigate",
}
_NEW = {
    "dataset-changed": "newly flagged on the new revision",
    "checker-changed": "newly flagged by the updated checker",
    "sample-changed": "flagged in this sample",
    "coverage-changed": "flagged now that the check ran",
    "revision-unknown": "newly flagged; the dataset revision is unknown, so a dataset change cannot be ruled out",
    "ambiguous": "newly flagged; more than one thing changed, so the cause is ambiguous",
    "unexplained": "newly flagged with nothing changed - a determinism bug to investigate",
}


def compare_rows(prev: dict | None, cur: dict, checks_prev: list[str], checks_cur: list[str]) -> dict:
    """One row's change between two snapshots: a label, the causes found,
    and per-check deltas worded by cause."""
    out = {"id": cur["id"]}
    if prev is None:
        return {**out, "label": "new-row"}
    if cur["status"] != "audited":
        if prev["status"] == "audited":
            return {**out, "label": "audit-failed" if cur["status"] != "gated" else "now-gated",
                    "note": "no audit this time: earlier findings are unknown now, not resolved"}
        return {**out, "label": "still-not-audited"}
    if prev["status"] != "audited":
        return {**out, "label": "audit-recovered"}
    if cur.get("audited_at") and cur.get("audited_at") == prev.get("audited_at"):
        return {**out, "label": "not-re-audited"}

    causes = []
    rp, rc = prev.get("revision"), cur.get("revision")
    if rp and rc and rp != rc:
        causes.append("dataset-changed")
    if prev.get("engine") != cur.get("engine"):
        causes.append("checker-changed")
    if _sample_key(prev.get("sample")) != _sample_key(cur.get("sample")):
        causes.append("sample-changed")
    vp = dict(zip(checks_prev, prev.get("v") or ""))
    vc = dict(zip(checks_cur, cur.get("v") or ""))
    skipped_p = {c for c, x in vp.items() if x in "S-"}
    skipped_c = {c for c, x in vc.items() if x in "S-"}
    if skipped_p != skipped_c:
        causes.append("coverage-changed")
    unknown_rev = not (rp and rc)

    deltas = []
    for cid in sorted(set(vp) | set(vc)):
        a, b = vp.get(cid, "-"), vc.get(cid, "-")
        fa, fb = (prev.get("fd") or {}).get(cid), (cur.get("fd") or {}).get(cid)
        if a == b and fa == fb:
            continue
        deltas.append({"check": cid, "from": _NAME.get(a, a), "to": _NAME.get(b, b),
                       "_a": a, "_b": b, "_same_verdict": a == b})
    if not deltas:
        return {**out, "label": "stable", "changed": causes or None,
                **({"note": "revision unknown on one side"} if unknown_rev else {})}

    if not causes:
        label = "revision-unknown" if unknown_rev else "unexplained"
    elif len(causes) == 1 and not (unknown_rev and causes[0] != "dataset-changed"):
        label = causes[0]
    else:
        label = "ambiguous"
    for d in deltas:
        a, b = d.pop("_a"), d.pop("_b")
        same = d.pop("_same_verdict")
        if b in "S-" and a not in "S-":
            d["reads"] = "not checked this run"
        elif same:
            d["reads"] = "same verdict, different findings"
        elif _RANK.get(a, -1) >= 2 and _RANK.get(b, -1) < 2:
            d["reads"] = _GONE[label]
        elif _RANK.get(b, -1) >= 2 and _RANK.get(a, -1) < 2:
            d["reads"] = _NEW[label]
        else:
            d["reads"] = f"{_NAME.get(a, a)} -> {_NAME.get(b, b)}"
    shown = causes + (["revision-unknown"] if unknown_rev else [])
    return {**out, "label": label, "changed": shown or None, "deltas": deltas}


def compare(prev: dict | None, cur: dict) -> dict:
    """All rows' changes since the previous snapshot, plus a label count."""
    if prev is None:
        return {"compared_to": None, "changes": [], "summary": {}}
    before = {r["id"]: r for r in prev["rows"]}
    changes = [compare_rows(before.get(r["id"]), r, prev["checks"], cur["checks"]) for r in cur["rows"]]
    gone = [{"id": i, "label": "row-removed"} for i in before if i not in {r["id"] for r in cur["rows"]}]
    changes += gone
    summary: dict[str, int] = {}
    for c in changes:
        summary[c["label"]] = summary.get(c["label"], 0) + 1
    # unchanged rows are counted, not listed
    listed = [c for c in changes if c["label"] not in ("stable", "not-re-audited", "still-not-audited")
              or c.get("changed")]
    return {"compared_to": prev["date"], "changes": listed, "summary": summary}

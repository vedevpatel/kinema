"""'Datasets like yours' retrieval over the kinema leaderboard corpus.

The public leaderboard's audits are a labelled corpus: ~200 real robot
datasets, each with a verdict per check, shape metadata and a score. Ranking
a query report against it answers 'has anyone shipped data like mine, and
what was wrong with it' without training anything - the corpus is the
model. The metric is deliberately simple (Jaccard distance over WARN check
sets, small penalties for shape mismatch) so every match can be explained
in one line.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from kinema.report import _MEASURED_HARM_PCT, warn_weight

BOARD_URL = "https://vedevpatel.github.io/kinema/data.json"
BOARD_PAGE = "https://vedevpatel.github.io/kinema/"
_CACHE_DIR = Path("~/.cache/kinema/board").expanduser()
_CACHE_MAX_AGE_S = 24 * 3600
_FETCH_TIMEOUT_S = 15

# verdicts that mark an evaluated problem; SKIP/INFO carry no defect signal
_PROBLEMS = {"WARN", "FAIL"}


def _warn_ids(report: dict[str, Any]) -> set[str]:
    return {c.get("id") for c in report.get("checks") or [] if c.get("verdict") in _PROBLEMS} - {None}


def _report_of(entry: dict[str, Any]) -> dict[str, Any]:
    """A board entry carries its audit under 'report'; a bare report is itself."""
    r = entry.get("report")
    return r if isinstance(r, dict) else entry


def distance(query: dict[str, Any], entry: dict[str, Any]) -> float:
    """How far a board entry is from the query report. 0 = identical failure
    profile and shape; dominated by which checks WARN on each side."""
    report = _report_of(entry)
    q_warn, e_warn = _warn_ids(query), _warn_ids(report)
    union = q_warn | e_warn
    jaccard = len(q_warn ^ e_warn) / len(union) if union else 0.0
    d = 2.0 * jaccard
    if query.get("format_version") and report.get("format_version"):
        d += 0.5 * (query["format_version"] != report["format_version"])
    q_fps, e_fps = query.get("fps"), report.get("fps")
    if q_fps and e_fps:
        d += 0.25 * min(abs(math.log2(q_fps / e_fps)), 2.0)
    if query.get("score") is not None and report.get("score") is not None:
        d += abs(query["score"] - report["score"]) / 200.0
    return d


def rank(query: dict[str, Any], entries: list[dict[str, Any]], k: int = 5,
         exclude_id: str | None = None) -> list[dict[str, Any]]:
    """The k closest board datasets, annotated so each match explains itself:
    which warnings are shared, and the measured MSE harm of those defects."""
    out: list[dict[str, Any]] = []
    exclude_id = exclude_id or query.get("source") or query.get("dataset")
    if exclude_id:
        exclude_id = exclude_id.removeprefix("hf://").split("@", 1)[0]
    for e in entries:
        if exclude_id and e.get("id") == exclude_id:
            continue
        report = _report_of(e)
        if not any(c.get("verdict") in {"PASS", "OK", "INFO", "WARN", "FAIL"}
                   for c in report.get("checks") or []):
            continue  # no evaluated evidence is not a perfect health match
        shared = sorted(_warn_ids(query) & _warn_ids(report))
        out.append({
            "id": e.get("id"),
            "score": report.get("score"),
            "verdict": report.get("verdict"),
            "downloads": e.get("downloads"),
            "distance": round(distance(query, e), 3),
            "shared_warnings": [
                {
                    "check": cid,
                    "basis": warn_weight(cid)[1],
                    **({"measured_mse_harm_pct": _MEASURED_HARM_PCT[cid]}
                       if cid in _MEASURED_HARM_PCT else {}),
                }
                for cid in shared
            ],
            "url": f"{BOARD_PAGE}dataset/{e['slug']}.html" if e.get("slug") else BOARD_PAGE,
        })
    out.sort(key=lambda m: (m["distance"], -(m["downloads"] or 0)))  # type: ignore[misc]
    return out[:k]


def load_query_report(path: str | Path) -> dict[str, Any]:
    """A kinema JSON report from disk, tolerating extra fields."""
    d = json.loads(Path(path).read_text())
    if not isinstance(d, dict) or not isinstance(d.get("checks"), list) or "verdict" not in d:
        raise ValueError(f"{path} is not a kinema JSON report (no verdict/checks)")
    return d


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as r:  # noqa: S310 - fixed https URL
        return r.read()


def fetch_board(url: str = BOARD_URL, cache_dir: Path = _CACHE_DIR,
                max_age_s: int = _CACHE_MAX_AGE_S,
                fetcher: Callable[[str], bytes] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The board's dataset entries plus provenance. A fresh fetch is cached
    for max_age_s; when the fetch fails, any cached copy (even stale) is
    served instead - the corpus a day old beats no corpus."""
    get = fetcher or _http_get
    cache = cache_dir / "kinema-board.json"
    try:
        raw = get(url)
        data = json.loads(raw)
        entries = data.get("datasets") or []
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(cache)
        return entries, {"url": url, "datasets": len(entries), "generated": data.get("generated"),
                         "cached": False}
    except Exception as e:  # noqa: BLE001 - any fetch/parse failure falls back to cache
        if cache.exists():
            try:
                data = json.loads(cache.read_text())
                entries = data.get("datasets") or []
                return entries, {"url": url, "datasets": len(entries),
                                 "generated": data.get("generated"), "cached": True,
                                 "cache_age_s": int(time.time() - cache.stat().st_mtime),
                                 "fetch_error": f"{type(e).__name__}: {e}"}
            except Exception:  # noqa: BLE001 - a corrupt cache is as good as none
                pass
        raise RuntimeError(
            f"could not fetch the kinema leaderboard corpus ({type(e).__name__}: {e}) "
            "and there is no cached copy") from e

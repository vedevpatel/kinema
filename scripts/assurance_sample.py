"""Freeze a deterministic, check-stratified sample of published findings.

Only the board JSON is the sampling frame. Dataset pages are never consulted.
Run: python3 scripts/assurance_sample.py docs/data.json audits/2026-11-22
"""

from __future__ import annotations

import hashlib
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEED = "kinema-assurance-phase1-2026-11-22-v1"
TARGET = 40
SEVERITIES = {"WARN", "FAIL"}


def digest(value: str) -> str:
    return hashlib.sha256((SEED + "\0" + value).encode()).hexdigest()


def frame(board: dict) -> list[dict]:
    entries = []
    for ri, row in enumerate(board["datasets"]):
        for ci, check in enumerate((row.get("report") or {}).get("checks", [])):
            for fi, finding in enumerate(check.get("findings", [])):
                if finding.get("severity") not in SEVERITIES:
                    continue
                entries.append({
                    "key": f"r{ri}:c{ci}:f{fi}", "dataset": row["id"],
                    "revision": row.get("revision"), "engine": row.get("engine"),
                    "check": check["id"], "severity": finding["severity"],
                    "message": finding["message"], "episode_index": finding.get("episode_index"),
                    "details": finding.get("details") or {}, "format_version": (row.get("report") or {}).get("format_version"),
                    "sampled_board_load": row.get("sampled"),
                })
    return entries


def select(entries: list[dict], target: int = TARGET) -> list[dict]:
    """Two per check, covering both severities where present; then fill FAILs.

    Hash order is fixed before examining evidence. Unused dataset IDs are
    preferred within each stratum to broaden coverage; ties use hash order.
    """
    by_check: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_check[e["check"]].append(e)
    chosen: list[dict] = []
    used: set[str] = set()

    def add(pool: list[dict]) -> bool:
        available = [e for e in pool if e["key"] not in {x["key"] for x in chosen}]
        if not available or len(chosen) >= target:
            return False
        e = min(available, key=lambda x: (x["dataset"] in used, digest(x["key"])))
        chosen.append(e)
        used.add(e["dataset"])
        return True

    for check in sorted(by_check):
        pool = by_check[check]
        severities = {e["severity"] for e in pool}
        if severities == SEVERITIES:
            add([e for e in pool if e["severity"] == "FAIL"])
            add([e for e in pool if e["severity"] == "WARN"])
        else:
            add(pool)
            add(pool)
    # Reserve the remaining slots for rarer FAIL findings. If exhausted, use
    # the global hash order over remaining findings.
    while len(chosen) < target and add([e for e in entries if e["severity"] == "FAIL"]):
        pass
    while len(chosen) < target and add(entries):
        pass
    return chosen


def make(board_path: Path, out: Path) -> dict:
    raw = gzip.decompress(board_path.read_bytes()) if board_path.suffix == ".gz" else board_path.read_bytes()
    board = json.loads(raw)
    entries = frame(board)
    selected = select(entries)
    all_findings = [f for row in board["datasets"] for check in (row.get("report") or {}).get("checks", [])
                    for f in check.get("findings", [])]
    excluded = [f for f in all_findings if f.get("severity") not in SEVERITIES]
    return {
        "schema": 1, "seed": SEED, "target": TARGET,
        "board_sha256": hashlib.sha256(raw).hexdigest(),
        "board_generated": board.get("generated"), "score_model": board.get("score_model"),
        "frame": {"board_rows": len(board["datasets"]), "all_findings": len(all_findings),
                  "eligible": len(entries), "excluded_findings": len(excluded),
                  "excluded_by_severity": dict(sorted(Counter(f.get("severity", "<missing>") for f in excluded).items())),
                  "eligible_datasets": len({e["dataset"] for e in entries}),
                  "checks": dict(sorted(Counter(e["check"] for e in entries).items())),
                  "severities": dict(sorted(Counter(e["severity"] for e in entries).items())),
                  "exclusions": "Only INFO, PASS and SKIP findings excluded; no WARN/FAIL rows excluded for absent dataset pages"},
        "selected": selected,
    }


def main() -> None:
    board_path, out = map(Path, sys.argv[1:3])
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out / "board.json.gz"
    source = board_path.read_bytes()
    if not snapshot.exists():
        with snapshot.open("wb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as gz:
            gz.write(source)
    elif gzip.decompress(snapshot.read_bytes()) != source:
        raise SystemExit("frozen board differs from input; refusing to replace it")
    doc = make(snapshot, out)
    sample = out / "sample.json"
    if sample.exists() and json.loads(sample.read_text()) != doc:
        raise SystemExit("sample differs from frozen artifact; refusing to replace it")
    sample.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"{len(doc['selected'])} selected from {doc['frame']['eligible']} findings")


if __name__ == "__main__":
    main()

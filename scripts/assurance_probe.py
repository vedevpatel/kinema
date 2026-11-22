"""Fetch pinned source metadata for the phase 1 audit, without detector code.

Run: .venv/bin/python scripts/assurance_probe.py audits/2026-11-22
The JSON is deliberately a provenance record, not an adjudication.
"""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def probe(item: tuple[str, str]) -> dict:
    dataset, revision = item
    if dataset.startswith("rlds/"):
        return {"dataset": dataset, "revision": revision, "status": "unsupported",
                "reason": "GCS board stamp is not an immutable object generation or source revision"}
    repo, _, subdir = dataset.partition("//")
    rec: dict = {"dataset": dataset, "repo": repo, "subdir": subdir,
                 "revision": revision, "status": "unresolved"}
    try:
        info = HfApi().dataset_info(repo, revision=revision)
        rec["resolved_sha"] = info.sha
        if info.sha != revision:
            rec["reason"] = "requested revision did not resolve exactly"
            return rec
        rec["status"] = "revision_resolved"
        # siblings are useful for targeted access, but may be truncated by
        # the Hub on large repos. Never treat them as a complete listing.
        rec["siblings_returned"] = len(info.siblings or [])
        names = {s.rfilename for s in info.siblings or []}
        rec["metadata"] = {}
        for name in ("meta/info.json", "meta/stats.json"):
            rel = f"{subdir}/{name}" if subdir else name
            try:
                path = Path(hf_hub_download(repo, rel, repo_type="dataset", revision=revision))
                raw = path.read_bytes()
                rec["metadata"][name] = {"path": rel, "sha256": hashlib.sha256(raw).hexdigest(),
                                         "bytes": len(raw), "json": json.loads(raw)}
            except Exception as exc:  # absence or network failure, not a finding
                rec["metadata"][name] = {"path": rel, "error": f"{type(exc).__name__}: {exc}"[:300]}
        rec["metadata_names_seen"] = sorted(n for n in names if n.startswith((subdir + "/meta/") if subdir else "meta/"))[:20]
    except Exception as exc:
        rec["reason"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def main() -> None:
    out = Path(sys.argv[1])
    sample = json.loads((out / "sample.json").read_text())
    pairs = sorted({(x["dataset"], x["revision"]) for x in sample["selected"]})
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = sorted(pool.map(probe, pairs), key=lambda r: r["dataset"])
    (out / "provenance.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(f"{len(records)} sources, {sum(x['status'] == 'revision_resolved' for x in records)} exact Hub revisions")


if __name__ == "__main__":
    main()

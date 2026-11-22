"""Record pinned Hub object IDs for raw files used in observations.

Run: .venv/bin/python scripts/assurance_artifacts.py audits/2026-11-22
LFS SHA-256 is recorded where the Hub exposes it; Git blob IDs otherwise.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

SUFFIXES = (".parquet", ".json", ".jsonl", ".mp4")


def paths(value, prefix: str) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        if value.endswith(SUFFIXES) and value.startswith(prefix):
            found.add(value)
    elif isinstance(value, dict):
        for v in value.values():
            found.update(paths(v, prefix))
    elif isinstance(value, list):
        for v in value:
            found.update(paths(v, prefix))
    return found


def main() -> None:
    root = Path(sys.argv[1])
    sample = json.loads((root / "sample.json").read_text())["selected"]
    sources = {s["dataset"]: s for s in json.loads((root / "provenance.json").read_text())}
    obs = json.loads((root / "observations.json").read_text())
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for c in sample:
        s = sources[c["dataset"]]
        if s["status"] != "revision_resolved":
            continue
        prefix = s["subdir"] + "/" if s["subdir"] else ""
        grouped[(s["repo"], s["revision"])].update(paths(obs[c["key"]].get("observed", {}), prefix))
        grouped[(s["repo"], s["revision"])].add(prefix + "meta/info.json")
        if "sha256" in s.get("metadata", {}).get("meta/stats.json", {}):
            grouped[(s["repo"], s["revision"])].add(prefix + "meta/stats.json")
    api = HfApi()
    records = []
    for (repo, revision), names in sorted(grouped.items()):
        try:
            files = {f.path: f for f in api.get_paths_info(repo, sorted(names), repo_type="dataset",
                                                            revision=revision)}
            for name in sorted(names):
                f = files.get(name)
                records.append({"repo": repo, "revision": revision, "path": name,
                                "present": f is not None, "bytes": getattr(f, "size", None),
                                "git_blob_id": getattr(f, "blob_id", None),
                                "lfs_sha256": getattr(getattr(f, "lfs", None), "sha256", None)})
        except Exception as exc:
            records.append({"repo": repo, "revision": revision,
                            "paths_requested": sorted(names), "error": f"{type(exc).__name__}: {exc}"[:300]})
    (root / "artifact_hashes.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    print(len(records), "pinned path records")


if __name__ == "__main__":
    main()

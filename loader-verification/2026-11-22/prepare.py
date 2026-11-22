"""Recover only pinned LeRobot metadata and selected episode files.

This program does not import kinema. Inputs live outside the repository and
are addressed by immutable Hub commits. Every downloaded byte gets SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path("/tmp/kinema-phase2-inputs")


def source_parts(dataset: str) -> tuple[str, str]:
    repo, separator, subdir = dataset.partition("//")
    return repo, f"{subdir}/" if separator else ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_dir(case_id: str, root: Path = DEFAULT_ROOT) -> Path:
    return root / case_id.replace(":", "_")


def recover(case: dict, relative_paths: list[str], root: Path = DEFAULT_ROOT) -> list[dict]:
    repo, prefix = source_parts(case["dataset"])
    destination = case_dir(case["case_id"], root)
    ledger = []
    for relative in relative_paths:
        remote = prefix + relative
        cached = Path(hf_hub_download(repo, remote, repo_type="dataset", revision=case["revision"]))
        local = destination / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() or local.is_symlink():
            local.unlink()
        local.symlink_to(cached)
        ledger.append({"path": relative, "remote_path": remote, "sha256": sha256(cached), "bytes": cached.stat().st_size})
    return ledger


def verify_frozen(case: dict, ledger: list[dict], artifact_hashes: list[dict]) -> None:
    repo, prefix = source_parts(case["dataset"])
    historical = {(entry["repo"], entry["revision"], entry["path"]): entry for entry in artifact_hashes}
    for entry in ledger:
        expected = historical.get((repo, case["revision"], entry["remote_path"]))
        if expected and expected.get("lfs_sha256") and entry["sha256"] != expected["lfs_sha256"]:
            raise ValueError(f"Frozen LFS hash mismatch: {entry['remote_path']}")
        if entry["path"] == "meta/info.json" and entry["sha256"] != case["pinned_inputs"]["info_sha256"]:
            raise ValueError(f"Frozen info hash mismatch: {entry['remote_path']}")
        if entry["path"] == "meta/stats.json" and entry["sha256"] != case["pinned_inputs"]["stats_sha256"]:
            raise ValueError(f"Frozen stats hash mismatch: {entry['remote_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--path", action="append", default=[])
    parser.add_argument("--metadata", action="store_true")
    args = parser.parse_args()
    cases = {case["case_id"]: case for case in json.loads((HERE / "expectations.json").read_text())["cases"]}
    case = cases[args.case_id]
    paths = list(args.path)
    if args.metadata:
        repo, prefix = source_parts(case["dataset"])
        entries = HfApi().list_repo_tree(repo, path_in_repo=prefix + "meta", repo_type="dataset", revision=case["revision"], recursive=True)
        paths.extend(sorted(entry.path.removeprefix(prefix) for entry in entries if hasattr(entry, "size")))
    paths = sorted(set(paths))
    ledger = recover(case, paths)
    frozen_hashes = json.loads((HERE.parents[1] / "audits" / "2026-11-22" / "artifact_hashes.json").read_text())
    verify_frozen(case, ledger, frozen_hashes)
    output = HERE / "evidence" / f"{args.case_id.replace(':', '_')}.json"
    output.parent.mkdir(exist_ok=True)
    prior = json.loads(output.read_text()) if output.exists() else []
    merged = {entry["path"]: entry for entry in prior}
    merged.update({entry["path"]: entry for entry in ledger})
    output.write_text(json.dumps(list(merged.values()), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case_id": args.case_id, "paths": len(ledger), "bytes": sum(entry["bytes"] for entry in ledger), "evidence": str(output)}))


if __name__ == "__main__":
    main()

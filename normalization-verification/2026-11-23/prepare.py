"""Acquire the bounded, pinned inputs for every probe and record their hashes.

Run with the phase-2 v2 environment (it has huggingface_hub):
    /tmp/kinema-phase2-v2/bin/python -I prepare.py

What is fetched (and nothing else):
- deployment cases: phase 2's already-acquired meta/ and episode-0 parquet,
  re-hashed against loader-verification/2026-11-22/evidence/<case>.json and
  copied into a private input root; plus a fresh meta/ listing at the pinned
  revision (names, sizes, hashes only).
- near-constant cases: every meta/ file the loaders under test read, and the
  data files chosen by the frozen sampling rule in expectations.json.
- checkpoints: only the small configuration / statistics / license / README
  files chosen by the frozen selection rule. No weights.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

EXP = common.load_json(common.EXPECTATIONS)
API = HfApi()
HF_CACHE = Path.home() / ".cache/huggingface/hub"


def listing(repo: str, rev: str, path: str) -> list[dict]:
    rows = []
    for it in API.list_repo_tree(repo, path_in_repo=path, repo_type="dataset", revision=rev, recursive=True):
        if hasattr(it, "size"):
            rows.append({"path": it.path, "bytes": it.size, "git_blob_id": it.blob_id,
                         "lfs_sha256": it.lfs.sha256 if it.lfs else None})
    return sorted(rows, key=lambda r: r["path"])


def fetch(repo: str, rev: str, path: str, repo_type: str = "dataset") -> Path:
    return Path(hf_hub_download(repo, path, repo_type=repo_type, revision=rev, cache_dir=str(HF_CACHE)))


def copy_into(src: Path, root: Path, rel: str) -> dict:
    dst = root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return {"path": rel, "bytes": dst.stat().st_size, "sha256": common.sha256_file(dst), "source_cache_path": str(src)}


def prepare_deployment(case: dict) -> dict:
    cid = case["case_id"]
    tag = cid.replace(":", "_")
    ledger = common.load_json(common.REPO / "loader-verification/2026-11-22/evidence" / f"{tag}.json")
    root = common.INPUTS / tag
    if root.exists():
        shutil.rmtree(root)
    files = []
    for entry in ledger:
        src = common.PHASE2_INPUTS / tag / entry["path"]
        got = common.sha256_file(src)
        if got != entry["sha256"]:
            raise SystemExit(f"{cid}: {entry['path']} hash {got} != phase-2 ledger {entry['sha256']}")
        rec = copy_into(src.resolve(), root, entry["path"])
        rec["phase2_sha256_match"] = True
        files.append(rec)
    src = case["source"]
    meta_listing = listing(src["repo"], src["revision"], src.get("subdir", "") + "meta")
    return {"case_id": cid, "root": str(root), "files": files, "meta_listing_at_revision": meta_listing}


def prepare_channel_case(case: dict, rule: dict) -> dict:
    cid = case["case_id"]
    tag = cid.replace(":", "_")
    src = case["source"]
    repo, rev, sub = src["repo"], src["revision"], src.get("subdir", "")
    root = common.INPUTS / tag
    if root.exists():
        shutil.rmtree(root)
    meta_rows = listing(repo, rev, sub + "meta")
    meta_files = []
    for row in meta_rows:
        rel = row["path"][len(sub):]
        if any(rel.startswith(p) for p in src["meta_skip_prefixes"]):
            continue
        p = fetch(repo, rev, row["path"])
        rec = copy_into(p, root, rel)
        expect = row["lfs_sha256"]
        rec["listing_lfs_sha256_match"] = (expect == rec["sha256"]) if expect else None
        rec["git_blob_id"] = row["git_blob_id"]
        meta_files.append(rec)
    data_rows = listing(repo, rev, sub + "data")
    chosen = common.select_files(rule, cid, data_rows)
    data_files = []
    for row in chosen:
        p = fetch(repo, rev, row["path"])
        got = common.sha256_file(p)
        if row["lfs_sha256"] and got != row["lfs_sha256"]:
            raise SystemExit(f"{cid}: {row['path']} sha256 {got} != listing {row['lfs_sha256']}")
        data_files.append({"path": row["path"][len(sub):], "bytes": row["bytes"], "sha256": got,
                           "listing_lfs_sha256": row["lfs_sha256"], "cache_path": str(p),
                           "rank": common.rank(rule["seed"], cid, row["path"])})
    return {"case_id": cid, "root": str(root), "meta_files": meta_files,
            "data_listing_count": len(data_rows), "data_listing_bytes": sum(r["bytes"] for r in data_rows),
            "selected_data_files": data_files, "meta_listing_at_revision": meta_rows}


def prepare_checkpoints() -> list[dict]:
    rules = EXP["checkpoint_selection"]
    out = []
    # 1. Hub checkpoints named explicitly by the rules, and those found by the
    #    VLABench-organisation asset rule.
    hub = list(rules["hub_explicit"])
    org = rules["vlabench_org_rule"]
    for m in API.list_models(author=org["author"], limit=500):
        info = API.model_info(m.id, files_metadata=True)
        assets = [s.rfilename for s in info.siblings
                  if s.rfilename.startswith("assets/") and s.rfilename.endswith("/norm_stats.json")]
        matched = [a for a in assets if a.split("/")[1] + "/" + a.split("/")[2] in org["asset_ids"]]
        out.append({"rule": "vlabench_org_scan", "repo": m.id, "revision": info.sha,
                    "norm_stats_assets": assets, "selected": bool(matched)})
        if matched:
            hub.append({"repo": m.id, "revision": info.sha,
                        "files": matched + [s.rfilename for s in info.siblings if s.rfilename in org["extra_files"]],
                        "role": "vlabench_org"})
    for entry in hub:
        info = API.model_info(entry["repo"], revision=entry["revision"], files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings}
        rec = {"repo": entry["repo"], "revision": entry["revision"], "role": entry["role"],
               "license": (info.card_data or {}).get("license") if info.card_data else None,
               "listing": sorted(sizes), "total_listing_bytes": sum(v or 0 for v in sizes.values()), "files": []}
        for f in entry["files"]:
            if f not in sizes:
                rec["files"].append({"path": f, "present": False})
                continue
            if (sizes[f] or 0) > rules["max_file_bytes"]:
                raise SystemExit(f"{entry['repo']}:{f} exceeds the small-file cap")
            p = Path(hf_hub_download(entry["repo"], f, revision=entry["revision"], cache_dir=str(common.CKPT)))
            rec["files"].append({"path": f, "present": True, "bytes": p.stat().st_size,
                                 "sha256": common.sha256_file(p), "local": str(p)})
        out.append(rec)
    # 2. openpi public GCS assets (small JSON only)
    for entry in rules["gcs_explicit"]:
        meta_url = ("https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix="
                    + entry["prefix"] + "&fields=items(name,size,md5Hash,generation,updated)")
        with urllib.request.urlopen(meta_url, timeout=60) as r:
            objects = json.load(r).get("items", [])
        rec = {"gcs_prefix": entry["prefix"], "role": entry["role"], "objects": objects, "files": []}
        for obj in objects:
            if not obj["name"].endswith("norm_stats.json"):
                continue
            url = "https://storage.googleapis.com/openpi-assets/" + obj["name"]
            dst = common.CKPT / "gcs" / obj["name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if len(data) > rules["max_file_bytes"]:
                raise SystemExit(f"{url} exceeds the small-file cap")
            dst.write_bytes(data)
            rec["files"].append({"name": obj["name"], "bytes": len(data), "sha256": common.sha256_bytes(data),
                                 "md5_base64": base64.b64encode(hashlib.md5(data).digest()).decode(),
                                 "local": str(dst)})
        out.append(rec)
    return out


def main() -> None:
    common.check_preregistration(EXP["frozen_files"])
    common.INPUTS.mkdir(parents=True, exist_ok=True)
    rule = EXP["sampling_rule"]
    result = {"schema": "kinema.normverify.inputs/1", "cases": [], "checkpoints": None}
    for case in EXP["cases"]:
        if case["kind"] == "missing-stats":
            result["cases"].append(prepare_deployment(case))
        else:
            result["cases"].append(prepare_channel_case(case, rule))
    result["checkpoints"] = prepare_checkpoints()
    common.write_json(common.EVIDENCE / "inputs.json", result)
    print("ok")


if __name__ == "__main__":
    main()

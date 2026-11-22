"""Independent raw-source observations for selected phase 1 cases.

Run: .venv/bin/python scripts/assurance_observe.py audits/2026-11-22
This code reads pinned Hub metadata and parquet; it never imports kinema.
It records observations only; the rubric determines adjudication separately.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem


def table(fs: HfFileSystem, repo: str, rev: str, rel: str, columns: list[str]):
    with fs.open(f"datasets/{repo}@{rev}/{rel}", "rb") as file:
        return pq.ParquetFile(file).read(columns=columns).to_pylist()


def tree(api: HfApi, repo: str, rev: str, path: str, recursive: bool = True) -> list[str]:
    return sorted(x.path for x in api.list_repo_tree(repo, repo_type="dataset", revision=rev,
                                                     path_in_repo=path or None, recursive=recursive))


def jsonl(fs: HfFileSystem, repo: str, rev: str, rel: str):
    with fs.open(f"datasets/{repo}@{rev}/{rel}", "rt") as file:
        for line in file:
            yield json.loads(line)


def observe(case: dict, meta: dict, api: HfApi, fs: HfFileSystem) -> dict:
    repo, rev, sub = meta["repo"], case["revision"], meta["subdir"]
    prefix = sub + "/" if sub else ""
    info = meta["metadata"]["meta/info.json"]["json"]
    cid = case["check"]
    out: dict = {"command": f".venv/bin/python scripts/assurance_observe.py audits/2026-11-22 # {case['key']}",
                 "source": f"hf://datasets/{repo}@{rev}/{prefix}", "observed": {}}
    obs = out["observed"]
    if cid == "task-strings":
        targets = set(case["details"]["episodes"])
        observed = {r["episode_index"]: r.get("tasks") for r in jsonl(fs, repo, rev, prefix + "meta/episodes.jsonl")
                    if r["episode_index"] in targets}
        obs.update({"metadata_path": prefix + "meta/episodes.jsonl", "target_count": len(targets),
                    "found_count": len(observed), "tasks_by_episode": observed})
    elif cid == "deployment-readiness":
        paths = tree(api, repo, rev, prefix + "meta")
        obs.update({"metadata_tree_count": len(paths), "metadata_tree": paths[:40],
                    "metadata_tree_complete": True, "stats_json_present": prefix + "meta/stats.json" in paths,
                    "per_episode_stats_paths": [p for p in paths if "stats" in p and p != prefix + "meta/stats.json"],
                    "feature_declared": case["details"]["feature"] in info.get("features", {})})
    elif case["key"] == "r413:c2:f2":
        mp = tree(api, repo, rev, "meta/episodes")
        dp = tree(api, repo, rev, "data")
        mfiles = [p for p in mp if p.endswith(".parquet")]
        dfiles = [p for p in dp if p.endswith(".parquet")]
        catalog = [r for p in mfiles for r in table(fs, repo, rev, p,
                   ["episode_index", "length", "data/chunk_index", "data/file_index"])
                   if r["episode_index"] == 50]
        rows = {p: sum(r["episode_index"] == 50 for r in table(fs, repo, rev, p, ["episode_index"]))
                for p in dfiles}
        obs.update({"metadata_files": mfiles, "data_files": dfiles, "episode_50_catalog": catalog,
                    "episode_50_rows_by_file": rows, "total_rows": sum(rows.values())})
    elif case["key"] == "r256:c2:f0":
        rel = prefix + info["data_path"].format(episode_chunk=0, episode_index=0)
        with fs.open(f"datasets/{repo}@{rev}/{rel}", "rb") as file:
            pf = pq.ParquetFile(file)
            obs.update({"data_path": rel, "columns": pf.schema_arrow.names,
                        "frame_index_present": "frame_index" in pf.schema_arrow.names})
    elif cid == "reward-done":
        stats = meta["metadata"]["meta/stats.json"]["json"]
        data_files = [p for p in tree(api, repo, rev, prefix + "data") if p.endswith(".parquet")]
        values: set[float | None] = set()
        rows = 0
        for rel in data_files:
            with fs.open(f"datasets/{repo}@{rev}/{rel}", "rb") as file:
                column = pq.ParquetFile(file).read(columns=["next.reward"]).column("next.reward")
                values.update(column.unique().to_pylist())
                rows += len(column)
        obs.update({"stats_path": prefix + "meta/stats.json",
                    "reward_stats": stats.get("next.reward") or stats.get("reward"),
                    "total_frames_declared": info.get("total_frames"),
                    "complete_data_listing": True, "data_files": data_files,
                    "raw_rows": rows, "raw_unique_rewards": sorted(values, key=str)})
    elif cid == "episode-file-pointers":
        ep = case["episode_index"]
        catalog_files = [p for p in tree(api, repo, rev, prefix + "meta/episodes") if p.endswith(".parquet")]
        cols = ["episode_index", "data/chunk_index", "data/file_index"]
        matches = [r for p in catalog_files for r in table(fs, repo, rev, p, cols)
                   if r["episode_index"] == ep]
        if not matches:
            obs.update({"catalog_files": catalog_files, "episode": ep, "catalog_rows": []})
        else:
            declared = sorted({prefix + info["data_path"].format(chunk_index=r["data/chunk_index"],
                                                            file_index=r["data/file_index"]) for r in matches})
            rosters = {p: sum(r["episode_index"] == ep for r in table(fs, repo, rev, p, ["episode_index"]))
                       for p in declared}
            obs.update({"catalog_files": catalog_files, "episode": ep, "catalog_rows": matches,
                        "declared_files": declared, "rows_in_declared_files": rosters})
    elif case["key"] == "r388:c0:f1":
        x = meta["metadata"]["meta/stats.json"]["json"]["observation.state"]
        d = case["details"]["dim"]
        rel = prefix + info["data_path"].format(episode_index=0)
        values = np.asarray([r["observation.state"][d]
                             for r in table(fs, repo, rev, rel, ["observation.state"])], dtype=float)
        obs.update({"stats_path": prefix + "meta/stats.json", "dim": d,
                    "min": x["min"][d], "max": x["max"][d], "mean": x["mean"][d],
                    "std": x["std"][d], "range_half": (x["max"][d] - x["min"][d]) / 2,
                    "data_path": rel, "episode_0_rows": len(values),
                    "episode_0_min": float(values.min()), "episode_0_max": float(values.max()),
                    "episode_0_mean": float(values.mean()), "episode_0_std": float(values.std())})
    elif case["key"] == "r177:c0:f3":
        stat = meta["metadata"]["meta/stats.json"]["json"]["observation.state"]
        d = case["details"]["dim"]
        episodes = []
        for n in range(3):
            rel = f"meta/episodes/chunk-000/file-{n:03d}.parquet"
            row = table(fs, repo, rev, rel, ["episode_index", "length", "stats/observation.state/mean",
                                              "stats/observation.state/std"])[0]
            data_rel = f"data/chunk-000/file-{n:03d}.parquet"
            raw = np.asarray([r["observation.state"][d]
                              for r in table(fs, repo, rev, data_rel, ["observation.state"])], dtype=float)
            episodes.append({"path": rel, "episode": row["episode_index"], "length": row["length"],
                             "mean_dim": row["stats/observation.state/mean"][d],
                             "std_dim": row["stats/observation.state/std"][d],
                             "data_path": data_rel, "raw_rows": len(raw),
                             "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                             "raw_mean": float(raw.mean()), "raw_std": float(raw.std())})
        obs.update({"stored_min": stat["min"][d], "stored_max": stat["max"][d],
                    "stored_std": stat["std"][d], "info_total_episodes": info["total_episodes"],
                    "catalog_examples": episodes})
    elif case["key"] == "r216:c5:f1":
        d = case["details"]["joint"]
        ep = case["details"]["episodes"][0]
        rel = prefix + info["data_path"].format(episode_chunk=ep // info.get("chunks_size", 1000),
                                                  episode_index=ep)
        rows = table(fs, repo, rev, rel, ["observation.state"])
        data = np.asarray([r["observation.state"] for r in rows], dtype=float)
        names = info["features"]["observation.state"]["names"]
        obs.update({"data_path": rel, "episode": ep, "frames": len(rows), "dim": d,
                    "feature_name": names[d], "dimension_kind": "quaternion component, not a motor",
                    "min": float(data[:, d].min()), "max": float(data[:, d].max()),
                    "std": float(data[:, d].std()),
                    "adjacent_feature_name": names[d - 1],
                    "adjacent_min": float(data[:, d - 1].min()),
                    "adjacent_max": float(data[:, d - 1].max())})
    elif cid == "nan-and-normalization" and case["details"].get("std") is not None:
        x = meta["metadata"]["meta/stats.json"]["json"][case["details"]["feature"]]
        d = case["details"]["dim"]
        obs.update({"stats_path": prefix + "meta/stats.json", "dim": d,
                    "stored_std": x["std"][d], "stored_min": x["min"][d],
                    "stored_max": x["max"][d]})
    elif case["key"] == "r202:c13:f1":
        roots = tree(api, repo, rev, "", recursive=False)
        ep_count = sum(1 for _ in jsonl(fs, repo, rev, "meta/episodes.jsonl"))
        obs.update({"root_listing": roots, "root_listing_complete": True,
                    "videos_directory_present": "videos" in roots, "catalog_episode_count": ep_count,
                    "video_path_template": info.get("video_path"),
                    "camera_declared": case["details"]["camera"] in info.get("features", {})})
    elif case["key"] == "r12:c13:f1":
        cam = case["details"]["camera"]
        cat_paths = [p for p in tree(api, repo, rev, "meta/episodes") if p.endswith(".parquet")]
        cols = ["episode_index", f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"]
        records = [r for p in cat_paths for r in table(fs, repo, rev, p, cols)]
        refs = {}
        for r in records:
            ci, fi = r[cols[1]], r[cols[2]]
            if ci is None or fi is None:
                continue
            rel = info["video_path"].format(video_key=cam, chunk_index=ci, file_index=fi)
            refs.setdefault(rel, set()).add(r["episode_index"])
        vroot = f"videos/{cam}"
        files = {p for p in tree(api, repo, rev, vroot) if p.endswith(".mp4")}
        missing = sorted(set(refs) - files)
        obs.update({"catalog_files": cat_paths, "catalog_rows": len(records),
                    "camera_listing_complete": True, "camera_files": len(files),
                    "referenced_files": len(refs), "missing_file_count": len(missing),
                    "affected_episodes": len(set().union(*(refs[p] for p in missing))) if missing else 0,
                    "missing_files_first": missing[:20]})
    elif case["key"] in {"r94:c1:f1", "r71:c1:f1"}:
        ep = case["details"]["episodes"][0]
        rel = prefix + info["data_path"].format(episode_chunk=ep // info.get("chunks_size", 1000),
                                                  episode_index=ep)
        rows = table(fs, repo, rev, rel, ["timestamp"])
        ts = [r["timestamp"] for r in rows]
        obs.update({"data_path": rel, "fps": info["fps"], "frames": len(ts),
                    "first_timestamp": ts[0], "second_timestamp": ts[1] if len(ts) > 1 else None,
                    "last_timestamp": ts[-1]})
    else:
        out["unsupported"] = "no independent raw verifier implemented for this claim"
    return out


def main() -> None:
    root = Path(sys.argv[1])
    selected = json.loads((root / "sample.json").read_text())["selected"]
    keys = set(sys.argv[2:])
    if keys:
        selected = [x for x in selected if x["key"] in keys]
    sources = {r["dataset"]: r for r in json.loads((root / "provenance.json").read_text())}
    api, fs = HfApi(), HfFileSystem()
    path = root / "observations.json"
    evidence = json.loads(path.read_text()) if path.exists() else {}
    for case in selected:
        key = case["key"]
        src = sources[case["dataset"]]
        if src["status"] != "revision_resolved":
            evidence[key] = {"unsupported": src.get("reason", src["status"])}
            continue
        try:
            evidence[key] = observe(case, src, api, fs)
        except Exception as exc:
            evidence[key] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
        print(key, "ok" if "observed" in evidence[key] and evidence[key]["observed"] else "unresolved", flush=True)
    path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

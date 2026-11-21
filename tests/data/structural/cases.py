"""Case definitions for the structural regression pack (see build_pack.py).

Ground truth here is computed from the Hub files at the pinned revision
with plain pyarrow - no kinema check code - and written to CASE.json:

- pointers: each kept episode's declared shard (meta/episodes data/chunk_index,
  data/file_index through info.json's data_path) against that shard's actual
  episode_index roster; a broken pointer records where the rows really are
  among the packed shards (None if in none), plus whether they exist in ANY
  shard of the repo (evidence).
- segments: per video camera, kept episodes whose meta/episodes segment
  chunk_index is null - no video recorded for that camera.
- video files: per camera, the file each segment names (info.json video_path)
  that the repo listing at the revision does not contain.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import build_pack as B

HERE = Path(__file__).resolve().parent

CASES = {
    # metadata catalogues 82 episodes; the data holds 80 - episodes 80/81 point
    # at file-002 with no rows anywhere (metadata committed ahead of data)
    "mendelbeans_pointer_no_rows": dict(
        repo="MendelBeans/lerobot_dataset_put_pen_in_the_holder",
        revision="0c0e6528c6b3441e6c6a309a7e725199fb6ddf89",
        episodes=None, shards="pointed"),
    # 83 episodes catalogued in file-006 whose rows live in file-002 (not packed:
    # the extract reproduces the board's view - no rows in the packed shards -
    # and CASE.json evidence records where the rows really are)
    "bitrobot_pointer_rows_unpacked": dict(
        repo="BitRobot/berkeley-frodobots-lerobot-7k",
        revision="096912cad3512e29d95dd809af7cb9fc020e4de7",
        episodes=("shard", (0, 6)), shards="pointed",
        # 750k rows: the pointer check reads episode_index only
        shard_columns=("episode_index", "frame_index")),
    # the v2.1->v3.0 converter rollover bug (lerobot#2401), for real: boundary
    # episodes point one shard early (2 -> file-000, rows in file-001; 4 ->
    # file-001, rows in file-002, ...)
    "vlabench_pointer_rollover": dict(
        repo="VLABench/vlabench_primitive_ft_lerobot_video",
        revision="9846a2f6bead3873251dc4fe3079359d57326b7c",
        episodes=("range", 0, 40), shards="pointed"),
    # 85 episodes from 9761 record no video segment for any of 7 cameras
    "yaak_segments_missing": dict(
        repo="yaak-ai/L2D-v3",
        revision="c9a9d0713e251068c83e2c43df4c8a9d1344b748",
        episodes=("range", 9700, 9860), shards="pointed"),
    # clean controls: every pointer, segment and video file verified present
    "pusht_clean": dict(repo="lerobot/pusht", revision="7628202a2180972f291ba1bc6723834921e72c19",
                        episodes=None, shards="pointed"),
    "svla_so101_clean": dict(repo="lerobot/svla_so101_pickplace",
                             revision="f641879e22172be7e8161d5e6c1503c2d2feb657",
                             episodes=None, shards="pointed"),
}


def _full_revision(repo: str, short: str) -> str:
    if len(short) == 40:
        return short
    from huggingface_hub import HfApi
    full = HfApi().dataset_info(repo, revision=short).sha
    assert full.startswith(short), (repo, short, full)
    return full


def _catalogue(repo: str, rev: str) -> dict:
    _, fs = B._api()
    at = f"datasets/{repo}@{rev}"
    tables = []
    for p in sorted(fs.glob(f"{at}/meta/episodes/**/*.parquet")):
        pf = pq.ParquetFile(p, filesystem=fs)
        tables.append(pf.read(columns=B._meta_columns(pf.schema_arrow.names)))
    return pa.concat_tables(tables, promote_options="default").to_pydict()


def _select(cat: dict, spec) -> set[int]:
    eps = cat["episode_index"]
    if spec is None:
        return set(eps)
    if spec[0] == "range":
        return {e for e in eps if spec[1] <= e < spec[2]}
    if spec[0] == "shard":
        c, f = spec[1]
        return {e for e, ci, fi in zip(eps, cat["data/chunk_index"], cat["data/file_index"]) if (ci, fi) == (c, f)}
    raise ValueError(spec)


def ground_truth(repo: str, rev: str, info: dict, cat: dict, keep: set[int], shards: set[str],
                 listing: set[str]) -> dict:
    tpl = info["data_path"]
    rows = [i for i, e in enumerate(cat["episode_index"]) if e in keep]
    rosters = {rel: B.roster(repo, rev, rel) for rel in sorted(shards)}
    # catalogued more than once: a defect of its own; with conflicting
    # pointers the "declared shard" has no single answer, so those episodes
    # are left out of the pointer truth (and named)
    from collections import Counter, defaultdict
    counts = Counter(cat["episode_index"][i] for i in rows)
    declared = defaultdict(set)
    for i in rows:
        declared[cat["episode_index"][i]].add((cat["data/chunk_index"][i], cat["data/file_index"][i]))
    duplicates = sorted(e for e, n in counts.items() if n > 1)
    conflicting = sorted(e for e in duplicates if len(declared[e]) > 1)
    pointers, seen = [], set()
    for i in rows:
        ep = cat["episode_index"][i]
        if ep in conflicting or ep in seen:
            continue
        seen.add(ep)
        rel = tpl.format(chunk_index=cat["data/chunk_index"][i], file_index=cat["data/file_index"][i])
        if ep not in rosters.get(rel, set()):
            actual = next((r for r, s in rosters.items() if ep in s), None)
            pointers.append({"episode": ep, "declared": rel, "actual": actual})
    cams = [k for k, f in (info.get("features") or {}).items() if isinstance(f, dict) and f.get("dtype") == "video"]
    segments, files = {}, {}
    vtpl = info.get("video_path")
    for cam in cams:
        ci, fi = f"videos/{cam}/chunk_index", f"videos/{cam}/file_index"
        if ci not in cat:
            continue
        miss = sorted(cat["episode_index"][i] for i in rows if cat[ci][i] is None)
        if miss:
            segments[cam] = miss
        gone, affected = set(), []
        for i in rows:
            if cat[ci][i] is None or not vtpl:
                continue
            path = vtpl.format(video_key=cam, chunk_index=cat[ci][i], file_index=cat[fi][i])
            if path not in listing:
                gone.add(path)
                affected.append(cat["episode_index"][i])
        if gone:
            files[cam] = {"missing_files": sorted(gone), "episodes": sorted(affected)}
    return {"pointers": sorted(pointers, key=lambda p: p["episode"]), "segments": segments, "video_files": files,
            "duplicate_catalogue": duplicates, "conflicting_duplicates": conflicting}


def _anywhere(repo: str, rev: str, eps: list[int], all_shards: list[str], cap: int = 400) -> dict:
    """Evidence only: do the broken episodes' rows exist in ANY repo shard?"""
    want, found = set(eps), {}
    for rel in all_shards[:cap]:
        hit = want & B.roster(repo, rev, rel)
        for e in hit:
            found[e] = rel
        want -= hit
        if not want:
            break
    return {"searched_shards": min(len(all_shards), cap), "of": len(all_shards),
            "found_elsewhere": {str(k): v for k, v in sorted(found.items())}}


def build(name: str) -> dict:
    spec = CASES[name]
    repo, rev = spec["repo"], _full_revision(spec["repo"], spec["revision"])
    lic = B.license_of(repo)
    if (lic or "").lower() not in B.REDISTRIBUTABLE:
        raise SystemExit(f"{name}: {repo} declares license {lic!r} - not packed (no redistribution permission)")
    _, fs = B._api()
    info = json.loads(fs.cat_file(f"datasets/{repo}@{rev}/meta/info.json"))
    cat = _catalogue(repo, rev)
    keep = _select(cat, spec["episodes"])
    tpl = info["data_path"]
    shards = {tpl.format(chunk_index=c, file_index=f) for e, c, f in
              zip(cat["episode_index"], cat["data/chunk_index"], cat["data/file_index"]) if e in keep}
    api, _ = B._api()
    all_paths = sorted(f.path for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision=rev)
                       if hasattr(f, "size"))
    existing = sorted(s for s in shards if s in set(all_paths))
    root = B.extract(repo, rev, name, lambda e: e in keep, set(existing),
                     **({"shard_columns": spec["shard_columns"]} if "shard_columns" in spec else {}))
    truth = ground_truth(repo, rev, info, cat, keep, set(existing), set(all_paths))
    evidence = {}
    if truth["pointers"]:
        data_shards = [p for p in all_paths if p.startswith("data/") and p.endswith(".parquet")]
        evidence["pointer_rows_anywhere"] = _anywhere(repo, rev, [p["episode"] for p in truth["pointers"]], data_shards)
    doc = {
        "source": repo, "revision": rev, "license": lic,
        "built": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}",
        "kept": {"episodes": len(keep), "of": len(cat["episode_index"]), "shards": existing,
                 "note": "meta/episodes trimmed to pointer and segment columns; shards reduced to index columns"},
        "truth": truth, "evidence": evidence,
    }
    (root / "CASE.json").write_text(json.dumps(doc, indent=1))
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"{name}: {repo}@{rev[:12]} ({lic}) | {len(keep)}/{len(cat['episode_index'])} episodes, "
          f"{len(existing)} shard(s), {size // 1024} KB | pointers {len(truth['pointers'])}, "
          f"segment gaps {sum(len(v) for v in truth['segments'].values())}, "
          f"missing video files {sum(len(v['missing_files']) for v in truth['video_files'].values())}")
    return doc


def main(names: list[str]) -> int:
    for n in names or CASES:
        build(n)
    return 0

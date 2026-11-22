"""Build the structural regression pack (tests/test_structural_pack.py).

    python tests/data/structural/build_pack.py [case ...]

Each case is a small directory under tests/data/structural/<case>/ cut from
a dataset at an exact Hub revision: meta/info.json and meta/tasks.parquet
verbatim, meta/episodes trimmed to the pointer and video-segment columns of
the kept episodes, the kept data shards reduced to their index columns
(episode_index, frame_index, timestamp, index, task_index - no state,
action or images), and listing.txt: the repo's file listing at that
revision for every path the kept episodes can reference.

The expected findings are computed here WITHOUT kinema's check code - plain
pyarrow over the Hub files at the pinned revision - and written to
CASE.json with the evidence, so the test compares the detector against an
independent ground truth. Only datasets whose card declares a license that
permits redistribution are packed; the license is recorded per case.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REDISTRIBUTABLE = {"mit", "apache-2.0", "cc-by-4.0", "cc0-1.0", "cc-by-sa-4.0", "odc-by", "bsd-3-clause"}
INDEX_COLS = ("episode_index", "frame_index", "timestamp", "index", "task_index")


def _api():
    from huggingface_hub import HfApi, HfFileSystem
    return HfApi(), HfFileSystem()


def license_of(repo: str, revision: str | None = None) -> str | None:
    """The card's license at revision (the pinned one: a card edited later
    says nothing about the files packed)."""
    api, _ = _api()
    cd = api.dataset_info(repo, revision=revision).card_data
    return (cd.to_dict() if cd else {}).get("license")


def _meta_columns(names: list[str]) -> list[str]:
    keep = {"episode_index", "length", "tasks", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index", "meta/episodes/chunk_index", "meta/episodes/file_index"}
    return [n for n in names if n in keep or (n.startswith("videos/") and n.split("/")[-1] in
            ("chunk_index", "file_index", "from_timestamp", "to_timestamp"))]


def extract(repo: str, rev: str, case: str, keep_episodes, keep_shards: set[str],
            shard_columns: tuple[str, ...] = INDEX_COLS) -> Path:
    """Write the trimmed case directory; keep_episodes(ep_index) -> bool."""
    api, fs = _api()
    root = HERE / case
    if root.exists():
        import shutil
        shutil.rmtree(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    at = f"datasets/{repo}@{rev}"
    for name in ("meta/info.json", "meta/tasks.parquet"):
        try:
            (root / name).write_bytes(fs.cat_file(f"{at}/{name}"))
        except FileNotFoundError:
            pass
    listing = sorted((f.path, f.size) for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision=rev)
                     if hasattr(f, "size"))
    meta_files = [p for p, _ in listing if p.startswith("meta/episodes/") and p.endswith(".parquet")]
    tables = []
    for p in meta_files:
        pf = pq.ParquetFile(f"{at}/{p}", filesystem=fs)
        t = pf.read(columns=_meta_columns(pf.schema_arrow.names))
        mask = [bool(keep_episodes(int(e))) for e in t.column("episode_index").to_pylist()]
        tables.append(t.filter(pa.array(mask)))
    meta = pa.concat_tables(tables, promote_options="default")
    pq.write_table(meta, root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    for rel in sorted(keep_shards):
        pf = pq.ParquetFile(f"{at}/{rel}", filesystem=fs)
        cols = [c for c in shard_columns if c in pf.schema_arrow.names]
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pf.read(columns=cols), out, compression="zstd")
    # the listing the pointer checks consult: meta, the kept shards, and every
    # video path the kept episodes' segments name (present or not: a missing
    # one is then absent from the listing, exactly as on the Hub)
    info = json.loads((root / "meta" / "info.json").read_text())
    vtpl = info.get("video_path")
    md = meta.to_pydict()
    cams = [k for k, f in (info.get("features") or {}).items() if isinstance(f, dict) and f.get("dtype") == "video"]
    named = set()
    for cam in cams:
        ci, fi = md.get(f"videos/{cam}/chunk_index"), md.get(f"videos/{cam}/file_index")
        if vtpl and ci is not None:
            named |= {vtpl.format(video_key=cam, chunk_index=c, file_index=f) for c, f in zip(ci, fi) if c is not None}
    # every data shard path too, packed or not: the board's listing is the
    # whole repo's, and the pointer check reads it to know what it did not fetch
    (root / "listing.txt").write_text("".join(f"{p}\t{s}\n" for p, s in listing
                                              if p.startswith(("meta/", "data/")) or p in named))
    return root


def roster(repo: str, rev: str, rel: str) -> set[int]:
    _, fs = _api()
    pf = pq.ParquetFile(f"datasets/{repo}@{rev}/{rel}", filesystem=fs)
    return set(pf.read(columns=["episode_index"]).column(0).to_pylist())


# --- v2.1 (one parquet and one mp4 per episode), optionally in a repo subfolder

def sub_listing(repo: str, rev: str, subdir: str = "") -> list[tuple[str, int]]:
    """(path relative to the dataset root, size) for every file under subdir
    at rev - list_repo_tree paginates; dataset_info's siblings cap at 100k
    entries, which a multi-dataset repo exceeds."""
    api, _ = _api()
    pre = f"{subdir}/" if subdir else ""
    return sorted((f.path[len(pre):], f.size) for f in api.list_repo_tree(
        repo, repo_type="dataset", path_in_repo=subdir or None, recursive=True, revision=rev)
        if hasattr(f, "size") and f.path.startswith(pre))


def extract_v21(repo: str, rev: str, case: str, keep: set[int], subdir: str,
                listing: list[tuple[str, int]], license_text: bytes | None = None,
                shard_columns: tuple[str, ...] = ("episode_index", "frame_index"),
                all_videos: bool = True) -> Path:
    """Write a trimmed v2.1 case: meta/info.json and meta/tasks.jsonl verbatim,
    meta/episodes.jsonl reduced to the kept episodes' lines (byte-for-byte),
    each kept episode's data parquet reduced to shard_columns, and listing.txt:
    every meta/ file of the dataset folder at rev, the kept episodes' data
    files (no v2.1 check reads the others; their count is in CASE.json) and,
    with all_videos, EVERY videos/ file, kept episodes or not - the evidence
    that trimming cannot make a video look absent. Clean controls list only
    the videos the kept episodes name, as the v3 extracts do."""
    _, fs = _api()
    root = HERE / case
    if root.exists():
        import shutil
        shutil.rmtree(root)
    (root / "meta").mkdir(parents=True)
    at = f"datasets/{repo}@{rev}" + (f"/{subdir}" if subdir else "")
    for name in ("meta/info.json", "meta/tasks.jsonl"):
        (root / name).write_bytes(fs.cat_file(f"{at}/{name}"))
    lines = fs.cat_file(f"{at}/meta/episodes.jsonl").splitlines(keepends=True)
    kept = [ln for ln in lines if ln.strip() and json.loads(ln)["episode_index"] in keep]
    assert {json.loads(ln)["episode_index"] for ln in kept} == keep, "a kept episode is not catalogued"
    (root / "meta" / "episodes.jsonl").write_bytes(b"".join(kept))
    info = json.loads((root / "meta" / "info.json").read_text())
    cs = int(info.get("chunks_size") or 1000)
    present = {p for p, _ in listing}
    kept_data, named = set(), set()
    cams = [k for k, f in (info.get("features") or {}).items() if isinstance(f, dict) and f.get("dtype") == "video"]
    for ep in sorted(keep):
        named |= {info["video_path"].format(episode_chunk=ep // cs, video_key=c, episode_index=ep) for c in cams}
        rel = info["data_path"].format(episode_chunk=ep // cs, episode_index=ep)
        assert rel in present, f"episode {ep}: data file {rel} absent - not a video-only case"
        kept_data.add(rel)
        pf = pq.ParquetFile(f"{at}/{rel}", filesystem=fs)
        cols = [c for c in shard_columns if c in pf.schema_arrow.names]
        out = root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pf.read(columns=cols), out, compression="zstd")
    (root / "listing.txt").write_text("".join(f"{p}\t{s}\n" for p, s in listing
                                              if p.startswith("meta/") or p in kept_data or p in named
                                              or (all_videos and p.startswith("videos/"))))
    if license_text is not None:
        (root / "LICENSE").write_bytes(license_text)
    return root


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))  # the case definitions sit next to this file
    import cases  # noqa: E402
    sys.exit(cases.main(sys.argv[1:]))

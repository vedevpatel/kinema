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
    # v2.1, one dataset per repo folder. heat_food_in_microwave catalogues 3156
    # episodes and holds all 3156 data parquets, but only 560 of the 3156
    # metadata-named MP4s: chunk-001 and chunk-002 have no videos/ folder at
    # all, and chunk-000 lacks 596 (irregular gaps from 189 - an interrupted
    # upload). The kept ranges straddle every kind of edge: isolated misses
    # (189, 289) between present episodes, runs, both chunk boundaries
    # (999|1000 missing|missing, 2999|3000 missing|present)
    "fastumi_video_files_missing": dict(
        repo="IPEC-COMMUNITY/FastUMI_100k_lerobot", subdir="single_arm/heat_food_in_microwave",
        revision="4cf7cca8fdc15d6f71ee0055553ce8a5e78aa9a0",
        episodes=("ranges", ((186, 243), (287, 293), (998, 1002), (2998, 3002))),
        license_grant="fastumi"),
    # the v2.1 clean control from the same release: every one of iron_clothes'
    # 2541 metadata-named MP4s is in the listing (1000 + 1000 + 541)
    "fastumi_v21_clean": dict(
        repo="IPEC-COMMUNITY/FastUMI_100k_lerobot", subdir="single_arm/iron_clothes",
        revision="4cf7cca8fdc15d6f71ee0055553ce8a5e78aa9a0",
        episodes=("ranges", ((0, 20), (998, 1002), (1998, 2002), (2535, 2541))),
        license_grant="fastumi"),
}

# License text and attribution shipped with each v3 extract: Apache-2.0
# section 4(a) asks for a copy of the license, MIT for its copyright and
# permission notice. Each of these sources declares its license only in the
# card YAML at the pinned revision - no LICENSE, NOTICE or copyright line in
# the repo - so the Apache cases carry the canonical text, and the MIT cases
# the LICENSE of the upstream project the card cites, whose copyright line
# is retained verbatim. Every text is pinned by sha256.
_APACHE = ("https://www.apache.org/licenses/LICENSE-2.0.txt",
           "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30")
_NO_NOTICE = "the source repo has no LICENSE, NOTICE or copyright line; license from its card at the revision"
NOTICES = {
    "bitrobot_pointer_rows_unpacked": dict(
        text=_APACHE, notice=_NO_NOTICE,
        attribution="BitRobot/berkeley-frodobots-lerobot-7k, published on the Hugging Face Hub by BitRobot "
                    "(the card names no paper or homepage). "
                    "https://huggingface.co/datasets/BitRobot/berkeley-frodobots-lerobot-7k"),
    "mendelbeans_pointer_no_rows": dict(
        text=_APACHE, notice=_NO_NOTICE,
        attribution="MendelBeans/lerobot_dataset_put_pen_in_the_holder, published on the Hugging Face Hub by "
                    "MendelBeans (the card names no paper or homepage). "
                    "https://huggingface.co/datasets/MendelBeans/lerobot_dataset_put_pen_in_the_holder"),
    "yaak_segments_missing": dict(
        text=_APACHE, notice=_NO_NOTICE,
        attribution="L2D (yaak-ai/L2D-v3), published on the Hugging Face Hub by Yaak (https://www.yaak.ai/); "
                    "described in 'LeRobot goes to driving school' "
                    "(https://huggingface.co/blog/lerobot-goes-to-driving-school). "
                    "https://huggingface.co/datasets/yaak-ai/L2D-v3"),
    "svla_so101_clean": dict(
        text=_APACHE, notice=_NO_NOTICE,
        attribution="lerobot/svla_so101_pickplace, published on the Hugging Face Hub by LeRobot (Hugging Face) "
                    "(the card names no paper or homepage). "
                    "https://huggingface.co/datasets/lerobot/svla_so101_pickplace"),
    "pusht_clean": dict(
        text=("https://raw.githubusercontent.com/real-stanford/diffusion_policy/"
              "5ba07ac6661db573af695b419a7947ecb704690f/LICENSE",
              "304e99a48271e10820f71a1b86ef395c06c100d7f68dd1d308790d1064fc4d41"),
        notice="the source repo has no LICENSE or copyright line (card: license mit, homepage "
               "diffusion-policy.cs.columbia.edu); LICENSE is the MIT LICENSE of the Diffusion Policy code "
               "repository the card cites, its copyright line retained verbatim",
        attribution="lerobot/pusht, published on the Hugging Face Hub by LeRobot (Hugging Face), from Diffusion "
                    "Policy - Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, "
                    "Russ Tedrake, Shuran Song. Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. "
                    "The International Journal of Robotics Research, 2024. https://diffusion-policy.cs.columbia.edu/ "
                    "https://huggingface.co/datasets/lerobot/pusht"),
    "vlabench_pointer_rollover": dict(
        text=("https://raw.githubusercontent.com/OpenMOSS/VLABench/"
              "cf588fe60c0c7282174fe979f5913170cfe69017/LICENSE.txt",
              "c28645822e5cdadf816144773ee153332449ae04aa6f0b6a6393e5611092f878"),
        notice="the source repo has no LICENSE or copyright line (card: license mit, cites VLABench); "
               "LICENSE is the MIT LICENSE.txt of the VLABench code repository, its copyright line "
               "('2024 simpler-env', as published there) retained verbatim",
        attribution="VLABench/vlabench_primitive_ft_lerobot_video, published on the Hugging Face Hub by "
                    "VLABench, from VLABench - Shiduo Zhang, Zhe Xu, Peiju Liu, Xiaopeng Yu, Yuan Li, Qinghui "
                    "Gao, Zhaoye Fei, Zhangyue Yin, Zuxuan Wu, Yu-Gang Jiang, et al. VLABench: A Large-Scale "
                    "Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks. "
                    "arXiv:2412.18194, 2024. "
                    "https://huggingface.co/datasets/VLABench/vlabench_primitive_ft_lerobot_video"),
}


def notice_fields(name: str, root: Path, lic: str) -> dict:
    """Fetch the case's pinned license text, check its sha256 and family,
    write it to root/LICENSE; return the CASE.json provenance fields."""
    import hashlib
    import urllib.request
    n = NOTICES[name]
    url, sha = n["text"]
    with urllib.request.urlopen(url, timeout=30) as r:
        text = r.read()
    if hashlib.sha256(text).hexdigest() != sha:
        raise SystemExit(f"{name}: {url} changed (sha256 mismatch) - not written")
    head = text.lstrip().split(b"\n", 1)[0].strip()
    if head != {"apache-2.0": b"Apache License", "mit": b"MIT License"}[lic.lower()]:
        raise SystemExit(f"{name}: {url} is not the {lic} text ({head!r})")
    (root / "LICENSE").write_bytes(text)
    return {"license_source": "dataset card", "license_notice": n["notice"],
            "license_file": f"LICENSE (verbatim from {url}, sha256 {sha[:16]})",
            "attribution": n["attribution"]}


# Redistribution grants for sources whose card YAML carries no license field.
# A grant is accepted only with the publisher's own statement at the pinned
# revision (checked verbatim at build time) and the license text it names.
LICENSE_GRANTS = {
    "fastumi": dict(
        license="apache-2.0",
        readme_line="This dataset is released under the **Apache-2.0 License**.",
        # the paper names this GitHub repo as the release link; its README
        # names the Hub repo as the dataset and carries the same grant
        license_file=("MrKeee/FastUMI-100K", "4d2bb3f23b5209bf34d1db457118969b6d3d4179", "LICENSE"),
        evidence=[
            "Hub README.md at the pinned revision, section 'License': "
            "'This dataset is released under the **Apache-2.0 License**.' (card YAML has no license field)",
            "arXiv:2510.08022 (FastUMI-100K, Liu et al., Shanghai AI Laboratory et al.) abstract: "
            "'The source code and dataset will be released in this link' -> https://github.com/MrKeee/FastUMI-100K",
            "github.com/MrKeee/FastUMI-100K@4d2bb3f23b52: README links "
            "https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot as [dataset] and states the "
            "same grant; LICENSE is the unmodified Apache-2.0 text; no NOTICE file in either repo",
        ],
        attribution="FastUMI-100K: Advancing Data-driven Robotic Manipulation with a Large-scale UMI-style "
                    "Dataset - Kehui Liu, Zhongjie Jia, Yang Li, Zhaxizhuoma, Pengan Chen, Song Liu, Xin Liu, "
                    "Pingrui Zhang, Haoming Song, Xinyi Ye, Nieqing Cao, Zhigang Wang, Jia Zeng, Dong Wang, "
                    "Yan Ding, Bin Zhao, Xuelong Li. arXiv:2510.08022, 2025. "
                    "https://huggingface.co/datasets/IPEC-COMMUNITY/FastUMI_100k_lerobot",
    ),
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


def _verified_grant(name: str, repo: str, rev: str, key: str) -> tuple[dict, bytes]:
    """The grant's README statement must be in the README at rev, verbatim,
    and the license file must be the license it names; else no packing."""
    import urllib.request
    g = LICENSE_GRANTS[key]
    _, fs = B._api()
    readme = fs.cat_file(f"datasets/{repo}@{rev}/README.md").decode()
    if g["readme_line"] not in readme:
        raise SystemExit(f"{name}: the README at {rev[:12]} no longer states {g['readme_line']!r} - not packed")
    gh, sha, path = g["license_file"]
    with urllib.request.urlopen(f"https://raw.githubusercontent.com/{gh}/{sha}/{path}", timeout=30) as r:
        text = r.read()
    if g["license"] == "apache-2.0" and not text.lstrip().startswith(b"Apache License\n                           Version 2.0"):
        raise SystemExit(f"{name}: {gh}@{sha[:12]}/{path} is not the Apache-2.0 text - not packed")
    return g, text


def _runs(eps) -> list[list[int]]:
    out: list[list[int]] = []
    for e in sorted(eps):
        if out and e == out[-1][1] + 1:
            out[-1][1] = e
        else:
            out.append([e, e])
    return out


def truth_v21(info: dict, episodes: list[int], listing: set[str]) -> dict:
    """v2.1 ground truth: render video_path for every (episode, camera) and
    look the path up in the listing. No declared shard and no segment
    columns exist in v2.1, so those two checks have nothing to compare."""
    cs = int(info.get("chunks_size") or 1000)
    cams = [k for k, f in (info.get("features") or {}).items() if isinstance(f, dict) and f.get("dtype") == "video"]
    files = {}
    for cam in cams:
        gone = {e: info["video_path"].format(episode_chunk=e // cs, video_key=cam, episode_index=e)
                for e in episodes}
        gone = {e: p for e, p in gone.items() if p not in listing}
        if gone:
            files[cam] = {"missing_files": sorted(gone.values()), "episodes": sorted(gone)}
    from collections import Counter
    return {"pointers": [], "segments": {}, "video_files": files,
            "duplicate_catalogue": sorted(e for e, n in Counter(episodes).items() if n > 1),
            "conflicting_duplicates": []}


def build_v21(name: str, spec: dict, repo: str, rev: str) -> dict:
    api, fs = B._api()
    sub = spec.get("subdir", "")
    at = f"datasets/{repo}@{rev}" + (f"/{sub}" if sub else "")
    lic, grant, license_text = B.license_of(repo, rev), None, None
    if (lic or "").lower() not in B.REDISTRIBUTABLE:
        if "license_grant" not in spec:
            raise SystemExit(f"{name}: {repo} declares license {lic!r} - not packed (no redistribution permission)")
        grant, license_text = _verified_grant(name, repo, rev, spec["license_grant"])
        lic = grant["license"]
    info = json.loads(fs.cat_file(f"{at}/meta/info.json"))
    assert str(info.get("codebase_version")).startswith("v2."), info.get("codebase_version")
    all_eps = [json.loads(ln)["episode_index"] for ln in fs.cat_file(f"{at}/meta/episodes.jsonl").splitlines()
               if ln.strip()]
    listing = B.sub_listing(repo, rev, sub)
    paths = {p for p, _ in listing}
    keep = {e for lo, hi in spec["episodes"][1] for e in all_eps if lo <= e < hi}
    root = B.extract_v21(repo, rev, name, keep, sub, listing, license_text,
                         all_videos=not name.endswith("_clean"))
    truth = truth_v21(info, [e for e in all_eps if e in keep], paths)
    # evidence over the WHOLE dataset folder, not the extract
    whole = truth_v21(info, all_eps, paths)
    cs = int(info.get("chunks_size") or 1000)
    data_named = {info["data_path"].format(episode_chunk=e // cs, episode_index=e) for e in all_eps}
    lower = {p.lower() for p in paths}
    import hashlib
    full_text = "".join(f"{p}\t{s}\n" for p, s in listing)
    evidence: dict = {
        "listing": {"method": "HfApi.list_repo_tree(recursive=True, path_in_repo=subdir) - paginated; "
                              "dataset_info siblings stop at 100000 entries for this repo",
                    "files": len(listing), "videos": sum(p.startswith("videos/") for p in paths),
                    "data": sum(p.startswith("data/") for p in paths),
                    "meta": sorted(p for p in paths if p.startswith("meta/")),
                    "other": sorted(p for p in paths if not p.startswith(("meta/", "data/", "videos/"))),
                    "sha256_of_full_listing": hashlib.sha256(full_text.encode()).hexdigest(),
                    "listing_txt": ("every meta/ entry and the kept episodes' data and video entries"
                                    if name.endswith("_clean") else
                                    "every meta/ and videos/ entry; data/ entries for the kept episodes only")},
        "whole_dataset": {
            "catalogued_episodes": len(all_eps), "info_total_episodes": info.get("total_episodes"),
            "info_total_videos": info.get("total_videos"),
            "data_files_named_and_present": len(data_named & paths), "data_files_named": len(data_named),
            "video_files_missing": {c: len(v["missing_files"]) for c, v in whole["video_files"].items()},
            "missing_episode_runs": {c: _runs(v["episodes"]) for c, v in whole["video_files"].items()},
            "missing_with_case_insensitive_match": sum(
                p.lower() in lower for v in whole["video_files"].values() for p in v["missing_files"]),
        },
    }
    miss_all = [p for v in whole["video_files"].values() for p in v["missing_files"]]
    if miss_all:
        # a second API, per path, on a spread of missing paths plus present controls
        pre = f"{sub}/" if sub else ""
        sample = sorted(set(miss_all[:: max(1, len(miss_all) // 12)] + miss_all[-1:]))
        controls = sorted(p for p in paths if p.startswith("videos/"))[:1] + sorted(
            p for p in paths if p.startswith("videos/"))[-1:]
        hit = {x.path[len(pre):] for x in api.get_paths_info(repo, [pre + p for p in sample + controls],
                                                               repo_type="dataset", revision=rev)}
        assert not (hit & set(sample)) and set(controls) <= hit, (hit & set(sample), controls)
        refs = api.list_repo_refs(repo, repo_type="dataset")
        evidence["second_api"] = {"method": "HfApi.get_paths_info at the revision",
                                  "missing_paths_checked": sample, "found": [], "present_controls_found": controls}
        evidence["refs"] = {"branches": {b.name: b.target_commit for b in refs.branches},
                            "tags": [t.name for t in refs.tags]}
    doc = {
        "source": repo, "subdir": sub, "revision": rev, "license": lic,
        **({"license_source": "publisher grant (card YAML has no license field)",
            "license_evidence": grant["evidence"], "attribution": grant["attribution"],
            "license_file": "LICENSE (verbatim from github.com/{}@{}/{})".format(*grant["license_file"])}
           if grant else {"license_source": "dataset card"}),
        "built": f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d}",
        "kept": {"episodes": len(keep), "of": len(all_eps), "ranges": [list(r) for r in spec["episodes"][1]],
                 "note": "v2.1: meta/info.json, meta/tasks.jsonl verbatim; meta/episodes.jsonl reduced to the "
                         "kept episodes' lines (byte-for-byte); each kept episode's data parquet reduced to "
                         "episode_index and frame_index; episodes_stats.jsonl and every video omitted; "
                         "listing.txt holds every meta/ and videos/ file of the folder at the revision, kept "
                         "episodes or not, and the kept episodes' data files"},
        "not_applicable": {
            "episode-file-pointers": "v2.1 names one parquet per episode: no declared shard to compare",
            "camera-coverage": "v2.1 records no video segments and the videos are not fetched",
        },
        "truth": truth, "evidence": evidence,
    }
    (root / "CASE.json").write_text(json.dumps(doc, indent=1))
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    print(f"{name}: {repo}/{sub}@{rev[:12]} ({lic}) | {len(keep)}/{len(all_eps)} episodes, {size // 1024} KB | "
          f"missing video files {sum(len(v['missing_files']) for v in truth['video_files'].values())} kept, "
          f"{len(miss_all)} in the whole folder")
    return doc


def build(name: str) -> dict:
    spec = CASES[name]
    repo, rev = spec["repo"], _full_revision(spec["repo"], spec["revision"])
    if spec.get("subdir") or spec["episodes"] and spec["episodes"][0] == "ranges":
        return build_v21(name, spec, repo, rev)
    lic = B.license_of(repo, rev)
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
        "source": repo, "revision": rev, "license": lic, **notice_fields(name, root, lic),
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


def refresh_notices(name: str) -> None:
    """Write LICENSE and the CASE.json license/attribution fields of an
    existing v3 extract without re-extracting it (no data is fetched); the
    card license at the pinned revision must still match CASE.json."""
    root = HERE / name
    doc = json.loads((root / "CASE.json").read_text())
    lic = B.license_of(doc["source"], doc["revision"])
    if (lic or "").lower() != (doc["license"] or "").lower():
        raise SystemExit(f"{name}: card license at {doc['revision'][:12]} is {lic!r}, CASE.json says {doc['license']!r}")
    fields = notice_fields(name, root, lic)
    out: dict = {}
    for k, v in doc.items():
        if k not in fields:
            out[k] = v
        if k == "license":
            out.update(fields)
    (root / "CASE.json").write_text(json.dumps(out, indent=1))
    print(f"{name}: {fields['license_file']}; attribution written")


def main(names: list[str]) -> int:
    if names[:1] == ["--notices"]:
        for n in names[1:] or NOTICES:
            refresh_notices(n)
        return 0
    for n in names or CASES:
        build(n)
    return 0

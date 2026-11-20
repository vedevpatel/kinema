#!/usr/bin/env python3
"""Nightly leaderboard: audit the most-downloaded LeRobot datasets on the
Hugging Face Hub and render a static site (docs/ -> GitHub Pages).

Usage: python scripts/leaderboard.py [--limit 50] [--out docs] [--max-episodes 50]
       python scripts/leaderboard.py --rescore [--out docs]

--rescore recomputes every stored score from its stored per-check verdicts
under the current score model (kinema.report.SCORE_MODEL) and re-renders
the site - no downloads, no check re-runs. Every normal run rescores the
board it resumes from the same way, so a score-model change lands in one run.

Datasets whose parquet payload exceeds --size-cap-mb are listed as
"not audited (too large for the sampled audit)" instead of being
downloaded - honesty over coverage for v0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from kinema.checks import ALL_CHECKS, CheckConfig, Severity  # noqa: E402
from kinema.dataset import DatasetError, _meta_shard_map, _target_video_files, load_dataset  # noqa: E402
from kinema.engine import engine_version  # noqa: E402
from kinema.report import (  # noqa: E402
    SCORE_MODEL,
    DatasetReport,
    report_from_dict,
    report_to_dict,
    rescore_report_dict,
)

SIZE_CAP_MB_DEFAULT = 200

CSS = """
:root { color-scheme: light dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; max-width: 1080px;
       margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #8884; }
th { position: sticky; top: 0; background: Canvas; }
tr:hover td { background: #8881; }
.badge { display: inline-block; padding: .05rem .5rem; border-radius: 1rem;
         font-size: .8rem; font-weight: 600; }
.b-PASS { background: #2da44e22; color: #2da44e; }
.b-INFO { background: #0969da22; color: #0969da; }
.b-WARN { background: #bf870022; color: #bf8700; }
.b-FAIL { background: #cf222e22; color: #cf222e; }
.b-SKIP, .b-NA { background: #8882; color: #888; }
.score { font-weight: 700; }
.dim { color: #888; }
a { color: #0969da; }
h1 { margin-bottom: 0; }
"""

VERDICTS = ["PASS", "INFO", "WARN", "FAIL", "SKIP"]


def esc(s) -> str:
    return html.escape(str(s))


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{CSS}</style>{body}"
    )


def badge(v: str) -> str:
    return f'<span class="badge b-{v}">{v}</span>'


def _probe_nested(api, repo_id: str, seats: int) -> list[dict]:
    """Board entries for the LeRobot datasets nested inside one repo.

    Any directory holding meta/info.json is a LeRobot root - the same gate
    the root-level probe applies. Discovery is a breadth-first walk with
    batched existence checks (a full recursive listing does not fit giant
    repos: X-Embodiment-Sim's tree is 200k+ items); sizing reads each
    root's data/ subtree only, and an over-cap or unreadable root sizes as
    0 and ranks last rather than costing the repo its rows. Gated repos contribute nothing (their
    subpath downloads 403 like the root's). Rows carry repo-level download
    counts (HF exposes no per-directory stats) marked with repo_scope, and
    ride the flagship lane so repo-level numbers never pollute the
    download-ranked top lane."""
    try:
        from huggingface_hub.hf_api import RepoFile
    except ImportError:  # older hub: duck-typing fallback below
        RepoFile = None
    try:
        info = api.dataset_info(repo_id)
    except Exception as e:  # noqa: BLE001 - e.g. a 429 after a heavy audit
        print(f"{repo_id}: dataset_info failed ({type(e).__name__}) - "
              "skipping nested seats this run", flush=True)
        return []
    if getattr(info, "gated", None):
        return []
    calls = 0

    def is_file(item) -> bool:
        if RepoFile is not None:
            return isinstance(item, RepoFile)
        return getattr(item, "size", None) is not None

    # breadth-first root discovery: list a directory, batch-check each child
    # for meta/info.json, descend into non-root children (containers like
    # Teleop-Sim's LeRobot/ dir) up to the depth cap
    roots: list[str] = []
    frontier = [""]
    while frontier:
        current = frontier.pop(0)
        try:
            items = list(api.list_repo_tree(repo_id, repo_type="dataset",
                                            path_in_repo=current or None,
                                            recursive=False))
        except Exception as e:  # noqa: BLE001 - one repo must not stall discovery
            print(f"{repo_id}: nested listing failed at {current or '/'!r} "
                  f"({type(e).__name__}) - skipping nested seats this run", flush=True)
            return []
        calls += 1
        cand = []
        for item in items:
            if is_file(item):
                continue
            p = item.path
            parts = p.split("/")
            if any(seg in _NESTED_SKIP_DIRS or seg.startswith(".") for seg in parts):
                continue
            cand.append(p)
        if not cand:
            continue
        try:
            found = api.get_paths_info(
                repo_id, [f"{c}/meta/info.json" for c in cand], repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"{repo_id}: nested existence check failed ({type(e).__name__}) - "
                  "skipping nested seats this run", flush=True)
            return []
        calls += 1
        have = {o.path[: -len("/meta/info.json")] for o in found if is_file(o)}
        for c in cand:
            depth = len(c.split("/"))
            if c in have:
                if depth <= _NESTED_MAX_DEPTH:
                    roots.append(c)
            elif depth < _NESTED_MAX_DEPTH:
                frontier.append(c)
        if calls > _NESTED_MAX_CALLS:
            print(f"{repo_id}: nested probe over {_NESTED_MAX_CALLS} API calls - "
                  "skipping nested seats this run", flush=True)
            return []
    if not roots:
        return []
    # seat order: subpath data bytes, from each root's data/ subtree only
    # (videos/ is where the item count explodes). Each root's listing is
    # capped: a giant repo (X-Embodiment-Sim: 132 roots, 100k+ data files
    # in total) must not stall discovery, so an over-cap or failed root
    # sizes as 0 and ranks last rather than costing the repo its rows.
    data_bytes: dict[str, int] = {}
    for r in sorted(roots):
        total = 0
        count = 0
        try:
            for item in api.list_repo_tree(repo_id, repo_type="dataset",
                                           path_in_repo=f"{r}/data", recursive=True):
                count += 1
                if count > _NESTED_ROOT_MAX_ITEMS:
                    print(f"{repo_id}: {r!r} data listing over "
                          f"{_NESTED_ROOT_MAX_ITEMS} items - sized as 0, "
                          "ranking it last", flush=True)
                    total = 0
                    break
                total += getattr(item, "size", None) or 0
        except Exception as e:  # noqa: BLE001 - one root must not stall the probe
            print(f"{repo_id}: data listing failed for {r!r} ({type(e).__name__}) - "
                  "sized as 0, ranking it last", flush=True)
            total = 0
        calls += 1
        data_bytes[r] = total
        if calls > _NESTED_MAX_CALLS:
            print(f"{repo_id}: nested probe over {_NESTED_MAX_CALLS} API calls - "
                  "remaining roots sized as 0", flush=True)
            break
    downloads = getattr(info, "downloads", 0) or 0
    likes = getattr(info, "likes", 0) or 0
    ranked = sorted(roots, key=lambda r: (-data_bytes[r], r))[:seats]
    return [{"id": f"{repo_id}//{sub}", "downloads": downloads, "likes": likes,
             "data_bytes": data_bytes[sub], "subpath": sub, "repo_scope": True}
            for sub in ranked]


RLDS_BUCKETS = [  # precedence order: the first bucket holding REAL data wins
    ("gdm-robotics-open-x-embodiment", ""),  # Open-X-official corpus
    ("gresearch", "robotics/"),  # Google Research mirror + extra robot sets
]
_RLDS_LANE_MAX_CALLS = 2500  # GCS JSON-API calls per bucket (gresearch's shard
# listings paginate: ~17 calls/dataset; 600 starved the pass at the d's)


def _rlds_shards_complete(ds) -> bool:
    """Every declared split's full shard set is present in this copy."""
    from kinema.checks.rlds_checks import _shard_parts

    lengths = getattr(ds, "shard_lengths", None)
    if not lengths:
        return True  # nothing declared to judge against: keep plain precedence
    if getattr(ds, "ambiguous_splits", None):
        return False
    try:
        parts = _shard_parts(ds)
    except Exception:  # noqa: BLE001 - completeness is a tiebreak, never a lane killer
        return True
    return all(len(parts.get(split, {})) == len(n)
               for split, n in lengths.items())


def probe_rlds_lane(workers: int = 8) -> list[dict]:
    """Board entries for the RLDS corpus on GCS (not Hugging Face).

    Discovery is an anonymous bucket listing per bucket in RLDS_BUCKETS;
    each dataset prefix is probed (newest TFDS semver, dataset_info.json +
    features.json). A dataset boards ONCE under rlds/<name>: the first
    bucket in precedence order holding its COMPLETE declared shard set claims
    the name; a partial copy boards only when no bucket has a complete one
    (the gdm and gresearch copies of a mirrored dataset DIVERGE, so the
    winning bucket is recorded on the row for reproducibility).
    A bucket-level listing failure zeroes the WHOLE lane, not just that
    bucket: falling through to a lower bucket on a transient outage would
    silently swap a mirrored dataset's bytes, which is worse than a missing
    row - the resume prune guard keeps the stored rlds board when the lane
    comes back empty. Per-dataset probe failures still skip just that
    dataset. Rows carry no downloads/likes (GCS exposes neither) and ride
    their own lane so zero-counts never pollute the download-ranked lanes.

    Per-prefix probes run through _ordered_pool (the serial pass measured at
    6-7 min of every run, 2026-11-18). Buckets stay serial in precedence
    order and results are consumed in sorted-prefix order, so claimed-set
    fall-through, board membership, and the call-budget trip point are
    identical to the serial pass. Each probe gets its own GcsClient (the
    constructor is stateless besides the call counter, which the consumer
    reads after the task finishes); the budget still bounds pathological
    pagination, with an overshoot of at most `workers` in-flight probes."""
    from kinema.rlds import GcsClient, RldsDataset

    out: list[dict] = []
    claimed: set[str] = set()
    partial: dict[str, dict] = {}
    try:
        for bucket, base in RLDS_BUCKETS:
            client = GcsClient(bucket)
            prefixes, _ = client.list(prefix=base, delimiter="/")
            calls = client.calls

            def probe_one(p: str, _bucket=bucket):
                task_client = GcsClient(_bucket)
                name = p.rstrip("/").split("/")[-1]
                try:
                    ds = RldsDataset.probe(task_client, p)
                    if not ds.shards:
                        return None, task_client.calls  # metadata-only cannot claim a mirror seat
                except Exception as e:  # noqa: BLE001 - one dataset must not stall the lane
                    print(f"rlds/{name}: probe failed in {_bucket} "
                          f"({type(e).__name__}) - skipping", flush=True)
                    PROBE_ERRORS.add(f"rlds/{name}")
                    return None, task_client.calls
                return {"id": f"rlds/{name}", "rlds": True, "rlds_prefix": p,
                        "rlds_bucket": _bucket,
                        "version": ds.version, "downloads": 0, "likes": 0,
                        "data_bytes": ds.data_bytes,
                        "_complete": _rlds_shards_complete(ds)}, task_client.calls

            todo = [p for p in sorted(prefixes)
                    if p.rstrip("/").split("/")[-1] not in claimed]
            stream = _ordered_pool(probe_one, todo, max(1, workers))
            try:
                for item in stream:
                    entry, probe_calls = item if item is not None else (None, 0)
                    calls += probe_calls
                    if entry is not None:
                        name = entry["rlds_prefix"].rstrip("/").split("/")[-1]
                        if entry.pop("_complete"):
                            claimed.add(name)
                            partial.pop(name, None)
                            out.append(entry)
                        else:
                            # a partial copy claims the name only if no later
                            # bucket holds the complete declared shard set
                            partial.setdefault(name, entry)
                    if calls > _RLDS_LANE_MAX_CALLS:
                        print(f"rlds lane over {_RLDS_LANE_MAX_CALLS} API calls in "
                              f"{bucket} - boarding what we have", flush=True)
                        break
            finally:
                stream.close()
        out.extend(e for n, e in sorted(partial.items()) if n not in claimed)
        return out
    except Exception as e:  # noqa: BLE001 - the lane dies quiet, the board lives
        print(f"rlds lane failed ({type(e).__name__}) - zero rlds rows this run "
              "(a bucket outage must not swap mirrored datasets' bytes)", flush=True)
        return []


def audit_rlds(entry: dict) -> DatasetReport:
    """The rlds audit: metadata checks over the GCS listing, no downloads."""
    from kinema.checks.rlds_checks import run_rlds_checks
    from kinema.rlds import GcsClient, RldsDataset

    client = GcsClient(entry.get("rlds_bucket", RLDS_BUCKETS[0][0]))
    ds = RldsDataset.probe(client, entry["rlds_prefix"])
    mirror, mirror_bucket = _probe_rlds_mirror(entry)
    results = run_rlds_checks(ds, client, mirror=mirror, mirror_bucket=mirror_bucket)
    rep = DatasetReport(
        source=entry["id"],
        format_version=f"rlds-{ds.version}",
        episodes=ds.episodes,
        frames=0,  # TFDS metadata declares no frame counts
        fps=None,
        results=results,
        video_audited=VIDEO_AUDIT_MARK,
    )
    rep.audit_meta = {"revision": rlds_revision(ds), "sample": None}
    return rep


def _probe_rlds_mirror(entry: dict):
    """The same dataset in the OTHER rlds bucket, for the drift check - or
    (None, "") when it has no copy there or the probe fails: a mirror hiccup
    must never fail the audit of the row's own bucket."""
    from kinema.rlds import GcsClient, RldsDataset

    name = entry["id"].removeprefix("rlds/")
    own = entry.get("rlds_bucket", RLDS_BUCKETS[0][0])
    for bucket, base in RLDS_BUCKETS:
        if bucket == own:
            continue
        try:
            return RldsDataset.probe(GcsClient(bucket), f"{base}{name}/"), bucket
        except Exception:  # noqa: BLE001 - no copy (or a hiccup): no drift signal
            return None, ""
    return None, ""


# ids whose discovery probe ERRORED this run (rate limit, timeout, outage):
# their absence from the listing is unknown, not evidence - reconcile_discovery
# must not count it as a miss. Repo ids for Hub rows, "rlds/<name>" for RLDS,
# and nested base repos (a failed nested probe hides all of its seats).
PROBE_ERRORS: set[str] = set()
_PROBE_RETRY_SLEEPS = (30.0, 90.0)


def _transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("429", "too many requests", "timeout", "timed out",
                                   "connection", "502", "503", "504", "temporarily"))


def _retrying(fn, what: str):
    """Call fn(); on a transient error back off and retry. Re-raises the last
    error, so callers can tell 'errored' apart from 'not a LeRobot dataset'."""
    import time

    for i, pause in enumerate((*_PROBE_RETRY_SLEEPS, None)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            if pause is None or not _transient(exc):
                raise
            print(f"{what}: {type(exc).__name__} - retry {i + 1} in {pause:.0f}s", flush=True)
            time.sleep(pause)
    raise RuntimeError("unreachable")


def _probe_dataset(api, repo_id: str, downloads: int, likes: int,
                   skip_gated: bool = False) -> dict | None:
    """The board entry for one Hub dataset, or None when it has no LeRobot
    metadata (meta/info.json) - the same gate both listing lanes apply.
    With skip_gated, gated repos also return None: info.json is public on
    gated repos, so the plain probe waves them onto the board and the audit
    only hits the 403 at fetch time. Lanes that cannot reallocate the seat
    (top, new) keep showing gated rows; the flagship lane skips them so the
    seat goes to the next auditable dataset."""
    try:
        info = _retrying(lambda: api.dataset_info(repo_id, files_metadata=True), repo_id)
    except Exception as exc:  # noqa: BLE001
        if _transient(exc):
            PROBE_ERRORS.add(repo_id)  # unknown, not absent
        return None
    if skip_gated and getattr(info, "gated", None):
        return None
    sib = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
    if len(sib) >= 100_000:
        # dataset_info's siblings cap: re-list with pagination so data_bytes
        # and the meta/info.json test see the whole repo (a data/ tree over
        # 100k files would otherwise hide meta/ and shrink the byte count)
        from huggingface_hub import list_repo_tree

        try:
            sib = _retrying(lambda: [(f.path, f.size or 0)
                                     for f in list_repo_tree(repo_id, repo_type="dataset",
                                                             recursive=True)
                                     if hasattr(f, "size")], f"{repo_id} listing")
        except Exception as exc:  # noqa: BLE001 - don't advertise incomplete inventory
            print(f"{repo_id}: complete discovery listing failed ({type(exc).__name__})",
                  flush=True)
            PROBE_ERRORS.add(repo_id)  # the repo exists; its listing errored
            return None
    if "meta/info.json" not in [n for n, _ in sib]:
        return None
    return {
        "id": repo_id,
        "downloads": downloads,
        "likes": likes,
        "data_bytes": sum(sz for n, sz in sib if n.startswith("data/")),
    }


# flagship publishers whose LeRobot-format datasets never surface under the
# "lerobot" search (their repos don't carry the name): a capped number of
# seats per org onboards their top datasets without flooding the board with
# one publisher's per-task sets (RoboCOIN alone has 810, which would evict
# most of the ecosystem under a pure download ranking)
# physical-intelligence (pi0/pi0.5: libero 51k downloads) and yaak-ai
# (L2D 87k) ship LeRobot format but never surface under the "lerobot"
# search either - 2026-11-18 recon
FLAGSHIP_ORG_SEATS = {"RoboCOIN": 25, "unitreerobotics": 15,
                      "physical-intelligence": 2, "yaak-ai": 2}
# individual flagship datasets the search misses for the same reason
FLAGSHIP_SEEDS = ("GT-111/agibotworld_alpha_v3_eef",)
# datasets nested inside a larger repo - a LeRobot root below the repo top,
# addressed as repo//subpath. Curated seats per repo, riding the flagship
# lane: verified structures only, no automatic nested discovery (spec and
# recon: docs say agibot-world/AgiBotWorld2026 is NOT one - raw tarballs)
NESTED_REPO_SEATS = {
    "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim": 10,
    "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim": 8,
    "Traly/AgiBotWorld-Alpha-lerobot": 8,
    "Traly/RoboMIND-lerobot": 4,
}
# bounds on one repo's nested probe: a giant repo skips and logs instead of
# stalling discovery (X-Embodiment-Sim's full tree is 200k+ items)
_NESTED_ROOT_MAX_ITEMS = 20_000  # files in one root's data/ listing before it sizes as 0
_NESTED_MAX_CALLS = 500  # Hub API calls per repo probe
_NESTED_MAX_DEPTH = 3  # subpath components (X-Embodiment-Sim is 1, Teleop-Sim 2)
# directory names that are never dataset roots
_NESTED_SKIP_DIRS = ("assets", "simulation", "HDF5", "data", "videos", "meta")


def _split_subpath(repo_id: str) -> tuple[str, str]:
    """repo//subpath -> (repo, subpath); '//' cannot appear in a Hub repo id."""
    if "//" in repo_id:
        repo, sub = repo_id.split("//", 1)
        return repo, sub.strip("/")
    return repo_id, ""


def _pref(prefix: str, name: str) -> str:
    """A root-relative path back to its repo-relative form under a subpath."""
    return f"{prefix}/{name}" if prefix else name


def _merge_lanes(top: list[dict], recent: list[dict], limit: int, new_lane: int,
                 flagship: list[dict] | None = None) -> list[dict]:
    """Top-downloaded board (lane "top"), a capped block of flagship
    publisher seats (lane "flagship"), plus a probation lane of the newest
    uploads (lane "new"): fresh datasets get audited before they are popular,
    which is exactly when a QA flag is cheapest to act on. No duplicates;
    the lane tag rides the entry into data.json."""
    out: list[dict] = []
    seen: set[str] = set()
    for d in top[:limit]:
        out.append({**d, "lane": "top"})
        seen.add(d["id"])
    for d in flagship or []:
        if d["id"] in seen:
            continue
        out.append({**d, "lane": "flagship"})
        seen.add(d["id"])
    for d in recent:
        if sum(1 for r in out if r["lane"] == "new") >= new_lane:
            break
        if d["id"] in seen:
            continue
        out.append({**d, "lane": "new"})
        seen.add(d["id"])
    return out


def _ordered_pool(fn, items, workers: int):
    """Yield fn(item) in INPUT order with up to `workers` calls in flight.

    Discovery probes are independent network calls; running them through a
    small pool takes the serial discovery phase (measured ~8 min of every
    run, 2026-11-18) down to about a minute without changing WHAT boards:
    consumers early-stop (lane caps, the rlds call budget) on the in-order
    result stream, so membership and ordering match the old serial loops
    exactly. Close the generator on early exit (try/finally) so the pool
    does not linger. A future that raises yields None: one bad probe must
    not kill discovery (probe functions already catch and return None on
    their own expected failures).
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if workers <= 1:
        yield from (fn(item) for item in items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        it = iter(items)
        pending: dict = {}
        results: dict = {}
        idx = 0
        exhausted = False

        def submit_more() -> None:
            nonlocal idx, exhausted
            while not exhausted and len(pending) < workers:
                try:
                    item = next(it)
                except StopIteration:
                    exhausted = True
                    return
                pending[pool.submit(fn, item)] = idx
                idx += 1

        submit_more()
        next_out = 0
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for f in done:
                i = pending.pop(f)
                try:
                    results[i] = f.result()
                except Exception:  # noqa: BLE001 - one probe must not kill discovery
                    results[i] = None
            submit_more()
            while next_out in results:
                yield results.pop(next_out)
                next_out += 1


def find_lerobot_datasets(limit: int, new_lane: int = 0, workers: int = 8) -> list[dict]:
    from huggingface_hub import HfApi

    api = HfApi()
    top: list[dict] = []
    candidates = list(api.list_datasets(search="lerobot", sort="downloads",
                                        limit=limit * 3))
    stream = _ordered_pool(
        lambda d: _probe_dataset(api, d.id, d.downloads or 0,
                                 getattr(d, "likes", 0) or 0),
        candidates, workers)
    try:
        for e in stream:
            if e:
                top.append(e)
            if len(top) >= limit:
                break
    finally:
        stream.close()
    recent: list[dict] = []
    if new_lane:
        candidates = list(api.list_datasets(search="lerobot", sort="created_at",
                                            limit=new_lane * 3))
        stream = _ordered_pool(
            lambda d: _probe_dataset(api, d.id, d.downloads or 0,
                                     getattr(d, "likes", 0) or 0),
            candidates, workers)
        try:
            for e in stream:
                if e:
                    recent.append(e)
                if len(recent) >= new_lane:
                    break
        finally:
            stream.close()
    flagship: list[dict] = []
    for org, seats in FLAGSHIP_ORG_SEATS.items():
        candidates = list(api.list_datasets(author=org, sort="downloads",
                                            limit=seats * 4))
        # skip_gated: a gated repo would take the seat and show "gated"
        # forever; the next auditable dataset makes better use of it
        stream = _ordered_pool(
            lambda d: _probe_dataset(api, d.id, d.downloads or 0,
                                     getattr(d, "likes", 0) or 0, skip_gated=True),
            candidates, workers)
        got = 0
        try:
            for e in stream:
                if e:
                    flagship.append(e)
                    got += 1
                if got >= seats:
                    break
        finally:
            stream.close()
    for seed in FLAGSHIP_SEEDS:
        try:
            d = api.dataset_info(seed)
        except Exception:
            continue
        e = _probe_dataset(api, seed, getattr(d, "downloads", 0) or 0,
                           getattr(d, "likes", 0) or 0)
        if e:
            flagship.append(e)

    def probe_one_nested(kv) -> list[dict]:
        nested_repo, seats = kv
        try:
            return _probe_nested(api, nested_repo, seats)
        except Exception as e:  # noqa: BLE001 - one repo must not stall discovery
            print(f"{nested_repo}: nested probe failed ({type(e).__name__}) - skipping",
                  flush=True)
            PROBE_ERRORS.add(nested_repo)
            return []

    if NESTED_REPO_SEATS:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
                max_workers=min(max(1, workers), len(NESTED_REPO_SEATS))) as pool:
            for entries in pool.map(probe_one_nested, NESTED_REPO_SEATS.items()):
                flagship.extend(entries)
    rlds: list[dict] = []
    try:
        rlds = probe_rlds_lane(workers=workers)
    except Exception as e:  # noqa: BLE001 - defense in depth; probe_rlds_lane logs
        print(f"rlds lane failed ({type(e).__name__}) - zero rlds rows", flush=True)
    return _merge_lanes(top, recent, limit, new_lane, flagship) + [
        {**e, "lane": "rlds"} for e in rlds]


def _classic_resolve_download(repo_id: str, name: str, dest: Path, revision: str | None = None) -> None:
    """Plain HTTP GET on the classic resolve endpoint - the fallback when the
    Xet CAS backend cannot reconstruct a file (upstream 404s)."""
    import os
    import shutil
    import urllib.request

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision or 'main'}/{name}"
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as r, open(out, "wb") as w:
        shutil.copyfileobj(r, w)


def _hub_download(repo_id: str, name: str, dest: Path, revision: str | None = None) -> None:
    from huggingface_hub import hf_hub_download

    try:
        hf_hub_download(repo_id, name, repo_type="dataset", local_dir=str(dest), revision=revision)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        if "reconstruction" in msg or "cas-server" in msg or "Xet" in msg:
            try:
                _classic_resolve_download(repo_id, name, dest, revision)
            except Exception:
                # both endpoints failed: the Hub cannot serve this file at all,
                # so any loader (LeRobot included) fails on it. Name the file -
                # the raw CAS error says only "404" and a hash
                raise DatasetError(
                    f"the Hub cannot serve {name}: Xet storage has no data for it "
                    f"({msg[:160]}) and the classic resolve endpoint failed too - "
                    "the repo needs that file re-uploaded") from None
        else:
            raise


def _meta_episode_indices(root: Path) -> list[int] | None:
    """Episode indices from an already-downloaded meta/, either format."""
    import json as _json

    import pyarrow.parquet as _pq

    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    try:
        if ep_dir.is_dir():
            files = sorted(ep_dir.glob("**/*.parquet"))
            if files:
                idx: list[int] = []
                for f in files:
                    idx.extend(_pq.read_table(f, columns=["episode_index"]).column(0).to_pylist())
                return [int(i) for i in idx]
        if jsonl.exists():
            return [
                int(_json.loads(line)["episode_index"])
                for line in jsonl.read_text().splitlines()
                if line.strip()
            ]
    except Exception:  # noqa: BLE001 - targeting is best-effort
        return None
    return None


def _meta_episode_lengths(root: Path) -> dict[int, int] | None:
    """{episode index: catalogued frame count} from an already-downloaded
    meta/, either format; None when the catalogue has no lengths."""
    import json as _json

    import pyarrow.parquet as _pq

    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    try:
        if ep_dir.is_dir():
            files = sorted(ep_dir.glob("**/*.parquet"))
            if files:
                out: dict[int, int] = {}
                for f in files:
                    t = _pq.read_table(f, columns=["episode_index", "length"]).to_pydict()
                    out.update(zip(map(int, t["episode_index"]), map(int, t["length"])))
                return out
        if jsonl.exists():
            rows = [_json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
            return {int(r["episode_index"]): int(r["length"]) for r in rows}
    except Exception:  # noqa: BLE001 - best-effort, like the index reader
        return None
    return None


def _complete_episodes(table, lengths: dict[int, int] | None):
    """Drop episodes a partial row-group read cut short. A truncated episode
    reads as "metadata says 300 frames but the data holds 100" - a
    dropped-frames FAIL the board itself caused (aloha_pen_uncap_diverse) -
    and every motion check would audit a fragment. Raises DatasetError when
    no episode is whole: the honest result is a failed load, not a verdict."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if not lengths or table.num_rows == 0:
        return table
    vc = pc.value_counts(table.column("episode_index")).to_pylist()
    whole = [int(v["values"]) for v in vc if lengths.get(int(v["values"])) is not None
             and v["counts"] >= lengths[int(v["values"])]]
    unknown = [int(v["values"]) for v in vc if lengths.get(int(v["values"])) is None]
    keep = whole + unknown
    if len(keep) == len(vc):
        return table
    if not whole:
        raise DatasetError("range-read sample holds no complete episode "
                           f"(read {table.num_rows} rows of {len(vc)} episode(s))")
    return table.filter(pc.is_in(table.column("episode_index"),
                                 value_set=pa.array(keep, type=table.schema.field("episode_index").type)))


def _range_read_parquet(repo_id: str, name: str, dest: Path, episodes: list[int] | None = None,
                        min_rows: int = 20000, max_bytes: int = 150 * 1024 * 1024,
                        lengths: dict[int, int] | None = None, fs=None, path: str | None = None,
                        revision: str | None = None) -> None:
    """Materialize row groups of a remote parquet via HTTP range reads (footer
    first, then only the needed row groups) - no full download. Used when every
    data shard exceeds the sample cap. When episode indices are known, prefer
    row groups whose episode_index statistics overlap them, streaming batchwise
    so one giant row group cannot exhaust memory; falls back to leading row
    groups."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = fs or HfFileSystem()
    # open via filesystem+path (not a stream handle) so pyarrow issues true
    # range requests instead of buffering the whole remote file
    at = f"@{revision}" if revision else ""
    pf = pq.ParquetFile(path or f"datasets/{repo_id}{at}/{name}", filesystem=fs)
    # project out pixel columns: frames embedded as images dominate these
    # shards, and v0 runs no visual checks
    columns = [
        n for n in pf.schema_arrow.names
        if "image" not in n.lower() and "video" not in n.lower()
    ] or None

    def rg_bytes(rg) -> int:
        # budget what is actually fetched: the projected columns' compressed
        # bytes. rg.total_byte_size counts the embedded image columns this
        # read never touches - one 100-row group of 4-camera ALOHA frames
        # spent the whole 64 MiB and cut a 300-frame episode to 100 rows
        if columns is None:
            return rg.total_byte_size
        keep = set(columns)
        return sum(rg.column(j).total_compressed_size for j in range(rg.num_columns)
                   if rg.column(j).path_in_schema.split(".")[0] in keep)

    table = None
    if episodes:
        want = sorted(set(episodes))
        groups, total, stats_ok = [], 0, True
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            bounds = None
            for j in range(rg.num_columns):
                col = rg.column(j)
                if col.path_in_schema == "episode_index":
                    if col.is_stats_set and col.statistics is not None:
                        bounds = (col.statistics.min, col.statistics.max)
                    break
            if bounds is None:
                stats_ok = False
                break
            lo, hi = bounds
            if any(lo <= e <= hi for e in want):
                groups.append(i)
                total += rg_bytes(rg)
                if total >= max_bytes:
                    break
        if stats_ok and groups:
            # stream batchwise: a single row group can dwarf memory, so filter
            # to catalogued episodes per batch and stop once the sample
            # covers enough of them
            import pyarrow as pa
            import pyarrow.compute as pc

            value_set = pa.array(want)
            kept, seen = [], set()
            for i in groups:
                for batch in pf.iter_batches(batch_size=1024, row_groups=[i], columns=columns):
                    t = pa.Table.from_batches([batch])
                    t = t.filter(pc.is_in(t.column("episode_index"), value_set=value_set))
                    if t.num_rows:
                        kept.append(t)
                        seen.update(t.column("episode_index").to_pylist())
                    if len(seen) >= 50:
                        break
                if len(seen) >= 50:
                    break
            if kept:
                table = pa.concat_tables(kept)
    if table is None:
        groups, rows, total = [], 0, 0
        for i in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(i)
            groups.append(i)
            rows += rg.num_rows
            total += rg_bytes(rg)
            if rows >= min_rows or total >= max_bytes:
                break
        table = pf.read_row_groups(groups, columns=columns)
    table = _complete_episodes(table, lengths)
    out = dest / name
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)

def download_sample(repo_id: str, cap_bytes: int, dest: Path,
                    unservable: list | None = None,
                    revision: str | None = None) -> tuple[Path, dict[str, int]]:
    """meta/** in full plus data files in name order until cap_bytes.

    v3 shards hold many episodes per file, so the first shard(s) are a
    real sample; v2.1 keeps one parquet per episode, so the first files
    are the first episodes. Either way the audit covers actual data.

    A repo//subpath address samples the dataset nested at subpath: the
    sibling list is filtered to the subpath and every returned path is
    relative to the nested root.

    Returns the sample root (dest, or dest/<subpath> for a nested dataset)
    and the Hub's file-size map keyed by root-relative paths, so the caller
    can budget a video subsample without a second dataset_info call.
    """
    from huggingface_hub import list_repo_tree

    base, prefix = _split_subpath(repo_id)
    root = dest / prefix if prefix else dest
    # list_repo_tree paginates past dataset_info's 100k-sibling cap: DROID,
    # Bridge and Fractal blow through 100,000 files, and the truncated
    # listing made video-file-pointers FAIL pointers the repo really holds
    # (13 false FAILs on the 2026-11-18 board) and skewed shard sampling
    sib = [
        (f.path, f.size or 0)
        for f in list_repo_tree(base, repo_type="dataset", recursive=True,
                                path_in_repo=prefix or None, revision=revision)
        if hasattr(f, "size")  # RepoFile, not RepoFolder
    ]
    if prefix:
        sib = [(n[len(prefix) + 1:], sz) for n, sz in sib if n.startswith(prefix + "/")]
    sizes = dict(sib)
    picked = [n for n, _ in sib if n.startswith("meta/")]
    # meta always comes down first: every branch needs it, both to audit and
    # to target range reads at row groups overlapping catalogued episodes
    for name in picked:
        _hub_download(base, _pref(prefix, name), dest, revision)
    budget = cap_bytes
    data_files = sorted((n, sz) for n, sz in sib if n.startswith("data/"))
    # v3 shard pointers are authoritative: skip stale dual-layout leftovers
    # (episode_*.parquet beside the packed shards) - they would burn the byte
    # cap and hundreds of rate-limited requests for zero audit value
    shard_map = _meta_shard_map(root)
    if shard_map:
        pointed = [(n, sz) for n, sz in data_files if n in set(shard_map.values())]
        if pointed:
            data_files = pointed
    # cap file count too: v2.1 keeps one small parquet per episode, and
    # hundreds of per-file requests are what trips the Hub's rate limits.
    # Candidates are spread evenly over the dataset (not the first files:
    # early episodes are often warm-up recordings).
    chosen = []
    n_files = len(data_files)
    spread = sorted({int(round(x)) for x in
                     (i * (n_files - 1) / 39 for i in range(40))} if n_files > 40 else range(n_files))
    for name, size in (data_files[i] for i in spread):
        if size > budget:
            continue
        chosen.append(name)
        budget -= size
    if not chosen:
        if not data_files:
            raise DatasetError(f"{repo_id}: no data files listed on the Hub")
        smallest, ssize = min(data_files, key=lambda x: x[1])
        try:
            # bound range-read memory: decompressed arrow tables run several
            # times the compressed size, and workers share one process
            _range_read_parquet(
                base, _pref(prefix, smallest), dest,
                episodes=_meta_episode_indices(root),
                max_bytes=min(cap_bytes, 64 * 1024 * 1024),
                lengths=_meta_episode_lengths(root),
                revision=revision,
            )
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            if "reconstruction" in msg or "cas-server" in msg or "Xet" in msg:
                _classic_resolve_download(base, _pref(prefix, smallest), dest, revision)
                return root, sizes
            raise DatasetError(
                f"smallest data shard is {ssize / 1e6:.0f} MB and range-read "
                f"sampling failed: {msg}"
            )
        return root, sizes
    # one data file the Hub cannot serve (arekborucki/bridge_orig_lerobot:
    # episode_000000 of 53k) must not block the audit of the rest: it is
    # recorded for dropped-frames' unreadable-file FAIL - LeRobot's own
    # download of it fails the same way - and the other files still audit
    got = 0
    for name in chosen:
        try:
            _hub_download(base, _pref(prefix, name), dest, revision)
            got += 1
        except DatasetError as e:
            if unservable is None or "cannot serve" not in str(e):
                raise
            unservable.append((name, str(e)[:200]))
    if chosen and not got:
        raise DatasetError(f"{repo_id}: the Hub could serve none of the sampled data files "
                           f"(e.g. {unservable[0][0] if unservable else chosen[0]})")
    return root, sizes


def _clean_reason(msg: str) -> str:
    """Never publish local paths (or anything else environment-specific) to Pages."""
    import re

    return re.sub(r"/[\w./-]*kinema-leaderboard-[\w-]+/?\S*", "<sample>", msg)


# failure taxonomy: permanent states stay on the board as-is, transient
# errors are retried on the next run (see the resume filter in main)
def _classify_reason(reason: str) -> str:
    low = reason.lower()
    if "gatedrepoerror" in low or "gated repo" in low:
        return "gated"
    if "not a lerobot dataset" in low or "no episode metadata" in low:
        return "unsupported"
    return "error"


_STATUS_LABELS = {
    "gated": "gated",
    "unsupported": "unsupported format",
    "error": "audit failed - retry queued",
}


# visual-quality on the board: a spread of the AUDITED episodes' video
# shards, hard-capped so the nightly run's bandwidth stays bounded. The
# targets MUST come from _target_video_files(dest, max_episodes) - the check
# only reads videos of the episodes load_dataset actually loaded, so any
# other episode sample downloads files the verdict ignores. Whole subsample
# or nothing at the cap (a partial pick would bias toward early shards);
# individual files that 404 are skipped - an unfetchable MP4 must never
# kill an otherwise-good audit.
VIDEO_SAMPLE_FILES = 24
VIDEO_CAP_BYTES = 400 * 1000 * 1000
# rows carry video_audited: 0 = pre-video board, 1 = v1 targeting bug (picked
# a 24-episode spread disjoint from the loaded episodes - nothing was
# audited), 2 = v2 lifecycle bug (checks ran after the sample dir was
# deleted, so no video file could ever be seen), 3 = 24-file spread with a
# hard cap (big-shard datasets skipped), 4 = single-shard fallback that
# gave up when the middle target alone was unbudgetable, 5 = current
# (fallback picks the middle BUDGETABLE shard). Lower marks re-audit once,
# then settle.
VIDEO_AUDIT_MARK = 5


def _video_subsample(targets: list[str], sizes: dict[str, int],
                     count: int, cap: int) -> list[str] | None:
    """A spread of up to `count` video files out of `targets`, or None when
    the pick exceeds `cap` bytes (or a size is unknown, which cannot be
    budgeted honestly). The ["videos/**"] sentinel (dataset smaller than the
    episode sample: targeting declined) expands to every videos/ path the
    Hub lists."""
    if targets == ["videos/**"]:
        targets = sorted(p for p in sizes if p.startswith("videos/"))
    if not targets:
        return None
    if len(targets) > count:
        idx = sorted({round(i * (len(targets) - 1) / (count - 1)) for i in range(count)})
        targets = [targets[i] for i in idx]
    if any(p not in sizes for p in targets):
        return None
    return targets if sum(sizes[p] for p in targets) <= cap else None


def _video_single_shard(targets: list[str], sizes: dict[str, int],
                        cap: int) -> str | None:
    """Over-cap fallback: one budgetable shard, picked from the middle of
    the sorted budgetable list (median episode range among the files that
    fit, deterministic - never biased toward the first shard, which would
    skew toward warm-up recordings). None when no target can be budgeted
    honestly (every file over the cap, or none has a known size)."""
    if targets == ["videos/**"]:
        targets = sorted(p for p in sizes if p.startswith("videos/"))
    budgetable = sorted(t for t in targets
                        if sizes.get(t) is not None and sizes[t] <= cap)
    if not budgetable:
        return None
    return budgetable[len(budgetable) // 2]


# Heavy rows (multi-GB on the Hub: GR00T sim tasks load 130-190k frames of
# ~200 dims each) queue next to each other by lane, and 8 of them in flight
# at once exhausted the runner and killed the run (2026-11-19, 06:25 UTC).
# At most HEAVY_SLOTS run concurrently; light rows fill the other workers.
HEAVY_BYTES = 2 * 1024 ** 3
HEAVY_SLOTS = 2


def is_heavy(d: dict) -> bool:
    return (d.get("data_bytes") or 0) >= HEAVY_BYTES


def next_row(it, deferred: list, in_flight: list) -> dict | None:
    """The next row to submit, or None. A heavy row with no free heavy slot
    is deferred (order kept) and the next light row goes instead; deferred
    rows go first once a slot frees."""
    heavy_now = sum(1 for d in in_flight if is_heavy(d))
    if deferred and heavy_now < HEAVY_SLOTS:
        return deferred.pop(0)
    for d in it:
        if is_heavy(d) and heavy_now >= HEAVY_SLOTS:
            deferred.append(d)
            continue
        return d
    return None


# --- dataset change tracking (docs/checks.md, "Change tracking") -------------

def _digest(items) -> str:
    import hashlib

    return hashlib.sha1("\n".join(map(str, items)).encode()).hexdigest()[:12]


def resolve_revision(repo: str) -> str | None:
    """The Hub commit a dataset is at right now; None if it cannot be
    resolved (the audit then records the revision as unknown, never guesses)."""
    from huggingface_hub import HfApi

    try:
        return _retrying(lambda: HfApi().dataset_info(repo).sha, f"{repo} revision")
    except Exception:  # noqa: BLE001
        return None


def revision_staleness(entries: list[dict], current: set[str],
                       workers: int = 8) -> tuple[set[str], set[str]]:
    """(changed, unknown) among the current rows: Hub datasets whose commit
    moved since their audit, and audits that recorded no revision. RLDS and
    gated rows are not probed (no commits / no access). A probe that errors
    leaves the row current - an error is not a change."""
    from concurrent.futures import ThreadPoolExecutor

    rows = [e for e in entries if e["id"] in current and e.get("report")
            and not e.get("rlds") and not e["id"].startswith("rlds/")]
    unknown = {e["id"] for e in rows if not e.get("revision")}
    known = [e for e in rows if e.get("revision")]

    def probe(e):
        return e["id"], e["revision"], resolve_revision(e["id"].split("//", 1)[0])

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(probe, known))
    changed = {i for i, was, now in results if now and now != was}
    return changed, unknown


def write_history(out: Path, entries: list[dict], generated: str) -> dict:
    """Today's snapshot (last run of the UTC day wins) compared with the
    latest earlier day, plus docs/history/index.json - kinema.history."""
    from kinema.history import compare, snapshot

    hist = out / "history"
    hist.mkdir(parents=True, exist_ok=True)
    date = generated[:10]
    checks = sorted({c["id"] for e in entries for c in (e.get("report") or {}).get("checks", [])})
    cur = snapshot(entries, checks, date, generated)
    earlier = sorted(p for p in hist.glob("????-??-??.json") if p.stem < date)
    prev = json.loads(earlier[-1].read_text()) if earlier else None
    doc = {**cur, **compare(prev, cur)}
    tmp = hist / f".{date}.json.tmp"
    tmp.write_text(json.dumps(doc, separators=(",", ":"), default=str))
    tmp.replace(hist / f"{date}.json")
    index_p = hist / "index.json"
    index = json.loads(index_p.read_text()) if index_p.exists() else {"days": {}}
    index["days"][date] = {"compared_to": doc["compared_to"], "summary": doc["summary"],
                           "rows": len(cur["rows"])}
    index_p.write_text(json.dumps(index, indent=1, sort_keys=True))
    return doc


def sample_signature(episodes, videos) -> dict:
    """What the audit actually read: same revision + same engine + same
    signature means a verdict change has no innocent explanation."""
    eps = sorted(int(e) for e in episodes)
    vids = sorted(videos or [])
    return {"episodes": len(eps), "episodes_digest": _digest(eps),
            "videos": len(vids), "videos_digest": _digest(vids) if vids else None}


def rlds_revision(ds) -> str:
    """RLDS buckets have no commits: the revision is a digest of the shard
    inventory and dataset_info the audit read."""
    inv = sorted((str(n), int(sz)) for n, sz in (getattr(ds, "raw_shards", None) or []))
    return "gcs:" + _digest([json.dumps(getattr(ds, "info", {}) or {}, sort_keys=True, default=str), *inv])


# --- robustness gate (docs/checks.md, "Robustness gate") --------------------
# A NEW WARN/FAIL - a check whose verdict rose against the row's current
# board verdict - is re-run under metamorphic probes (kinema.probes) before
# it is published. A verdict that moves under a probe is an artifact of the
# representation, the episode order or the sample, so it publishes as INFO
# with the reason. Each check also admits at most `cap` new WARN/FAIL
# verdicts per run; the rest publish as INFO, held for review in
# held-transitions.json (a workflow artifact, not the public site).
PROBE_MAX_BYTES = 64 << 20  # probes copy the arrays: light, in-RAM loads only
_RANK = {"SKIP": -1, "OK": 0, "PASS": 0, "INFO": 1, "WARN": 2, "FAIL": 3}
_NOT_PROBED = {"visual-quality"}  # reads MP4s again; never representation-bound


class TransitionGate:
    def __init__(self, cap: int):
        import threading

        self.cap = cap
        self.admitted: dict[str, int] = {}
        self.held: list[dict] = []
        self.unstable: list[dict] = []
        self._lock = threading.Lock()

    def admit(self, check_id: str) -> bool:
        with self._lock:
            n = self.admitted.get(check_id, 0)
            if self.cap and n >= self.cap:
                return False
            self.admitted[check_id] = n + 1
            return True

    def note(self, kind: str, row: dict) -> None:
        with self._lock:
            (self.held if kind == "held" else self.unstable).append(row)


def _nightly_probes(ctx, flagged: set[int]):
    from kinema import probes

    out = [("unit-milli", lambda: probes.affine(ctx, 1e-3)),
           ("unit-kilo", lambda: probes.affine(ctx, 1e3)),
           ("order-reversed", lambda: probes.reverse_episodes(ctx))]
    if ctx.truncated:
        # a finding on a sample must survive a smaller sample that still
        # holds the episodes it names
        rest = [e for e in ctx.episode_ids() if e not in flagged]
        if len(rest) >= 4:
            keep = set(flagged) | set(rest[::2])
            out.append(("half-sample", lambda: probes.keep_episodes(ctx, keep)))
    return out


def _demote(result, why: str) -> None:
    for f in result.findings:
        if f.severity >= Severity.WARN:
            f.severity = Severity.INFO
            f.message += f" [published as INFO: {why}]"
    result.summary = (result.summary + f" [published as INFO: {why}]").strip()


def robustness_gate(ctx, rep, prior: dict | None, gate: TransitionGate | None,
                    repo_id: str = "", probe: bool = True) -> None:
    """Apply the gate to a fresh report in place (see the block comment)."""
    from kinema import probes
    from kinema.report import flagged_episodes

    if gate is None:
        return
    by_id = {c.id: c for c in ALL_CHECKS}
    small = (ctx.state is None or getattr(ctx.state, "nbytes", 0) <= PROBE_MAX_BYTES) and \
        (ctx.action is None or getattr(ctx.action, "nbytes", 0) <= PROBE_MAX_BYTES) and \
        not isinstance(ctx.state, np.memmap) and not isinstance(ctx.action, np.memmap)
    cfg = CheckConfig()
    for r in rep.results:
        now = r.verdict.name
        before = (prior or {}).get(r.check_id)
        if _RANK.get(now, 0) < 2 or (before is not None and _RANK.get(before, 0) >= _RANK.get(now, 0)):
            continue  # not a WARN/FAIL, or not newly one
        r.details = dict(r.details or {})
        if probe and small and r.check_id not in _NOT_PROBED and r.check_id in by_id:
            base = probes.signature(r)
            for name, make in _nightly_probes(ctx, flagged_episodes(r)):
                try:
                    moved = probes.signature(by_id[r.check_id].run(make(), cfg))
                except Exception:  # noqa: BLE001 - a probe that cannot run proves nothing
                    continue
                if moved != base:
                    r.details["robustness"] = {"unstable_under": name, "was": now,
                                               "probe_verdict": moved[0]}
                    _demote(r, f"verdict not robust under the {name} probe")
                    gate.note("unstable", {"id": repo_id, "check": r.check_id, "was": now, "probe": name})
                    break
            else:
                r.details["robustness"] = {"probed": [n for n, _ in _nightly_probes(ctx, flagged_episodes(r))]}
            if r.details.get("robustness", {}).get("unstable_under"):
                continue
        else:
            r.details["robustness"] = {"probed": [], "skipped": "large or memory-mapped load"
                                       if not small else "not probed"}
        if not gate.admit(r.check_id):
            r.details["robustness"]["held"] = True
            _demote(r, f"held for review - more than {gate.cap} new {r.check_id} "
                       "WARN/FAIL verdicts this run")
            gate.note("held", {"id": repo_id, "check": r.check_id, "was": now})


def audit_one(repo_id: str, max_episodes: int, cap_bytes: int,
              prior: dict | None = None, gate: "TransitionGate | None" = None,
              probe: bool = True) -> DatasetReport:
    import shutil
    import tempfile

    base, prefix = _split_subpath(repo_id)
    tmp_dir = Path(tempfile.mkdtemp(prefix="kinema-leaderboard-"))
    dest = tmp_dir
    try:
        unservable: list = []
        # one revision for the whole audit: listing, data, range reads and
        # videos all come from the same commit, so the audit describes exactly
        # it even if the dataset is pushed to mid-run
        revision = resolve_revision(base)
        dest, sizes = download_sample(repo_id, cap_bytes, dest, unservable, revision=revision)
        targets_all = _target_video_files(dest, max_episodes) or []
        subset = _video_subsample(targets_all, sizes, VIDEO_SAMPLE_FILES, VIDEO_CAP_BYTES)
        if subset is None and targets_all:
            # big-shard dataset: the 24-file spread busts the cap, so grade
            # the loaded episodes living in ONE middle shard instead
            single = _video_single_shard(targets_all, sizes, VIDEO_CAP_BYTES)
            if single:
                subset = [single]
                print(f"{repo_id}: 24-file video spread over the "
                      f"{VIDEO_CAP_BYTES / 1e6:.0f} MB cap - falling back to "
                      f"single shard {single}", flush=True)
        video = False
        fetched_videos: list[str] = []
        if subset:
            got = 0
            for rel in subset:
                try:
                    # dest is the nested root after download_sample; fetch
                    # through the temp dir's top to land back at dest/rel
                    _hub_download(base, _pref(prefix, rel), tmp_dir, revision)
                    got += 1
                    fetched_videos.append(rel)
                except Exception as e:  # noqa: BLE001
                    print(f"{repo_id}: video fetch failed for {rel}: "
                          f"{type(e).__name__} - skipping the file", flush=True)
            video = got > 0
        elif targets_all:
            print(f"{repo_id}: video subsample over the "
                  f"{VIDEO_CAP_BYTES / 1e6:.0f} MB cap and no single shard fits - "
                  f"visual-quality will skip", flush=True)
        ctx = load_dataset(str(dest), max_episodes=max_episodes, video=video)
        # the board always fetches a targeted video subset (the capped
        # subsample or the single-shard fallback), never the full tree:
        # disk-based camera coverage must defer to the metadata
        ctx.video_partial = True
        # the full repo file listing lets metadata checks verify file
        # pointers against the repo itself, not the partial local download
        ctx.repo_files = sorted(sizes)
        ctx.unreadable_files = list(ctx.unreadable_files) + unservable
        # the checks MUST run before the finally's rmtree: visual-quality
        # reads the downloaded MP4s off disk; every other check works from
        # the context's in-memory tables
        rep = DatasetReport(
            source=repo_id,
            format_version=ctx.format_version,
            episodes=ctx.total_episodes,
            frames=ctx.row_count,
            fps=ctx.fps,
            truncated=ctx.truncated,
            video_audited=VIDEO_AUDIT_MARK,
        )
        cfg = CheckConfig(video=video)
        for check in ALL_CHECKS:
            try:
                rep.results.append(check.run(ctx, cfg))
            except Exception as e:  # noqa: BLE001
                rep.results.append(check.skip(f"check crashed: {type(e).__name__}: {e}"))
        robustness_gate(ctx, rep, prior, gate, repo_id=repo_id, probe=probe)
        rep.audit_meta = {"revision": revision,
                          "sample": sample_signature(ctx.episode_ids(), fetched_videos)}
        return rep
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _waiver_note(charge: dict) -> str:
    up = charge.get("waived_by")
    if not up:
        return ""
    return f" <span class=dim>(charge waived - same episodes as {esc(up)})</span>"


def _revision_note(source: str, revision: str | None) -> str:
    if not revision:
        return "dataset revision: unknown (audited before revisions were recorded)"
    if revision.startswith("gcs:"):
        return f"dataset revision: shard-inventory digest <code>{esc(revision[4:])}</code>"
    base = source.split("//", 1)[0]
    return (f"dataset revision: <a href='https://huggingface.co/datasets/{esc(base)}/tree/{esc(revision)}'>"
            f"<code>{esc(revision[:12])}</code></a>")


def render_dataset_page(rep: DatasetReport, audited: str | None = None,
                        rlds_ref: tuple[str, str] | None = None,
                        revision: str | None = None) -> str:
    charges = rep.charges()  # the same waiver the stored charge/waived_by fields record
    rows = "".join(
        f"<tr><td><code>{esc(r.check_id)}</code></td>"
        f"<td>{badge('SKIP' if r.skipped else r.verdict.name)}</td>"
        f"<td>{esc(r.skip_reason if r.skipped else r.summary)}"
        f"{_waiver_note(charges.get(r.check_id, {}))}</td></tr>"
        for r in rep.results
    )
    findings = []
    for r in rep.results:
        for f in r.findings:
            if f.severity >= Severity.WARN:
                findings.append(
                    f"<li>{badge(f.severity.name)} <code>{esc(r.check_id)}</code> {esc(f.message)}</li>"
                )
    if rep.source.startswith("rlds/"):
        bucket, prefix = rlds_ref or (RLDS_BUCKETS[0][0], rep.source.removeprefix("rlds/") + "/")
        hub_url = ("https://console.cloud.google.com/storage/browser/"
                   f"{bucket}/{prefix.rstrip('/')}")
    else:
        hub_base, hub_sub = _split_subpath(rep.source)
        hub_url = (f"https://huggingface.co/datasets/{hub_base}/tree/main/{hub_sub}"
                   if hub_sub else f"https://huggingface.co/datasets/{hub_base}")
    body = (
        f"<p><a href='../index.html'>&larr; leaderboard</a></p>"
        f"<h1>{esc(rep.source)}</h1>"
        f"<p class=dim>format {esc(rep.format_version)} | {rep.episodes} episodes | "
        f"{rep.frames} frames | fps {esc(rep.fps or '?')}"
        f"{' | sampled subset' if rep.truncated else ''}"
        f" | <a href='{esc(hub_url)}'>hub</a></p>"
        f"<p class=dim>{_revision_note(rep.source, revision)}</p>"
        f"<h2>Health score: {rep.score}/100</h2>"
        f"<table><tr><th>check</th><th>verdict</th><th>summary</th></tr>{rows}</table>"
        f"<h2>Warnings &amp; failures</h2><ul>{''.join(findings) or '<li>none</li>'}</ul>"
        f"<p class=dim>audited {esc(audited or f'{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}')} by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a>"
        f" | score model <code>{esc(SCORE_MODEL)}</code></p>"
    )
    return page(f"{rep.source} - kinema audit", body)


PRUNE_AFTER_MISSES = 3  # fresh discoveries in a row without the dataset


def reconcile_discovery(entries: list[dict], datasets: list[dict],
                        fresh: bool = False, prune_after: int = PRUNE_AFTER_MISSES,
                        uncertain: set[str] | None = None) -> list[dict]:
    """Merge discovery metadata into stored rows; prune only on evidence.

    One omitted probe is not evidence of deletion (a 429, a timeout). A row
    is dropped only after `prune_after` FRESH discoveries in a row did not
    list it - a cached worklist proves nothing and never counts - and only
    when its lane's discovery looks healthy: a lane that came back with
    under half its stored rows is a listing failure, not mass deletion, so
    no miss is counted for that lane at all."""
    current = {d["id"]: d for d in datasets}
    unknown = uncertain or set()

    def is_uncertain(row_id: str) -> bool:
        # an errored probe (rate limit, outage) says nothing about the row;
        # a nested seat is uncertain when its base repo's probe errored
        return row_id in unknown or row_id.split("//", 1)[0] in unknown

    healthy: dict[str, bool] = {}
    if fresh:
        lanes = {e.get("lane") for e in entries}
        for lane in lanes:
            stored = sum(1 for e in entries if e.get("lane") == lane)
            found = sum(1 for d in datasets if d.get("lane") == lane) + sum(
                1 for e in entries if e.get("lane") == lane and e["id"] not in current
                and is_uncertain(e["id"]))
            healthy[lane] = found >= 0.5 * stored
    out = []
    for old in entries:
        row = current.get(old["id"])
        metadata = ({k: v for k, v in row.items()
                     if k not in {"report", "engine", "reason", "status"}} if row else {})
        merged = {**old, **metadata, "discovery_current": row is not None}
        if fresh:
            if row is not None:
                merged["discovery_misses"] = 0
            elif is_uncertain(old["id"]):
                merged["discovery_uncertain"] = True  # counter left as it was
            elif healthy.get(old.get("lane"), False):
                merged["discovery_misses"] = int(old.get("discovery_misses") or 0) + 1
                if merged["discovery_misses"] >= prune_after:
                    print(f"pruning {old['id']}: absent from {prune_after} fresh discoveries",
                          flush=True)
                    continue
        out.append(merged)
    return out


def preserve_failed_retry(old: dict, new: dict) -> dict:
    if old.get("report") and not new.get("report") and new.get("status") == "error":
        return {**old, "last_audit_error": new.get("reason", "unknown audit failure")}
    return new


def audit_stale(entry: dict, engine: str, max_episodes: int) -> bool:
    """Whether a stored board entry is queued for re-audit: older check logic
    (engine mismatch), a row below the current video_audited mark
    (unset, or from a superseded video-audit implementation), or a
    truncated audit that a higher --max-episodes would deepen
    (report.episodes below the current flag). max_episodes=0 disables the
    depth check (rescore-only renders queue nothing). Gated rows are never
    queued; unsupported rows are retried on an engine rotation - a newer
    engine may support them."""
    if entry.get("status") == "error" or entry.get("last_audit_error"):
        return True
    if entry.get("status") == "gated":
        return False  # gated repos stay gated whatever the engine
    rep = entry.get("report")
    if not rep:
        return entry.get("engine") != engine  # unsupported: retry per engine
    if entry.get("engine") != engine:
        return True
    if (rep.get("video_audited") or 0) < VIDEO_AUDIT_MARK:
        # rows from before board video audits (0/missing) or from the v1
        # targeting bug (1): re-audit once so visual-quality runs on the
        # capped subsample (or settles as no-video / over-cap)
        return True
    depth = max(int(rep.get("episodes") or 0), int(entry.get("audit_max_episodes") or 0))
    return bool(rep.get("truncated")) and depth < max_episodes


def invalidate_capped_video_results(entries: list[dict], capped_ids: set[str],
                                    corrected_at: str) -> list[str]:
    """Invalidate every finding severity backed by a known-capped listing.

    capped_ids must come from a verified inventory check, never a name or
    dataset-size guess. Existing verified PASS and pending SKIP stay intact.
    """
    changed = []
    for entry in entries:
        if entry["id"] not in capped_ids or not entry.get("report"):
            continue
        rep = entry["report"]
        for check in rep.get("checks", []):
            if check["id"] != "video-file-pointers" or not check.get("findings"):
                continue
            reason = ("Previous video-pointer result invalidated: the Hub file listing was "
                      "truncated at 100,000 entries. Complete-listing verification is pending "
                      "re-audit; missing files are not established.")
            check.update(verdict="SKIP", skipped=True, skip_reason=reason,
                         summary=reason, findings=[],
                         details={"invalidated": "truncated-hub-listing", "pending_reaudit": True})
            entry["video_pointer_correction"] = {
                "at": corrected_at, "scope": "video-file-pointers only",
                "status": "invalidated-pending-reaudit"}
            rescore_report_dict(rep)
            verdict = report_from_dict(rep).verdict
            rep["verdict"] = "PASS" if verdict == Severity.OK else verdict.name
            changed.append(entry["id"])
    return changed


def _correction_summary(entries: list[dict]) -> tuple[str, int, int]:
    corrections = [e["video_pointer_correction"] for e in entries
                   if e.get("video_pointer_correction")]
    return (max((c["at"] for c in corrections), default=""),
            sum(c["status"] == "verified" for c in corrections),
            sum(c["status"] == "invalidated-pending-reaudit" for c in corrections))


def render_index(entries: list[dict], generated: str, harm_table: bool = False, max_episodes: int = 0) -> str:
    engine = engine_version()
    corrected_at, verified, pending = _correction_summary(entries)
    correction_note = (
        f". Targeted video-pointer corrections through {esc(corrected_at)}: "
        f"{verified} verified, {pending} pending complete re-audit"
        if corrected_at else ""
    )
    rows = []
    n_stale = 0
    entries = sorted(
        entries,
        key=lambda e: (
            -((e.get("report") or {}).get("score", -1)),
            -(e.get("downloads") or 0),
        ),
    )  # health score first, downloads break ties, unaudited last
    for i, e in enumerate(entries, 1):
        stale = audit_stale(e, engine, max_episodes)
        n_stale += bool(stale)
        if e.get("report"):
            r = e["report"]
            link = f"<a href='dataset/{esc(e['slug'])}.html'>{esc(e['id'])}</a>"
            if e.get("lane") == "new":
                link += ' <span class=dim title="newest-uploads probation lane">(new)</span>'
            if e.get("lane") == "flagship":
                link += ' <span class=dim title="flagship publisher seats">(flagship)</span>'
            if e.get("subpath"):
                link += (' <span class=dim title="dataset nested inside a larger repo; '
                         'downloads are repo-level">(nested)</span>')
            if e.get("lane") == "rlds":
                link += (' <span class=dim title="Open X RLDS dataset on GCS; '
                         'metadata audit only">(rlds)</span>')

            if e.get("discovery_current") is False:
                link += ' <span class=dim>(not rediscovered this run)</span>'
            if e.get("last_audit_error"):
                link += f' <span class=dim title="{esc(e["last_audit_error"])}">(retry failed)</span>'
            score = f"<span class=score>{r['score']}</span>"
            model = r.get("score_model", "legacy-v0")
            if model != SCORE_MODEL:
                score += f' <span class=dim>({esc(model)})</span>'

            verdict = badge(r["verdict"])
            checks = " ".join(
                badge(c["verdict"]) for c in r["checks"] if c["verdict"] in ("WARN", "FAIL")
            ) or '<span class=dim>all clear</span>'
            if stale:
                why = ("audited by older check logic" if e.get("engine") != engine
                       else f"audited at {r.get('episodes')} episodes - deepening to {max_episodes} queued")
                checks += f' <span class=dim title="{why}">(stale - re-audit needed)</span>'
        else:
            link = esc(e["id"])
            score = '<span class=dim>-</span>'
            verdict = badge("NA")
            label = _STATUS_LABELS.get(e.get("status") or "error", _STATUS_LABELS["error"])
            detail = esc(e.get("reason", ""))
            checks = f'<span class=dim title="{detail}">{label}</span>'
        downloads_cell = "-" if e.get("rlds") else f"{e['downloads']:,}"
        rows.append(
            f"<tr><td>{i}</td><td>{link}</td><td>{downloads_cell}</td>"
            f"<td>{score}</td><td>{verdict}</td><td>{checks}</td></tr>"
        )
    body = (
        "<h1>LeRobot dataset health leaderboard</h1>"
        f"<p class=dim>The most-downloaded LeRobot datasets on Hugging Face, audited nightly by "
        "<a href='https://github.com/vedevpatel/kinema'>kinema</a>. "
        f"Stored audit snapshot: {esc(generated)}. Current checker engine "
        f"<code>{esc(engine)}</code>; per-result engine stamps are retained"
        + correction_note
        + (f"; {n_stale} result(s) need re-audit (not a running-job status)" if n_stale else "")
        + f". Scores use model <code>{esc(SCORE_MODEL)}</code>: 100 minus 25 per failing "
        "check and a per-check weight for each warning check (10 / 6 / 2 by measured "
        "training harm where the harm table has measured it, hand-tuned otherwise; "
        "<a href='https://github.com/vedevpatel/kinema/blob/main/docs/checks.md#health-score'>"
        "details</a>"
        + ("; <a href='harm-table.html'>harm table</a>" if harm_table else "")
        + "). Ranked by health score; downloads break ties. Rows marked (nested) live "
        "inside a larger repo; their download counts are repo-level. Scores are not "
        "comparable across score models. Video-pointer checks use repo listings; decoded-video "
        "coverage varies by result. See individual checks, or run "
        "<code>kinema check --video</code> locally.</p>"
        "<p id=pager><button onclick=pg(-1)>&larr; Prev</button> "
        "<span id=pglabel></span> "
        "<button onclick=pg(1)>Next &rarr;</button></p>"
        "<table id=lb><thead><tr><th>#</th><th>dataset</th><th>downloads</th>"
        f"<th>score <span class=dim>({esc(SCORE_MODEL)})</span></th>"
        "<th>verdict</th><th>flags</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
        "<script>"
        "const PER=50;let cur=0;"
        "const rows=[...document.querySelectorAll('#lb tbody tr')];"
        "function show(){rows.forEach((r,i)=>r.style.display="
        "(i>=cur*PER&&i<(cur+1)*PER)?'':'none');"
        "document.getElementById('pglabel').textContent="
        "(cur+1)+' / '+Math.max(1,Math.ceil(rows.length/PER));}"
        "function pg(d){cur=Math.min(Math.ceil(rows.length/PER)-1,Math.max(0,cur+d));show();}"
        "show();"
        "</script>"
    )
    return page("LeRobot dataset health leaderboard", body)


HARM_TABLE_SCHEMA = "kinema.harm-table/v1"
_CHECKS_MD = "https://github.com/vedevpatel/kinema/blob/main/docs/checks.md"


def _fmt(x, unit: str) -> str:
    return "-" if x is None else f"{x:+.1f}{unit}"


def render_harm_table(doc: dict) -> str:
    """docs/harm-table.html from docs/harm-table.json (schema v1)."""
    n_seeds = len(doc.get("seeds", []))
    steps = doc.get("steps", 0)
    steps_s = f"{steps / 1000:g}k" if steps >= 1000 else str(steps)
    evals = doc.get("eval_episodes")
    paired_counts = [len(r.get("seeds") or []) for r in doc.get("rows", [])]
    lo, hi = (min(paired_counts), max(paired_counts)) if paired_counts else (n_seeds, n_seeds)
    seed_scope = (f"{lo} seeds" if lo == hi else f"{lo}-{hi} paired seeds per defect")
    scope = (f"{seed_scope} / "
             + (f"{evals} eval episodes / " if evals else "no rollouts / ")
             + f"{steps_s} steps")
    pol, ds, comp = doc.get("policy", {}), doc.get("dataset", {}), doc.get("compute", {})
    rows = []
    for r in doc.get("rows", []):
        mse, succ = r["mse_delta_pct"], r["success_delta_pts"]
        rows.append(
            f"<tr><td><code>{esc(r['defect'])}</code></td>"
            f"<td><a href='{_CHECKS_MD}#{esc(r['check'])}'>{esc(r['check'])}</a></td>"
            f"<td>{esc(r['recipe']['description'])}</td>"
            f"<td class=score>{_fmt(mse['mean'], '%')}</td>"
            f"<td class=dim>{' / '.join(_fmt(v, '%') for v in mse['per_seed'])}</td>"
            f"<td class=score>{_fmt(succ['mean'], ' pts')}</td>"
            f"<td class=dim>{' / '.join(_fmt(v, '') for v in succ['per_seed'])}</td></tr>"
        )
    gpu = ", ".join(comp.get("gpu", [])) if isinstance(comp.get("gpu"), list) else comp.get("gpu", "")
    body = (
        "<h1>Harm table</h1>"
        f"<p><b>{esc(scope)} - indicative ordering evidence, not claim-grade.</b> "
        "Each injected defect is trained against a clean arm under the same seed; deltas are "
        "defect minus clean. Cross-accelerator pairs are confounded, as noted below. "
        "Health-score weights (score model tiered-v3) are anchored on these rows where a check has one, "
        "two of them capped below their tier; a new table needs a new score-model version "
        f"(<a href='{_CHECKS_MD}#harm-table'>protocol</a>, "
        f"<a href='{_CHECKS_MD}#health-score'>weights</a>, <a href='index.html'>leaderboard</a>).</p>"
        f"<p class=dim>Policy: {esc(pol.get('name', ''))} ({esc(pol.get('source', ''))}), "
        f"batch {esc(pol.get('batch_size', '-'))}. Dataset: <code>{esc(ds.get('repo_id', ''))}"
        f"@{esc(str(ds.get('revision', ''))[:12])}</code>, {len(ds.get('train_episodes') or [])} train / "
        f"{len(ds.get('heldout_episodes') or [])} held-out episodes ({esc(ds.get('episode_sample', ''))}). "
        f"Seeds {esc(doc.get('seeds'))}. "
        f"Generated {esc(doc.get('generated', ''))}"
        + (f"; {esc(comp['table_runs_gpu_hours'])} GPU-hours across listed runs on {esc(gpu)}"
           if "table_runs_gpu_hours" in comp else "")
        + (f"; accelerators {esc(gpu)}" if gpu and "table_runs_gpu_hours" not in comp else "")
        + (f"; partial ledger {esc(comp['ledger_gpu_hours'])} GPU-hours (one manifest's "
           "spend, not the run total)"
           if "ledger_gpu_hours" in comp else "")
        + ".</p>"
        + (f"<p class=dim>{esc(doc['notes'])}</p>" if doc.get("notes") else "")
        + "<table><thead><tr><th>defect</th><th>check</th><th>recipe</th>"
        "<th>held-out MSE &Delta;</th><th class=dim>per seed</th>"
        "<th>rollout success &Delta;</th><th class=dim>per seed</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    return page("Harm table", body)


def write_harm_table(out: Path) -> bool:
    """Render harm-table.html when harm-table.json is present (absent: skip
    silently); True when the page exists."""
    src = out / "harm-table.json"
    if not src.exists():
        return False
    try:
        doc = json.loads(src.read_text())
        if doc.get("schema") != HARM_TABLE_SCHEMA:
            raise ValueError(f"schema {doc.get('schema')!r}")
        text = render_harm_table(doc)
    except Exception as e:  # noqa: BLE001 - a bad table must not take the board down
        print(f"harm table not rendered: {type(e).__name__}: {e}")
        return False
    tmp = out / ".harm-table.html.tmp"
    tmp.write_text(text)
    tmp.replace(out / "harm-table.html")
    return True


_DETAIL_LIST_CAP = 100


def _cap_detail_lists(entries: list) -> list:
    """Return a serialization copy of entries with oversized detail lists
    capped at 100 (+ a <key>_total count). Keeps data.json under GitHub's
    100 MB hard file limit: episode lists were 52% of the compact payload
    (0.6 MB each on the DROID-scale rows) and pretty-printing doubled the
    rest. The board renders counts, not rosters. The in-memory entries keep
    their full evidence for page rendering and same-run scoring;
    episode evidence for checks participating in scoring waivers stays
    complete, so future rescoring never uses a truncated overlap set."""
    import copy

    from kinema.report import WAIVER_EDGES

    scoring_checks = {cid for edge in WAIVER_EDGES for cid in edge}
    capped = copy.deepcopy(entries)
    for e in capped:
        for chk in (e.get("report") or {}).get("checks") or []:
            dets = [chk.get("details") or {}]
            dets += [f.get("details") or {} for f in chk.get("findings") or []]
            for det in dets:
                for k, v in list(det.items()):
                    if k == "episodes" and chk.get("id") in scoring_checks:
                        continue
                    if isinstance(v, list) and len(v) > _DETAIL_LIST_CAP:
                        det[k] = v[:_DETAIL_LIST_CAP]
                        det.setdefault(f"{k}_total", len(v))
    return capped


def _write(out: Path, entries: list, generated: str, max_episodes: int = 0) -> None:
    """Atomic writes so a kill mid-run never leaves a corrupt or missing site."""
    data = json.dumps(
        {"generated": generated, "corrections_updated_at": _correction_summary(entries)[0] or None,
         "score_model": SCORE_MODEL, "datasets": _cap_detail_lists(entries)},
        separators=(",", ":"), default=str,
    )
    idx = render_index(entries, generated, harm_table=write_harm_table(out), max_episodes=max_episodes)
    for name, text in (("data.json", data), ("index.html", idx)):
        tmp = out / f".{name}.tmp"
        tmp.write_text(text)
        tmp.replace(out / name)


def rescore_entries(entries: list[dict]) -> int:
    """Rescore every audited entry in place from its stored per-check verdicts
    (never per finding, never a check re-run); returns how many scores moved."""
    moved = 0
    for e in entries:
        if e.get("report"):
            before = e["report"].get("score")
            rescore_report_dict(e["report"])
            moved += e["report"]["score"] != before
    return moved


_AUDITED_RE = re.compile(r"audited (\d{4}-\d\d-\d\d \d\d:\d\d UTC)")


def rewrite_dataset_pages(out: Path, entries: list[dict]) -> None:
    """Re-render each audited entry's dataset page + report JSON from its
    stored report, keeping the page's original audit stamp."""
    for e in entries:
        if not e.get("report"):
            continue
        page_path = out / "dataset" / f"{e['slug']}.html"
        m = _AUDITED_RE.search(page_path.read_text()) if page_path.exists() else None
        rep = report_from_dict(e["report"])
        # the page renders from the round-tripped report; it must agree with
        # the stored verdicts the board score was computed from
        if rep.score != e["report"]["score"]:
            raise RuntimeError(f"{e['id']}: round-tripped score {rep.score} != stored "
                               f"{e['report']['score']}")
        for name, text in (
            (f"{e['slug']}.json", json.dumps(e["report"], separators=(",", ":"), default=str)),
            (f"{e['slug']}.html", render_dataset_page(rep, audited=m.group(1) if m else "unknown",
                                                       revision=e.get("revision"),
                                                       rlds_ref=(e.get("rlds_bucket", RLDS_BUCKETS[0][0]),
                                                                 e.get("rlds_prefix", ""))
                                                       if e["id"].startswith("rlds/") else None)),
        ):
            tmp = out / "dataset" / f".{name}.tmp"
            tmp.write_text(text)
            tmp.replace(out / "dataset" / name)


def rescore_site(out: Path) -> int:
    """--rescore: recompute data.json, index.html and every dataset page from
    the stored verdicts. Keeps the board's generated stamp and each page's
    audit stamp - nothing was re-audited."""
    data_json = out / "data.json"
    board = json.loads(data_json.read_text())
    entries = board.get("datasets", [])
    moved = rescore_entries(entries)
    rewrite_dataset_pages(out, entries)
    _write(out, entries, board.get("generated", "unknown"))
    n = sum(1 for e in entries if e.get("report"))
    print(f"rescored {n} datasets under {SCORE_MODEL}: {moved} score(s) changed")
    return 0


def load_worklist(path: Path) -> tuple[list[dict], float]:
    """(datasets, age in hours). Accepts the old bare-list format (age = file mtime)."""
    import time

    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        gen = dt.datetime.fromisoformat(raw["generated"])
        age = (dt.datetime.now(dt.timezone.utc) - gen).total_seconds() / 3600
        return list(raw["datasets"]), age
    return list(raw), (time.time() - path.stat().st_mtime) / 3600


def save_worklist(path: Path, datasets: list[dict]) -> None:
    payload = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "datasets": datasets}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":")))
    tmp.replace(path)


def main() -> int:
    import socket
    import time

    started = time.monotonic()  # total soft budget includes discovery and resume
    socket.setdefaulttimeout(150)  # no download op may hang the whole run
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--new-lane", type=int, default=25,
                    help="also audit this many of the newest LeRobot uploads (probation lane)")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--size-cap-mb", type=int, default=SIZE_CAP_MB_DEFAULT)
    ap.add_argument("--time-budget", type=float, default=0, help="stop starting new audits after N seconds")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--worklist", default="", help="cache file for the dataset list")
    ap.add_argument("--worklist-max-age-hours", type=float, default=0,
                    help="rediscover when the worklist is older than this (0: never expires)")
    ap.add_argument("--discover-only", action="store_true",
                    help="refresh the worklist (and stored rows' discovery state), audit nothing")
    ap.add_argument("--no-revision-probe", action="store_true",
                    help="skip the nightly check for datasets that changed upstream since their audit")
    ap.add_argument("--transition-cap", type=int, default=10,
                    help="publish at most N new WARN/FAIL verdicts per check per run; the rest are held as INFO (0 = no cap)")
    ap.add_argument("--no-probes", action="store_true",
                    help="skip the robustness probes on new WARN/FAIL verdicts (the cap still applies)")
    ap.add_argument("--hard-deadline", type=float, default=0,
                    help="seconds from start after which in-flight audits are abandoned and the "
                         "run writes what it has and exits (keep below the shell timeout)")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute stored scores under the current score model; no audits")
    args = ap.parse_args()

    out = Path(args.out)
    if args.rescore:
        return rescore_site(out)
    (out / "dataset").mkdir(parents=True, exist_ok=True)

    worklist_path = Path(args.worklist) if args.worklist else None
    datasets, fresh = None, False
    if worklist_path and worklist_path.exists() and not args.discover_only:
        datasets, age_h = load_worklist(worklist_path)
        if args.worklist_max_age_hours and age_h > args.worklist_max_age_hours:
            print(f"worklist is {age_h:.0f}h old (> {args.worklist_max_age_hours:.0f}h): rediscovering")
            datasets = None
        else:
            print(f"loaded {len(datasets)} datasets from worklist cache ({age_h:.0f}h old)")
    if datasets is None:
        print(f"listing top {args.limit} lerobot datasets on the Hub"
              + (f" + {args.new_lane} newest..." if args.new_lane else "..."))
        datasets = find_lerobot_datasets(args.limit, args.new_lane,
                                         workers=args.workers)
        fresh = True
        print(f"found {len(datasets)} LeRobot-format datasets "
              f"({time.monotonic() - started:.0f}s)")
        if worklist_path:
            save_worklist(worklist_path, datasets)

    # resume: skip entries already audited BY THIS ENGINE. Results from older
    # check logic stay on the board (marked stale) until they are replaced,
    # and are re-audited first; gated repos stay gated whatever the engine.
    engine = engine_version()
    entries = []
    data_json = out / "data.json"
    done_ids: set[str] = set()
    unknown_rev: set[str] = set()
    if data_json.exists():
        try:
            prev = json.loads(data_json.read_text()).get("datasets", [])
            entries = [
                e for e in prev
                if e.get("report") or (e.get("reason") and e.get("status") in ("gated", "unsupported", "error"))
            ]
            entries = reconcile_discovery(entries, datasets, fresh=fresh,
                                          uncertain=set(PROBE_ERRORS) if fresh else None)
            if fresh and PROBE_ERRORS:
                print(f"{len(PROBE_ERRORS)} discovery probe(s) errored: those rows keep "
                      "their miss counters (an error is not an absence)", flush=True)
            # stored results carry their verdicts: rescore them under the
            # current model so the whole board swaps models in this run
            n_moved = rescore_entries(entries)
            done_ids = {
                e["id"] for e in entries
                if e.get("status") == "gated" or not audit_stale(e, engine, args.max_episodes)
            }
            n_stale = sum(1 for e in entries if e["id"] not in done_ids)
            print(f"resuming: {len(done_ids)} current, {n_stale} stale (engine {engine})")
            # a current row is still stale when its dataset moved upstream,
            # or when its audit predates recorded revisions
            if not args.discover_only and not args.no_revision_probe:
                changed_rev, unknown_rev = revision_staleness(entries, done_ids)
                done_ids -= changed_rev | unknown_rev
                print(f"revisions: {len(changed_rev)} dataset(s) changed upstream, "
                      f"{len(unknown_rev)} audit(s) with no recorded revision queued", flush=True)
        except Exception:
            entries, done_ids, n_moved = [], set(), 0
        if n_moved:
            # outside the try: a page-render error must not wipe the board
            rewrite_dataset_pages(out, entries)
            print(f"rescored stored results under {SCORE_MODEL}: {n_moved} score(s) changed")
    stale_ids = {e["id"] for e in entries if e["id"] not in done_ids}
    if args.discover_only:
        # discovery state (misses, pruning, fresh metadata) is the whole job
        _write(out, entries, f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}",
               max_episodes=args.max_episodes)
        print(f"discovery only: {len(datasets)} listed, {len(entries)} board rows kept")
        return 0

    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    # the board's current per-check verdicts: the gate judges only verdicts
    # that rise against them (a fresh board has none, so everything is new)
    prior_verdicts = {e["id"]: {c["id"]: c.get("verdict") for c in e["report"].get("checks", [])}
                      for e in entries if e.get("report")}
    gate = TransitionGate(args.transition_cap)

    def process(d):
        slug = d["id"].replace("/", "--")
        entry = {**d, "slug": slug, "engine": engine,
                 "audit_max_episodes": args.max_episodes}
        if d["data_bytes"] > args.size_cap_mb * 1024 * 1024:
            entry["sampled"] = True
        try:
            if d.get("rlds"):
                entry["sampled"] = True  # metadata-only audit, never downloads
                rep = audit_rlds(d)
            else:
                rep = audit_one(d["id"], args.max_episodes, args.size_cap_mb * 1024 * 1024,
                                prior=prior_verdicts.get(d["id"]), gate=gate,
                                probe=not args.no_probes and not is_heavy(d))
            entry["report"] = report_to_dict(rep)
            meta = getattr(rep, "audit_meta", None) or {}
            entry["revision"] = meta.get("revision")  # None: unknown, never guessed
            entry["sample"] = meta.get("sample")
            entry["audited_at"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
            (out / "dataset" / f"{slug}.html").write_text(render_dataset_page(
                rep, rlds_ref=(d.get("rlds_bucket", RLDS_BUCKETS[0][0]), d.get("rlds_prefix", ""))
                if d.get("rlds") else None, revision=entry["revision"]))
            (out / "dataset" / f"{slug}.json").write_text(json.dumps(entry["report"], indent=2, default=str))
            print(f"    {d['id']}: score {rep.score}/100 ({rep.verdict.name})", flush=True)
        except DatasetError as e:
            entry["reason"] = _clean_reason(str(e))
            entry["status"] = _classify_reason(entry["reason"])
        except Exception as e:  # noqa: BLE001
            entry["reason"] = _clean_reason(f"audit error: {type(e).__name__}: {e}")
            entry["status"] = _classify_reason(entry["reason"])
        return entry

    todo = [d for d in datasets if d["id"] not in done_ids]
    discovered = {d["id"] for d in datasets}
    # A failed discovery probe must not strand the preserved stale evidence.
    # Reuse only its recorded route and discovery metadata, never the old
    # report/engine stamp as if it were a fresh execution.
    for e in entries:
        if e["id"] not in discovered and e["id"] not in done_ids:
            d = {k: v for k, v in e.items() if k in {
                "id", "downloads", "likes", "data_bytes", "lane", "subpath",
                "repo_scope", "rlds", "rlds_prefix", "rlds_bucket", "version"}}
            d.setdefault("data_bytes", 0)
            todo.append(d)
    # stale results first (engine, error, dataset changed upstream); audits
    # that only lack a recorded revision after them; never-audited rows last
    todo.sort(key=lambda d: 0 if d["id"] in stale_ids and d["id"] not in unknown_rev
              else 1 if d["id"] in unknown_rev else 2)
    max_pending = max(1, args.workers) * 2

    def record(entry) -> None:
        for i, e in enumerate(entries):
            if e["id"] == entry["id"]:
                entries[i] = preserve_failed_retry(e, entry)
                break
        else:
            entries.append(entry)
        generated = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
        _write(out, entries, generated, max_episodes=args.max_episodes)

    def past_deadline() -> bool:
        return bool(args.hard_deadline) and time.monotonic() - started >= args.hard_deadline

    # submit incrementally: the queue stays shallow so --time-budget actually
    # bounds the run (at most max_pending audits in flight when it expires).
    # Not a `with` block: on the hard deadline the pool is abandoned rather
    # than joined, so a stuck download cannot outlive the shell timeout.
    pool = ThreadPoolExecutor(max_workers=max(1, args.workers))
    abandoned = False
    try:
        pending = {}
        it = iter(todo)
        deferred: list[dict] = []  # heavy rows waiting for a heavy slot

        def submit_next() -> bool:
            if args.time_budget and time.monotonic() - started >= args.time_budget:
                return False
            d = next_row(it, deferred, list(pending.values()))
            if d is None:
                return False
            pending[pool.submit(process, d)] = d
            return True

        for _ in range(min(len(todo), max_pending)):
            if not submit_next():
                break
        while pending:
            if args.time_budget and time.monotonic() - started > args.time_budget:
                print("time budget reached; letting in-flight audits finish", flush=True)
                break
            # never sleep past the budget or the deadline waiting on a slow audit
            limits = [30.0]
            for lim in (args.time_budget, args.hard_deadline):
                if lim:
                    limits.append(max(0.5, lim - (time.monotonic() - started)))
            done, _ = wait(pending, timeout=min(limits), return_when=FIRST_COMPLETED)
            for fut in done:
                pending.pop(fut)
                record(fut.result())
            while len(pending) < max_pending and submit_next():
                pass
        for fut in list(pending):
            if args.hard_deadline:
                remaining = args.hard_deadline - (time.monotonic() - started)
                if remaining <= 0:
                    abandoned = True
                    break
                wait_s = min(600.0, remaining)
            else:
                wait_s = 600.0
            try:
                record(fut.result(timeout=wait_s))
            except Exception as e:  # noqa: BLE001
                print(f"    dropped in-flight audit: {type(e).__name__}", flush=True)
                if past_deadline():
                    abandoned = True
                    break
    finally:
        pool.shutdown(wait=not abandoned, cancel_futures=True)

    generated = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
    _write(out, entries, generated, max_episodes=args.max_episodes)
    n_ok = sum(1 for e in entries if e.get("report"))
    try:
        hist = write_history(out, entries, generated)
        print(f"history: {hist['date']} vs {hist['compared_to'] or 'nothing earlier (first snapshot)'}"
              + (": " + ", ".join(f"{n} {k}" for k, n in sorted(hist["summary"].items()))
                 if hist["summary"] else ""), flush=True)
    except Exception as e:  # noqa: BLE001 - history must never sink the board
        print(f"history: not written ({type(e).__name__}: {e})", flush=True)
    # what the gate kept off the scored board this run: a workflow artifact
    # for review, never published (the rows themselves show the INFO note)
    if gate.held or gate.unstable:
        Path("held-transitions.json").write_text(json.dumps(
            {"generated": generated, "cap": gate.cap, "admitted": gate.admitted,
             "held": gate.held, "unstable": gate.unstable}, indent=1))
    print(f"robustness gate: {sum(gate.admitted.values())} new WARN/FAIL verdict(s) published, "
          f"{len(gate.unstable)} not robust under a probe, {len(gate.held)} held by the cap", flush=True)
    print(f"done: {n_ok}/{len(entries)} audited -> {out}/index.html")
    if abandoned:
        # worker threads are still blocked in downloads; a normal interpreter
        # exit would join them. The site is written (atomic writes): leave now.
        import os
        print(f"hard deadline: abandoned {sum(1 for f in pending if not f.done())} "
              "in-flight audit(s); results so far are written", flush=True)
        sys.stdout.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""LeRobot dataset loading without a lerobot dependency.

Supports v3.0 (file-based: many episodes per Parquet file) and v2.1
(one Parquet per episode, episodes.jsonl), from a local directory, a
.zip, or the Hugging Face Hub ("owner/name" or "hf://owner/name").

Only meta/** and data/** are fetched from the Hub; videos are skipped
unless the caller asks for them.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from kinema.stats import aggregate, as_stat_dict, feature_names

STATE_KEY = "observation.state"
ACTION_KEY = "action"
# column names seen in the wild for the same features (openpi / GR00T-style
# conversions use "state" / "actions"); resolved to the canonical keys
_ALIASES = {
    STATE_KEY: ("observation.state", "state", "observation.states", "observation.proprio",
                "qpos", "joint_positions", "observation.qpos"),
    ACTION_KEY: ("action", "actions", "commands", "action_commands"),
}
# reward / done are optional extras, loaded when present under any of these
_REWARD_COLS = ("next.reward", "reward")
_DONE_COLS = ("next.done", "done", "terminated")


def _resolve_keys(info: dict) -> dict[str, str]:
    """canonical key -> the column name this dataset actually uses."""
    feats = info.get("features") or {}
    out = {}
    for canon, names in _ALIASES.items():
        out[canon] = next((n for n in names if n in feats), canon)
    return out


class DatasetError(Exception):
    pass


class CancelledError(Exception):
    """The caller cancelled the load (or the run built on it) - raised at
    the next cooperative checkpoint: after the download, between row groups
    while reading, between checks."""


def _raise_if_cancelled(cancelled, where: str) -> None:
    if cancelled is not None and cancelled.is_set():
        raise CancelledError(f"cancelled {where}")


def _remove_quietly(path: str) -> None:
    for rm in (os.unlink, lambda q: os.rmdir(os.path.dirname(q))):
        try:
            rm(path)
        except OSError:
            pass


def _spill_map(path: str, dtype, shape) -> np.memmap:
    """A writable memmap whose disk space goes away with its last view.

    POSIX: the file is unlinked as soon as it is mapped. The mapping keeps
    the data reachable, and the kernel frees the space when the last array
    view is gone - even if the process is killed. Without this, a
    long-lived MCP server left one full frame copy in the temp dir per
    large load. Windows cannot unlink a mapped file, so there removal is
    deferred until the map is garbage-collected (best effort)."""
    m = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    if os.name == "posix":
        os.unlink(path)
    else:
        weakref.finalize(m, _remove_quietly, path)
    return m


def _extract_zip(p: Path) -> Path:
    """Extract a .zip dataset once per (path, size, mtime) and reuse it.

    A long-lived MCP server sees the same archive many times; extracting to
    a fresh temp dir per call leaked one full copy per call. Extraction
    lands in a scratch dir and is renamed into place, so the final dir
    existing means the extraction completed."""
    import hashlib
    import zipfile

    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
    final = Path(tempfile.gettempdir()) / f"kinema-zip-{key}"
    if not final.is_dir():
        scratch = Path(tempfile.mkdtemp(prefix=f"kinema-zip-{key}-partial-"))
        try:
            with zipfile.ZipFile(p) as z:
                z.extractall(scratch)
            try:
                os.rename(scratch, final)
            except OSError:
                if not final.is_dir():  # not a lost race with a concurrent extraction
                    raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
    # unwrap a single top-level directory
    children = list(final.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return final


_FRAME_BUDGET_ENV = "KINEMA_FRAME_BUDGET_BYTES"
_FRAME_BUDGET_DEFAULT = 1 << 30  # 1 GiB


def _frame_budget() -> int:
    """Resident-byte budget for the frame arrays. A load estimated above it
    spills the arrays into memory-mapped temp files instead of RAM (a full
    DROID load is ~27M frames ≈ 3.5 GB of float64 - without a budget that
    is an OOM on smaller machines). 0 disables spilling."""
    try:
        return int(os.environ.get(_FRAME_BUDGET_ENV, _FRAME_BUDGET_DEFAULT))
    except ValueError:
        return _FRAME_BUDGET_DEFAULT


# rows per Arrow read batch while loading frame data: bounds the transient
# memory of a read to one batch no matter how large the shard's row groups
# are (a DROID-scale group is larger than a small machine's RAM)
_READ_BATCH_ROWS = 65536


@dataclass
class EpisodeMeta:
    index: int
    length: int
    tasks: list[str] = field(default_factory=list)


@dataclass
class DatasetContext:
    root: Path
    repo_id: str | None
    format_version: str
    info: dict
    stats: dict
    fps: float | None
    episodes: list[EpisodeMeta]
    episode_index: np.ndarray
    frame_index: np.ndarray
    timestamp: np.ndarray
    state: np.ndarray | None
    action: np.ndarray | None
    task_index: np.ndarray | None
    reward: np.ndarray | None = None
    done: np.ndarray | None = None
    truncated: bool = False  # True when --max-episodes limited the load
    # True when only a subset of the dataset's episode videos was fetched
    # (targeted/sampled Hub downloads, or --video not passed for a Hub
    # dataset): what is on disk then says nothing about the episodes that
    # were not fetched, so disk-based camera coverage does not apply
    video_partial: bool = False
    # full repo file listing (root-relative paths) when the caller holds one
    # (the board's audit passes dataset_info's siblings): metadata checks
    # can then verify file pointers against the repo itself instead of a
    # partial local download
    repo_files: list[str] | None = None
    # per-episode {feature: {stat: array}} for EVERY catalogued episode
    # (meta is always complete, even when the frame data is sampled)
    episode_stats: dict = field(default_factory=dict, repr=False)
    stats_json: dict = field(default_factory=dict, repr=False)  # raw meta/stats.json
    stats_source: str = "meta/stats.json"  # where ctx.stats came from (all features)
    # per feature: "meta/stats.json" or "aggregated per-episode stats" - one
    # feature falling back must not relabel (and un-gate) the other
    stats_sources: dict = field(default_factory=dict)
    state_names: list | None = None
    action_names: list | None = None
    # canonical key -> the column name this dataset uses ("actions", ...)
    feature_keys: dict = field(default_factory=dict)
    # episode index -> times catalogued, when the metadata lists it > once
    duplicate_episode_ids: dict = field(default_factory=dict)
    # catalogued episode ids by position, when the frames number episodes
    # 0..N-1 instead (matched by position; see the loader); None otherwise
    episode_ids_by_position: list | None = None
    # None for manually constructed contexts; loader records actual parquet
    # columns so checks distinguish missing fields from present zero arrays.
    present_columns: frozenset[str] | None = None
    # column -> null cells read (filled: -1 for ints, NaN for floats)
    null_counts: dict = field(default_factory=dict)
    unreadable_files: list = field(default_factory=list)  # [(relpath, error)]
    # v3: episode -> camera -> (chunk_index, file_index, from_s, to_s)
    video_segments: dict = field(default_factory=dict, repr=False)
    _row_slices: dict = field(default_factory=dict, repr=False)
    _meta_index: dict = field(default_factory=dict, repr=False)

    @property
    def total_episodes(self) -> int:
        return len(self.episodes)

    @property
    def row_count(self) -> int:
        return int(self.episode_index.shape[0])

    @property
    def state_dim(self) -> int:
        return 0 if self.state is None else int(self.state.shape[1])

    @property
    def action_dim(self) -> int:
        return 0 if self.action is None else int(self.action.shape[1])

    def rows_for(self, ep_index: int) -> np.ndarray:
        """Row indices (into the frame arrays) belonging to an episode, in
        frame_index order."""
        if not self._row_slices:
            order = np.argsort(self.episode_index, kind="stable")
            sorted_eps = self.episode_index[order]
            starts = np.flatnonzero(
                np.r_[True, sorted_eps[1:] != sorted_eps[:-1]]
            )
            for i, s in enumerate(starts):
                e = int(sorted_eps[s])
                end = int(starts[i + 1]) if i + 1 < len(starts) else len(order)
                rows = order[s:end]
                # timestamp breaks ties: some converters write frame_index=0
                # for every frame, and file order is not time order
                rows = rows[np.lexsort((self.timestamp[rows], self.frame_index[rows]))]
                self._row_slices[e] = rows
        return self._row_slices.get(int(ep_index), np.empty(0, dtype=int))

    def episode_ids(self) -> list[int]:
        return [e.index for e in self.episodes]

    def meta_for(self, ep_index: int) -> EpisodeMeta | None:
        # indexed: checks call this per episode, and a linear scan made that
        # quadratic (hours on a 90k-episode dataset)
        if len(self._meta_index) != len(self.episodes):
            self._meta_index = {e.index: e for e in self.episodes}
        return self._meta_index.get(int(ep_index))


def _meta_shard_map(root: Path) -> dict[int, str] | None:
    """episode_index -> repo-relative packed data shard, from v3 metadata.

    None when the metadata or its shard-location columns are missing or
    unreadable - callers then fall back to layout guessing. Repos converted
    twice can carry BOTH the packed file-*.parquet the metadata points to
    and stale episode_*.parquet leftovers; only the metadata's list is
    authoritative."""
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir():
        return None
    out: dict[int, str] = {}
    cols = ("episode_index", "data/chunk_index", "data/file_index")
    for f in sorted(ep_dir.glob("**/*.parquet")):
        try:
            names = pq.read_schema(f).names
            if not all(c in names for c in cols):
                return None  # episodes parquet without shard locations
            t = pq.read_table(f, columns=list(cols)).to_pydict()
        except (OSError, pa.ArrowInvalid, ValueError):
            return None  # unreadable meta: best-effort only
        template = _data_path_template(root)
        for ep, ch, fi in zip(t[cols[0]], t[cols[1]], t[cols[2]]):
            if ch is not None and fi is not None:
                try:
                    out[int(ep)] = template.format(chunk_index=int(ch), file_index=int(fi),
                                                   episode_chunk=int(ch), episode_index=int(ep))
                except (KeyError, IndexError, ValueError):
                    out[int(ep)] = f"data/chunk-{int(ch):03d}/file-{int(fi):03d}.parquet"
    return out or None


_DEFAULT_V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _data_path_template(root: Path) -> str:
    """info.json's data_path (the converter may not use the default naming)."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return _DEFAULT_V3_DATA_PATH
    tpl = info.get("data_path")
    return tpl if isinstance(tpl, str) and "{" in tpl else _DEFAULT_V3_DATA_PATH


def _target_data_files(root: Path, max_episodes: int) -> list[str] | None:
    """Repo-relative data files holding a spread sample of episodes, so a
    sampled Hub audit downloads only the shards it will read.

    v3.0 records each episode's shard in meta/episodes (data/chunk_index,
    data/file_index); v2.1 names one parquet per episode. None when the
    layout says neither - the caller falls back to a full download."""
    try:
        episodes = _load_episode_meta(root)
    except DatasetError:
        return None
    ids = sorted({e.index for e in episodes})
    if not ids or max_episodes >= len(ids):
        return ["data/**"]
    wanted = set(_spread(ids, max_episodes))
    shard_map = _meta_shard_map(root)
    if shard_map is not None:
        files = {shard_map[ep] for ep in wanted if ep in shard_map}
        return sorted(files) if files else None
    if (root / "meta" / "episodes.jsonl").exists():
        # render info.json's data_path: converters pick their own padding
        # (GR00T-X-Embodiment-Sim writes episode_{episode_index:08d}, and the
        # hard-coded 6-digit pattern matched nothing - the sampled load failed)
        try:
            info = json.loads((root / "meta" / "info.json").read_text())
        except (OSError, ValueError):
            info = {}
        tpl = info.get("data_path")
        chunk_size = int(info.get("chunks_size") or 1000)
        out = []
        for i in sorted(wanted):
            try:
                out.append(tpl.format(episode_chunk=i // chunk_size, episode_index=i))
            except (AttributeError, KeyError, IndexError, ValueError):
                out.append(f"data/**/episode_{i:06d}.parquet")
        return out
    return None


def _target_video_files(root: Path, max_episodes: int) -> list[str] | None:
    """Repo-relative video files holding the same spread sample of episodes,
    so a sampled --video audit downloads only the shards it will read.

    Without this, --video fetches every video in the repo before the sampler
    runs: on DROID that is 823 files for a 4-episode sample. v3.0 records
    each episode's video shard per camera in meta/episodes
    (videos/<cam>/chunk_index, videos/<cam>/file_index); v2.1 names one mp4
    per episode. None when the layout says neither - the caller falls back
    to a full videos/ download."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
        episodes = _load_episode_meta(root)
    except (OSError, json.JSONDecodeError, DatasetError):
        return None
    template = info.get("video_path")
    if not template:
        return None
    cams = [k for k, f in (info.get("features") or {}).items()
            if isinstance(f, dict) and f.get("dtype") == "video"]
    if not cams:
        return None
    ids = sorted({e.index for e in episodes})
    if not ids or max_episodes >= len(ids):
        return ["videos/**"]
    wanted = set(_spread(ids, max_episodes))
    chunk_size = int(info.get("chunks_size") or 1000)

    def render(cam: str, ep: int, ch: int, fi: int) -> str | None:
        try:
            # alias both naming generations: v3 uses chunk_index/file_index,
            # v2 templates use episode_chunk/episode_index
            return template.format(video_key=cam, episode_index=ep,
                                   episode_chunk=ep // chunk_size,
                                   chunk_index=ch, file_index=fi)
        except (KeyError, IndexError, ValueError):
            return None

    ep_dir = root / "meta" / "episodes"
    if ep_dir.is_dir():
        files: set[str] = set()
        for f in sorted(ep_dir.glob("**/*.parquet")):
            try:
                names = pq.read_schema(f).names
                cam_cols = {c: (f"videos/{c}/chunk_index", f"videos/{c}/file_index")
                            for c in cams
                            if f"videos/{c}/chunk_index" in names
                            and f"videos/{c}/file_index" in names}
                if not cam_cols:
                    continue  # episodes parquet without video shard locations
                cols = ["episode_index"] + [c for pair in cam_cols.values() for c in pair]
                t = pq.read_table(f, columns=cols).to_pydict()
            except (OSError, pa.ArrowInvalid, ValueError):
                return None  # unreadable meta: targeting is best-effort
            for row, ep in enumerate(t["episode_index"]):
                if int(ep) not in wanted:
                    continue
                for cam, (ccol, fcol) in cam_cols.items():
                    ch, fi = t[ccol][row], t[fcol][row]
                    if ch is None or fi is None:
                        continue
                    path = render(cam, int(ep), int(ch), int(fi))
                    if path is None:
                        return None
                    files.add(path)
        return sorted(files) if files else None
    if (root / "meta" / "episodes.jsonl").exists():
        out = []
        for i in sorted(wanted):
            for cam in cams:
                path = render(cam, i, i // chunk_size, i)
                if path is None:
                    return None
                out.append(path)
        return out
    return None


def _hub_address(src: str) -> tuple[str, str | None, str]:
    """Validate and split a Hub address, never a disk path. Both orders work:
    owner/name@revision//subpath and owner/name//subpath@revision."""
    revision = None
    head, sep, tail = src.partition("//")
    if "@" in head:  # owner/name@rev//sub - the revision belongs to the repo
        head, revision = head.split("@", 1)
    elif sep and "@" in tail:  # owner/name//sub@rev
        tail, revision = tail.rsplit("@", 1)
    if revision is not None and (not revision or any(c.isspace() for c in revision)):
        raise DatasetError("invalid Hub revision")
    tail = tail.rstrip("/")  # a trailing slash on a subpath is harmless
    src = head + (sep + tail if sep else "")
    base, _, subpath = src.partition("//")
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", base):
        raise DatasetError("Hub source must be owner/name, optionally //subpath and @revision")
    if "//" in src and (not subpath or any(seg in ("", ".", "..") for seg in subpath.split("/"))):
        raise DatasetError("invalid nested Hub dataset path")
    return base, revision, subpath


def _resolve_source(
    source: str, video: bool = False, max_episodes: int | None = None
) -> tuple[Path, str | None, str | None]:
    """Return (local_root, repo_id, revision). Downloads from the Hub when needed."""
    src = source.strip()
    forced_hub = src.startswith("hf://")
    if forced_hub:
        src = src[len("hf://"):]
    p = Path(src).expanduser()
    if not forced_hub and p.exists():
        if p.suffix == ".zip":
            return _extract_zip(p), None, None
        return p, None, None
    # Treat as a Hub repo id.
    if "/" not in src:
        raise DatasetError(
            f"{source!r} is neither a local path nor a Hub repo id (owner/name)"
        )
    from huggingface_hub import snapshot_download

    # Parse after deciding local-vs-Hub: hf:// must never consult local paths.
    src, revision, subpath = _hub_address(src)
    patterns = ["meta/**", "*.json", "*.jsonl"]
    if video and max_episodes is None:
        patterns.append("videos/**")
    if max_episodes is None:
        patterns.append("data/**")
    if subpath:
        patterns = [f"{subpath}/{p}" for p in patterns]
    try:
        local = Path(
            snapshot_download(
                repo_id=src, repo_type="dataset", allow_patterns=patterns, revision=revision
            )
        )
        if max_episodes is not None:
            # sampled audit: fetch only the shards the sampled episodes live
            # in (droid's full data/ is ~14 GB; a 50-episode sample is not)
            target_root = local / subpath if subpath else local
            targets = _target_data_files(target_root, max_episodes) or ["data/**"]
            if video:
                # sampled --video audits fetch only the video shards of the
                # sampled episodes, not the whole videos/ tree (823 files
                # for a 4-episode DROID sample before this)
                targets += _target_video_files(target_root, max_episodes) or ["videos/**"]
            if subpath:
                targets = [f"{subpath}/{t}" for t in targets]
            snapshot_download(
                repo_id=src,
                repo_type="dataset",
                allow_patterns=targets,
                revision=revision,
            )
    except Exception as e:  # noqa: BLE001 - surface a clean message
        raise DatasetError(f"could not download {src} from the Hugging Face Hub: {e}")
    if subpath:
        local = local / subpath
        if not local.is_dir():
            raise DatasetError(f"{src}//{subpath}: nothing under {subpath!r} in the repo")
        return local, f"{src}//{subpath}", revision
    return local, src, revision


def _load_episode_meta(root: Path) -> list[EpisodeMeta]:
    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    episodes: list[EpisodeMeta] = []
    if ep_dir.is_dir():
        files = sorted(ep_dir.glob("**/*.parquet"))
        if not files:
            raise DatasetError(f"meta/episodes exists but holds no parquet files in {root}")
        # only the columns we use: meta/episodes also carries per-episode
        # stats for every feature (images included), often most of its bytes
        def _meta(f):
            names = pq.read_schema(f).names
            return pq.read_table(f, columns=[c for c in ("episode_index", "length", "tasks") if c in names])

        try:
            table = pa.concat_tables([_meta(f) for f in files], promote_options="default")
        except (OSError, pa.ArrowInvalid) as e:
            raise DatasetError(f"meta/episodes cannot be read ({type(e).__name__}: {e})") from e
        cols = table.column_names
        idx = table.column("episode_index").to_pylist()
        length = table.column("length").to_pylist() if "length" in cols else [0] * len(idx)
        if "tasks" in cols:
            raw_tasks = table.column("tasks").to_pylist()
            # a list column holds one task per entry; a plain string column is
            # ONE task per episode (iterating it gave single characters)
            tasks = [[ts] if isinstance(ts, str) else [str(t) for t in (ts or [])]
                     for ts in raw_tasks]
        else:
            tasks = [[] for _ in idx]
        episodes = [
            EpisodeMeta(index=int(i), length=int(n), tasks=t)
            for i, n, t in zip(idx, length, tasks)
        ]
    elif jsonl.exists():
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                episodes.append(
                    EpisodeMeta(
                        index=int(rec["episode_index"]),
                        length=int(rec.get("length", 0)),
                        tasks=[str(t) for t in rec.get("tasks", [])],
                    )
                )
    else:
        raise DatasetError(f"no episode metadata found (meta/episodes or episodes.jsonl) in {root}")
    episodes.sort(key=lambda e: e.index)
    return episodes


def _catalogue_file_order(root: Path) -> list[int] | None:
    """Catalogued episode ids in the order the catalogue stores them - the
    order a positional loader numbers them (episode_meta sorts by id, which
    would hide a scrambled catalogue). v3 rows must also be strictly
    increasing in dataset_from_index when it is present. None when the order
    cannot be established."""
    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    try:
        if ep_dir.is_dir():
            ids: list[int] = []
            starts: list[int] = []
            for f in sorted(ep_dir.glob("**/*.parquet")):
                names = pq.read_schema(f).names
                cols = ["episode_index"] + (["dataset_from_index"] if "dataset_from_index" in names else [])
                t = pq.read_table(f, columns=cols).to_pydict()
                ids.extend(int(i) for i in t["episode_index"])
                if "dataset_from_index" in t:
                    starts.extend(int(x) for x in t["dataset_from_index"])
            if starts and (len(starts) != len(ids) or any(a >= b for a, b in zip(starts, starts[1:]))):
                return None
            return ids
        if jsonl.exists():
            return [int(json.loads(line)["episode_index"])
                    for line in jsonl.read_text().splitlines() if line.strip()]
    except (OSError, ValueError, KeyError, pa.ArrowInvalid):
        return None
    return None


def _load_episode_stats(
    root: Path, keys: dict[str, str] | None = None
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """Per-episode stats for observation.state and action, keyed by episode.

    v3.0 stores them as stats/<feature>/<stat> columns in meta/episodes;
    v2.1 as meta/episodes_stats.jsonl. Empty dict when neither exists."""
    out: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes_stats.jsonl"
    keys = keys or {STATE_KEY: STATE_KEY, ACTION_KEY: ACTION_KEY}
    feats = (STATE_KEY, ACTION_KEY)
    if ep_dir.is_dir():
        wanted = [f"stats/{keys[f]}/{k}" for f in feats for k in ("mean", "std", "min", "max", "count")]
        for f in sorted(ep_dir.glob("**/*.parquet")):
            names = set(pq.read_schema(f).names)
            cols = [c for c in wanted if c in names]
            if not cols:
                continue
            t = pq.read_table(f, columns=["episode_index", *cols]).to_pydict()
            for i, ep in enumerate(t["episode_index"]):
                rec = {}
                for feat in feats:
                    entry = {
                        c.rsplit("/", 1)[1]: t[c][i]
                        for c in cols
                        if c.startswith(f"stats/{keys[feat]}/")
                    }
                    d = as_stat_dict(entry)
                    if d:
                        rec[feat] = d
                if rec:
                    out[int(ep)] = rec
    elif jsonl.exists():
        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                st = r.get("stats") or {}
                rec = {feat: d for feat in feats if (d := as_stat_dict(st.get(keys[feat])))}
                if rec:
                    out[int(r["episode_index"])] = rec
    return out


def _spread(ids: list[int], n: int) -> list[int]:
    """n ids evenly spaced over the sorted list (first and last included)."""
    if n >= len(ids):
        return list(ids)
    pos = np.unique(np.round(np.linspace(0, len(ids) - 1, n)).astype(int))
    return [ids[i] for i in pos]


def _load_video_segments(root: Path) -> dict[int, dict[str, tuple]]:
    """v3.0 records each episode's [from, to) segment inside the shared MP4
    of every camera. Metadata only - no video is needed to read it."""
    out: dict[int, dict[str, tuple]] = {}
    ep_dir = root / "meta" / "episodes"
    if not ep_dir.is_dir():
        return out
    for f in sorted(ep_dir.glob("**/*.parquet")):
        names = pq.read_schema(f).names
        cams = sorted({c.split("/")[1] for c in names
                       if c.startswith("videos/") and c.endswith("/from_timestamp")})
        keys = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        cams = [c for c in cams if all(f"videos/{c}/{k}" in names for k in keys)]
        if not cams:
            continue
        t = pq.read_table(f, columns=["episode_index"] + [f"videos/{c}/{k}" for c in cams for k in keys]).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            rec = out.setdefault(int(ep), {})
            for c in cams:
                vals = [t[f"videos/{c}/{k}"][i] for k in keys]
                if None not in vals:
                    rec.setdefault(c, (int(vals[0]), int(vals[1]), float(vals[2]), float(vals[3])))
    return out


def _effective_stats(
    stats_json: dict, episode_stats: dict, version: str, episode_ids: list[int]
) -> tuple[dict, str, dict]:
    """The stats a LeRobot policy would actually normalize with.

    v2.1 ignores meta/stats.json and aggregates episodes_stats.jsonl at load
    time; v2.0/v3.0 read meta/stats.json. A feature missing from stats.json
    falls back to the aggregated per-episode stats when those exist. Like
    LeRobot, the aggregate is over the per-episode entries that exist - one
    missing line must not leave the dataset with no stats at all.

    Returns (stats, overall source, per-feature sources)."""
    stats = dict(stats_json)
    sources: dict[str, str] = {}
    v21 = version.startswith("v2.1")
    for feat in (STATE_KEY, ACTION_KEY):
        if feat in stats and not v21:
            sources[feat] = "meta/stats.json"
            continue
        per_ep = [episode_stats[e][feat] for e in episode_ids if feat in episode_stats.get(e, {})]
        agg = aggregate(per_ep) if per_ep else None
        if agg:
            stats[feat] = {k: v.tolist() for k, v in agg.items()}
            sources[feat] = "aggregated per-episode stats"
        elif feat in stats:
            sources[feat] = "meta/stats.json"
    overall = ("meta/stats.json" if all(v == "meta/stats.json" for v in sources.values())
               else "aggregated per-episode stats")
    return stats, overall, sources


def _column(table: "pa.Table | pa.RecordBatch", name: str) -> np.ndarray | None:
    """A vector column as a (rows, dim) float64 array, straight from Arrow
    buffers. Going through Python lists costs ~10x the memory and time."""
    if name not in table.column_names:
        return None
    col = table.column(name)
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    typ = col.type
    is_list = pa.types.is_fixed_size_list(typ) or pa.types.is_list(typ) or pa.types.is_large_list(typ)
    if not is_list:
        return col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    if col.null_count == 0 and pa.types.is_primitive(typ.value_type):
        if pa.types.is_fixed_size_list(typ):
            width = typ.list_size
        else:
            lengths = np.diff(col.offsets.to_numpy())
            width = int(lengths[0]) if len(lengths) else 0
            if len(lengths) and not (lengths == width).all():
                width = -1  # ragged rows: fall back below
        if width >= 0:
            flat = col.flatten().to_numpy(zero_copy_only=False)  # nulls -> NaN
            return flat.astype(np.float64, copy=False).reshape(len(col), width)
    # null rows, ragged or nested lists: the slow, general path. A null ROW
    # becomes a NaN row (nan-and-normalization reports it) rather than a raw
    # ValueError; genuinely ragged rows are a dataset error with a name.
    rows = col.to_pylist()
    width = next((len(r) for r in rows if r is not None), 0)
    if any(r is not None and len(r) != width for r in rows):
        raise DatasetError(f"column {name!r} has rows of different lengths")
    return np.asarray([r if r is not None else [float("nan")] * width for r in rows],
                      dtype=np.float64).reshape(len(rows), width)


def _flat_scalar(col):
    """A scalar column, unwrapping width-1 list columns ([[t0], [t1], ...])."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    typ = col.type
    if pa.types.is_fixed_size_list(typ) or pa.types.is_list(typ) or pa.types.is_large_list(typ):
        if col.null_count == 0:
            flat = col.flatten()
            if len(flat) == len(col):
                return flat
        col = pa.array([(v[0] if v else None) if v is not None else None
                        for v in col.to_pylist()])
    return col


def _ints(table: "pa.Table | pa.RecordBatch", name: str, fill: int = -1) -> np.ndarray:
    """An integer column; nulls become `fill` (they would otherwise make a
    healthy shard read as unreadable - the zero-copy conversion refuses them)."""
    col = _flat_scalar(table.column(name))
    if col.null_count:
        col = pc.fill_null(col, fill)
    return col.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)


def _floats(table: "pa.Table | pa.RecordBatch", name: str) -> np.ndarray:
    """A float column; nulls become NaN, width-1 lists are unwrapped."""
    col = _flat_scalar(table.column(name))
    return col.to_numpy(zero_copy_only=False).astype(np.float64, copy=False)


def load_dataset(
    source: str,
    max_episodes: int | None = None,
    video: bool = False,
    cancelled=None,
) -> DatasetContext:
    """Load a LeRobot dataset. `cancelled` (a threading.Event) is checked
    cooperatively - after the download and between row groups - and raises
    CancelledError; a Hub download in flight is not interrupted."""
    _raise_if_cancelled(cancelled, "before loading")
    root, repo_id, revision = _resolve_source(source, video=video, max_episodes=max_episodes)
    _raise_if_cancelled(cancelled, "after download")
    full_videos = video and max_episodes is None
    repo_files = None
    if repo_id is not None and not full_videos:
        # the pointer checks need the repo's own file listing: the local tree
        # holds only what the targeted fetch brought down. Paginate with
        # list_repo_tree - dataset_info's siblings cap at 100k entries, which
        # false-FAILed DROID/Bridge-scale repos. A listing failure (gated
        # repo, offline) leaves repo_files None and those checks skip.
        base, _, sub = repo_id.partition("//")
        try:
            from huggingface_hub import list_repo_tree

            repo_files = sorted(
                (f.path[len(sub) + 1:] if sub else f.path)
                for f in list_repo_tree(base, repo_type="dataset", path_in_repo=sub or None,
                                        revision=revision, recursive=True)
                if hasattr(f, "size") and (not sub or f.path.startswith(sub + "/"))
            )
        except Exception:  # noqa: BLE001 - no listing means skip, not crash
            repo_files = None
        _raise_if_cancelled(cancelled, "after repo listing")
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise DatasetError(f"{root} is not a LeRobot dataset (meta/info.json missing)")
    info = json.loads(info_path.read_text())
    stats_path = root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    fps = info.get("fps")
    fps = float(fps) if fps else None
    version = str(info.get("codebase_version", "unknown"))

    episodes = _load_episode_meta(root)
    counts: dict[int, int] = {}
    for e in episodes:
        counts[e.index] = counts.get(e.index, 0) + 1
    duplicate_ids = {i: n for i, n in counts.items() if n > 1}
    if duplicate_ids:  # audit each index once; dropped-frames reports the rest
        first: dict[int, EpisodeMeta] = {}
        for e in episodes:
            first.setdefault(e.index, e)
        episodes = list(first.values())
    keys = _resolve_keys(info)
    episode_stats = _load_episode_stats(root, keys)
    # expose stats under the canonical keys whatever the dataset calls them
    stats_json = dict(stats)
    for canon, col in keys.items():
        if col != canon and col in stats_json and canon not in stats_json:
            stats_json[canon] = stats_json[col]
    stats, stats_source, stats_sources = _effective_stats(
        stats_json, episode_stats, version, [e.index for e in episodes]
    )

    # the metadata's shard pointers (v3) are authoritative over what lies
    # under data/: twice-converted repos can carry stale episode_*.parquet
    # files beside the packed shards, and reading both counts the shared
    # episodes' frames twice. Glob only when the metadata names nothing.
    shard_map = _meta_shard_map(root)
    data_files: list[Path] = []
    if shard_map:
        data_files = sorted({root / rel for rel in set(shard_map.values())
                             if (root / rel).is_file()})
    if not data_files:
        data_files = sorted(root.glob("data/**/*.parquet"))
    if not data_files:
        raise DatasetError(f"no frame data found under data/ in {root}")

    # read files one at a time, keeping only rows for the episodes we audit -
    # bounds peak memory instead of materializing every shard at once (OOM
    # on wide multi-file datasets). With --max-episodes, the audited episodes
    # are spread evenly over the ones present: the first N are often warm-up
    # recordings and are not representative of the dataset.
    meta_ids = sorted(e.index for e in episodes)
    targets = meta_ids
    unreadable: list[tuple[str, str]] = []

    def _bad(f: Path, e: Exception) -> None:
        rel = f.relative_to(root).as_posix()
        if not any(name == rel for name, _ in unreadable):
            unreadable.append((rel, str(e)[:120]))

    # Only the columns the checks use: datasets that embed images in the
    # parquet (and wide auxiliary columns) would otherwise be read in full.
    feats_all = info.get("features") or {}
    reward_col = next((c for c in _REWARD_COLS if c in feats_all), None)
    done_col = next((c for c in _DONE_COLS if c in feats_all), None)
    wanted_cols = ["episode_index", "frame_index", "timestamp", "task_index",
                   keys[STATE_KEY], keys[ACTION_KEY],
                   *([reward_col] if reward_col else []),
                   *([done_col] if done_col else [])]

    # Never concatenate differently-shaped column inventories. Missing columns
    # otherwise shorten only one array (or leave zero-filled memmap slots),
    # assigning a later episode's values to an earlier episode. Reject that
    # unsupported schema explicitly before reading any frames.
    column_inventory: frozenset[str] | None = None

    # Which episodes each row group holds, from the footer statistics alone
    # (no data read). None for a row group whose stats are missing.
    files: list[tuple[Path, pq.ParquetFile, list[tuple[int, int] | None]]] = []
    for f in data_files:
        try:
            pf = pq.ParquetFile(f)
            inventory = frozenset(pf.schema_arrow.names) & frozenset(wanted_cols)
            if column_inventory is None:
                column_inventory = inventory
            elif inventory != column_inventory:
                raise DatasetError(
                    f"inconsistent checked columns in {f.relative_to(root)}: "
                    f"missing {sorted(column_inventory - inventory)}, "
                    f"extra {sorted(inventory - column_inventory)}; "
                    "mixed shard schemas cannot be aligned safely")
            md = pf.metadata
            col_idx = md.schema.names.index("episode_index")
            ranges = []
            for g in range(md.num_row_groups):
                st = md.row_group(g).column(col_idx).statistics
                ok = st is not None and st.has_min_max
                ranges.append((int(st.min), int(st.max)) if ok else None)
            files.append((f, pf, ranges))
        except (OSError, pa.ArrowInvalid, ValueError) as e:  # corrupt footer
            _bad(f, e)

    meta_arr = np.asarray(meta_ids, dtype=np.int64)

    def _candidates(rng: tuple[int, int] | None) -> np.ndarray:
        if rng is None:
            return meta_arr  # unknown: assume it may hold any episode
        lo, hi = np.searchsorted(meta_arr, [rng[0], rng[1] + 1])
        return meta_arr[lo:hi]

    # Positional episode indexing (lerobot/behavior1k-task0045): meta/episodes
    # catalogues source ids (450010, 450020, ...) while the frames number
    # episodes 0..N-1 by position - LeRobot's default loader looks episodes
    # up by position and trains on it. Matched by position only when the
    # footer statistics prove it: every row group's frame indices within
    # 0..N-1, none overlapping the catalogue, the catalogue strictly
    # increasing in file order. A wrong match would still surface:
    # dropped-frames compares each episode's catalogued length to its rows.
    remap: np.ndarray | None = None
    all_ranges = [r for _, _, rs in files for r in rs]
    known = [r for r in all_ranges if r is not None]
    file_order = _catalogue_file_order(root)
    if (known and len(known) == len(all_ranges)
            and min(r[0] for r in known) >= 0 and max(r[1] for r in known) < len(meta_ids)
            and not any(len(_candidates(r)) for r in known)
            and meta_ids != list(range(len(meta_ids)))
            and file_order is not None and len(file_order) == len(meta_ids)
            and all(a < b for a, b in zip(file_order, file_order[1:]))):
        remap = meta_arr  # position p -> catalogued id
        files = [(f, pf, [(int(remap[r[0]]), int(remap[r[1]])) if r else None for r in rs])
                 for f, pf, rs in files]

    if max_episodes is not None and repo_id is not None:
        # Hub downloads target the metadata spread. Extra cached shards must
        # not resample those anchors into a different, cache-dependent set.
        targets = _spread(meta_ids, max_episodes)
    elif max_episodes is not None and max_episodes < len(meta_ids):
        present_ids: set[int] = set()
        for _, _, ranges in files:
            for rng in ranges:
                present_ids.update(_candidates(rng).tolist())
        targets = _spread(sorted(present_ids), max_episodes)
    value_set = pa.array(targets, type=pa.int64())
    tgt = np.asarray(sorted(targets), dtype=np.int64)

    # the read plan: which row groups of each shard can hold target episodes
    plan: list[tuple[Path, pq.ParquetFile, list[int]]] = []
    row_bound = 0
    for f, pf, ranges in files:
        groups = [g for g, rng in enumerate(ranges)
                  if rng is None or np.searchsorted(tgt, rng[1], side="right") > np.searchsorted(tgt, rng[0])]
        if not groups:
            continue
        plan.append((f, pf, groups))
        row_bound += sum(pf.metadata.row_group(g).num_rows for g in groups)

    def _feat_dim(col: str) -> int:
        shape = (info.get("features", {}).get(col) or {}).get("shape") or []
        return int(shape[0]) if shape else 0

    est_bytes = row_bound * 8 * (4 + _feat_dim(keys[STATE_KEY]) + _feat_dim(keys[ACTION_KEY]))
    est_bytes += row_bound * ((8 if reward_col else 0) + (8 if done_col else 0))
    spill = est_bytes > _frame_budget() > 0
    spill_dir = tempfile.mkdtemp(prefix="kinema-frames-") if spill else ""
    dtypes: dict[str, type[np.generic]] = {
        "episode_index": np.int64, "frame_index": np.int64,
        "timestamp": np.float64, "task_index": np.int64,
        keys[STATE_KEY]: np.float64, keys[ACTION_KEY]: np.float64,
    }
    if reward_col:
        dtypes[reward_col] = np.float64
    if done_col:
        dtypes[done_col] = np.float64  # bools stored as 0/1 for memmap simplicity
    chunks: dict[str, list[np.ndarray]] = {}
    maps: dict[str, np.memmap] = {}
    widths: dict[str, int] = {}
    seen: set[str] = set()
    nulls: dict[str, int] = {}
    n = 0

    def _put(name: str, arr: np.ndarray) -> None:
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if spill:
            if name not in maps:
                maps[name] = _spill_map(
                    os.path.join(spill_dir, name.replace("/", "_") + ".dat"),
                    dtypes[name], (row_bound, arr.shape[1]),
                )
            m = maps[name]
            m[n:n + arr.shape[0]] = arr
            # force writeback to start now: streaming gigabytes of memmap
            # writes without flushing lets dirty pages pile up until the
            # OOM killer steps in on small machines (DROID-scale, #11)
            m.flush()
        else:
            chunks.setdefault(name, []).append(arr)

    # file at a time, batch at a time: each batch is dropped before the
    # next is read, so peak memory is one read batch plus the output
    # arrays. Reading whole row groups at once spikes to the full
    # (decompressed) group - on huge shards (DROID-scale, finding #11)
    # that alone OOMs small machines even when the frame arrays spill.
    try:
        for f, pf, groups in plan:
            cols = [c for c in wanted_cols if c in pf.schema_arrow.names]
            for g in groups:
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError(f"cancelled while reading {f.name}")
                # a row group lands whole or not at all: an error part-way
                # used to leave its episodes half-loaded while dropped-frames
                # said they were not audited
                n_before = n
                chunks_before = {k: len(v) for k, v in chunks.items()}
                try:
                    for t in pf.iter_batches(batch_size=_READ_BATCH_ROWS, row_groups=[g], columns=cols):
                        if remap is not None:
                            pos = _ints(t, "episode_index")
                            ids = np.where((pos >= 0) & (pos < len(remap)),
                                           remap[np.clip(pos, 0, len(remap) - 1)], -1)
                            t = pa.Table.from_batches([t])
                            t = t.set_column(t.column_names.index("episode_index"), "episode_index",
                                             pa.array(ids, type=pa.int64()))
                        t = t.filter(pc.is_in(t.column("episode_index").cast(pa.int64()), value_set=value_set))
                        if not t.num_rows:
                            continue
                        for name in ("episode_index", "frame_index", "task_index"):
                            if name in t.column_names:
                                seen.add(name)
                                nulls[name] = nulls.get(name, 0) + t.column(name).null_count
                                _put(name, _ints(t, name))
                        if "timestamp" in t.column_names:
                            seen.add("timestamp")
                            nulls["timestamp"] = nulls.get("timestamp", 0) + t.column("timestamp").null_count
                            _put("timestamp", _floats(t, "timestamp"))
                        for extra in (reward_col, done_col):
                            if extra and extra in t.column_names:
                                seen.add(extra)
                                # bool/null/width-1-list safe; nulls -> NaN
                                _put(extra, _floats(t, extra))
                        for canon in (STATE_KEY, ACTION_KEY):
                            name = keys[canon]
                            if name in t.column_names:
                                arr = _column(t, name)
                                if arr is not None:
                                    seen.add(name)
                                    widths.setdefault(name, arr.shape[1] if arr.ndim > 1 else 1)
                                    _put(name, arr)
                        n += t.num_rows
                except (OSError, pa.ArrowInvalid) as e:  # truncated / corrupt data pages
                    n = n_before
                    for k in list(chunks):
                        del chunks[k][chunks_before.get(k, 0):]
                    _bad(f, e)
                    break
    except BaseException:
        maps.clear()  # drop the maps now so their space is released with them
        raise
    finally:
        if spill_dir:  # POSIX: already empty, the spill files were unlinked when mapped
            shutil.rmtree(spill_dir, ignore_errors=True)

    if n == 0 and unreadable:
        raise DatasetError(
            f"none of the data files needed could be read ({len(unreadable)} unreadable, "
            f"e.g. {unreadable[0][0]}: {unreadable[0][1]})"
        )
    if n == 0:
        raise DatasetError(
            f"no frames in {root} match the episode metadata "
            f"(metadata catalogues episodes {meta_ids[0]}-{meta_ids[-1]}; "
            f"data files hold none of them - the dataset's metadata and data "
            "disagree on episode indexing)"
        )

    def _take(name: str) -> np.ndarray:
        if spill:
            m = maps[name]
            m.flush()
            return m[:n] if name in widths else m[:n].reshape(n)
        cl = chunks[name]
        out = cl[0] if len(cl) == 1 else np.concatenate(cl)
        return out if name in widths else out.reshape(-1)

    episode_index = _take("episode_index")
    truncated = False

    # partial downloads: restrict episode metadata to episodes whose frames
    # are actually present BEFORE capping, so a sample holding later shards
    # still audits the episodes it has instead of capping to absent ones
    present = set(np.unique(episode_index).tolist())
    if len(present) < len(episodes):
        # A catalogued episode absent from an available authoritative shard
        # is missing data, not an episode to erase from the audit. Keep it so
        # dropped-frames can compare its declared length with zero rows.
        target_set = set(targets)  # was a list: quadratic on huge catalogues
        bad_rel = {name for name, _ in unreadable}
        episodes = [e for e in episodes if e.index in present or (
            e.index in target_set and shard_map and e.index in shard_map
            and shard_map[e.index] not in bad_rel  # its shard is reported unreadable already
            and (root / shard_map[e.index]).is_file())]
        truncated = True

    if max_episodes is not None and len(episodes) > max_episodes:
        keep_ids = set(_spread([e.index for e in episodes], max_episodes))
        episodes = [e for e in episodes if e.index in keep_ids]
        truncated = True
    keep = {e.index for e in episodes}
    if not present <= keep:
        # some loaded rows belong to capped-away episodes (sampled partial
        # downloads); drop them. Fancy-indexing copies, so skip when the
        # loaded episodes are exactly the kept ones - the full-load case.
        # n keeps the written row count: _take slices maps with it below.
        mask = np.isin(episode_index, list(keep))
        episode_index = episode_index[mask]

        def _masked(arr: np.ndarray) -> np.ndarray:
            return arr[mask]

    else:
        def _masked(arr: np.ndarray) -> np.ndarray:
            return arr

    if episode_index.shape[0] == 0:
        raise DatasetError(
            f"no frames in {root} match the episode metadata "
            "(partial download without overlapping episodes)"
        )

    frame_index = (
        _masked(_take("frame_index"))
        if "frame_index" in seen
        else np.zeros_like(episode_index)
    )
    timestamp = (
        _masked(_take("timestamp"))
        if "timestamp" in seen
        else np.zeros(len(episode_index), dtype=np.float64)
    )
    task_index = (
        _masked(_take("task_index"))
        if "task_index" in seen
        else None
    )
    reward = (
        _masked(_take(reward_col))
        if reward_col and reward_col in seen
        else None
    )
    done = (
        _masked(_take(done_col))
        if done_col and done_col in seen
        else None
    )
    state = (
        _masked(_take(keys[STATE_KEY]))
        if keys[STATE_KEY] in seen
        else None
    )
    action = (
        _masked(_take(keys[ACTION_KEY]))
        if keys[ACTION_KEY] in seen
        else None
    )

    return DatasetContext(
        root=root,
        repo_id=repo_id,
        format_version=version,
        info=info,
        stats=stats,
        fps=fps,
        episodes=episodes,
        episode_index=episode_index,
        frame_index=frame_index,
        timestamp=timestamp,
        state=state,
        action=action,
        task_index=task_index,
        reward=reward,
        done=done,
        truncated=truncated,
        # a Hub source whose videos were not fully fetched holds an
        # incomplete local tree: disk-based camera coverage and video pointer
        # existence must not read absence as defect. _resolve_source fetches
        # the full videos tree only without an episode cap; sampled --video
        # fetches a targeted subset.
        video_partial=repo_id is not None and not full_videos,
        repo_files=repo_files,
        episode_stats=episode_stats,
        stats_json=stats_json,
        stats_source=stats_source,
        stats_sources=stats_sources,
        feature_keys=keys,
        duplicate_episode_ids=duplicate_ids,
        episode_ids_by_position=[int(x) for x in remap] if remap is not None else None,
        video_segments=_load_video_segments(root),
        present_columns=frozenset(seen),
        null_counts={k: v for k, v in nulls.items() if v},
        unreadable_files=unreadable,
        state_names=feature_names(info, keys[STATE_KEY]),
        action_names=feature_names(info, keys[ACTION_KEY]),
    )

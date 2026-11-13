"""LeRobot dataset loading without a lerobot dependency.

Supports v3.0 (file-based: many episodes per Parquet file) and v2.1
(one Parquet per episode, episodes.jsonl), from a local directory, a
.zip, or the Hugging Face Hub ("owner/name" or "hf://owner/name").

Only meta/** and data/** are fetched from the Hub; videos are skipped
unless the caller asks for them.
"""

from __future__ import annotations

import json
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
    STATE_KEY: ("observation.state", "state", "observation.states", "observation.proprio"),
    ACTION_KEY: ("action", "actions"),
}


def _resolve_keys(info: dict) -> dict[str, str]:
    """canonical key -> the column name this dataset actually uses."""
    feats = info.get("features") or {}
    out = {}
    for canon, names in _ALIASES.items():
        out[canon] = next((n for n in names if n in feats), canon)
    return out


class DatasetError(Exception):
    pass


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
    truncated: bool = False  # True when --max-episodes limited the load
    # per-episode {feature: {stat: array}} for EVERY catalogued episode
    # (meta is always complete, even when the frame data is sampled)
    episode_stats: dict = field(default_factory=dict, repr=False)
    stats_json: dict = field(default_factory=dict, repr=False)  # raw meta/stats.json
    stats_source: str = "meta/stats.json"  # where ctx.stats came from
    state_names: list | None = None
    action_names: list | None = None
    _row_slices: dict = field(default_factory=dict, repr=False)

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
        for e in self.episodes:
            if e.index == ep_index:
                return e
        return None


def _resolve_source(source: str, video: bool = False) -> tuple[Path, str | None]:
    """Return (local_root, repo_id). Downloads from the Hub when needed."""
    src = source.strip()
    if src.startswith("hf://"):
        src = src[len("hf://"):]
    p = Path(src).expanduser()
    if p.exists():
        if p.suffix == ".zip":
            import zipfile
            import tempfile

            out = Path(tempfile.mkdtemp(prefix="kinema-ds-"))
            with zipfile.ZipFile(p) as z:
                z.extractall(out)
            # unwrap a single top-level directory
            children = list(out.iterdir())
            if len(children) == 1 and children[0].is_dir():
                out = children[0]
            return out, None
        return p, None
    # Treat as a Hub repo id.
    if "/" not in src:
        raise DatasetError(
            f"{source!r} is neither a local path nor a Hub repo id (owner/name)"
        )
    from huggingface_hub import snapshot_download

    patterns = ["meta/**", "data/**", "*.json", "*.jsonl"]
    if video:
        patterns.append("videos/**")
    try:
        local = snapshot_download(repo_id=src, repo_type="dataset", allow_patterns=patterns)
    except Exception as e:  # noqa: BLE001 - surface a clean message
        raise DatasetError(f"could not download {src} from the Hugging Face Hub: {e}")
    return Path(local), src


def _load_episode_meta(root: Path) -> list[EpisodeMeta]:
    ep_dir = root / "meta" / "episodes"
    jsonl = root / "meta" / "episodes.jsonl"
    episodes: list[EpisodeMeta] = []
    if ep_dir.is_dir():
        files = sorted(ep_dir.glob("**/*.parquet"))
        if not files:
            raise DatasetError(f"meta/episodes exists but holds no parquet files in {root}")
        table = pa.concat_tables([pq.read_table(f) for f in files])
        cols = table.column_names
        idx = table.column("episode_index").to_pylist()
        length = table.column("length").to_pylist() if "length" in cols else [0] * len(idx)
        if "tasks" in cols:
            raw_tasks = table.column("tasks").to_pylist()
            tasks = [[str(t) for t in (ts or [])] for ts in raw_tasks]
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


def _effective_stats(
    stats_json: dict, episode_stats: dict, version: str, episode_ids: list[int]
) -> tuple[dict, str]:
    """The stats a LeRobot policy would actually normalize with.

    v2.1 ignores meta/stats.json and aggregates episodes_stats.jsonl at load
    time; v2.0/v3.0 read meta/stats.json. A feature missing from stats.json
    falls back to the aggregated per-episode stats when those exist."""
    stats = dict(stats_json)
    source = "meta/stats.json"
    v21 = version.startswith("v2.1")
    for feat in (STATE_KEY, ACTION_KEY):
        if feat in stats and not v21:
            continue
        per_ep = [episode_stats[e][feat] for e in episode_ids if feat in episode_stats.get(e, {})]
        agg = aggregate(per_ep) if len(per_ep) == len(episode_ids) else None
        if agg:
            stats[feat] = {k: v.tolist() for k, v in agg.items()}
            source = "aggregated per-episode stats"
    return stats, source


def _column(table: pa.Table, name: str) -> np.ndarray | None:
    if name not in table.column_names:
        return None
    col = table.column(name)
    if pa.types.is_fixed_size_list(col.type) or pa.types.is_list(col.type):
        return np.asarray(col.to_pylist(), dtype=np.float64)
    return col.to_numpy()


def load_dataset(
    source: str,
    max_episodes: int | None = None,
    video: bool = False,
) -> DatasetContext:
    root, repo_id = _resolve_source(source, video=video)
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
    keys = _resolve_keys(info)
    episode_stats = _load_episode_stats(root, keys)
    # expose stats under the canonical keys whatever the dataset calls them
    stats_json = dict(stats)
    for canon, col in keys.items():
        if col != canon and col in stats_json and canon not in stats_json:
            stats_json[canon] = stats_json[col]
    stats, stats_source = _effective_stats(
        stats_json, episode_stats, version, [e.index for e in episodes]
    )

    data_files = sorted(root.glob("data/**/*.parquet"))
    if not data_files:
        raise DatasetError(f"no frame data found under data/ in {root}")

    # read files one at a time, keeping only rows for catalogued episodes,
    # and stop once enough episodes are covered - bounds peak memory instead
    # of materializing every shard at once (OOM on wide multi-file datasets)
    meta_ids = sorted(e.index for e in episodes)
    value_set = pa.array(meta_ids)
    tables: list[pa.Table] = []
    seen: set[int] = set()
    need = max_episodes if max_episodes is not None else len(meta_ids)
    for f in data_files:
        t = pq.read_table(f)
        t = t.filter(pc.is_in(t.column("episode_index"), value_set=value_set))
        if t.num_rows:
            tables.append(t)
            seen.update(t.column("episode_index").to_pylist())
        if len(seen) >= need:
            break
    if not tables:
        raise DatasetError(
            f"no frames in {root} match the episode metadata "
            f"(metadata catalogues episodes {meta_ids[0]}-{meta_ids[-1]}; "
            f"data files hold none of them - the dataset's metadata and data "
            "disagree on episode indexing)"
        )
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    truncated = False

    # partial downloads: restrict episode metadata to episodes whose frames
    # are actually present BEFORE capping, so a sample holding later shards
    # still audits the episodes it has instead of capping to absent ones
    present = set(table.column("episode_index").to_pylist())
    if len(present) < len(episodes):
        episodes = [e for e in episodes if e.index in present]
        truncated = True

    if max_episodes is not None and len(episodes) > max_episodes:
        episodes = episodes[:max_episodes]
        truncated = True
    keep = [e.index for e in episodes]
    mask = pc.is_in(table.column("episode_index"), value_set=pa.array(keep))
    table = table.filter(mask)

    if table.num_rows == 0:
        raise DatasetError(
            f"no frames in {root} match the episode metadata "
            "(partial download without overlapping episodes)"
        )

    episode_index = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    frame_index = (
        np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
        if "frame_index" in table.column_names
        else np.zeros_like(episode_index)
    )
    timestamp = (
        np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        if "timestamp" in table.column_names
        else np.zeros(len(episode_index), dtype=np.float64)
    )
    task_index = (
        np.asarray(table.column("task_index").to_pylist(), dtype=np.int64)
        if "task_index" in table.column_names
        else None
    )
    state = _column(table, keys[STATE_KEY])
    action = _column(table, keys[ACTION_KEY])

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
        truncated=truncated,
        episode_stats=episode_stats,
        stats_json=stats_json,
        stats_source=stats_source,
        state_names=feature_names(info, keys[STATE_KEY]),
        action_names=feature_names(info, keys[ACTION_KEY]),
    )

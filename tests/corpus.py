"""The adjudicated corpus: small frozen extracts of real datasets whose
verdicts were checked against the raw data, each with the verdict a check
must (or must not) give. tests/test_corpus.py asserts them on every push,
offline; tests/test_invariance.py also runs its probes over them.

Each case is one .npz under tests/data/corpus/, built from a pinned Hub
revision by tests/data/corpus/build_corpus.py (which records where every
array came from). materialize() writes a case out as a minimal LeRobot v3
dataset the normal loader reads.

npz fields: state, action (float32, frames x dims), episode_index,
frame_index (int32), fps, state_names, action_names; optionally the stored
statistics when they are the phenomenon (stats_<feat>_<stat>: dims;
epstats_<feat>_<stat>: catalogue x dims) and a catalogue wider than the data
(catalogue_ids, catalogue_lengths) - catalogued episodes without frames
point at a shard that does not exist, so the load is a partial sample, as
on the board.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinema.fixtures.synth import make_dataset

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"
FEATS = {"state": "observation.state", "action": "action"}
STATS = ("mean", "std", "min", "max")


def cases() -> list[str]:
    return sorted(p.stem for p in CORPUS_DIR.glob("*.npz"))


def load_case(name: str) -> dict:
    with np.load(CORPUS_DIR / f"{name}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    return {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0)}


def materialize(root: Path, name: str) -> Path:
    c = load_case(name)
    ep = c["episode_index"].astype(np.int64)
    fi = c["frame_index"].astype(np.int64)
    fps = float(c["fps"])
    arrays = {"state": c["state"].astype(np.float64), "action": c["action"].astype(np.float64)}
    data_eps = sorted(set(ep.tolist()))
    cat_ids = [int(e) for e in c["catalogue_ids"]] if "catalogue_ids" in c else data_eps
    cat_len = ([int(n) for n in c["catalogue_lengths"]] if "catalogue_lengths" in c
               else [int((ep == e).sum()) for e in data_eps])

    # skeleton from the synthetic writer (tasks table, info layout), then
    # every frame, catalogue and statistic replaced by the case's own
    make_dataset(root, episodes=1)
    order = np.lexsort((fi, ep))
    ep, fi = ep[order], fi[order]
    arrays = {k: v[order] for k, v in arrays.items()}
    table = {
        "episode_index": ep, "frame_index": fi, "timestamp": fi / fps,
        "action": [list(r) for r in arrays["action"]],
        "task_index": np.zeros(len(ep), dtype=np.int64),
        "observation.state": [list(r) for r in arrays["state"]],
    }
    pq.write_table(pa.table(table), root / "data" / "chunk-000" / "file-000.parquet")

    info_p = root / "meta" / "info.json"
    info = json.loads(info_p.read_text())
    info["fps"] = fps
    info["total_episodes"] = len(cat_ids)
    info["total_frames"] = int(sum(cat_len))
    for short, feat in FEATS.items():
        info["features"][feat]["shape"] = [arrays[short].shape[1]]
        info["features"][feat]["names"] = [str(n) for n in c[f"{short}_names"]]
    info_p.write_text(json.dumps(info))

    stats_json = {}
    for short, feat in FEATS.items():
        if f"stats_{short}_mean" in c:
            st = {k: c[f"stats_{short}_{k}"].astype(np.float64) for k in STATS}
        else:
            st = _stats(arrays[short])
        stats_json[feat] = {k: v.tolist() for k, v in st.items()} | {"count": [int(sum(cat_len))]}
    if "no_stats_json" in c:
        (root / "meta" / "stats.json").unlink()  # the real dataset ships none
    else:
        (root / "meta" / "stats.json").write_text(json.dumps(stats_json))

    present = set(data_eps)
    meta = {"episode_index": cat_ids, "length": cat_len,
            "tasks": [["task"] for _ in cat_ids],
            "data/chunk_index": [0] * len(cat_ids),
            # catalogued episodes with no frames point at a shard that does
            # not exist: a partial sample, not missing rows in a local shard
            "data/file_index": [0 if e in present else 999 for e in cat_ids]}
    for short, feat in FEATS.items():
        for k in STATS:
            key = f"epstats_{short}_{k}"
            if key in c:
                vals = c[key].astype(np.float64)
            else:
                vals = np.stack([_stats(arrays[short][ep == e])[k] if e in present
                                 else np.full(arrays[short].shape[1], np.nan) for e in cat_ids])
            meta[f"stats/{feat}/{k}"] = [list(r) for r in vals]
        meta[f"stats/{feat}/count"] = [[n] for n in cat_len]
    pq.write_table(pa.table(meta), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return root

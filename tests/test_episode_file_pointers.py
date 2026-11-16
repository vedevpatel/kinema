"""episode-file-pointers: stale v3 shard metadata (the lerobot#2401 converter bug).

At the file-size rollover the v2.1->v3.0 converter wrote the boundary
episode into the new shard but left its meta/episodes pointer aimed at the
previous one. LeRobot loads only the metadata-named file, so the episode
silently vanishes from training while the dataset looks complete on disk -
and kinema's own loader reads the shards directly, so every other check
sees the data as healthy. Verified live against allenai/MolmoAct2-RT-1-Dataset
(3 of 87212 episodes), unitreerobotics/G1_Dex3_ToastedBread_Dataset (1 of 418)
and LightwheelAI/Lightwheel-Tasks-X7S (36 of 10300); the sibling G1_Dex3
exports without a rollover are clean.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import DatasetContext, load_dataset
from kinema.fixtures.synth import make_dataset

CHECK = {c.id: c for c in ALL_CHECKS}["episode-file-pointers"]


def _run(root):
    return CHECK.run(load_dataset(str(root)), CheckConfig())


def test_clean_sharded_dataset_passes(tmp_path):
    res = _run(make_dataset(tmp_path / "clean", episodes=10, shards=2))
    assert res.verdict == Severity.OK, res.summary
    assert "10 episodes, 2 shard(s)" in res.summary


def test_single_shard_dataset_passes(tmp_path):
    res = _run(make_dataset(tmp_path / "one", episodes=8))
    assert res.verdict == Severity.OK, res.summary


def test_stale_pointer_is_flagged_with_declared_and_actual(tmp_path):
    root = make_dataset(tmp_path / "broken", episodes=10, shards=2,
                        defects=["stale-file-pointer"])
    res = _run(root)
    assert res.verdict == Severity.WARN
    hits = [f for f in res.findings if f.severity == Severity.WARN]
    assert len(hits) == 1, [f.message for f in res.findings]
    hit = hits[0]
    # episodes fill shards contiguously: 0-4 in file-000, 5-9 in file-001;
    # the converter bug leaves episode 5 pointing at file-000
    assert hit.episode_index == 5
    assert hit.details["declared"].endswith("file-000.parquet")
    assert hit.details["actual"].endswith("file-001.parquet")
    assert "silently drops out of training" in hit.message
    assert "lerobot/issues/2401" in res.summary
    assert res.details["broken"] == [5]


def test_stale_pointer_needs_two_shards(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="shards >= 2"):
        make_dataset(tmp_path / "x", episodes=8, defects=["stale-file-pointer"])


def _stub_ctx(root, version="v3.0"):
    n = np.zeros((4, 2))
    return DatasetContext(
        root=Path(root),
        repo_id=None,
        format_version=version,
        info={"data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"},
        stats={},
        fps=10.0,
        episodes=[],
        episode_index=np.zeros(4, dtype=np.int64),
        frame_index=np.zeros(4, dtype=np.int64),
        timestamp=np.zeros(4),
        state=n,
        action=n,
        task_index=None,
    )


def test_v21_layout_skips(tmp_path):
    res = CHECK.run(_stub_ctx(tmp_path, version="v2.1"), CheckConfig())
    assert res.skipped and "v2.x" in res.skip_reason


def test_meta_without_shard_columns_skips(tmp_path):
    root = make_dataset(tmp_path / "noshard", episodes=8)
    f = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    pq.write_table(t.select([c for c in t.column_names if not c.startswith("data/")]), f)
    res = _run(root)
    assert res.skipped and "shard" in res.skip_reason


def test_undownloaded_shards_are_outside_coverage(tmp_path):
    # a sampled audit downloads only some shards: episodes declared in a
    # shard that is not local cannot be judged, and the summary says so
    root = tmp_path / "partial"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({
            "episode_index": pa.array([0, 1], type=pa.int64()),
            "frame_index": pa.array([0, 0], type=pa.int64()),
            "timestamp": pa.array([0.0, 0.0]),
            "action": pa.array([[0.0], [0.0]], type=pa.list_(pa.float64())),
        }),
        root / "data" / "chunk-000" / "file-000.parquet",
    )
    pq.write_table(
        pa.table({
            "episode_index": pa.array([0, 1, 2, 3], type=pa.int64()),
            "data/chunk_index": pa.array([0, 0, 0, 0], type=pa.int64()),
            "data/file_index": pa.array([0, 0, 1, 1], type=pa.int64()),
        }),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    res = CHECK.run(_stub_ctx(root), CheckConfig())
    assert res.verdict == Severity.OK, res.summary
    assert "checked 2 of 4" in res.summary


def test_no_local_shards_skips(tmp_path):
    root = tmp_path / "empty"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(
        pa.table({
            "episode_index": pa.array([0], type=pa.int64()),
            "data/chunk_index": pa.array([0], type=pa.int64()),
            "data/file_index": pa.array([0], type=pa.int64()),
        }),
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
    )
    res = CHECK.run(_stub_ctx(root), CheckConfig())
    assert res.skipped and "no declared data shards" in res.skip_reason

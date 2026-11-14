"""Sampled Hub downloads, threshold config files, and stale help text."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinema.checks import CheckConfig, load_config_overrides
from kinema.dataset import _spread, _target_data_files


def _v3_meta(root, mapping):
    """meta/episodes with shard locations, as LeRobot v3 writes them."""
    d = root / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([m[0] for m in mapping], type=pa.int64()),
                "length": pa.array([10] * len(mapping), type=pa.int64()),
                "data/chunk_index": pa.array([m[1] for m in mapping], type=pa.int64()),
                "data/file_index": pa.array([m[2] for m in mapping], type=pa.int64()),
            }
        ),
        d / "file-000.parquet",
    )


def _v21_meta(root, ids):
    (root / "meta").mkdir(parents=True, exist_ok=True)
    with open(root / "meta" / "episodes.jsonl", "w") as f:
        for i in ids:
            f.write(json.dumps({"episode_index": i, "length": 10}) + "\n")


def test_v3_targets_only_the_shards_holding_the_sample(tmp_path):
    # 200 episodes over 20 shards; a 10-episode sample touches a few shards
    mapping = [(ep, ep // 10, 0) for ep in range(200)]
    _v3_meta(tmp_path, mapping)
    targets = _target_data_files(tmp_path, 10)
    assert targets is not None
    assert all(t.startswith("data/chunk-") for t in targets)
    assert len(targets) <= 10
    # first and last episodes are always in a spread sample
    assert "data/chunk-000/file-000.parquet" in targets
    assert "data/chunk-019/file-000.parquet" in targets


def test_v21_targets_one_file_per_sampled_episode(tmp_path):
    _v21_meta(tmp_path, list(range(100)))
    targets = _target_data_files(tmp_path, 5)
    assert len(targets) == 5
    assert "data/**/episode_000000.parquet" in targets
    assert "data/**/episode_000099.parquet" in targets


def test_sample_at_or_above_total_downloads_everything(tmp_path):
    _v21_meta(tmp_path, list(range(10)))
    assert _target_data_files(tmp_path, 50) == ["data/**"]


def test_unknown_layout_falls_back_to_full_download(tmp_path):
    # episodes parquet without shard locations
    d = tmp_path / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pq.write_table(
        pa.table({"episode_index": pa.array([0, 1, 2], type=pa.int64()),
                  "length": pa.array([5, 5, 5], type=pa.int64())}),
        d / "file-000.parquet",
    )
    assert _target_data_files(tmp_path, 1) is None
    # no episode metadata at all
    assert _target_data_files(tmp_path / "elsewhere", 1) is None


def test_spread_includes_first_and_last():
    got = _spread(list(range(206)), 50)
    assert got[0] == 0 and got[-1] == 205 and len(got) == 50


def test_config_overrides_apply(tmp_path):
    p = tmp_path / "tuning.json"
    p.write_text(json.dumps({"stats_std_fail": 0.4, "idle_min_seconds": 5}))
    cfg = CheckConfig(**load_config_overrides(str(p)))
    assert cfg.stats_std_fail == 0.4
    assert cfg.idle_min_seconds == 5
    assert cfg.stats_std_warn == CheckConfig().stats_std_warn


def test_config_rejects_unknown_keys_and_flags(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"stats_std_fial": 0.4}))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config_overrides(str(p))
    p.write_text(json.dumps({"video": True}))
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config_overrides(str(p))
    p.write_text(json.dumps({"stats_std_fail": "high"}))
    with pytest.raises(ValueError, match="must be numbers"):
        load_config_overrides(str(p))


def test_spilled_load_matches_in_memory(tmp_path, monkeypatch):
    import numpy as np

    from kinema.checks import ALL_CHECKS, CheckConfig
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset

    root = make_dataset(tmp_path / "ds", episodes=12,
                        defects=["idle-segments", "timestamp-grid", "calibration-drift"])
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "0")  # never spill
    a = load_dataset(str(root))
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "1")  # always spill
    b = load_dataset(str(root))

    assert isinstance(b.state, np.memmap) or isinstance(b.episode_index, np.memmap)
    np.testing.assert_array_equal(a.state, np.asarray(b.state))
    np.testing.assert_array_equal(a.action, np.asarray(b.action))
    np.testing.assert_array_equal(a.episode_index, np.asarray(b.episode_index))
    np.testing.assert_array_equal(a.timestamp, np.asarray(b.timestamp))
    assert a.total_episodes == b.total_episodes and a.truncated == b.truncated

    cfg = CheckConfig()
    va = {c.id: c.run(a, cfg).verdict for c in ALL_CHECKS}
    vb = {c.id: c.run(b, cfg).verdict for c in ALL_CHECKS}
    assert va == vb


def test_spilled_sampled_load(tmp_path, monkeypatch):
    import numpy as np

    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset

    root = make_dataset(tmp_path / "ds", episodes=24)
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "1")
    ctx = load_dataset(str(root), max_episodes=8)
    assert ctx.total_episodes == 8 and ctx.truncated
    assert np.unique(np.asarray(ctx.episode_index)).size == 8


def _v3_meta_with_video(root, mapping):
    """v3 meta/episodes with data AND per-camera video shard locations."""
    d = root / "meta" / "episodes" / "chunk-000"
    d.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([m[0] for m in mapping], type=pa.int64()),
                "length": pa.array([10] * len(mapping), type=pa.int64()),
                "data/chunk_index": pa.array([m[1] for m in mapping], type=pa.int64()),
                "data/file_index": pa.array([m[2] for m in mapping], type=pa.int64()),
                "videos/cam/chunk_index": pa.array([m[1] for m in mapping], type=pa.int64()),
                "videos/cam/file_index": pa.array([m[2] for m in mapping], type=pa.int64()),
            }
        ),
        d / "file-000.parquet",
    )
    (root / "meta" / "info.json").write_text(json.dumps({
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "chunks_size": 1000,
        "features": {"cam": {"dtype": "video"}},
    }))


def test_v3_video_targets_only_shards_holding_the_sample(tmp_path):
    from kinema.dataset import _target_video_files

    mapping = [(ep, ep // 10, 0) for ep in range(200)]
    _v3_meta_with_video(tmp_path, mapping)
    targets = _target_video_files(tmp_path, 10)
    assert targets is not None
    assert all(t.startswith("videos/cam/chunk-") for t in targets)
    assert len(targets) <= 10  # not the whole 20-shard tree
    assert "videos/cam/chunk-000/file-000.mp4" in targets
    assert "videos/cam/chunk-019/file-000.mp4" in targets


def test_v21_video_targets_one_mp4_per_sampled_episode(tmp_path):
    from kinema.dataset import _target_video_files

    _v21_meta(tmp_path, list(range(100)))
    (tmp_path / "meta" / "info.json").write_text(json.dumps({
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "features": {"observation.images.top": {"dtype": "video"}},
    }))
    targets = _target_video_files(tmp_path, 5)
    assert targets is not None and len(targets) == 5
    assert "videos/chunk-000/observation.images.top/episode_000000.mp4" in targets
    assert "videos/chunk-000/observation.images.top/episode_000099.mp4" in targets


def test_video_sample_at_or_above_total_downloads_everything(tmp_path):
    from kinema.dataset import _target_video_files

    _v21_meta(tmp_path, list(range(10)))
    (tmp_path / "meta" / "info.json").write_text(json.dumps({
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {"cam": {"dtype": "video"}},
    }))
    assert _target_video_files(tmp_path, 50) == ["videos/**"]


def test_video_targets_none_without_video_path(tmp_path):
    from kinema.dataset import _target_video_files

    _v21_meta(tmp_path, list(range(10)))
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"features": {}}))
    assert _target_video_files(tmp_path, 5) is None


def _spill_dirs(tmp):
    return [p for p in tmp.iterdir() if p.name.startswith("kinema-frames-")]


def test_spilled_load_leaves_nothing_in_the_temp_dir(tmp_path, monkeypatch):
    """Regression: every spilled load left its full frame copy under
    kinema-frames-* forever - gigabytes per call in a long-lived MCP server.
    On POSIX the files are unlinked as soon as they are mapped, so nothing
    is on disk even while the context is alive."""
    import gc
    import os
    import tempfile

    import numpy as np

    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset

    root = make_dataset(tmp_path / "ds", episodes=12)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "0")
    a = load_dataset(str(root))
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "1")
    b = load_dataset(str(root))
    assert isinstance(b.state, np.memmap) or isinstance(b.episode_index, np.memmap)
    if os.name == "posix":
        assert _spill_dirs(tmp) == []  # gone while the arrays are still in use
    np.testing.assert_array_equal(a.state, np.asarray(b.state))  # data still readable
    np.testing.assert_array_equal(a.action, np.asarray(b.action))
    del b
    gc.collect()
    assert _spill_dirs(tmp) == []


def test_cancelled_spilled_load_leaves_nothing(tmp_path, monkeypatch):
    import tempfile

    import pyarrow.parquet as pq
    import pytest

    from kinema.dataset import CancelledError, load_dataset
    from kinema.fixtures.synth import make_dataset

    root = make_dataset(tmp_path / "ds", episodes=12)
    # several row groups, so the cancel lands after some frames were spilled
    for f in (root / "data").rglob("*.parquet"):
        pq.write_table(pq.read_table(f), f, row_group_size=50)
    tmp = tmp_path / "tmp"
    tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    monkeypatch.setenv("KINEMA_FRAME_BUDGET_BYTES", "1")

    class CancelAt:
        """is_set() turns true on call `at`; counts calls otherwise."""

        def __init__(self, at=None):
            self.at, self.calls = at, 0

        def is_set(self):
            self.calls += 1
            return self.at is not None and self.calls >= self.at

    probe = CancelAt()
    load_dataset(str(root), cancelled=probe)
    assert probe.calls > 4  # 2 pre-read checkpoints + one per row group
    with pytest.raises(CancelledError, match="while reading"):
        load_dataset(str(root), cancelled=CancelAt(at=probe.calls))  # the last row group
    assert _spill_dirs(tmp) == []

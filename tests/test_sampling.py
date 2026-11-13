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

"""Regressions for the loader items from the 2026-11-18 re-audit."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinema.dataset import DatasetError, _hub_address, load_dataset
from kinema.fixtures.synth import make_dataset


@pytest.mark.parametrize("src, want", [
    ("lerobot/pusht@v3.0//sub/ds", ("lerobot/pusht", "v3.0", "sub/ds")),
    ("lerobot/pusht//sub/ds@v3.0", ("lerobot/pusht", "v3.0", "sub/ds")),
    ("lerobot/pusht//sub/ds/", ("lerobot/pusht", None, "sub/ds")),
    ("owner/n@refs/pr/1", ("owner/n", "refs/pr/1", "")),
])
def test_hub_address_both_orders(src, want):
    assert _hub_address(src) == want


def _rewrite(root, fn):
    f = root / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(fn(pq.read_table(f)), f)


def test_null_timestamp_and_frame_index_read_and_are_reported(tmp_path):
    root = make_dataset(tmp_path / "n", episodes=6)

    def nullify(t):
        ts = t.column("timestamp").to_pylist()
        ts[3] = None
        fi = t.column("frame_index").to_pylist()
        fi[5] = None
        t = t.set_column(t.schema.get_field_index("timestamp"), "timestamp", pa.array(ts, pa.float64()))
        return t.set_column(t.schema.get_field_index("frame_index"), "frame_index", pa.array(fi, pa.int64()))
    _rewrite(root, nullify)
    ctx = load_dataset(str(root))
    assert not ctx.unreadable_files
    assert ctx.null_counts == {"timestamp": 1, "frame_index": 1}
    from kinema.checks import ALL_CHECKS, CheckConfig, Severity
    res = {c.id: c for c in ALL_CHECKS}["dropped-frames"].run(ctx, CheckConfig())
    assert sum(1 for f in res.findings if f.severity == Severity.WARN and "null value" in f.message) == 2


def test_null_state_row_becomes_nan_not_a_crash(tmp_path):
    root = make_dataset(tmp_path / "s", episodes=4)

    def null_row(t):
        rows = t.column("observation.state").to_pylist()
        rows[2] = None
        i = t.schema.get_field_index("observation.state")
        return t.set_column(i, "observation.state", pa.array(rows, pa.list_(pa.float64())))
    _rewrite(root, null_row)
    ctx = load_dataset(str(root))
    assert np.isnan(ctx.state[2]).all() and np.isfinite(ctx.state[3]).all()


def test_string_tasks_column_is_one_task_per_episode(tmp_path):
    root = make_dataset(tmp_path / "t", episodes=3)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m["tasks"] = ["pick cube"] * len(m["episode_index"])
    pq.write_table(pa.table(m), mf)
    assert load_dataset(str(root)).episodes[0].tasks == ["pick cube"]


def test_corrupt_episode_metadata_is_a_dataset_error(tmp_path):
    root = make_dataset(tmp_path / "c", episodes=3)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    mf.write_bytes(mf.read_bytes()[:40])
    with pytest.raises(DatasetError):
        load_dataset(str(root))


@pytest.mark.parametrize("bad", ["0", "-1", "x"])
def test_cli_rejects_bad_max_episodes(tmp_path, bad):
    root = make_dataset(tmp_path / "m", episodes=3)
    proc = subprocess.run([sys.executable, "-m", "kinema.cli", "check", str(root), "--max-episodes", bad],
                          capture_output=True, text=True)
    assert proc.returncode == 2 and "max-episodes" in proc.stderr


def test_fetch_board_serves_a_fresh_cache_without_fetching(tmp_path):
    from kinema.similar import fetch_board
    calls = []

    def fetcher(url):
        calls.append(url)
        return json.dumps({"generated": "x", "datasets": [{"id": "a/b"}]}).encode()
    fetch_board(cache_dir=tmp_path, fetcher=fetcher, max_age_s=3600)
    entries, prov = fetch_board(cache_dir=tmp_path, fetcher=fetcher, max_age_s=3600)
    assert len(calls) == 1 and prov["cached"] and entries == [{"id": "a/b"}]


def test_shard_map_follows_info_data_path(tmp_path):
    from kinema.dataset import _meta_shard_map
    root = make_dataset(tmp_path / "dp", episodes=3)
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m["data/chunk_index"] = [0] * len(m["episode_index"])
    m["data/file_index"] = [0] * len(m["episode_index"])
    pq.write_table(pa.table(m), mf)
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["data_path"] = "data/episodes-{file_index:04d}.parquet"
    p.write_text(json.dumps(info))
    assert set(_meta_shard_map(root).values()) == {"data/episodes-0000.parquet"}


@pytest.mark.parametrize("body, msg", [
    ('{"idle_min_seconds": NaN}', "finite"),
    ('{"stats_sample_bootstrap": 0.5}', "whole numbers"),
])
def test_config_rejects_nan_and_fractional_integer_settings(tmp_path, body, msg):
    from kinema.checks.base import load_config_overrides
    p = tmp_path / "c.json"
    p.write_text(body)
    with pytest.raises(ValueError, match=msg):
        load_config_overrides(str(p))

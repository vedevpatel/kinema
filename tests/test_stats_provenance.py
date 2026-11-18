"""stats-f32-fingerprint: stored per-episode stats that reproduce a naive
float32 accumulation bitwise are converter numerics, not data statistics."""

from __future__ import annotations

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset

CFG = CheckConfig()
CHECK = {c.id: c for c in ALL_CHECKS}["stats-f32-fingerprint"]


def run(ctx):
    return CHECK.run(ctx, CFG)


def test_fingerprint_fires_on_naive_f32_stats(tmp_path):
    root = make_dataset(tmp_path / "d", episodes=8, defects=["f32-naive-episode-stats"])
    res = run(load_dataset(str(root)))
    infos = [f for f in res.findings if f.severity == Severity.INFO]
    assert len(infos) == 1
    assert "naive float32 accumulation" in infos[0].message
    assert "accumulation-rounded, not data-exact" in infos[0].message
    # no WARN/FAIL ever: provenance is informational
    assert not [f for f in res.findings if f.severity >= Severity.WARN]


def test_exact_f64_stats_stay_silent(tmp_path):
    root = make_dataset(tmp_path / "d", episodes=8)
    res = run(load_dataset(str(root)))
    assert res.findings == []


def test_drifted_stats_match_neither_and_stay_silent(tmp_path):
    # stats-drift multiplies mean/std by 1.08: matches neither the naive-f32
    # recomputation nor the data - stats-drift's job, not this check's
    root = make_dataset(tmp_path / "d", episodes=8, defects=["stats-drift"])
    res = run(load_dataset(str(root)))
    assert res.findings == []


def test_single_episode_dataset_fires(tmp_path):
    root = make_dataset(tmp_path / "d", episodes=1, defects=["f32-naive-episode-stats"])
    res = run(load_dataset(str(root)))
    assert len([f for f in res.findings if f.severity == Severity.INFO]) == 1


def test_no_episode_stats_skips(tmp_path):
    root = make_dataset(tmp_path / "d", episodes=4)
    ctx = load_dataset(str(root))
    ctx.episode_stats = {}
    res = run(ctx)
    assert res.skipped

"""Glitch frames that set a dim's normalization (IPEC-COMMUNITY/dobbe_lerobot:
odometry glitches to z = -132 m in 6 of 5208 episodes inflate z's std 15x)."""
import numpy as np

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset
from test_false_positives import _rewrite_data

CHECK = {c.id: c for c in ALL_CHECKS}["nan-and-normalization"]


def dom(res):
    return [f for f in res.findings if f.details.get("kind") == "dominating-outliers"]


def test_glitch_frames_that_inflate_the_std_are_named(tmp_path):
    root = make_dataset(tmp_path / "g", episodes=24)

    def glitch(s, a, ep):
        s = s.copy()
        rows = np.flatnonzero(ep == 7)[:6]
        s[rows, 2] = -132.0  # a few frames, far outside the bulk
        return s, a
    _rewrite_data(root, glitch)
    hits = dom(CHECK.run(load_dataset(str(root)), CheckConfig()))
    assert len(hits) == 1 and hits[0].severity == Severity.WARN
    assert hits[0].details["dim"] == 2 and hits[0].details["episodes"] == [7]


def test_rare_moves_of_a_parked_joint_are_its_signal(tmp_path):
    root = make_dataset(tmp_path / "p", episodes=24)

    def parked(s, a, ep):
        s = s.copy()
        s[:, 3] = 0.5  # parked: flat bulk
        rows = np.flatnonzero(ep == 3)[:10]
        s[rows, 3] = 2.0  # one rare move
        return s, a
    _rewrite_data(root, parked)
    assert dom(CHECK.run(load_dataset(str(root)), CheckConfig())) == []


def test_two_level_dims_and_common_extremes_are_not_flagged(tmp_path):
    root = make_dataset(tmp_path / "t", episodes=24)

    def shapes(s, a, ep):
        s, a = s.copy(), a.copy()
        a[:, 5] = np.where(np.sin(np.arange(len(a)) / 7) > 0, 1.0, 0.0)  # gripper
        rows = np.flatnonzero(np.isin(ep, [1, 9, 17]))  # ~12% of frames: a mode, not glitches
        s[rows, 1] += 50.0
        return s, a
    _rewrite_data(root, shapes)
    assert dom(CHECK.run(load_dataset(str(root)), CheckConfig())) == []


def test_sampled_load_names_a_stored_range_far_beyond_the_sample(tmp_path):
    import json

    from test_findings_batch4 import _set_true_stats, _strip_episode_stats
    root = make_dataset(tmp_path / "s", episodes=40)

    def glitch(a, ep):
        a = a.copy()
        a[np.flatnonzero(ep == 13)[:3], 0] = -500.0  # unloaded by a 6-episode spread
        return a
    _set_true_stats(root, glitch)
    _strip_episode_stats(root)
    ctx = load_dataset(str(root), max_episodes=6)
    assert 13 not in ctx.episode_ids()
    res = {c.id: c for c in ALL_CHECKS}["stats-drift"].run(ctx, CheckConfig())
    hint = [f for f in res.findings if "dominate this dim's normalization" in f.message]
    assert hint and all(f.severity == Severity.INFO for f in hint), [f.message for f in res.findings]
    assert json.dumps(hint[0].details)  # serializable

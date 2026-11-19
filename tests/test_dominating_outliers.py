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


def test_diagnosis_names_the_dominating_outliers_cause(tmp_path):
    # the MCP layer must not explain it as a constant / near-zero-variance dim
    from kinema.diagnose import diagnose
    root = make_dataset(tmp_path / "d", episodes=24)

    def glitch(s, a, ep):
        s = s.copy()
        s[np.flatnonzero(ep == 7)[:6], 2] = -132.0
        return s, a
    _rewrite_data(root, glitch)
    out = diagnose(str(root), "pi0", "policy fails")
    claim = [c["cause_id"] for c in out["causes"]
             if any("inflate the std" in e["message"] for e in c["evidence"])]
    assert claim == ["outlier-dominated-normalization"]


# --- shapes from the 2026-11-19 board run --------------------------------

def _set_dim(root, fn, feat="state"):
    def apply(s, a, ep):
        s, a = s.copy(), a.copy()
        fn(s if feat == "state" else a, ep)
        return s, a
    _rewrite_data(root, apply)


def test_parked_dims_with_numeric_noise_are_not_judged(tmp_path):
    # ucsd dim 5 parked at 0.90 moving in one episode; GR00T dims parked at
    # 100 flipping to -100; both carry tiny numeric noise in the bulk
    root = make_dataset(tmp_path / "pk", episodes=24)
    rng = np.random.default_rng(0)

    def parked(s, ep):
        s[:, 3] = 0.9 + 1e-6 * rng.normal(size=len(s))
        s[np.flatnonzero(ep == 5)[:13], 3] = 1.278
        s[:, 4] = 100.0 + 1e-7 * rng.normal(size=len(s))
        s[np.flatnonzero(ep == 9)[:8], 4] = -100.0
    _set_dim(root, parked)
    assert dom(CHECK.run(load_dataset(str(root)), CheckConfig())) == []


def test_dropout_from_a_live_value_is_still_flagged(tmp_path):
    # hggforget / austin_buds: a live value near 0.95 drops to 0 for a few frames
    root = make_dataset(tmp_path / "do", episodes=24)
    rng = np.random.default_rng(1)

    def dropout(s, ep):
        s[:, 3] = 0.95 + 0.005 * rng.normal(size=len(s))
        s[np.flatnonzero(ep == 11)[[10, 20, 30]], 3] = 0.0
    _set_dim(root, dropout)
    hits = dom(CHECK.run(load_dataset(str(root)), CheckConfig()))
    assert [(h.details["dim"], h.details["episodes"], h.details["at_episode_start"]) for h in hits] == [(3, [11], False)]


def test_reset_transient_is_named_as_one(tmp_path):
    # behavior1k dim 146: every (here: every other) episode's frame 0 sits far
    # outside a tight but live bulk
    root = make_dataset(tmp_path / "rt", episodes=24)
    rng = np.random.default_rng(2)

    def transient(s, ep):
        s[:, 3] = -0.0015 + 2e-5 * rng.normal(size=len(s))
        for e in range(0, 24, 2):
            s[np.flatnonzero(ep == e)[0], 3] = -0.0083
    _set_dim(root, transient)
    hits = dom(CHECK.run(load_dataset(str(root)), CheckConfig()))
    assert len(hits) == 1 and hits[0].details["at_episode_start"]
    assert "reset transient" in hits[0].message


def test_numerically_zero_dim_is_left_to_the_near_zero_rule(tmp_path):
    # behavior1k dim 120: std 1e-11 with first-frame spikes - already a
    # near-zero-variance WARN; not judged twice
    root = make_dataset(tmp_path / "nz", episodes=24)
    rng = np.random.default_rng(3)

    def zeroish(s, ep):
        s[:, 3] = 2e-12 * rng.normal(size=len(s))
        # dims 102/106: a bulk in [0, 2e-8] (wider than the old absolute
        # floor) with frame-0 spikes to -0.021
        s[:, 4] = 2e-8 * rng.random(size=len(s))
        for e in range(0, 24, 3):
            s[np.flatnonzero(ep == e)[0], 3] = 5e-11
            s[np.flatnonzero(ep == e)[0], 4] = -0.021
    _set_dim(root, zeroish)
    res = CHECK.run(load_dataset(str(root)), CheckConfig())
    assert dom(res) == [], [f.message for f in dom(res)]
    assert any(f.details.get("dim") == 3 and "near-zero variance" in f.message for f in res.findings)

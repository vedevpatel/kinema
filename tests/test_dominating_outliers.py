"""Glitch frames that set a dim's normalization (IPEC-COMMUNITY/dobbe_lerobot:
odometry glitches to z = -132 m in 6 of 5208 episodes inflate z's std 15x)."""
import numpy as np
import pytest

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


def test_dropout_on_the_same_dim_of_state_and_action_is_flagged_in_both(tmp_path):
    # the shape hggforget/lerobot_2026-10-26_12-09-32 showed on the board (its
    # extract left the corpus: no declared license): a live value ~2% wide
    # around 0.95 drops to 0 for a few frames mid-episode, in the command and
    # the measurement alike. austin_buds (MIT) keeps the real state-side case.
    root = make_dataset(tmp_path / "sa", episodes=24)
    rng = np.random.default_rng(4)

    def dropout(s, a, ep):
        s, a = s.copy(), a.copy()
        live = 0.95 + 0.005 * rng.normal(size=len(s))
        s[:, 3] = live
        a[:, 3] = live + 0.0005 * rng.normal(size=len(s))
        for e in (4, 11, 19):
            rows = np.flatnonzero(ep == e)[[15, 30]]
            s[rows, 3] = 0.0
            a[rows, 3] = 0.0
        return s, a
    _rewrite_data(root, dropout)
    res = CHECK.run(load_dataset(str(root)), CheckConfig())
    for feat in ("observation.state", "action"):
        hits = [h for h in dom(res) if h.details["feature"] == feat]
        assert [(h.details["dim"], h.details["episodes"], h.details["at_episode_start"]) for h in hits] \
            == [(3, [4, 11, 19], False)], feat


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


# --- heavy tails vs separated populations (2026-11-20 board) ---------------
# GR00T-sim joint_qvel / base_qvel and behavior1k dim 146: frames beyond 20
# bulk widths that continue a tail as dense just inside the cut - the tip of
# a continuum, not a handful of glitches. Built at board-like frame counts
# (the decision counts frames per octave; a 1440-frame fixture is too thin
# to show a tail and falls back to flagging). Every shape runs at several
# unit scales: the rule is in bulk widths and frame counts only.

from corpus import write_case  # noqa: E402

SCALES = (1e-6, 1e-3, 1.0, 180 / np.pi, 1e3, 1e6)
_EPS, _LEN = 40, 1500


def _case(tmp_path, state, *, no_frame_index=False):
    n = _EPS * _LEN
    ep = np.repeat(np.arange(_EPS), _LEN).astype(np.int32)
    c = {"state": state.astype(np.float64), "action": np.zeros((n, 1)),
         "episode_index": ep, "frame_index": np.tile(np.arange(_LEN), _EPS).astype(np.int32),
         "fps": np.float64(30.0), "state_names": np.array([f"s{i}" for i in range(state.shape[1])]),
         "action_names": np.array(["a0"])}
    if no_frame_index:
        c["no_frame_index"] = np.int8(1)
    return load_dataset(str(write_case(tmp_path / "ds", c)))


def _live(rng):
    # an ordinary moving joint: smooth per-episode motion plus sensor noise
    t = np.arange(_EPS * _LEN) / _LEN
    return np.sin(2 * np.pi * t + rng.uniform(0, 6)) + 0.01 * rng.normal(size=len(t))


def _heavy_tail(rng):
    # 0.6% spikes whose magnitudes are log-uniform from the bulk out to
    # ~100 bulk widths: every octave holds about as many frames as the next
    x = rng.normal(size=_EPS * _LEN)
    k = rng.random(len(x)) < 0.006
    x[k] = rng.choice([-1, 1], k.sum()) * 10 ** rng.uniform(0.3, 2.7, k.sum())
    return x


def _old_rule_fires(x):
    # the shape must sit inside the detection zone the cut-at-20 rule used,
    # or a test that it is not flagged proves nothing
    q01, q50, q99 = np.percentile(x, [1, 50, 99])
    far = np.abs(x - q50) > 20 * (q99 - q01)
    return 0 < far.sum() <= 0.01 * len(x) and x.std() >= 3 * x[~far].std()


def _hits(ctx, dim):
    return [f for f in dom(CHECK.run(ctx, CheckConfig())) if f.details["dim"] == dim]


@pytest.mark.parametrize("scale", SCALES)
def test_a_heavy_tail_is_not_a_handful_of_glitches(tmp_path, scale):
    # also the "legitimate sparse signal" of impulsive channels: contact
    # forces whose spikes come in every size
    rng = np.random.default_rng(10)
    tail = _heavy_tail(rng)
    assert _old_rule_fires(tail)
    ctx = _case(tmp_path, np.stack([_live(rng), tail * scale, (tail + 3.0) * scale], axis=1))
    assert not _hits(ctx, 1) and not _hits(ctx, 2), [f.message for f in dom(CHECK.run(ctx, CheckConfig()))]


@pytest.mark.parametrize("scale", SCALES)
def test_near_zero_padding_jitter_is_not_judged(tmp_path, scale):
    # GR00T base_qvel: a padding-like channel at ~1e-7 whose jitter is
    # heavy-tailed - above the near-zero rule, not flat enough to be parked
    rng = np.random.default_rng(11)
    pad = 1e-7 * _heavy_tail(rng)
    assert _old_rule_fires(pad)
    ctx = _case(tmp_path, np.stack([_live(rng) * scale, pad * scale], axis=1))
    assert not _hits(ctx, 1)


@pytest.mark.parametrize("scale", SCALES)
def test_glitches_on_a_heavy_tailed_channel_are_still_found(tmp_path, scale):
    # the cut walks out past the tail and finds the separated population
    rng = np.random.default_rng(12)
    x = _heavy_tail(rng)
    # ~400 bulk widths: past the tail (which ends near 100) but inside the
    # parked rule's 1000-width ceiling (see test_parked_exemption_...)
    glitch = np.concatenate([3 * _LEN + np.arange(100, 108), 17 * _LEN + np.arange(50, 56),
                             29 * _LEN + np.arange(700, 706)])
    x[glitch] = 2000.0
    hits = _hits(_case(tmp_path, np.stack([_live(rng), x * scale], axis=1)), 1)
    assert len(hits) == 1
    assert hits[0].details["frames"] == len(glitch) and hits[0].details["episodes"] == [3, 17, 29]
    assert hits[0].details["cut_bulks"] > 20 and not hits[0].details["at_episode_start"]


@pytest.mark.parametrize("scale", SCALES)
def test_real_extreme_glitches_are_found_at_every_unit_scale(tmp_path, scale):
    # dobbe (one far value), austin/hggforget (dropouts to 0 from a live
    # value) and a glitch population spread over three decades of distance
    rng = np.random.default_rng(13)
    n = _EPS * _LEN
    odo = _live(rng)
    odo[np.r_[5 * _LEN + np.arange(10, 25), 22 * _LEN + np.arange(400, 415)]] = -132.0
    drop = 0.95 + 0.005 * _live(rng)
    drop[rng.choice(n, 30, replace=False)] = 0.0
    spread = _live(rng)
    sp = rng.choice(n, 25, replace=False)
    spread[sp] = rng.choice([-1, 1], 25) * 10 ** rng.uniform(1.7, 2.7, 25)
    ctx = _case(tmp_path, np.stack([odo, drop, spread], axis=1) * scale)
    assert {f.details["dim"] for f in dom(CHECK.run(ctx, CheckConfig()))} == {0, 1, 2}
    assert _hits(ctx, 0)[0].details["episodes"] == [5, 22]


def test_parked_exemption_still_holds_beside_the_tail_rule(tmp_path):
    # a padding dim at exactly 0 with three frames at 5.0, and a joint parked
    # at 0.9 moving once: flat in their bulk, so not judged at all
    rng = np.random.default_rng(14)
    pad = np.zeros(_EPS * _LEN)
    pad[[100, 200, 300]] = 5.0
    parked = 0.9 + 1e-6 * rng.normal(size=_EPS * _LEN)
    parked[7 * _LEN + 10: 7 * _LEN + 60] = 1.278
    ctx = _case(tmp_path, np.stack([_live(rng), pad, parked], axis=1))
    assert not _hits(ctx, 1) and not _hits(ctx, 2)


@pytest.mark.xfail(strict=True, reason="known gap: a smooth excursion of a jittery-but-idle channel "
                   "(GR00T task-0028's one 10 s grasp, gripper bulk 0.16% of its range) is separated "
                   "and not parked, so it is still flagged; see docs/checks.md")
def test_a_smooth_burst_on_an_idle_channel_is_its_signal(tmp_path):
    rng = np.random.default_rng(15)
    x = 0.0497 + 2e-5 * rng.normal(size=_EPS * _LEN)
    s = 2 * _LEN + 6959 % _LEN
    x[s:s + 300] = 0.0497 * (1 - np.sin(np.linspace(0, np.pi, 300)))  # close and reopen
    assert _old_rule_fires(x)
    assert not _hits(_case(tmp_path, np.stack([_live(rng), x], axis=1)), 1)


def test_a_missing_frame_index_is_not_read_as_episode_starts(tmp_path):
    # GR00T-sim / behavior1k ship no frame_index column; the loader's zero
    # fill made every glitch an "episode start". Position comes from time
    # order instead, both ways round.
    rng = np.random.default_rng(16)
    mid, first = _live(rng), _live(rng)
    mid[np.arange(20) * _LEN + 700] = 200.0
    first[np.arange(0, _EPS, 2) * _LEN] = 200.0
    ctx = _case(tmp_path, np.stack([mid, first], axis=1), no_frame_index=True)
    assert ctx.present_columns is not None and "frame_index" not in ctx.present_columns
    assert not _hits(ctx, 0)[0].details["at_episode_start"]
    assert _hits(ctx, 1)[0].details["at_episode_start"] and "reset transient" in _hits(ctx, 1)[0].message

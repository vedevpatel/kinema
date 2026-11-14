"""Regression tests for findings batch 4: #14 (calibration on delta
commands), #15 (gripper slot prior parked-joint false positives), #19
(sampled loads without per-episode stats get a bootstrap reference), #20
(commands-only idle detection treats held velocity commands as motion)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset
from kinema.stats import inferred_gripper_dims

CHECKS = {c.id: c for c in ALL_CHECKS}
CFG = CheckConfig()


def run(check_id, ctx, cfg=CFG):
    return CHECKS[check_id].run(ctx, cfg)


def gated(res):
    return [f for f in res.findings if f.severity >= Severity.WARN]


# --- finding #20: idle from commands alone ----------------------------------

def test_held_velocity_command_is_motion_not_idle(tmp_path):
    # mobile base driven at a held 0.6 for the back half of episode 4, arm
    # still: every command is constant, but the base is MOVING. The old
    # commands-only fallback flagged this as idle.
    root = make_dataset(tmp_path / "vel", episodes=24, with_state=False, velocity_dim=True)
    res = run("idle-segments", load_dataset(str(root)))
    assert gated(res) == [], [f.message for f in res.findings]
    assert "velocity" in res.summary


def test_parked_base_and_still_arm_is_still_idle(tmp_path):
    # same layout, but the injected idle stretch holds the base at 0: a real
    # idle segment must still be caught without recorded state
    root = make_dataset(tmp_path / "vel2", episodes=24, defects=["idle-segments"],
                        with_state=False, velocity_dim=True)
    res = run("idle-segments", load_dataset(str(root)))
    hits = [f for f in res.findings if f.severity >= Severity.WARN and f.episode_index == 2]
    assert hits, "a true idle stretch must still flag with commands only"


# --- finding #15: gripper slot prior -----------------------------------------

def test_slot_prior_rejects_a_joint_parked_at_one_end():
    # the finding #15 false positive: an unused DOF parked at its limit with
    # a little drift nearby reads as "dwells at an extreme" - but it never
    # VISITS the other end (the quieter end holds ~3% of the dwell)
    rng = np.random.default_rng(0)
    n, d = 4000, 7
    a = rng.normal(size=(n, d))
    parked = int(n * 0.97)
    a[:parked, -1] = -2.00 + 0.01 * rng.normal(size=parked)
    a[parked:, -1] = -1.90 + 0.02 * rng.normal(size=n - parked)
    assert inferred_gripper_dims(a, None) == set()


def test_slot_prior_still_catches_real_grippers():
    rng = np.random.default_rng(0)
    n, d = 4000, 7
    a = rng.normal(size=(n, d))
    # a two-end dweller (scripted open/shut) ...
    a[:, -1] = np.where((np.arange(n) // 20) % 2 == 0, -1.0, 1.0) + 0.01 * rng.normal(size=n)
    assert inferred_gripper_dims(a, None) == {6}
    # ... and a continuous teleop gripper that dwells at both ends but moves
    # between them
    a[:, -1] = np.clip(np.sin(np.arange(n) / 30.0) * 3.0, -1, 1) + 0.01 * rng.normal(size=n)
    assert inferred_gripper_dims(a, None) == {6}


# --- finding #19: sampled stats bootstrap -------------------------------------

def _strip_episode_stats(root):
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf)
    keep = [c for c in m.column_names if not c.startswith("stats/")]
    pq.write_table(m.select(keep), mf)


def _shift_stored_mean(root, sigma):
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    m = np.asarray(stats["action"]["mean"])
    s = np.asarray(stats["action"]["std"])
    stats["action"]["mean"] = (m + sigma * s).tolist()
    p.write_text(json.dumps(stats))


def test_sampled_load_without_episode_stats_catches_shifted_stored_mean(tmp_path):
    root = make_dataset(tmp_path / "s", episodes=40)
    _strip_episode_stats(root)
    # +1.5 std devs: below the old 2-sigma gross limit, far outside the
    # bootstrap interval of a 24-episode sample
    _shift_stored_mean(root, 1.5)
    ctx = load_dataset(str(root), max_episodes=24)
    assert ctx.truncated and not ctx.episode_stats
    res = run("stats-drift", ctx)
    assert any(f.severity == Severity.WARN and f.details.get("reference") == "bootstrap"
               for f in res.findings), [f.message for f in res.findings]


def test_sampled_load_without_episode_stats_clean_data_stays_quiet(tmp_path):
    root = make_dataset(tmp_path / "s2", episodes=40)
    _strip_episode_stats(root)
    ctx = load_dataset(str(root), max_episodes=24)
    res = run("stats-drift", ctx)
    assert gated(res) == [], [f.message for f in res.findings]


# --- finding #14: calibration on delta commands -------------------------------

def test_delta_scale_mismatch_is_caught(tmp_path):
    # conversion-units bug: actions stored at 3x the real steps (measured
    # steps regress on commands with slope 1/3)
    root = make_dataset(tmp_path / "d", episodes=24, action_mode="delta", delta_scale=3.0)
    res = run("calibration-drift", load_dataset(str(root)))
    hits = [f for f in res.findings
            if f.details.get("kind") == "delta_scale" and f.severity >= Severity.WARN]
    assert hits, [f.message for f in res.findings]
    assert abs(hits[0].details["step_ratio"] - 1 / 3) < 0.05


def test_clean_delta_actions_get_a_clean_scale_verdict(tmp_path):
    root = make_dataset(tmp_path / "d2", episodes=24, action_mode="delta")
    res = run("calibration-drift", load_dataset(str(root)))
    assert not res.skipped
    assert res.details["compared_joints"] == []
    assert len(res.details["compared_deltas"]) >= 4
    assert gated(res) == [], [f.message for f in res.findings]

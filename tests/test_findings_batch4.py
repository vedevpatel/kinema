"""Regression tests for findings batch 4: #14 (calibration on delta
commands), #15 (gripper slot prior parked-joint false positives), #19
(sampled loads without per-episode stats get a bootstrap reference), #20
(commands-only idle detection treats held velocity commands as motion)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
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
    # gates again (regression N1): outside the interval AND beyond it by more
    # than the full-load WARN tolerance; the summary must not contradict it
    assert any(f.severity == Severity.WARN and f.details.get("reference") == "bootstrap"
               for f in res.findings), [f.message for f in res.findings]
    assert "matches" not in res.summary


def test_sampled_load_without_episode_stats_clean_data_stays_quiet(tmp_path):
    root = make_dataset(tmp_path / "s2", episodes=40)
    _strip_episode_stats(root)
    ctx = load_dataset(str(root), max_episodes=24)
    res = run("stats-drift", ctx)
    assert gated(res) == [], [f.message for f in res.findings]


# --- finding #14: calibration on delta commands -------------------------------

def test_delta_scale_without_unit_contract_is_informational(tmp_path):
    # The same action/state ratio can be normalization or a conversion bug.
    # Without a common-unit contract, preserve the evidence but don't gate.
    root = make_dataset(tmp_path / "d", episodes=24, action_mode="delta", delta_scale=3.0)
    res = run("calibration-drift", load_dataset(str(root)))
    hits = [f for f in res.findings
            if f.details.get("kind") == "delta_scale" and f.severity == Severity.INFO]
    assert hits, [f.message for f in res.findings]
    assert abs(hits[0].details["step_ratio"] - 1 / 3) < 0.05


def test_clean_delta_actions_get_a_clean_scale_verdict(tmp_path):
    root = make_dataset(tmp_path / "d2", episodes=24, action_mode="delta")
    res = run("calibration-drift", load_dataset(str(root)))
    assert not res.skipped
    assert res.details["compared_joints"] == []
    assert len(res.details["compared_deltas"]) >= 4
    assert gated(res) == [], [f.message for f in res.findings]


# --- audit regressions N1-N4 (a8b3512) -----------------------------------------

def test_sampled_bootstrap_gates_across_material_shifts_and_stays_quiet_when_clean(tmp_path):
    # N1: 1-1.9 sigma stored-mean errors on a sampled load gate (WARN), clean stays ungated
    for sigma in (0.0, 1.0, 1.9):
        root = make_dataset(tmp_path / f"b{sigma}", episodes=40)
        _strip_episode_stats(root)
        if sigma:
            _shift_stored_mean(root, sigma)
        res = run("stats-drift", load_dataset(str(root), max_episodes=24))
        gated = [f for f in res.findings if f.severity >= Severity.WARN]
        assert bool(gated) == bool(sigma), (sigma, [f.message for f in res.findings])


def _constant_dim_with_f32_stats(root, value):
    """Dim 3 constant at `value` in the data, stored stats from a naive float32 sum."""
    from kinema.checks.stats_provenance import _naive_f32_stats
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    for k in ("observation.state", "action"):
        t[k] = [r[:3] + [value] + r[4:] for r in t[k]]
    pq.write_table(pa.table(t), f)
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    n = len(t["action"])
    m, s = _naive_f32_stats(np.full((n, 1), value))
    for k in ("observation.state", "action"):
        stats[k]["mean"][3] = float(m[0])
        stats[k]["std"][3] = float(s[0])
        stats[k]["min"][3] = value
        stats[k]["max"][3] = value
    p.write_text(json.dumps(stats))
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    mt = pq.read_table(mf)
    pq.write_table(mt.select([c for c in mt.column_names if not c.startswith("stats/")]), mf)


def test_constant_large_offset_dim_with_f32_stats_is_inert(tmp_path):
    # N3: RoboCOIN-shaped channel at 4096.7 whose converter-written stored std
    # is ~0.5 of float32 noise: inert (INFO), not a FAIL
    root = make_dataset(tmp_path / "c", episodes=24)
    _constant_dim_with_f32_stats(root, 4096.7)
    res = run("stats-drift", load_dataset(str(root)))
    assert not any(f.severity >= Severity.WARN for f in res.findings), [f.message for f in res.findings]


def test_offset_dim_with_real_motion_still_compared():
    # the bug the |mean| floor once caused (audit #7) must stay fixed
    from kinema.checks.stats_drift import _errors
    actual = {"mean": np.array([1000.0]), "std": np.array([0.9]),
              "min": np.array([997.0]), "max": np.array([1003.0])}
    ref = {**actual, "std": np.array([0.09])}  # stored std 10x too small
    errs, _ = _errors(actual, ref)
    assert errs["std"][0] > 1.0


def test_zero_row_episodes_do_not_blame_correct_episode_stats(tmp_path):
    # N2: rows of 8 of 24 episodes deleted; stats.json and per-episode stats correct
    root = make_dataset(tmp_path / "z", episodes=24)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f)
    import pyarrow.compute as pc
    t = t.filter(pc.invert(pc.is_in(t["episode_index"], value_set=pa.array(list(range(0, 24, 3)), pa.int64()))))
    pq.write_table(t, f)
    ctx = load_dataset(str(root))
    res = run("stats-drift", ctx)
    assert not any(f.severity >= Severity.WARN for f in res.findings), [f.message for f in res.findings]
    assert run("dropped-frames", ctx).verdict == Severity.FAIL  # the missing rows still fail


# --- board re-audit regressions (run 44449956247) ---------------------------

def _set_true_stats(root, fn):
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    ep = np.asarray(t["episode_index"])
    a = fn(np.asarray(t["action"]), ep)
    t["action"] = a.tolist()
    pq.write_table(pa.table(t), f)
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    stats["action"].update(mean=a.mean(0).tolist(), std=a.std(0).tolist(),
                           min=a.min(0).tolist(), max=a.max(0).tolist())
    p.write_text(json.dumps(stats))


def test_small_sample_that_missed_rare_excursions_is_not_wrong_stats(tmp_path):
    # GR00T-sim shaped: a few episodes hold a dim's excursions; correct stored
    # stats; a 6-episode sample misses them and sees a far smaller std
    root = make_dataset(tmp_path / "r", episodes=40)
    rare = (5, 13, 27)

    def excursion(a, ep):
        a = a.copy()
        a[np.isin(ep, rare), 0] += 50.0
        return a
    _set_true_stats(root, excursion)
    _strip_episode_stats(root)
    ctx = load_dataset(str(root), max_episodes=6)
    assert not set(rare) & set(ctx.episode_ids())
    res = run("stats-drift", ctx)
    assert gated(res) == [], [f.message for f in res.findings]
    assert any(f.details.get("range_coverage") is not None for f in res.findings)


def test_small_sample_still_flags_a_gross_stored_std_error(tmp_path):
    # the coverage guard must not excuse wrong stats the sample DID span
    root = make_dataset(tmp_path / "g", episodes=40)
    _strip_episode_stats(root)
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    stats["action"]["std"] = (np.asarray(stats["action"]["std"]) * 10).tolist()
    p.write_text(json.dumps(stats))
    res = run("stats-drift", load_dataset(str(root), max_episodes=6))
    assert any(f.severity == Severity.WARN and f.details.get("reference") == "sampled"
               for f in res.findings), [f.message for f in res.findings]

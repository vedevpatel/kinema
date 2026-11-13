"""Regression tests for false positives found auditing the most-downloaded
Hub datasets: each test builds the benign pattern a real dataset showed
and asserts the check stays quiet, alongside a defect it must still catch."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset

CHECKS = {c.id: c for c in ALL_CHECKS}
CFG = CheckConfig()


def run(check_id, ctx, cfg=CFG):
    return CHECKS[check_id].run(ctx, cfg)


def gated(res):
    return [f for f in res.findings if f.severity >= Severity.WARN]


def _rewrite_data(root, fn):
    """Apply fn(state, action, episode_index) -> (state, action) to the frames
    and refresh stats.json + per-episode stats so they stay truthful."""
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    ep = np.asarray(t["episode_index"])
    s, a = fn(np.asarray(t["observation.state"]), np.asarray(t["action"]), ep)
    t["observation.state"], t["action"] = s.tolist(), a.tolist()
    pq.write_table(pa.table(t), f)
    stats = {}
    for feat, arr in (("observation.state", s), ("action", a)):
        stats[feat] = {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                       "min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "count": [len(arr)]}
    (root / "meta" / "stats.json").write_text(json.dumps(stats))
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    for feat, arr in (("observation.state", s), ("action", a)):
        for i, e in enumerate(m["episode_index"]):
            x = arr[ep == e]
            for k, v in (("mean", x.mean(0)), ("std", x.std(0)), ("min", x.min(0)), ("max", x.max(0))):
                m[f"stats/{feat}/{k}"][i] = v.tolist()
    pq.write_table(pa.table(m), mf)


def _set_names(root, state_names, action_names):
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["features"]["observation.state"]["names"] = state_names
    info["features"]["action"]["names"] = action_names
    p.write_text(json.dumps(info))


@pytest.fixture()
def base(tmp_path):
    return make_dataset(tmp_path / "ds", episodes=24)


# --- stats-drift ------------------------------------------------------------

def test_subset_load_does_not_fake_stats_drift(base):
    # lerobot/pusht: correct stats, but a 50-of-206-episode load differs from
    # whole-dataset stats. The subset is compared to its own per-episode stats.
    def shift(s, a, ep):
        a = a.copy()
        a[ep >= 12] += 3.0  # later episodes live elsewhere -> subset stats differ
        return s, a
    _rewrite_data(base, shift)
    ctx = load_dataset(str(base), max_episodes=8)
    assert ctx.truncated
    assert gated(run("stats-drift", ctx)) == []


def test_subset_still_catches_stale_per_episode_stats(base):
    mf = base / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m["stats/action/std"] = [[v * 1.5 for v in row] for row in m["stats/action/std"]]
    pq.write_table(pa.table(m), mf)
    res = run("stats-drift", load_dataset(str(base), max_episodes=8))
    # caught, but blamed on the per-episode stats (stats.json is right), and
    # only a WARN since v3 training normalizes with stats.json
    assert any(f.severity == Severity.WARN and f.details["stat"] == "std"
               and f.details["reference"] == "per-episode stats" for f in res.findings)
    assert not any(f.details.get("reference") == "meta/stats.json" for f in res.findings)


def test_subset_catches_stored_std_too_small(base):
    p = base / "meta" / "stats.json"
    st = json.loads(p.read_text())
    st["action"]["std"] = [v * 0.5 for v in st["action"]["std"]]  # amplifies inputs 2x
    p.write_text(json.dumps(st))
    res = run("stats-drift", load_dataset(str(base)))
    assert any(f.severity == Severity.FAIL and f.details["stat"] == "std" for f in res.findings)


def test_subset_checks_stats_json_against_episode_stats(base):
    p = base / "meta" / "stats.json"
    st = json.loads(p.read_text())
    st["action"]["mean"] = [v + 5 * s for v, s in zip(st["action"]["mean"], st["action"]["std"])]
    p.write_text(json.dumps(st))
    res = run("stats-drift", load_dataset(str(base), max_episodes=8))
    assert any(f.severity == Severity.FAIL and f.details.get("reference") == "meta/stats.json"
               for f in res.findings)


def test_mean_near_zero_is_not_drift(base):
    # BitRobot: stored mean 0.0012 vs 0.034 on a dim with std ~0.5 was
    # reported as "2777% off". In std units it is noise.
    p = base / "meta" / "stats.json"
    st = json.loads(p.read_text())
    st["action"]["mean"] = [m + 0.01 * s for m, s in zip(st["action"]["mean"], st["action"]["std"])]
    p.write_text(json.dumps(st))
    assert gated(run("stats-drift", load_dataset(str(base)))) == []


def test_v21_episode_stats_count_as_stats(base, tmp_path):
    # v2.1 has no meta/stats.json; LeRobot aggregates episodes_stats.jsonl.
    root = tmp_path / "v21"
    shutil.copytree(base, root)
    m = pq.read_table(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pydict()
    with open(root / "meta" / "episodes.jsonl", "w") as f, open(root / "meta" / "episodes_stats.jsonl", "w") as g:
        for i, e in enumerate(m["episode_index"]):
            f.write(json.dumps({"episode_index": e, "length": m["length"][i], "tasks": m["tasks"][i]}) + "\n")
            stats = {feat: {k: m[f"stats/{feat}/{k}"][i] for k in ("mean", "std", "min", "max", "count")}
                     for feat in ("observation.state", "action")}
            g.write(json.dumps({"episode_index": e, "stats": stats}) + "\n")
    shutil.rmtree(root / "meta" / "episodes")
    (root / "meta" / "stats.json").unlink()
    info = json.loads((root / "meta" / "info.json").read_text())
    info["codebase_version"] = "v2.1"
    (root / "meta" / "info.json").write_text(json.dumps(info))
    ctx = load_dataset(str(root))
    assert ctx.stats_source == "aggregated per-episode stats"
    assert gated(run("deployment-readiness", ctx)) == []
    assert gated(run("stats-drift", ctx)) == []


# --- calibration-drift -------------------------------------------------------

def test_gripper_offset_is_not_calibration_drift(base):
    # ALOHA / SO-100: a gripper commanded shut on an object never gets there.
    def squeeze(s, a, ep):
        a = a.copy()
        a[:, 5] = s[:, 5] - 0.4 * np.ptp(s[:, 5])
        return s, a
    _rewrite_data(base, squeeze)
    res = run("calibration-drift", load_dataset(str(base)))
    assert gated(res) == []
    assert res.details["excluded_joints"]["5"] == "gripper"


def test_ee_state_vs_joint_action_is_not_compared(base):
    # berkeley_mvp: 8-dim EE pose state vs 8-dim joint action
    def ee(s, a, ep):
        rng = np.random.default_rng(1)
        return s, rng.standard_normal(a.shape) * 3
    _rewrite_data(base, ee)
    _set_names(base, ["x", "y", "z", "rx", "ry", "gripper"],
               ["motor_0", "motor_1", "motor_2", "motor_3", "motor_4", "gripper"])
    res = run("calibration-drift", load_dataset(str(base)))
    assert res.skipped


def test_unnamed_delta_actions_are_not_compared(base):
    # xarm / kuka: generic motor_N names, delta actions (no correlation with state)
    def delta(s, a, ep):
        return s, np.r_[np.zeros((1, s.shape[1])), np.diff(s, axis=0)]
    _rewrite_data(base, delta)
    _set_names(base, [f"motor_{i}" for i in range(6)], [f"motor_{i}" for i in range(6)])
    assert run("calibration-drift", load_dataset(str(base))).skipped


def test_real_calibration_drift_still_caught_with_generic_names(tmp_path):
    root = make_dataset(tmp_path / "c", episodes=24, defects=["calibration-drift"])
    _set_names(root, [f"motor_{i}" for i in range(6)], [f"motor_{i}" for i in range(6)])
    res = run("calibration-drift", load_dataset(str(root)))
    assert any(f.severity >= Severity.WARN and f.details["joint"] == 2 for f in res.findings)


# --- stuck-actuator -----------------------------------------------------------

def test_piecewise_constant_commands_are_not_stuck(base):
    # quantized / low-rate teleop: commands hold for 8 frames, then jump
    def hold(s, a, ep):
        a = a.copy()
        for e in np.unique(ep):
            rows = np.flatnonzero(ep == e)
            for k in range(0, len(rows), 8):
                a[rows[k:k + 8]] = a[rows[k]]
        return s, a
    _rewrite_data(base, hold)
    assert gated(run("stuck-actuator", load_dataset(str(base)))) == []


def test_empty_episode_is_flagged(base):
    def freeze(s, a, ep):
        a = a.copy()
        a[ep == 4] = a[np.flatnonzero(ep == 4)[0]]
        return s, a
    _rewrite_data(base, freeze)
    res = run("stuck-actuator", load_dataset(str(base)))
    assert any(f.severity == Severity.WARN and f.details.get("episodes") == [4] and "no joint" in f.message
               for f in res.findings)


# --- action-saturation -------------------------------------------------------

def test_binary_gripper_and_near_limit_motion_are_not_saturation(base):
    def benign(s, a, ep):
        a = a.copy()
        a[:, 5] = np.where(np.sin(np.arange(len(a)) / 7) > 0, 1.0, 0.0)  # open/shut gripper
        return s, a
    _rewrite_data(base, benign)
    assert gated(run("action-saturation", load_dataset(str(base)))) == []

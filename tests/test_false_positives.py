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

def _gripper(s, a, closed_offset, open_offset):
    """Gripper dim 5 that alternates open/closed; offsets in units of its range."""
    s, a = s.copy(), a.copy()
    t = np.arange(len(s))
    closed = (t // 15) % 2 == 0
    s[:, 5] = np.where(closed, 0.0, 1.0) + 0.002 * np.sin(t)
    a[:, 5] = np.where(closed, 0.0 + closed_offset, 1.0 + open_offset)
    return s, a


def test_gripper_blocked_on_object_is_not_calibration_drift(base):
    # ALOHA / SO-100: commanded shut on an object, the gripper never gets
    # there - a big offset at the closed end only
    _rewrite_data(base, lambda s, a, ep: _gripper(s, a, closed_offset=-0.4, open_offset=0.0))
    res = run("calibration-drift", load_dataset(str(base)))
    assert gated(res) == []
    assert 5 in res.details["compared_grippers"]


def test_gripper_offset_at_its_free_end_is_drift(base):
    _rewrite_data(base, lambda s, a, ep: _gripper(s, a, closed_offset=-0.4, open_offset=0.3))
    res = run("calibration-drift", load_dataset(str(base)))
    assert any(f.severity == Severity.WARN and f.details.get("kind") == "gripper" for f in res.findings)


def test_scale_error_is_told_apart_from_offset(base):
    def scale(s, a, ep):
        a = a.copy()
        mid = np.median(s[:, 1])
        a[:, 1] = mid + 1.25 * (s[:, 1] - mid)  # command overshoots by 25% across the range
        return s, a
    _rewrite_data(base, scale)
    res = run("calibration-drift", load_dataset(str(base)))
    hit = [f for f in res.findings if f.details.get("joint") == 1]
    assert hit and hit[0].details["kind"] == "scale" and hit[0].severity >= Severity.WARN


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
    res = run("calibration-drift", load_dataset(str(base)))
    assert res.skipped and "delta command" in res.skip_reason


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


# --- idle-segments ------------------------------------------------------------

def _freeze(a, s, ep, e, sl):
    rows = np.flatnonzero(ep == e)[sl]
    a[rows] = a[rows[0]]
    s[rows] = s[rows[0]]


def test_short_still_tail_is_info_not_warn(base):
    # stopping after the task is done is correct behaviour (synth episodes
    # are 6s, so use a 1s minimum run and a 1.2s = 20% tail)
    def tail(s, a, ep):
        a, s = a.copy(), s.copy()
        _freeze(a, s, ep, 3, slice(-12, None))
        return s, a
    _rewrite_data(base, tail)
    res = run("idle-segments", load_dataset(str(base)), CheckConfig(idle_min_seconds=1.0))
    assert gated(res) == []
    assert any(f.episode_index == 3 and f.severity == Severity.INFO for f in res.findings)


def test_idle_start_is_flagged(base):
    def start(s, a, ep):
        a, s = a.copy(), s.copy()
        _freeze(a, s, ep, 3, slice(0, 25))
        return s, a
    _rewrite_data(base, start)
    res = run("idle-segments", load_dataset(str(base)))
    assert any(f.episode_index == 3 and f.severity == Severity.WARN and "starts idle" in f.message
               for f in res.findings)


def test_held_command_while_arm_moves_is_not_idle(base):
    # action held constant (target pose) while the follower is still travelling
    def hold(s, a, ep):
        a = a.copy()
        rows = np.flatnonzero(ep == 3)[:40]
        a[rows] = a[rows[0]]
        return s, a
    _rewrite_data(base, hold)
    res = run("idle-segments", load_dataset(str(base)))
    assert not any(f.episode_index == 3 for f in res.findings)


def test_frames_ordered_by_timestamp_when_frame_index_is_broken(base):
    # converters that write frame_index=0 everywhere: fall back to time order
    f = base / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    t["frame_index"] = [0] * len(t["frame_index"])
    order = np.random.default_rng(0).permutation(len(t["frame_index"]))
    t = {k: [v[i] for i in order] for k, v in t.items()}
    pq.write_table(pa.table(t), f)
    ctx = load_dataset(str(base))
    for e in ctx.episode_ids():
        assert (np.diff(ctx.timestamp[ctx.rows_for(e)]) > 0).all()


# --- outlier-episodes -----------------------------------------------------------

def test_single_frame_spike_is_flagged(base):
    # columbia_cairlab_pusht_real: a -27 sigma / +25 sigma one-frame glitch
    def spike(s, a, ep):
        a = a.copy()
        r = np.flatnonzero(ep == 7)[30]
        a[r, 1] += 30 * a[:, 1].std()
        return s, a
    _rewrite_data(base, spike)
    res = run("outlier-episodes", load_dataset(str(base)))
    assert any(f.episode_index == 7 and f.severity == Severity.WARN for f in res.findings)


def test_smooth_or_typical_episodes_are_not_outliers(base):
    # pusht ep40 / taco ep39: flagged for LOW jerk or an end pose CLOSER
    # to the median than usual - the harmless direction
    def smooth(s, a, ep):
        a, s = a.copy(), s.copy()
        rows = np.flatnonzero(ep == 7)
        a[rows] = a[rows].mean(axis=0) + 0.05 * (a[rows] - a[rows].mean(axis=0))
        return s, a
    _rewrite_data(base, smooth)
    res = run("outlier-episodes", load_dataset(str(base)))
    assert not any(f.episode_index == 7 and f.severity == Severity.WARN for f in res.findings)


def test_gripper_toggles_do_not_dominate_velocity(base):
    def toggles(s, a, ep):
        a = a.copy()
        a[:, 5] = 0.0
        a[ep == 7, 5] = np.where(np.arange((ep == 7).sum()) % 6 < 3, 0.0, 50.0)  # open/shut
        return s, a
    _rewrite_data(base, toggles)
    res = run("outlier-episodes", load_dataset(str(base)))
    assert not any(f.episode_index == 7 and "velocity" in f.message for f in res.findings)


def test_multitask_end_poses_compared_within_task(base):
    # different tasks end in different places; only a within-task deviation counts
    mf = base / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    left = {0, 5, 10, 15, 20}  # a minority task: globally these look like outliers
    m["tasks"] = [["place left"] if e in left else ["place right"] for e in m["episode_index"]]
    pq.write_table(pa.table(m), mf)

    def ends(s, a, ep):
        s = s.copy()
        for e in np.unique(ep):
            r = np.flatnonzero(ep == e)[-5:]
            s[r] = 5.0 if e in left else -5.0
        return s, a
    _rewrite_data(base, ends)
    res = run("outlier-episodes", load_dataset(str(base)))
    assert not any("end_pose" in f.message for f in res.findings)


# --- timestamps / frame bookkeeping -------------------------------------------

def _set_timestamps(root, fn):
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    t["timestamp"] = fn(np.asarray(t["timestamp"]), np.asarray(t["episode_index"])).tolist()
    pq.write_table(pa.table(t), f)


def test_constant_offset_without_video_is_not_jitter(base):
    # a constant start offset leaves frame spacing intact
    _set_timestamps(base, lambda t, ep: t + 0.05)
    res = run("timestamp-grid", load_dataset(str(base)))
    assert gated(res) == []


def test_wallclock_timestamps_on_video_dataset_fail(base):
    # openarm-packingbench: Unix-time timestamps; video lookup by timestamp misses
    _set_timestamps(base, lambda t, ep: t + 1.78e9)
    p = base / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["features"]["observation.images.top"] = {"dtype": "video", "shape": [8, 8, 3]}
    p.write_text(json.dumps(info))
    res = run("timestamp-grid", load_dataset(str(base)))
    assert any(f.severity == Severity.FAIL and "start at 0" in f.message for f in res.findings)


def test_dropped_frame_gap_is_reported(base):
    def gap(t, ep):
        t = t.copy()
        rows = np.flatnonzero(ep == 4)
        t[rows[30:]] += 0.1  # one missing frame at 10 fps
        return t
    _set_timestamps(base, gap)
    res = run("timestamp-grid", load_dataset(str(base)))
    assert any(f.episode_index == 4 and f.details.get("gaps") == 1 for f in res.findings)


def test_duplicate_catalogued_episode_ids_fail_once(base):
    # koch-arm: merged sessions restarting at 0 list the same index twice
    mf = base / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m2 = {k: v + v[:3] for k, v in m.items()}  # episodes 0-2 catalogued twice
    pq.write_table(pa.table(m2), mf)
    ctx = load_dataset(str(base))
    assert ctx.duplicate_episode_ids == {0: 2, 1: 2, 2: 2}
    res = run("dropped-frames", ctx)
    assert any(f.severity == Severity.FAIL and "more than once" in f.message for f in res.findings)
    dup = run("duplicate-and-leakage", ctx)
    assert not any(len(set(f.details.get("episodes", []))) == 1 for f in dup.findings)


def test_unreadable_data_file_is_reported_not_a_crash(base, tmp_path):
    f = base / "data" / "chunk-000" / "file-000.parquet"
    good = pq.read_table(f)
    half = good.num_rows // 2
    pq.write_table(good.slice(0, half), f)
    pq.write_table(good.slice(half), base / "data" / "chunk-000" / "file-001.parquet")
    broken = base / "data" / "chunk-000" / "file-001.parquet"
    broken.write_bytes(broken.read_bytes()[: len(broken.read_bytes()) // 2])  # truncated
    ctx = load_dataset(str(base))
    assert [n for n, _ in ctx.unreadable_files] == ["data/chunk-000/file-001.parquet"]
    res = run("dropped-frames", ctx)
    assert any(f.severity == Severity.FAIL and "cannot be read" in f.message for f in res.findings)


def test_sampled_episodes_are_spread_not_first(base):
    ctx = load_dataset(str(base), max_episodes=4)
    assert ctx.episode_ids() == [0, 8, 15, 23]


# --- commanded but not followed ---------------------------------------------

def test_dead_follower_motor_is_flagged(base):
    # the leader moves joint 3, the follower's measured position stays put
    def dead(s, a, ep):
        s = s.copy()
        rows = np.flatnonzero(ep == 5)
        s[rows, 3] = s[rows[0], 3]
        return s, a
    _rewrite_data(base, dead)
    res = run("stuck-actuator", load_dataset(str(base)))
    hit = [f for f in res.findings if f.details.get("kind") == "not_followed"]
    assert hit and hit[0].details["joint"] == 3 and hit[0].details["episodes"] == [5]
    assert hit[0].severity == Severity.WARN


def test_follower_never_moving_across_dataset_is_flagged_by_name(base):
    def never(s, a, ep):
        s = s.copy()
        s[:, 2] = 0.0
        return s, a
    _rewrite_data(base, never)
    res = run("stuck-actuator", load_dataset(str(base)))
    assert any(f.details.get("kind") == "never_followed" and f.details["joint"] == 2 for f in res.findings)


def test_delta_actions_with_mostly_generic_names_are_not_paired(base):
    # berkeley_rpt: motor_0..motor_4 + "gripper"; one real name must not vouch
    # for the whole list - delta actions vs joint positions are different spaces
    def delta(s, a, ep):
        return s, 0.01 * np.r_[np.zeros((1, s.shape[1])), np.diff(s, axis=0)]
    _rewrite_data(base, delta)
    _set_names(base, [f"motor_{i}" for i in range(5)] + ["gripper"],
               [f"motor_{i}" for i in range(5)] + ["gripper"])
    res = run("stuck-actuator", load_dataset(str(base)))
    assert not any(f.details.get("kind") in ("not_followed", "never_followed") for f in res.findings)


def test_brief_command_blip_is_not_a_dead_motor(base):
    # Dongkkka linear_x: a 2-frame base command the follower lags behind
    def blip(s, a, ep):
        a, s = a.copy(), s.copy()
        rows = np.flatnonzero(ep == 5)
        s[rows, 3] = s[rows[0], 3]
        a[rows, 3] = a[rows[0], 3]
        a[rows[20:22], 3] += 3 * np.ptp(a[:, 3])
        return s, a
    _rewrite_data(base, blip)
    res = run("stuck-actuator", load_dataset(str(base)))
    assert not any(f.details.get("kind") == "not_followed" and 5 in f.details["episodes"] for f in res.findings)


def test_oversized_action_std_fails_but_state_only_warns(base):
    p = base / "meta" / "stats.json"
    st = json.loads(p.read_text())
    st["action"]["std"] = [v * 4 for v in st["action"]["std"]]
    st["observation.state"]["std"] = [v * 4 for v in st["observation.state"]["std"]]
    p.write_text(json.dumps(st))
    res = run("stats-drift", load_dataset(str(base)))
    sev = {f.details["feature"]: f.severity for f in res.findings if f.details.get("stat") == "std"}
    assert sev["action"] == Severity.FAIL and "down-weighted" in [f.message for f in res.findings
                                                                    if f.details.get("feature") == "action"
                                                                    and f.details.get("stat") == "std"][0]
    assert sev["observation.state"] == Severity.WARN


def test_video_segment_shorter_than_data_fails(base):
    cam = "observation.images.top"
    mf = base / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    n = len(m["episode_index"])
    m[f"videos/{cam}/chunk_index"] = [0] * n
    m[f"videos/{cam}/file_index"] = [0] * n
    m[f"videos/{cam}/from_timestamp"] = [e * 6.0 for e in m["episode_index"]]
    m[f"videos/{cam}/to_timestamp"] = [e * 6.0 + (5.0 if e == 3 else 6.0) for e in m["episode_index"]]
    pq.write_table(pa.table(m), mf)
    res = run("dropped-frames", load_dataset(str(base)))
    bad = [f for f in res.findings if f.details.get("camera") == cam]
    assert len(bad) == 1 and bad[0].severity == Severity.FAIL and bad[0].details["episodes"] == [3]


# --- duplicates ---------------------------------------------------------------

def _paste(a, ep, src, dst, start, n):
    """Copy n frames of episode src (from frame start) into episode dst."""
    a = a.copy()
    rs, rd = np.flatnonzero(ep == src), np.flatnonzero(ep == dst)
    a[rd[start:start + n]] = a[rs[start:start + n]]
    return a


def test_partial_copy_is_flagged(base):
    _rewrite_data(base, lambda s, a, ep: (s, _paste(a, ep, 2, 9, 10, 30)))  # same offset, not the opening
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    assert any(f.details.get("kind") == "partial" and f.details["episodes"] == [2, 9] for f in res.findings)


def test_shared_rest_pose_is_not_a_partial_copy(base):
    # episodes that sit in the same home pose share identical frames - not a copy
    def home(s, a, ep):
        a = a.copy()
        for e in (2, 9):
            a[np.flatnonzero(ep == e)[:30]] = 0.123
        return s, a
    _rewrite_data(base, home)
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    assert not any(f.details.get("kind") == "partial" for f in res.findings)


def test_copy_across_declared_splits_is_leakage(base):
    _rewrite_data(base, lambda s, a, ep: (s, _paste(a, ep, 2, 20, 0, 60)))
    p = base / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["splits"] = {"train": "0:18", "val": "18:24"}
    p.write_text(json.dumps(info))
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    leak = [f for f in res.findings if f.details.get("kind") == "leakage"]
    assert leak and leak[0].details["pairs"] == [[2, 20]]


def test_velocity_command_steps_are_not_outliers(base):
    # Dongkkka linear_x: a base velocity command steps 0 -> v -> 0 by design
    def steps(s, a, ep):
        a = a.copy()
        a[:, 4] = 0.0
        rows = np.flatnonzero(ep == 7)
        a[rows[20:40], 4] = 5.0
        return s, a
    _rewrite_data(base, steps)
    _set_names(base, ["j0", "j1", "j2", "j3", "linear_x", "gripper"],
               ["j0", "j1", "j2", "j3", "linear_x", "gripper"])
    res = run("outlier-episodes", load_dataset(str(base)))
    assert not any(f.episode_index == 7 and f.severity == Severity.WARN for f in res.findings)


def test_backlash_step_is_not_a_scale_error(base):
    # intern-pour joint_6: offset -2% on one side of the range, +3% on the
    # other - a step (backlash / load reversal), not a line
    def step(s, a, ep):
        a = a.copy()
        mid = np.median(s[:, 1])
        a[:, 1] = s[:, 1] + np.where(s[:, 1] < mid, -0.02, 0.03) * np.ptp(s[:, 1])
        return s, a
    _rewrite_data(base, step)
    res = run("calibration-drift", load_dataset(str(base)))
    assert not any(f.details.get("kind") == "scale" for f in res.findings)


def test_big_offset_on_misdetected_joint_still_gated(base):
    # generic names put dim 5 in the gripper slot; if it is really a joint with
    # a large constant offset, the free-end test still catches it
    _set_names(base, [f"motor_{i}" for i in range(6)], [f"motor_{i}" for i in range(6)])

    def offset(s, a, ep):
        a = a.copy()
        a[:, 5] = s[:, 5] + 0.3 * np.ptp(s[:, 5])
        return s, a
    _rewrite_data(base, offset)
    res = run("calibration-drift", load_dataset(str(base)))
    assert any(f.severity >= Severity.WARN and f.details.get("joint") == 5 for f in res.findings)


def test_identical_scripted_opening_is_info_not_warn(base):
    # robomme: scripted demos replay the same opening motion from frame 0
    _rewrite_data(base, lambda s, a, ep: (s, _paste(a, ep, 2, 9, 0, 25)))
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    hit = [f for f in res.findings if f.details.get("kind") == "partial"]
    assert hit and all(f.severity == Severity.INFO and f.details["aligned_opening"] for f in hit)


def test_short_splice_at_another_offset_is_flagged(base):
    def splice(s, a, ep):
        a = a.copy()
        rs, rd = np.flatnonzero(ep == 2), np.flatnonzero(ep == 9)
        a[rd[30:50]] = a[rs[5:25]]  # 20 frames, different offset
        return s, a
    _rewrite_data(base, splice)
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    assert any(f.severity == Severity.WARN and f.details.get("kind") == "partial"
               and f.details["episodes"] == [2, 9] for f in res.findings)


def test_motion_repeated_within_episodes_is_not_a_copy(base):
    # robomme "retrace the pattern": each episode replays its demo motion, and
    # episodes of one task share the pattern - by design, not a copy
    def pattern(s, a, ep):
        a = a.copy()
        motif = a[np.flatnonzero(ep == 0)[5:25]].copy()
        for e in (3, 9):
            r = np.flatnonzero(ep == e)
            a[r[0:20]] = motif
            a[r[30:50]] = motif
        return s, a
    _rewrite_data(base, pattern)
    res = run("duplicate-and-leakage", load_dataset(str(base)))
    assert not any(f.severity >= Severity.WARN and f.details.get("kind") == "partial"
                   and set(f.details["episodes"]) == {3, 9} for f in res.findings)


def test_tiny_episodes_sit_out_jerk_and_velocity(base):
    # utokyo: 8-frame episodes, where p95 jerk is just the single largest step
    f = base / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    ep = np.asarray(t["episode_index"])
    fi = np.asarray(t["frame_index"])
    keep = (ep != 4) | (fi < 8)  # episode 4 becomes 8 frames long
    t = {k: [v for v, m in zip(vals, keep) if m] for k, vals in t.items()}
    a = np.asarray(t["action"])
    rows = np.flatnonzero(np.asarray(t["episode_index"]) == 4)
    a[rows[4], 1] += 3 * a[:, 1].std()  # one big step inside the tiny episode
    t["action"] = a.tolist()
    pq.write_table(pa.table(t), f)
    mf = base / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    m["length"] = [8 if e == 4 else n for e, n in zip(m["episode_index"], m["length"])]
    pq.write_table(pa.table(m), mf)
    res = run("outlier-episodes", load_dataset(str(base)))
    hit = [f for f in res.findings if f.episode_index == 4]
    assert not any(m in ("p95_jerk", "peak_velocity") for f in hit for m in f.details["metrics"])


def test_two_level_toggle_does_not_drive_outliers(base):
    # utokyo motor_3: an unnamed gripper on an odd-sized layout jumps by the
    # same gap every time it switches - not a glitch
    def toggle(s, a, ep):
        a = a.copy()
        for e in np.unique(ep):
            r = np.flatnonzero(ep == e)
            a[r, 2] = np.where(np.arange(len(r)) >= (10 + 3 * (e % 4)), 1.0, 0.0)
        return s, a
    _set_names(base, [f"motor_{i}" for i in range(6)], [f"motor_{i}" for i in range(6)])
    _rewrite_data(base, toggle)
    res = run("outlier-episodes", load_dataset(str(base)))
    assert not any(f.severity == Severity.WARN for f in res.findings)

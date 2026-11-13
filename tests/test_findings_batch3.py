"""Regression tests for the Nov-13 findings batch: outlier tightening (13),
near-duplicate cap lifted (18), structural velocity commands (21),
resolution-relative blur + overexposure (22), metadata checks (24),
wider column aliases (25)."""

from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY, load_dataset
from kinema.fixtures.synth import make_dataset
from kinema.stats import structural_velocity_dims

CHECKS = {c.id: c for c in ALL_CHECKS}
CFG = CheckConfig()


def run(check_id, ctx, cfg=CFG):
    return CHECKS[check_id].run(ctx, cfg)


def _edit_info(root, fn):
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    fn(info)
    p.write_text(json.dumps(info))


def _rewrite_data(root, fn):
    """Apply fn(pydict) -> pydict to the data shard and write it back."""
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    pq.write_table(pa.table(fn(t)), f)


# --- 13. outlier-episodes: fewer, higher-confidence flags -----------------

def test_static_alone_is_no_longer_a_warn(tmp_path):
    """A partly-still episode (static_fraction far out on its own) is
    idle-segments' job, not an outlier WARN."""
    root = make_dataset(tmp_path / "ds", episodes=24)

    def freeze(t):
        a = np.asarray(t["action"])
        a[np.asarray(t["episode_index"]) == 4] = a[0]
        t["action"] = a.tolist()
        return t

    _rewrite_data(root, freeze)
    res = run("outlier-episodes", load_dataset(str(root)))
    warns = [f for f in res.findings if f.severity >= Severity.WARN
             and f.episode_index == 4]
    assert not warns, [f.message for f in warns]
    assert any(f.episode_index == 4 for f in res.findings), \
        "the still episode should still be noted at INFO"


def test_end_pose_far_out_needs_an_absolute_bar(tmp_path):
    """3x the median end-pose deviation only WARNs when the deviation is also
    >= half the end-pose range: ep 5 (0.33 of range) is INFO, ep 6 (1.0) WARNs."""
    root = make_dataset(tmp_path / "ds", episodes=24)

    def bend(t):
        s = np.asarray(t["observation.state"])
        ep = np.asarray(t["episode_index"])
        fi = np.asarray(t["frame_index"])
        for e, shift in ((5, 0.35), (6, 1.05)):
            last = (ep == e) & (fi == fi[ep == e].max())
            s[last] = shift
        # everyone else ends in a tight cluster so the median deviation is tiny
        other = fi == np.array([fi[ep == e].max() for e in range(24)])[ep]
        s[other & (ep != 5) & (ep != 6)] = 0.01 * np.arange(s.shape[1])
        t["observation.state"] = s.tolist()
        return t

    _rewrite_data(root, bend)
    res = run("outlier-episodes", load_dataset(str(root)))
    sev = {f.episode_index: f.severity for f in res.findings
           if "end_pose_deviation" in f.details.get("metrics", [])}
    assert sev.get(5, Severity.OK) <= Severity.INFO
    assert sev.get(6) == Severity.WARN


def test_multi_metric_agreement_still_warns(tmp_path):
    """The 2-family rule and far-out duration still fire."""
    root = make_dataset(tmp_path / "ds", episodes=24, defects=["outlier-episodes"])
    res = run("outlier-episodes", load_dataset(str(root)))
    assert any(f.severity == Severity.WARN and f.episode_index == 23
               for f in res.findings)


# --- 18. near-duplicate scan beyond 10k episodes --------------------------

def test_near_dup_blocks_adapt_and_find_pairs(tmp_path, monkeypatch):
    """A small distance-matrix budget shrinks the block, not the coverage."""
    import kinema.checks.duplicates as dup

    monkeypatch.setattr(dup, "_NEAR_BLOCK_BYTES", 8 * 64)  # block floor 64
    root = make_dataset(tmp_path / "ds", episodes=24)

    def noisy_copy(t):
        rng = np.random.default_rng(0)
        ep = np.asarray(t["episode_index"])
        for key in ("observation.state", "action"):
            a = np.asarray(t[key])
            a[ep == 1] = a[ep == 0] + 0.001 * rng.standard_normal(a[ep == 0].shape)
            t[key] = a.tolist()
        return t

    _rewrite_data(root, noisy_copy)  # near-identical, not exact: only the near scan finds it
    res = run("duplicate-and-leakage", load_dataset(str(root)))
    assert any(f.details.get("kind") == "near" and set(f.details["episodes"]) == {0, 1}
               for f in res.findings)
    assert not any("evenly-spaced sample" in f.message or "skipped" in f.message
                   for f in res.findings)


def test_near_dup_samples_above_the_exact_cap(tmp_path, monkeypatch):
    """Over the exact cap the scan runs on an even sample and says so; the
    exact/partial scans still cover every episode."""
    import kinema.checks.duplicates as dup

    monkeypatch.setattr(dup, "_NEAR_EXACT_MAX", 10)
    root = make_dataset(tmp_path / "ds", episodes=24, defects=["duplicates"])
    res = run("duplicate-and-leakage", load_dataset(str(root)))
    assert any("evenly-spaced sample" in f.message and f.severity == Severity.INFO
               for f in res.findings)
    assert any(set(f.details.get("episodes", [])) == {0, 1} for f in res.findings), \
        "the exact-copy pair is found by the window scan, sampling or not"


# --- 21. velocity commands without the name -------------------------------

class _MiniCtx:
    """Just what structural_velocity_dims reads."""

    def __init__(self, state, action, bounds):
        self.state, self.action, self._bounds = state, action, bounds

    def episode_ids(self):
        return list(range(len(self._bounds)))

    def rows_for(self, ep):
        lo, hi = self._bounds[ep]
        return np.arange(lo, hi)


def _velocity_dataset(n_eps=12, n=60, seed=0):
    """Dim 0 is a velocity command (state integrates it); dim 1 is a position
    command (state tracks it)."""
    rng = np.random.default_rng(seed)
    states, actions, bounds = [], [], []
    for _ in range(n_eps):
        u = np.convolve(rng.standard_normal(n), np.ones(5) / 5, mode="same")
        s0 = np.cumsum(0.1 * u)
        s1 = np.sin(np.linspace(0, 4 * np.pi, n) + rng.uniform(0, 2 * np.pi))
        states.append(np.stack([s0, s1], axis=1))
        actions.append(np.stack([u, s1 + 0.005 * rng.standard_normal(n)], axis=1))
    bounds = [(i * n, (i + 1) * n) for i in range(n_eps)]
    return _MiniCtx(np.concatenate(states), np.concatenate(actions), bounds)


def test_structural_velocity_dims():
    ctx = _velocity_dataset()
    assert structural_velocity_dims(ctx) == {0}


def test_position_commands_are_not_structural_velocity():
    ctx = _velocity_dataset()
    # swap roles: both dims track their state directly
    ctx = _MiniCtx(ctx.state, ctx.state + 0.005 * np.random.default_rng(1).standard_normal(ctx.state.shape),
                   ctx._bounds)
    assert structural_velocity_dims(ctx) == set()


def test_outliers_exclude_structural_velocity_from_motion_metrics(tmp_path):
    """A dataset with generic names and one velocity-command dim: the check
    must not treat the command's steps as jerk outliers."""
    ctx = _velocity_dataset()
    assert structural_velocity_dims(ctx) == {0}
    # and a clean positional dataset yields nothing
    clean = load_dataset(str(make_dataset(tmp_path / "clean", episodes=24)))
    assert structural_velocity_dims(clean) == set()
    res = run("outlier-episodes", clean)
    assert res.verdict <= Severity.INFO, [f.message for f in res.findings]


# --- 24. metadata checks ---------------------------------------------------

def test_embedded_images_flagged(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    _edit_info(root, lambda info: info["features"].update(
        {"observation.images.wrist": {"dtype": "image", "shape": [480, 640, 3]}}))
    res = run("embedded-images", load_dataset(str(root)))
    assert any("observation.images.wrist" in f.message for f in res.findings)


def test_embedded_images_quiet_on_video_layout(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    res = run("embedded-images", load_dataset(str(root)))
    assert res.verdict == Severity.OK


def test_camera_coverage_v21_files(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    cam = "observation.images.top"
    _edit_info(root, lambda info: info["features"].update(
        {cam: {"dtype": "video", "shape": [480, 640, 3]}}))
    d = root / "videos" / "chunk-000" / cam
    d.mkdir(parents=True)
    for ep in range(24):
        if ep not in (3, 7):
            (d / f"episode_{ep:06d}.mp4").write_bytes(b"")
    res = run("camera-coverage", load_dataset(str(root)))
    assert len(res.findings) == 1
    assert res.findings[0].details["episodes"] == [3, 7]
    assert res.findings[0].severity == Severity.INFO  # 2/24 < 10%


def test_camera_coverage_warns_when_many_missing(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    cam = "observation.images.top"
    _edit_info(root, lambda info: info["features"].update(
        {cam: {"dtype": "video", "shape": [480, 640, 3]}}))
    d = root / "videos" / "chunk-000" / cam
    d.mkdir(parents=True)
    for ep in range(18):  # 6/24 = 25% missing
        (d / f"episode_{ep:06d}.mp4").write_bytes(b"")
    res = run("camera-coverage", load_dataset(str(root)))
    assert res.findings[0].severity == Severity.WARN
    assert res.findings[0].details["episodes"] == list(range(18, 24))


def test_camera_coverage_v3_segments(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    cam = "observation.images.top"
    _edit_info(root, lambda info: (info["features"].update(
        {cam: {"dtype": "video", "shape": [480, 640, 3]}}),
        info.update({"video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"})))
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    n = len(m["episode_index"])
    m[f"videos/{cam}/chunk_index"] = [0 if e not in (5, 9) else None for e in m["episode_index"]]
    m[f"videos/{cam}/file_index"] = [0 if e not in (5, 9) else None for e in m["episode_index"]]
    m[f"videos/{cam}/from_timestamp"] = [e * 6.0 if e not in (5, 9) else None for e in m["episode_index"]]
    m[f"videos/{cam}/to_timestamp"] = [(e + 1) * 6.0 if e not in (5, 9) else None for e in m["episode_index"]]
    assert n == 24
    pq.write_table(pa.table(m), mf)
    res = run("camera-coverage", load_dataset(str(root)))
    assert res.findings and res.findings[0].details["episodes"] == [5, 9]


def _set_episode_tasks(root, updates: dict[int, list[str]]):
    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    m = pq.read_table(mf).to_pydict()
    for i, e in enumerate(m["episode_index"]):
        if e in updates:
            m["tasks"][i] = updates[e]
    pq.write_table(pa.table(m), mf)


def test_task_strings_empty(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    _set_episode_tasks(root, {3: [""]})
    res = run("task-strings", load_dataset(str(root)))
    assert any(f.details.get("episodes") == [3] and f.severity == Severity.INFO
               for f in res.findings)


def test_task_strings_dangling_index(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)

    def retask(t):
        ti = np.asarray(t["task_index"])
        ti[np.asarray(t["episode_index"]) == 2] = 7  # no task 7 in meta/tasks
        t["task_index"] = ti.tolist()
        return t

    _rewrite_data(root, retask)
    res = run("task-strings", load_dataset(str(root)))
    assert any(f.severity == Severity.WARN and 7 in f.details.get("task_indices", [])
               for f in res.findings)


def test_task_strings_unknown_task_name(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    _set_episode_tasks(root, {4: ["do something unlisted"]})
    res = run("task-strings", load_dataset(str(root)))
    assert any("no entry in meta/tasks" in f.message and f.severity == Severity.INFO
               for f in res.findings)


def _add_reward_done(root, reward, done):
    def add(t):
        t["reward"] = reward
        t["done"] = done
        return t

    _rewrite_data(root, add)
    _edit_info(root, lambda info: info["features"].update(
        {"reward": {"dtype": "float64", "shape": [1]},
         "done": {"dtype": "bool", "shape": [1]}}))


def _healthy_reward_done(root):
    t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    n = len(t["episode_index"])
    rng = np.random.default_rng(0)
    reward = (0.5 + rng.standard_normal(n) * 0.1).tolist()
    done = np.zeros(n, dtype=bool)
    done[np.asarray(t["frame_index"]) == 59] = True
    _add_reward_done(root, reward, done.tolist())


def test_reward_done_healthy_is_quiet(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    _healthy_reward_done(root)
    ctx = load_dataset(str(root))
    assert ctx.reward is not None and ctx.done is not None
    res = run("reward-done", ctx)
    assert res.verdict == Severity.OK, [f.message for f in res.findings]


def test_reward_done_constant_and_never_terminating(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    n = len(t["episode_index"])
    _add_reward_done(root, [0.0] * n, [False] * n)
    res = run("reward-done", load_dataset(str(root)))
    msgs = " | ".join(f.message for f in res.findings)
    assert "constant" in msgs and "never True" in msgs


def test_reward_done_termination_mid_episode(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    t = pq.read_table(root / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    n = len(t["episode_index"])
    done = np.zeros(n, dtype=bool)
    done[(np.asarray(t["episode_index"]) == 2) & (np.asarray(t["frame_index"]) == 30)] = True
    _add_reward_done(root, (0.5 + np.random.default_rng(0).standard_normal(n) * 0.1).tolist(),
                     done.tolist())
    res = run("reward-done", load_dataset(str(root)))
    assert any("continues after termination" in f.message and f.details["episodes"] == [2]
               for f in res.findings)


def test_reward_done_skips_without_fields(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    res = run("reward-done", load_dataset(str(root)))
    assert res.skipped


# --- 25. wider column aliases ----------------------------------------------

def test_qpos_commands_aliases(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)

    def rename(t):
        t["qpos"] = t.pop("observation.state")
        t["commands"] = t.pop("action")
        return t

    _rewrite_data(root, rename)

    def rename_info(info):
        info["features"]["qpos"] = info["features"].pop("observation.state")
        info["features"]["commands"] = info["features"].pop("action")

    _edit_info(root, rename_info)
    ctx = load_dataset(str(root))
    assert ctx.state is not None and ctx.action is not None
    assert ctx.feature_keys[STATE_KEY] == "qpos"
    assert ctx.feature_keys[ACTION_KEY] == "commands"
    res = run("outlier-episodes", ctx)
    assert res.verdict <= Severity.INFO, [f.message for f in res.findings]


def test_spike_in_a_velocity_command_still_flags(tmp_path):
    """Velocity/delta dims sit out the jerk metrics, but a one-frame spike far
    beyond the dim's own step scale is a glitch, not a step."""
    root = make_dataset(tmp_path / "ds", episodes=24)

    def delta_and_spike(t):
        a = np.asarray(t["action"])
        ep = np.asarray(t["episode_index"])
        rng = np.random.default_rng(0)
        for e in range(24):
            rows = ep == e
            n = rows.sum()
            a[rows, 0] = np.convolve(rng.standard_normal(n), np.ones(5) / 5, mode="same")
        idx = np.where(ep == 10)[0]
        a[idx[len(idx) // 2], 0] += 30 * np.nanstd(a[:, 0])
        t["action"] = a.tolist()
        return t

    _rewrite_data(root, delta_and_spike)
    _edit_info(root, lambda info: info["features"]["action"].update(
        {"names": ["cmd_vel_0", "joint_1", "joint_2", "joint_3", "joint_4", "gripper"]}))
    res = run("outlier-episodes", load_dataset(str(root)))
    assert any(f.episode_index == 10 and f.severity == Severity.WARN
               and "velocity_spike" in f.details.get("metrics", [])
               for f in res.findings), [f.message for f in res.findings]

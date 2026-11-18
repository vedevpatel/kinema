"""Pins for scripts/harm_table.py (schema, ceiling refusal, resume, the
stuck_joint recipe) and the leaderboard's harm-table page. No GPU, no
network: the training function is a stub, data is synthetic."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"_test_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


ht = _load("harm_table")


def _eps(n=40, frames=30, sdim=2, adim=2, seed=7):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((frames, sdim)).cumsum(0), rng.standard_normal((frames, adim)).cumsum(0))
            for _ in range(n)]


def _results(seeds=(0, 1, 2), success=True):
    out = {}
    for s in seeds:
        for i, arm in enumerate(ht.ARMS):
            out[(arm, s)] = {"heldout_mse": 100.0 + 10 * i + s, "success_pct": (50.0 - 5 * i) if success else None,
                             "gpu_seconds": 600.0, "gpu": "Test GPU"}
    return out


SPLIT = (list(range(0, 240, 2)), list(range(240, 288, 2)))  # 120 train + 24 held-out ids


def _doc(**kw):
    return ht.build_table(_results(), policy={"name": "diffusion", "source": ht.POLICY_SOURCE, "rollouts": True,
                                              "batch_size": 32}, steps=5000, eval_episodes=20, split=SPLIT,
                          date="2026-11-14", **kw)


# --- schema --------------------------------------------------------------------

def test_schema_pin():
    doc = _doc()
    assert ht.validate_table(doc) == []
    assert doc["schema"] == "kinema.harm-table/v1"
    assert [r["defect"] for r in doc["rows"]] == [
        "duplicates_20pct", "idle_start_30f", "scale_mismatch_0.5x_25pct", "stuck_joint_30pct",
        "action_saturation_30pct", "dropped_frames_10pct_25pct",
        "outlier_burst_5pct", "stale_stats_offset_0.5sigma"]
    assert {r["defect"]: r["check"] for r in doc["rows"]} == {
        "duplicates_20pct": "duplicate-and-leakage", "idle_start_30f": "idle-segments",
        "scale_mismatch_0.5x_25pct": "calibration-drift", "stuck_joint_30pct": "stuck-actuator",
        "action_saturation_30pct": "action-saturation",
        "dropped_frames_10pct_25pct": "dropped-frames",
        "outlier_burst_5pct": "outlier-episodes", "stale_stats_offset_0.5sigma": "stats-drift"}
    row = doc["rows"][0]
    assert set(row) == {"defect", "check", "recipe", "policy", "dataset_revision", "steps", "seeds",
                        "mse_delta_pct", "success_delta_pts", "date", "confidence", "accelerator_pairs"}
    assert row["dataset_revision"] == "7628202a2180972f291ba1bc6723834921e72c19"
    assert row["seeds"] == [0, 1, 2] and row["confidence"] == "indicative (3 seeds)"
    # duplicates arm: mse 110+s vs clean 100+s; success 45 vs 50
    assert row["mse_delta_pct"]["per_seed"] == [10.0, round(1000 / 101, 3), round(1000 / 102, 3)]
    assert row["success_delta_pts"] == {"per_seed": [-5.0, -5.0, -5.0], "mean": -5.0}
    assert len(doc["runs"]) == 27
    assert len(json.dumps(doc)) < 100_000


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(schema="kinema.harm-table/v0"),
    lambda d: d["rows"][0].pop("confidence"),
    lambda d: d["rows"][0].update(check="idle-segments"),
    lambda d: d["rows"][0]["mse_delta_pct"].update(mean=99.0),
    lambda d: d["rows"][0]["mse_delta_pct"]["per_seed"].pop(),
    lambda d: d["rows"][0]["success_delta_pts"].update(per_seed=[None, None, None], mean=None),
    lambda d: d["rows"][0].update(confidence="claim-grade"),
    lambda d: d["rows"][0]["recipe"]["params"].update(duplicate_fraction=0.5),
    lambda d: d["rows"][1]["mse_delta_pct"].update(per_seed=[float("nan")] * 3),
    lambda d: d["dataset"].update(heldout_episodes=d["dataset"]["train_episodes"][:24]),
    lambda d: d["dataset"]["train_episodes"].pop(),
])
def test_schema_rejects(mutate):
    doc = _doc()
    mutate(doc)
    assert ht.validate_table(doc)
    with pytest.raises(ValueError):
        ht.write_table(Path("/nonexistent/never-written.json"), doc)


def test_dry_run_rows_have_no_success_and_validate():
    doc = ht.build_table(_results(success=False), policy={"name": "ridge-bc (dry-run)", "rollouts": False},
                         steps=0, eval_episodes=None, split=SPLIT)
    assert ht.validate_table(doc) == []
    assert all(r["success_delta_pts"]["mean"] is None for r in doc["rows"])


def test_partial_results_only_pair_complete_seeds():
    res = {k: v for k, v in _results().items() if not (k[0] == "idle_start_30f" and k[1] == 2)}
    res.pop(("stuck_joint_30pct", 0)), res.pop(("stuck_joint_30pct", 1)), res.pop(("stuck_joint_30pct", 2))
    doc = ht.build_table(res, policy={"name": "diffusion", "rollouts": True}, steps=5000, eval_episodes=20,
                         split=SPLIT)
    rows = {r["defect"]: r for r in doc["rows"]}
    assert "stuck_joint_30pct" not in rows
    assert rows["idle_start_30f"]["seeds"] == [0, 1]
    assert rows["idle_start_30f"]["confidence"] == "indicative (2 seeds)"
    assert ht.validate_table(doc) == []


# --- manifest: ceiling refusal and resume -----------------------------------------

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _stub_train(clock, seconds=3600.0, calls=None):
    def train(arm, seed, cfg, heartbeat):
        if calls is not None:
            calls.append((arm, seed))
        clock.t += seconds
        heartbeat()
        return {"heldout_mse": 1.0 + ht.ARMS.index(arm), "success_pct": 40.0, "train_seconds": seconds * 0.9,
                "eval_seconds": seconds * 0.1, "gpu": "Test GPU"}
    return train


def test_ceiling_refusal_launches_nothing(tmp_path):
    clock, calls = Clock(), []
    m = ht.Manifest(tmp_path / "m.json")
    m.add_overhead("earlier sessions", 14.5 * 3600)
    status = ht.run_plan(m, ht.RunConfig(), ht.ARMS, ht.SEEDS, _stub_train(clock, calls=calls),
                         prior_hours=1.0, clock=clock, log=lambda s: None)
    assert status == "refused" and calls == []
    assert m.data["runs"] == {}


def test_arm_projection_refuses_an_arm_that_cannot_finish(tmp_path):
    """3 h left, 1 h runs: clean (3 runs) fits exactly, the next arm is refused
    before any of its runs start - not after it has spent part of the budget."""
    clock, calls = Clock(), []
    m = ht.Manifest(tmp_path / "m.json")
    m.add_overhead("earlier sessions", 12.0 * 3600 - 0.31 * 3600)
    status = ht.run_plan(m, ht.RunConfig(), ht.ARMS, ht.SEEDS, _stub_train(clock, seconds=1800, calls=calls),
                         prior_hours=1.0, clock=clock, log=lambda s: None)
    # clean projected 11.69 + 3 x 1.0 (prior) <= 15; then est = 0.5 h x 1.1
    assert status == "refused"
    assert m.spent_seconds() <= m.ceiling_seconds
    assert calls[:3] == [("clean", 0), ("clean", 1), ("clean", 2)]
    arms_started = {a for a, _ in calls}
    for arm in arms_started:  # every started arm finished all its seeds
        assert sum(1 for a, _ in calls if a == arm) == 3


def test_resume_never_retrains_and_keeps_abandoned_time(tmp_path):
    path = tmp_path / "m.json"
    cfg = ht.RunConfig(steps=500, eval_episodes=2)
    clock, calls = Clock(), []
    m = ht.Manifest(path)
    ht.run_plan(m, cfg, ["clean", "stuck_joint_30pct"], [0, 1], _stub_train(clock, 600, calls),
                clock=clock, log=lambda s: None)
    assert calls == [("clean", 0), ("clean", 1), ("stuck_joint_30pct", 0), ("stuck_joint_30pct", 1)]
    done_before = json.loads(path.read_text())["runs"]

    # a session killed mid-run: the in-flight run is still "running" on disk
    data = json.loads(path.read_text())
    data["runs"][ht.run_key(cfg, "idle_start_30f", 0)] = {
        "status": "running", "config": cfg.key, "arm": "idle_start_30f", "seed": 0, "gpu_seconds": 900.0}
    path.write_text(json.dumps(data))

    calls.clear()
    m2 = ht.Manifest(path)
    assert [a["arm"] for a in m2.data["abandoned"]] == ["idle_start_30f"]
    assert m2.spent_seconds() == pytest.approx(4 * 600 + 900)  # abandoned time still spent
    ht.run_plan(m2, cfg, ["clean", "idle_start_30f", "stuck_joint_30pct"], [0, 1],
                _stub_train(clock, 600, calls), clock=clock, log=lambda s: None)
    assert calls == [("idle_start_30f", 0), ("idle_start_30f", 1)]  # nothing completed was retrained
    after = json.loads(path.read_text())["runs"]
    for k, rec in done_before.items():
        assert after[k] == rec
    table = ht.table_from_manifest(m2, cfg, SPLIT)
    assert ht.validate_table(table) == []
    assert {r["defect"] for r in table["rows"]} == {"idle_start_30f", "stuck_joint_30pct"}


def test_crashed_run_time_stays_on_the_ledger(tmp_path):
    clock = Clock()
    m = ht.Manifest(tmp_path / "m.json")

    def boom(arm, seed, cfg, heartbeat):
        clock.t += 1234.0
        raise RuntimeError("CUDA OOM")

    with pytest.raises(RuntimeError):
        ht.run_plan(m, ht.RunConfig(), ["clean"], [0], boom, clock=clock, log=lambda s: None)
    m2 = ht.Manifest(tmp_path / "m.json")
    assert m2.spent_seconds() == pytest.approx(1234.0)
    assert not m2.done(ht.run_key(ht.RunConfig(), "clean", 0))


def test_session_budget_stops_before_overrun(tmp_path):
    clock, calls = Clock(), []
    m = ht.Manifest(tmp_path / "m.json")
    status = ht.run_plan(m, ht.RunConfig(), ht.ARMS, ht.SEEDS, _stub_train(clock, 1800, calls),
                         session_hours=2.0, clock=clock, log=lambda s: None)
    assert status == "session-budget"
    assert clock.t <= 2.0 * 3600


def test_estimate_scales_other_configs_and_prefers_same_config(tmp_path):
    clock = Clock()
    m = ht.Manifest(tmp_path / "m.json")
    smoke = ht.RunConfig(steps=500, eval_episodes=2)
    ht.run_plan(m, smoke, ["clean"], [0], _stub_train(clock, 100), clock=clock, log=lambda s: None)
    # 100 s run: 90 s train, 10 s eval -> 5000 steps, 20 episodes: 900 + 100
    assert m.estimate_seconds(ht.RunConfig(), prior_hours=9) == pytest.approx(1000 * ht.ESTIMATE_MARGIN)
    assert m.estimate_seconds(smoke, prior_hours=9) == pytest.approx(100 * ht.ESTIMATE_MARGIN)


# --- the stuck_joint recipe ------------------------------------------------------

def test_stuck_joint_self_check_pin():
    eps = _eps(n=40)
    out = ht.inject_stuck_joint(eps, np.random.default_rng(3))
    frozen = [(i, d) for i, ((s, a), (s0, a0)) in enumerate(zip(out, eps))
              for d in range(a.shape[1]) if not np.array_equal(a[:, d], a0[:, d])]
    assert len(frozen) == 12  # 30% of 40
    assert len({d for _, d in frozen}) == 1  # one dim
    for i, d in frozen:
        assert np.all(out[i][1][:, d] == eps[i][1][0, d])  # its episode-start value
    for (s, a), (s0, a0) in zip(out, eps):
        assert np.array_equal(s, s0)  # states untouched
    again = ht.inject_stuck_joint(eps, np.random.default_rng(3))
    assert all(np.array_equal(a, b) for (_, a), (_, b) in zip(out, again))  # seed-deterministic
    assert ht.stuck_joint_self_check(eps, 3) == []
    assert ht.stuck_joint_self_check(_eps(n=120), 0) == []


def test_stuck_joint_self_check_catches_a_broken_recipe(monkeypatch):
    def two_dims(eps, rng, fraction=0.3):
        return [(s, np.repeat(a[:1], len(a), axis=0)) for s, a in eps]

    monkeypatch.setattr(ht, "inject_stuck_joint", two_dims)
    assert ht.stuck_joint_self_check(_eps(n=20), 0)


def test_recipes_are_harm_studys():
    """The three v0 recipes are harm_study.py's functions, imported - not copies."""
    assert Path(ht.hs.__file__).resolve() == SCRIPTS / "harm_study.py"
    for arm in ("duplicates_20pct", "idle_start_30f", "scale_mismatch_0.5x_25pct"):
        assert ht.DEFECTS[arm].inject is ht.hs.DEFECTS[arm]


def test_dry_run_math_and_frame_ids_ride_with_state():
    eps = _eps(n=30)
    res = ht.dry_run(eps[:24], eps[24:], [0, 1])
    assert set(res) == {(a, s) for a in ht.ARMS for s in (0, 1)}
    assert all(np.isfinite(r["heldout_mse"]) for r in res.values())
    # GPU path: the frame id column follows the injectors (idle pads repeat frame 0's image)
    state = np.arange(20, dtype=np.float32).reshape(10, 2)
    fr = ht.PushtFrames(state, state.copy(), np.zeros((10, 3, 2, 2), np.uint8), [(0, 4), (4, 10)])
    padded = ht.inject("idle_start_30f", fr.episodes(0, 2), 0)
    st, act, ids, lengths = ht.arm_arrays(padded)
    assert lengths == [34, 36]
    assert list(ids[:31]) == [0] * 31 and list(ids[30:34]) == [0, 1, 2, 3]
    assert np.all(act[:30] == 0) and np.array_equal(st[30:34], state[0:4])


def test_windows_follow_lerobot_delta_semantics():
    obs, act, pad, cur = ht._windows([3, 2], [-1, 0], list(range(-1, 3)))
    assert obs.tolist() == [[0, 0], [0, 1], [1, 2], [3, 3], [3, 4]]
    assert act[0].tolist() == [0, 0, 1, 2] and pad[0].tolist() == [True, False, False, False]
    assert act[2].tolist() == [1, 2, 2, 2] and pad[2].tolist() == [False, False, True, True]
    assert act[3].tolist() == [3, 3, 4, 4] and cur.tolist() == [0, 1, 2, 3, 4]


# --- renderer ---------------------------------------------------------------------

def test_renderer_pin(tmp_path):
    lb = _load("leaderboard")
    out = tmp_path / "docs"
    out.mkdir()
    lb._write(out, [], "2026-11-14 00:00 UTC")  # no table: skipped silently
    assert not (out / "harm-table.html").exists()
    assert "harm-table.html" not in (out / "index.html").read_text()

    ht.write_table(out / "harm-table.json", _doc(compute={"gpu": ["NVIDIA H100"], "ledger_gpu_hours": 2.5}))
    lb._write(out, [], "2026-11-14 00:00 UTC")
    page = (out / "harm-table.html").read_text()
    assert "3 seeds / 20 eval episodes / 5k steps - indicative ordering evidence, not claim-grade." in page
    for defect in ht.DEFECTS:
        assert defect in page
    assert "NVIDIA H100" in page and "120 train / 24 held-out episodes" in page
    index = (out / "index.html").read_text()
    assert "<a href='harm-table.html'>harm table</a>" in index
    assert index.index("checks.md#health-score") < index.index("harm-table.html")


def test_renderer_skips_a_foreign_table(tmp_path):
    lb = _load("leaderboard")
    (tmp_path / "harm-table.json").write_text(json.dumps({"schema": "something-else"}))
    assert lb.write_harm_table(tmp_path) is False
    assert not (tmp_path / "harm-table.html").exists()


def test_arm_stats_weight_frames_by_use():
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(7, 3, 4, 4), dtype=np.uint8)
    ids = np.array([0, 0, 0, 3, 5, 5])  # frame 0 three times (idle padding), 3 once, 5 twice
    st = rng.standard_normal((6, 2)).astype(np.float32)
    stats = ht.arm_stats(st, st * 2, ids, ht.frame_channel_moments(image, chunk=2))
    ref = image[ids].astype(np.float64) / 255.0
    np.testing.assert_allclose(stats["observation.image"]["mean"].ravel(), ref.mean(axis=(0, 2, 3)), rtol=1e-5)
    np.testing.assert_allclose(stats["observation.image"]["std"].ravel(), ref.std(axis=(0, 2, 3)), rtol=1e-4)
    np.testing.assert_allclose(stats["observation.image"]["min"].ravel(), ref.min(axis=(0, 2, 3)), rtol=1e-6)
    assert stats["observation.image"]["mean"].shape == (3, 1, 1)
    np.testing.assert_allclose(stats["action"]["max"], (st * 2).max(0))


def test_heartbeat_thread_keeps_the_ledger_current(tmp_path):
    """A run killed in a phase that never calls heartbeat() (LeRobot's eval
    loop) still has its time on disk."""
    import threading

    clock = Clock()
    seen = threading.Event()
    path = tmp_path / "m.json"
    m = ht.Manifest(path)

    def silent(arm, seed, cfg, heartbeat):
        clock.t += 500.0
        for _ in range(200):  # wait for the background beat to write
            if json.loads(path.read_text())["runs"].get(ht.run_key(cfg, arm, seed), {}).get("gpu_seconds") == 500.0:
                seen.set()
                break
            threading.Event().wait(0.01)
        return {"heldout_mse": 1.0, "success_pct": 0.0}

    ht.run_plan(m, ht.RunConfig(), ["clean"], [0], silent, clock=clock, log=lambda s: None,
                heartbeat_seconds=0.01)
    assert seen.is_set()


def test_renderer_notes_pin(tmp_path):
    """A top-level "notes" field renders as a dim paragraph; absent, nothing."""
    lb = _load("leaderboard")
    out = tmp_path / "docs"
    out.mkdir()
    doc = _doc(compute={"gpu": ["NVIDIA H100"], "ledger_gpu_hours": 2.5})
    ht.write_table(out / "harm-table.json", doc)
    lb._write(out, [], "2026-11-15 00:00 UTC")
    assert "notes-probe" not in (out / "harm-table.html").read_text()

    doc["notes"] = "notes-probe: success deltas unmeasurable"
    ht.write_table(out / "harm-table.json", doc)
    lb._write(out, [], "2026-11-15 00:00 UTC")
    assert "notes-probe: success deltas unmeasurable" in (out / "harm-table.html").read_text()

def test_harm_table_page_names_the_weight_source(tmp_path):
    """tiered-v2: the harm-table page says weights derive from it where measured."""
    lb = _load("leaderboard")
    out = tmp_path / "docs"
    out.mkdir()
    ht.write_table(out / "harm-table.json", _doc())
    assert lb.write_harm_table(out)
    page = (out / "harm-table.html").read_text()
    assert "historical v1.1 measurements" in page
    assert "not derived from this table yet" not in page


# --- v1.2 arms: action_saturation_30pct and dropped_frames_10pct_25pct --------

def test_action_saturation_injector_pins_at_bounds():
    eps = _eps(n=30, frames=60, adim=3)
    out = ht.inject_action_saturation(eps, np.random.default_rng(0))
    assert [len(s) for s, _ in out] == [len(s) for s, _ in eps]  # rows kept, values clipped
    a_all = np.concatenate([a for _, a in out])
    raw = np.concatenate([a for _, a in eps])
    lo, hi = np.quantile(raw, 0.35, axis=0), np.quantile(raw, 0.65, axis=0)
    assert (a_all >= lo - 1e-12).all() and (a_all <= hi + 1e-12).all()
    for j in range(a_all.shape[1]):  # every dim piles up at BOTH bounds
        assert (np.abs(a_all[:, j] - hi[j]) < 1e-9).mean() > 0.25
        assert (np.abs(a_all[:, j] - lo[j]) < 1e-9).mean() > 0.25
    assert ht.action_saturation_self_check(eps, 0) == []


def test_action_saturation_injector_trips_the_check(tmp_path):
    """The arm is the defect the action-saturation check measures: injecting
    it into a clean synthetic dataset flips that check to WARN."""
    from kinema.checks import ALL_CHECKS, CheckConfig, Severity
    from kinema.dataset import load_dataset
    from kinema.fixtures.synth import make_dataset

    root = make_dataset(tmp_path / "sat", episodes=32)
    ctx = load_dataset(str(root))
    eps = [(ctx.state[ctx.rows_for(ep)], ctx.action[ctx.rows_for(ep)])
           for ep in ctx.episode_ids()]
    out = ht.inject_action_saturation(eps, np.random.default_rng(0))
    ctx.action = np.concatenate([a for _, a in out])
    res = {c.id: c for c in ALL_CHECKS}["action-saturation"].run(ctx, CheckConfig())
    assert res.verdict == Severity.WARN, [f.message for f in res.findings]


def test_dropped_frames_injector_drops_interior_only():
    eps = _eps(n=40, frames=50)
    out = ht.inject_dropped_frames(eps, np.random.default_rng(0))
    dropped = [i for i in range(len(eps)) if len(out[i][0]) < len(eps[i][0])]
    assert dropped, "the recipe must drop somewhere"
    for i in dropped:  # ~10% of interior frames, first/last survive
        assert len(out[i][0]) == pytest.approx(len(eps[i][0]) * 0.9, abs=2)
        assert np.array_equal(out[i][0][0], eps[i][0][0]) and np.array_equal(out[i][0][-1], eps[i][0][-1])
        assert len(out[i][0]) == len(out[i][1])  # state and action drop together
    assert ht.dropped_frames_self_check(eps, 0) == []


def test_new_injectors_deterministic_per_seed():
    eps = _eps(n=20, frames=40)
    for fn in (ht.inject_action_saturation, ht.inject_dropped_frames):
        a = fn(eps, np.random.default_rng(3))
        b = fn(eps, np.random.default_rng(3))
        assert all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
                   for x, y in zip(a, b, strict=True))


def test_table_from_manifest_keeps_each_arms_measured_date(tmp_path, monkeypatch):
    """A regen must not re-stamp old measurements: each row's date is the day
    the arm's runs finished (from the manifest), not the table's build day."""
    m = ht.Manifest(tmp_path / "m.json")
    m.save()
    cfg = ht.RunConfig(steps=20000, eval_episodes=50)
    days = iter(["2026-11-15T01:00:00Z"] * 12 + ["2026-11-16T02:00:00Z"] * 6)
    monkeypatch.setattr(ht, "_utcnow", lambda: next(days))
    for arm, v in (("clean", 100.0), ("duplicates_20pct", 110.0), ("idle_start_30f", 175.0)):
        for s in (0, 1, 2):
            key = ht.run_key(cfg, arm, s)
            m.start(key, cfg, arm, s)
            m.finish(key, {"heldout_mse": v, "success_pct": 50.0}, gpu_seconds=1.0)
    doc = ht.table_from_manifest(m, cfg, (list(range(120)), list(range(120, 144))))
    rows = {r["defect"]: r for r in doc["rows"]}
    assert rows["duplicates_20pct"]["date"] == "2026-11-15"
    assert rows["idle_start_30f"]["date"] == "2026-11-16"
    assert ht.validate_table(doc) == []


def test_outlier_burst_self_check_pin():
    assert ht.outlier_burst_self_check(_eps(), seed=7) == []


def test_outlier_burst_self_check_catches_a_broken_recipe(monkeypatch):
    orig = ht.inject_outlier_burst

    def broken(eps, rng, fraction=0.05, burst=6, sigma=8.0):
        return orig(eps, rng, fraction, burst, sigma * 0.5)

    monkeypatch.setattr(ht, "inject_outlier_burst", broken)
    problems = ht.outlier_burst_self_check(_eps(), seed=7)
    assert any("offset" in p for p in problems)


def test_shifted_stats_arrays_moves_mean_only():
    rng = np.random.default_rng(0)
    state = rng.standard_normal((500, 3)).cumsum(0)
    action = rng.standard_normal((500, 2))
    s2, a2 = ht.shifted_stats_arrays(state, action, 0.5)
    assert np.allclose(s2.mean(0), state.mean(0) + 0.5 * state.std(0))
    assert np.allclose(a2.mean(0), action.mean(0) + 0.5 * action.std(0))
    assert np.allclose(s2.std(0), state.std(0)) and np.allclose(a2.std(0), action.std(0))
    assert ht.DEFECTS["stale_stats_offset_0.5sigma"].stats_shift_sigma == 0.5
    assert ht.DEFECTS["outlier_burst_5pct"].check == "outlier-episodes"


def test_status_is_read_only(tmp_path, capsys):
    """--status prints the ledger without the Manifest constructor's
    running->abandoned rewrite - a live run's ledger must survive a peek."""
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "schema": ht.MANIFEST_SCHEMA, "created": "2026-11-16T00:00:00Z",
        "ceiling_gpu_hours": 18,
        "runs": {"k/clean/seed0": {"status": "done", "gpu_seconds": 3600.0},
                 "k/arm/seed1": {"status": "running", "gpu_seconds": 900.0}},
        "abandoned": [{"key": "k/old/seed0", "gpu_seconds": 600.0}],
        "overhead": []}))
    rc = ht.main_status(type("A", (), {"manifest": str(manifest)})())
    out = capsys.readouterr().out
    assert rc == 0
    assert "running   k/arm/seed1 0.25 h" in out
    assert "done-hours 1.00, ledger-spent 1.42 of 18" in out
    after = json.loads(manifest.read_text())
    assert after["runs"]["k/arm/seed1"]["status"] == "running"  # untouched
    assert after["runs"]["k/clean/seed0"]["status"] == "done"


def test_published_deltas_reproduce_raw_same_seed_runs():
    doc = json.loads((SCRIPTS.parent / "docs/harm-table.json").read_text())
    assert ht.validate_table(doc) == []
    rows = {r["defect"]: r for r in doc["rows"]}
    assert rows["outlier_burst_5pct"]["mse_delta_pct"]["mean"] == 107.722
    assert not all(p["same_gpu"] for p in rows["outlier_burst_5pct"]["accelerator_pairs"])


def test_validator_rejects_internally_consistent_but_raw_false_deltas():
    doc = _doc()
    row = doc["rows"][0]
    row["mse_delta_pct"]["per_seed"] = [999.] * len(row["seeds"])
    row["mse_delta_pct"]["mean"] = 999.
    assert any("raw same-seed" in e for e in ht.validate_table(doc))

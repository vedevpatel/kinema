"""One test per check: clean datasets pass, injected defects get caught."""

from __future__ import annotations

import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.fixtures.synth import KNOWN_DEFECTS, make_dataset
from kinema.report import DatasetReport, render_json, render_markdown

CHECKS = {c.id: c for c in ALL_CHECKS}


@pytest.fixture()
def clean(tmp_path):
    root = make_dataset(tmp_path / "clean", episodes=24)
    return load_dataset(str(root))


@pytest.fixture()
def cfg():
    return CheckConfig()


def run(check_id, ctx, cfg):
    return CHECKS[check_id].run(ctx, cfg)


def verdict(check_id, ctx, cfg):
    return run(check_id, ctx, cfg).verdict


def defective(tmp_path, *defects):
    root = make_dataset(tmp_path / ("def_" + "_".join(defects)), episodes=24, defects=list(defects))
    return load_dataset(str(root))


# --- the clean dataset must come back clean -------------------------------

def test_clean_dataset_all_clear(clean, cfg):
    for check_id in (
        "stats-drift",
        "timestamp-grid",
        "dropped-frames",
        "calibration-drift",
        "idle-segments",
        "stuck-actuator",
        "action-saturation",
        "nan-and-normalization",
        "duplicate-and-leakage",
        "outlier-episodes",
        "deployment-readiness",
    ):
        res = run(check_id, clean, cfg)
        assert res.verdict in (Severity.OK, Severity.INFO), (
            f"{check_id} flagged a clean dataset: "
            f"{[f.message for f in res.findings]}"
        )


def test_visual_check_skips_cleanly_without_video(clean, cfg):
    res = run("visual-quality", clean, cfg)
    assert res.skipped and res.verdict == Severity.SKIP


# --- each injected defect must be caught ----------------------------------

def test_stats_drift(tmp_path, cfg):
    ctx = defective(tmp_path, "stats-drift")
    assert verdict("stats-drift", ctx, cfg) == Severity.WARN


def test_timestamp_grid_off_grid(tmp_path, cfg):
    ctx = defective(tmp_path, "timestamp-grid")
    res = run("timestamp-grid", ctx, cfg)
    assert res.verdict >= Severity.WARN
    assert any("frame spacing deviates" in f.message and f.episode_index == 6 for f in res.findings)
    assert any(f.severity == Severity.FAIL and f.episode_index == 7 for f in res.findings)


def test_dropped_frames(tmp_path, cfg):
    ctx = defective(tmp_path, "dropped-frames")
    res = run("dropped-frames", ctx, cfg)
    # frame_index gap AND metadata length lie on episode 9
    assert any(f.episode_index == 9 and f.severity == Severity.WARN for f in res.findings)
    assert any(f.episode_index == 9 and f.severity == Severity.FAIL for f in res.findings)


def test_calibration_drift(tmp_path, cfg):
    ctx = defective(tmp_path, "calibration-drift")
    res = run("calibration-drift", ctx, cfg)
    assert res.verdict >= Severity.WARN
    assert any(f.details.get("joint") == 2 for f in res.findings)


def test_calibration_clean_is_quiet(clean, cfg):
    assert verdict("calibration-drift", clean, cfg) == Severity.OK


def test_idle_segments(tmp_path, cfg):
    ctx = defective(tmp_path, "idle-segments")
    res = run("idle-segments", ctx, cfg)
    assert any(f.episode_index == 2 and f.severity == Severity.WARN and "ends with" in f.message
               for f in res.findings)


def test_stuck_actuator(tmp_path, cfg):
    ctx = defective(tmp_path, "stuck-actuator")
    res = run("stuck-actuator", ctx, cfg)
    assert any(
        f.severity == Severity.WARN and f.details.get("joint") == 4 and f.details["episodes"] == [3]
        for f in res.findings
    )


def test_action_saturation(tmp_path, cfg):
    ctx = defective(tmp_path, "action-saturation")
    res = run("action-saturation", ctx, cfg)
    assert res.verdict >= Severity.WARN
    assert any(f.details.get("dim") == 1 for f in res.findings)


def test_nan_detected(tmp_path, cfg):
    ctx = defective(tmp_path, "nan")
    res = run("nan-and-normalization", ctx, cfg)
    assert res.verdict == Severity.FAIL
    assert 5 in res.findings[0].details["episodes"]


def test_duplicates(tmp_path, cfg):
    ctx = defective(tmp_path, "duplicates")
    res = run("duplicate-and-leakage", ctx, cfg)
    assert res.verdict >= Severity.WARN
    assert any(set(f.details.get("episodes", [])) == {0, 1} for f in res.findings)


def test_outlier_episodes(tmp_path, cfg):
    ctx = defective(tmp_path, "outlier-episodes")
    res = run("outlier-episodes", ctx, cfg)
    assert any(f.episode_index == 23 for f in res.findings)


def test_deployment_short_episode(tmp_path, cfg):
    ctx = defective(tmp_path, "deployment")
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0"))
    assert any("shorter than" in f.message for f in res.findings)


def test_deployment_quantile_stats(tmp_path, cfg):
    ctx = defective(tmp_path)
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0"))
    assert any("q01/q99" in f.message for f in res.findings)


# --- reporting -------------------------------------------------------------

def test_report_score_and_exit_semantics(tmp_path, cfg):
    ctx = defective(tmp_path, "nan", "idle-segments")
    rep = DatasetReport("x", "v3.0", ctx.total_episodes, ctx.row_count, ctx.fps)
    for check in ALL_CHECKS:
        rep.results.append(check.run(ctx, cfg))
    assert rep.verdict == Severity.FAIL
    assert rep.score < 100
    js = render_json(rep)
    assert '"verdict": "FAIL"' in js
    md = render_markdown(rep)
    assert "Health score" in md and "nan-and-normalization" in md


def test_synth_rejects_unknown_defect(tmp_path):
    with pytest.raises(ValueError):
        make_dataset(tmp_path / "x", defects=["not-a-defect"])
    assert "visual" not in " ".join(KNOWN_DEFECTS)  # video defects need real MP4s


# --- deployment readiness: policy declarations from policies.json ----------

def test_policy_declarations_come_from_the_data_file():
    from kinema.checks.deployment import POLICY_ALIASES, POLICY_DEFAULTS, _load_policies
    raw = _load_policies()
    policies = {k: v for k, v in raw.items() if not k.startswith("_")}
    assert policies == POLICY_DEFAULTS
    for alias, target in raw.get("_aliases", {}).items():
        assert target in policies, alias
        assert POLICY_ALIASES[alias.lower()] == target
    for name, p in policies.items():
        assert isinstance(p["chunk_size"], int) and p["chunk_size"] > 0, name
        assert isinstance(p["quantile_norm"], bool), name


def test_policy_aliases_resolve_to_the_family_entry():
    from kinema.checks.deployment import canonical_policy_name, resolve_policy
    assert canonical_policy_name("pi0.5") == "pi05"
    name, entry = resolve_policy("pi0_5")
    assert name == "pi05" and entry is not None
    # pi05, verified against openpi's Pi0Config(pi05=True): quantile norm,
    # horizon default 50, 224x224 cameras, language-conditioned
    assert entry["quantile_norm"] is True and entry["chunk_size"] == 50
    assert entry["typical_resolution"] == [224, 224] and entry["needs_language"] is True


def test_per_run_policy_overrides_merge_and_validate(tmp_path):
    from kinema.checks.deployment import resolve_policy
    root = make_dataset(tmp_path / "ds", episodes=24)
    ctx = load_dataset(str(root))
    # chunk override: a 100-frame chunk flags the 60-frame synth episodes
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0", policy_overrides={"chunk_size": 100}))
    assert any("action chunk (100 frames)" in f.message for f in res.findings)
    # quantile_norm off: the quantile warning disappears for this run only
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0", policy_overrides={"quantile_norm": False}))
    assert not [f for f in res.findings if "q01/q99" in f.message]
    assert resolve_policy("pi0")[1]["quantile_norm"] is True  # defaults untouched
    # unknown keys fail loudly instead of tuning nothing
    import pytest
    with pytest.raises(ValueError, match="unknown policy override"):
        run("deployment-readiness", ctx, CheckConfig(policy="pi0", policy_overrides={"chunk": 10}))


def test_policy_without_camera_input_warns(tmp_path):
    # a vision-language policy cannot be served by a state-only dataset
    root = make_dataset(tmp_path / "ds", episodes=24)
    ctx = load_dataset(str(root))
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0"))
    assert any(f.severity == Severity.WARN and "camera input" in f.message for f in res.findings)
    # a policy the data file does not declare images for: no such finding
    res = run("deployment-readiness", ctx, CheckConfig(policy=None))
    assert not [f for f in res.findings if "camera input" in f.message]


def test_language_policy_without_task_strings_warns(tmp_path):
    root = make_dataset(tmp_path / "ds", episodes=24)
    ctx = load_dataset(str(root))
    # the synth dataset carries task strings: quiet for a language policy
    res = run("deployment-readiness", ctx, CheckConfig(policy="smolvla"))
    assert not [f for f in res.findings if "language" in f.message]

    mf = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    m = pq.read_table(mf).to_pydict()
    m["tasks"] = [[] for _ in m["episode_index"]]
    pq.write_table(pa.table(m), mf)
    ctx = load_dataset(str(root))
    res = run("deployment-readiness", ctx, CheckConfig(policy="smolvla"))
    assert any(f.severity == Severity.WARN and "language" in f.message for f in res.findings)


def test_resolution_below_policy_typical_is_info(tmp_path):
    import json
    root = make_dataset(tmp_path / "ds", episodes=24)
    p = root / "meta" / "info.json"
    info = json.loads(p.read_text())
    info["features"]["observation.images.wrist"] = {
        "dtype": "video", "shape": [3, 96, 96], "names": ["channel", "height", "width"],
    }
    p.write_text(json.dumps(info))
    ctx = load_dataset(str(root))
    # act is usually trained on 640x480: a 96x96 camera is usable but smaller
    res = run("deployment-readiness", ctx, CheckConfig(policy="act"))
    assert any(f.severity == Severity.INFO and "640x480" in f.message for f in res.findings)
    # diffusion is typically 96x96: no resolution note
    res = run("deployment-readiness", ctx, CheckConfig(policy="diffusion"))
    assert not [f for f in res.findings if "usually trained at" in f.message]


def test_quantile_warn_reads_raw_stats_json(tmp_path):
    # v2.1 case: the effective stats (per-episode aggregates) keep only
    # mean/std/min/max/count, so checking them cries wolf when the raw
    # meta/stats.json ships quantiles - openpi reads stats.json directly
    import json
    root = make_dataset(tmp_path / "ds", episodes=24)
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    for feat in ("observation.state", "action"):
        e = stats[feat]
        e["q01"] = e["min"]
        e["q99"] = e["max"]
    p.write_text(json.dumps(stats))
    ctx = load_dataset(str(root))
    # simulate the v2.1 aggregate: effective stats without quantiles,
    # raw stats.json carrying them -> no warning
    ctx.stats = {k: {s: v for s, v in e.items() if s not in ("q01", "q99")}
                 for k, e in ctx.stats.items()}
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0"))
    assert not [f for f in res.findings if "q01/q99" in f.message]
    # stats.json without quantiles at all -> the warning stands
    stats = json.loads(p.read_text())
    for feat in ("observation.state", "action"):
        stats[feat] = {k: v for k, v in stats[feat].items() if k not in ("q01", "q99")}
    p.write_text(json.dumps(stats))
    ctx = load_dataset(str(root))
    res = run("deployment-readiness", ctx, CheckConfig(policy="pi0"))
    assert any(f.severity == Severity.WARN and "q01/q99" in f.message for f in res.findings)

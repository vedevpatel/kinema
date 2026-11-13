"""One test per check: clean datasets pass, injected defects get caught."""

from __future__ import annotations

import numpy as np
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

"""Regression tests for scripts/revalidate_gripper_prior.py (finding #15):
the scoring harness that backs the numbers quoted in stats.py's docstring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "revalidate_gripper_prior.py"


def _load():
    spec = importlib.util.spec_from_file_location("revalidate_gripper_prior", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bimodal(rng, lo_frac=0.5, n=4000):
    """A dim dwelling at both ends: gripper-like."""
    x = np.empty(n)
    k = int(n * lo_frac)
    x[:k] = -1.0 + 0.01 * rng.standard_normal(k)
    x[k:] = 1.0 + 0.01 * rng.standard_normal(n - k)
    return x


def test_slot_dims():
    mod = _load()
    assert mod.slot_dims(6) == {5}
    assert mod.slot_dims(14) == {6, 13}
    assert mod.slot_dims(7) == {6}  # odd layouts: mid slot never applies


def test_score_dataset_flags_parked_joint_fp_only_under_old_rule():
    mod = _load()
    rng = np.random.default_rng(0)
    grip = _bimodal(rng)  # true gripper, last dim
    parked = np.full(4000, 2.0) + 0.01 * rng.standard_normal(4000)
    parked[:100] = 0.0  # parked at one end with a blip: not constant, not both-ends
    action = np.stack([rng.standard_normal(4000), parked, grip], axis=1)
    names = ["shoulder_pan", "waist_lift", "gripper"]
    row = mod.score_dataset(action, names)
    assert row["fp"] == [] and row["fn"] == [] and row["tp"] == [2]
    # the old either-end rule would also have picked the parked joint... but
    # it is not in a slot here (3-dim -> slot {2}), so neither rule can


def test_score_dataset_name_length_mismatch_is_unlabelled():
    mod = _load()
    rng = np.random.default_rng(1)
    action = np.stack([_bimodal(rng), rng.standard_normal(4000)], axis=1)
    row = mod.score_dataset(action, ["actions"])  # one name for two dims
    assert row["labelled"] == 0 and row["fp"] == []


def test_aggregate_counts_fp_rate_and_old_rule():
    mod = _load()
    rows = [
        {"tp": [1], "fp": [], "fn": [], "layout_miss": [], "old_picks": [1], "old_fp": [], "old_fn": []},
        {"tp": [], "fp": [3], "fn": [2], "layout_miss": [4], "old_picks": [3], "old_fp": [3], "old_fn": []},
    ]
    s = mod.aggregate(rows)
    assert s["datasets"] == 2
    assert s["slot_picks"] == 2 and s["fp"] == 1 and s["fp_rate"] == 0.5
    assert s["fn"] == 1 and s["recall"] == 0.5
    assert s["layout_misses"] == 1
    assert s["old_rule"]["slot_picks"] == 2 and s["old_rule"]["fp"] == 1

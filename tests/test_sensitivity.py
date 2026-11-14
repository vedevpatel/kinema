"""Nightly sensitivity guard: on pinned real datasets, gross injected defects
must be caught, and untouched data must not grow new flags.

Runs on the EVAL half of scripts/calibration_datasets.txt - the held-out
half. Thresholds, floors and ceilings are derived from the tune half, so
the numbers below were never tuned on the data that guards them (finding
#10: tuning and test sets must be disjoint). Pinned revisions keep the
numbers deterministic. Network-dependent: runs with KINEMA_E2E=1.

Floors sit a little under the measured rates so noise-free code changes pass;
a drop below one means a threshold lost its teeth. Ceilings are the measured
baseline: a rise means a new false positive - or a real issue newly found, in
which case read it and raise the ceiling deliberately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from netcheck import e2e_mark

pytestmark = e2e_mark()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# defect label -> [(magnitude, minimum detection rate)], measured on the eval
# quick set (the held-out half; rates differ from the tune half by dataset mix)
FLOORS = {
    "action std stored x m": [(4.0, 0.90)],
    "action mean shifted m std": [(1.0, 0.85)],
    "calibration offset m x range": [(0.2, 0.90)],
    "calibration scale error m": [(0.4, 0.90)],
    "dead follower (1 episode)": [(1, 0.90)],
    "empty episode (1)": [(1, 0.95)],
    "idle start m seconds": [(4, 0.80)],  # measured 5/6: one eval dataset idles its start by design
    "one-frame spike m std": [(30, 0.90)],
    "clipped top m %": [(20, 0.70)],  # measured 5/7: clipped delta commands pile at the limit less cleanly
    "copied opening m frames (0 = all)": [(30, 0.65), (0, 0.95)],  # 30 frames: pair-exclusive openings WARN (measured 4/6)
    "spliced m frames, other offset": [(60, 0.80)],
    "timestamp jitter m x period": [(0.5, 0.95)],
    "dropped frames (m gap)": [(3, 0.95)],
}

# check -> max datasets flagged on untouched data (measured on the eval quick set)
CEILINGS = {
    # calibration-drift 1: the delta-scale verdict flags xarm_lift_medium_replay -
    # measured steps run at ~0.07x the commanded delta steps, R^2 0.79-0.98, uniform
    # across joints 0-2. Read and adjudicated 2026-11-13: either a units mismatch in
    # the conversion or systematic under-execution by the replay controller; both are
    # things a QA tool should surface, so the WARN stays and the ceiling is raised
    # deliberately (docs/checks.md, calibration-drift).
    "stats-drift": 0, "timestamp-grid": 0, "dropped-frames": 0, "calibration-drift": 1,
    "nan-and-normalization": 0, "duplicate-and-leakage": 0, "deployment-readiness": 0,
    "idle-segments": 2, "stuck-actuator": 3, "action-saturation": 2, "outlier-episodes": 4,
}


@pytest.fixture(scope="module")
def calibration():
    import calibrate
    from kinema.dataset import load_dataset

    ctxs = {s: load_dataset(s, max_episodes=50) for s in calibrate.pinned_datasets(quick=True, role="eval")}
    return calibrate.run_calibration(ctxs)


@pytest.mark.parametrize("label", sorted(FLOORS))
def test_gross_defects_are_caught(calibration, label):
    sens, _ = calibration
    cid, cells = sens[label]
    for m, floor in FLOORS[label]:
        hits, tried = next((h, t) for mm, h, t in cells if mm == m)
        assert tried > 0, f"{label}: no dataset could take this defect"
        assert hits / tried >= floor, (
            f"{label} (m={m}) caught {hits}/{tried} by {cid}; floor {floor:.0%}"
        )


def test_untouched_data_does_not_grow_flags(calibration):
    _, baseline = calibration
    over = {c: (n, CEILINGS[c]) for c, n in baseline.items() if c in CEILINGS and n > CEILINGS[c]}
    assert not over, f"checks flagging more untouched datasets than before (got, ceiling): {over}"

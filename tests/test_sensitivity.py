"""Nightly sensitivity guard: on pinned real datasets, gross injected defects
must be caught, and untouched data must not grow new flags.

Uses scripts/calibrate.py's `quick` subset (~35 MB, pinned revisions, so the
numbers are deterministic). Network-dependent: runs with KINEMA_E2E=1.

Floors sit a little under the measured rates so noise-free code changes pass;
a drop below one means a threshold lost its teeth. Ceilings are the measured
baseline: a rise means a new false positive - or a real issue newly found, in
which case read it and raise the ceiling deliberately.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("KINEMA_E2E", os.environ.get("ROLLOUTQA_E2E")) != "1",
    reason="set KINEMA_E2E=1 to run",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# defect label -> (magnitude, minimum detection rate)
FLOORS = {
    "action std stored x m": (4.0, 0.90),
    "action mean shifted m std": (1.0, 0.85),
    "calibration offset m x range": (0.2, 0.90),
    "calibration scale error m": (0.4, 0.90),
    "dead follower (1 episode)": (1, 0.90),
    "empty episode (1)": (1, 0.95),
    "idle start m seconds": (4, 0.85),
    "one-frame spike m std": (30, 0.90),
    "clipped top m %": (20, 0.80),
    "copied opening m frames (0 = all)": (0, 0.95),
    "spliced m frames, other offset": (60, 0.80),
    "timestamp jitter m x period": (0.5, 0.95),
    "dropped frames (m gap)": (3, 0.95),
}

# check -> max datasets flagged on untouched data (measured on the quick set)
CEILINGS = {
    "stats-drift": 0, "timestamp-grid": 0, "dropped-frames": 0, "calibration-drift": 0,
    "nan-and-normalization": 0, "duplicate-and-leakage": 0, "deployment-readiness": 0,
    "idle-segments": 3, "stuck-actuator": 3, "action-saturation": 3, "outlier-episodes": 8,
}


@pytest.fixture(scope="module")
def calibration():
    import calibrate
    from kinema.dataset import load_dataset

    ctxs = {s: load_dataset(s, max_episodes=50) for s in calibrate.pinned_datasets(quick=True)}
    return calibrate.run_calibration(ctxs)


@pytest.mark.parametrize("label", sorted(FLOORS))
def test_gross_defects_are_caught(calibration, label):
    sens, _ = calibration
    cid, cells = sens[label]
    m, floor = FLOORS[label]
    hits, tried = next((h, t) for mm, h, t in cells if mm == m)
    assert tried > 0, f"{label}: no dataset could take this defect"
    assert hits / tried >= floor, f"{label} (m={m}) caught {hits}/{tried} by {cid}; floor {floor:.0%}"


def test_untouched_data_does_not_grow_flags(calibration):
    _, baseline = calibration
    over = {c: (n, CEILINGS[c]) for c, n in baseline.items() if c in CEILINGS and n > CEILINGS[c]}
    assert not over, f"checks flagging more untouched datasets than before (got, ceiling): {over}"

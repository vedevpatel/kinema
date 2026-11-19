"""Nightly sensitivity guard: on pinned real datasets, gross injected defects
must be caught, and untouched data must not grow new flags.

Runs on the eval regression group in scripts/calibration_datasets.txt.
Floors and ceilings were measured on that group. It is a regression guard,
not a pristine held-out generalization estimate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from netcheck import e2e_mark

pytestmark = e2e_mark()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# defect label -> [(magnitude, minimum detection rate)], measured on the eval
# quick regression group; rates differ from the tune group by dataset mix
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
    # calibration-drift 0 (was 1): the xarm_lift_medium_replay delta-scale WARN
    # it allowed for is now INFO (05f3016: command scaling on normalized delta
    # commands is informational), and a free slot would let one new false
    # positive through unnoticed. Re-measured on the eval quick set 2026-11-18.
    "stats-drift": 0, "timestamp-grid": 0, "dropped-frames": 0, "calibration-drift": 0,
    "nan-and-normalization": 0, "duplicate-and-leakage": 0, "deployment-readiness": 0,
    # outlier-episodes 2 (was 4): the 2026-11-15 tuning moved the far-out bar to
    # 6x the median and stopped static_fraction earning agreement. The two that
    # remain are real: svla_so100_sorting's dead episode 23 (stuck-actuator's
    # known defect) and openarm-2-cell's two 2-family physics glitches in sim.
    "idle-segments": 2, "stuck-actuator": 3, "action-saturation": 2, "outlier-episodes": 2,
    # episode-file-pointers: 0 - every eval-half dataset's shard pointers match
    # their files (measured 2026-11-15); the lerobot#2401 rollover bug is a WARN
    "episode-file-pointers": 0,
    # every registered check has a ceiling (measured on the eval quick set
    # 2026-11-18, all 0/7): a check without one could start flagging healthy
    # data and the guard would never notice. visual-quality is the exception -
    # the calibration run does not fetch videos, so it always skips here.
    "video-file-pointers": 0, "task-strings": 0, "reward-done": 0,
    "embedded-images": 0, "camera-coverage": 0, "stats-f32-fingerprint": 0,
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


def test_every_registered_check_has_a_ceiling():
    from kinema.checks import ALL_CHECKS

    missing = {c.id for c in ALL_CHECKS} - set(CEILINGS) - {"visual-quality"}
    assert not missing, f"checks without a ceiling: {sorted(missing)}"


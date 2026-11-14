"""Real-data guard: known-healthy Hub datasets must not FAIL, and known
defects must still be caught.

Each dataset was flagged by an earlier engine for a false positive (the
reason is next to it); pinning the revision keeps the test about kinema,
not about datasets changing upstream. Network-dependent: runs with
KINEMA_E2E=1 (CI nightly).
"""

from __future__ import annotations

import pytest

from netcheck import e2e_mark

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset

pytestmark = e2e_mark()

HEALTHY = {
    # 50-of-206-episode sample was compared to whole-dataset stats
    "lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19": "subset stats-drift",
    # gripper offset read as calibration drift
    "lerobot/aloha_sim_insertion_human@cc571a3c661df81b566dbfde3d5c1e85fcdf7884": "gripper calibration",
    "lerobot/svla_so100_pickplace@728583b5eaf9e739a7f119e2def466fa1d552402": "gripper calibration",
    # generic motor_N names, delta actions compared to state
    "lerobot/xarm_lift_medium_replay@1da06eafab3a2e6ad52e36edfba6583bd6378d19": "delta-action calibration",
    # v2.1 without stats.json; EE-pose state vs joint actions
    "IPEC-COMMUNITY/berkeley_mvp_lerobot@3ed881744758cf9701adbc0451d48dcee6a04e28": "v2.1 stats, EE vs joints",
}


def _run_all(ctx):
    results = {}
    for check in ALL_CHECKS:
        results[check.id] = check.run(ctx, CheckConfig())
    return results


@pytest.mark.parametrize("source", sorted(HEALTHY))
@pytest.mark.parametrize("max_episodes", [None, 20])
def test_known_healthy_dataset_has_no_fail(source, max_episodes):
    ctx = load_dataset(source, max_episodes=max_episodes)
    results = _run_all(ctx)
    fails = {
        cid: [f.message for f in r.findings if f.severity == Severity.FAIL]
        for cid, r in results.items()
        if r.verdict == Severity.FAIL
    }
    assert not fails, f"{source} ({HEALTHY[source]}) FAILs: {fails}"


def test_known_defect_still_caught():
    # svla_so100_sorting episodes 8 and 23: nothing moves at all
    ctx = load_dataset("lerobot/svla_so100_sorting@13870ca969084d8ce4d8a0391b06cead78e804ba")
    res = _run_all(ctx)["stuck-actuator"]
    assert any(
        f.severity == Severity.WARN and {8, 23} <= set(f.details.get("episodes", []))
        for f in res.findings
    ), [f.message for f in res.findings]

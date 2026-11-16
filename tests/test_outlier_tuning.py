"""outlier-episodes tuning (2026-11-15): the far-out bar moves 3x -> 6x the
median, and static stillness no longer earns multi-metric agreement.

Measured on the calibration halves before the change: known-good datasets'
far-out episodes cluster at 3.3-4.1x the median (wide-outcome tasks like
pusht end where the object lands), so the 3x bar WARNed on 8 of 14
untouched quick datasets; an injected one-frame 30-std spike lands at
10-198x, so 6x keeps the spike floor with >2x of headroom. And on real
teleop (berkeley_mvp), spike+static pairs were a paused operator, not a
glitch: static_fraction was already barred from a single-metric WARN for
that reason and is now barred from earning agreement too.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.checks.outliers import _FAR_OUT_REL, _far_out
from kinema.dataset import load_dataset
from kinema.fixtures.synth import make_dataset

CHECKS = {c.id: c for c in ALL_CHECKS}


def run(check_id, ctx):
    return CHECKS[check_id].run(ctx, CheckConfig())


def test_far_out_bar_is_six_times_median():
    base = [1.0] * 20
    for value, expected in ((4.5, False), (7.0, True)):
        flags = _far_out(np.array(base + [value]), "high", 0.0)
        assert bool(flags[-1]) is expected, (value, _FAR_OUT_REL)
    # two-sided (duration): a short episode must clear med / 6 as well
    flags = _far_out(np.array([1.0] * 20 + [1 / 7.0]), "both", 0.0)
    assert flags[-1]
    flags = _far_out(np.array([1.0] * 20 + [1 / 4.5]), "both", 0.0)
    assert not flags[-1]


def test_static_plus_spike_no_longer_earns_agreement(tmp_path):
    """Episode 4 holds still for its first half, then steps: static_fraction
    and peak_velocity both flag, but stillness no longer counts toward the
    2-family agreement, so the episode is INFO (idle-segments' territory)."""
    root = make_dataset(tmp_path / "ds", episodes=24)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    a = np.asarray(t["action"])
    s = np.asarray(t["observation.state"])
    ep = np.asarray(t["episode_index"])
    n_half = (ep == 4).sum() // 2
    first = np.where(ep == 4)[0][:n_half]
    a[first] = a[first[0]]
    s[first] = s[first[0]]
    a[np.where(ep == 4)[0][n_half]] += 5.0  # the step back into motion
    t["action"] = a.tolist()
    t["observation.state"] = s.tolist()
    pq.write_table(pa.table(t), f)
    res = run("outlier-episodes", load_dataset(str(root)))
    ep4 = [f for f in res.findings if f.episode_index == 4]
    assert ep4, "the paused-then-stepping episode should be on the review list"
    assert all(f.severity < Severity.WARN for f in ep4), \
        [f.message for f in ep4]
    assert res.verdict < Severity.WARN


def test_static_still_counts_as_evidence_for_a_real_outlier(tmp_path):
    """Static exclusion is about AGREEMENT, not visibility: a still episode
    still lands on the review list at INFO (the batch-3 behaviour)."""
    root = make_dataset(tmp_path / "ds", episodes=24)
    f = root / "data" / "chunk-000" / "file-000.parquet"
    t = pq.read_table(f).to_pydict()
    a = np.asarray(t["action"])
    a[np.asarray(t["episode_index"]) == 4] = a[0]
    t["action"] = a.tolist()
    pq.write_table(pa.table(t), f)
    res = run("outlier-episodes", load_dataset(str(root)))
    assert any(f.episode_index == 4 for f in res.findings)

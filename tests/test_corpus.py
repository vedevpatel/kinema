"""The adjudicated corpus (tests/corpus.py): real-data extracts whose verdicts
were checked against the raw data. Each must keep the verdict it was
adjudicated to - a change that re-introduces one of these false positives,
or loses one of these true positives, fails on every push, offline."""

from __future__ import annotations

import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset

from corpus import cases, materialize

C = {c.id: c for c in ALL_CHECKS}
CFG = CheckConfig()


@pytest.fixture(scope="module")
def ctx_of(tmp_path_factory):
    cache: dict = {}

    def get(name: str):
        if name not in cache:
            cache[name] = load_dataset(str(materialize(tmp_path_factory.mktemp(name) / "ds", name)))
        return cache[name]
    return get


def _dominating(res, feature: str):
    return [f for f in res.findings
            if f.details.get("kind") == "dominating-outliers" and f.details.get("feature") == feature]


def test_every_case_has_an_expectation():
    assert set(cases()) == set(EXPECT), "a corpus case without an adjudicated verdict tests nothing"


def _robocoin(ctx):
    # converter f32 noise on near-constant channels is not drift
    res = C["stats-drift"].run(ctx, CFG)
    assert res.verdict <= Severity.INFO, [f.message for f in res.findings if f.severity >= Severity.WARN]


def _openarm(ctx):
    # two-level grippers named like joints are not clipped commands
    res = C["action-saturation"].run(ctx, CFG)
    gated = [f for f in res.findings if f.severity >= Severity.WARN and f.details.get("dim") in (14, 15)]
    assert not gated, [f.message for f in gated]


def _ucsd(ctx):
    # a parked joint's one move is its signal, not a glitch
    res = C["nan-and-normalization"].run(ctx, CFG)
    assert not _dominating(res, "observation.state"), [f.message for f in _dominating(res, "observation.state")]


def _dobbe(ctx):
    hits = _dominating(C["nan-and-normalization"].run(ctx, CFG), "action")
    assert hits and any({2414, 4123} & set(f.details["episodes"]) for f in hits)


def _austin(ctx):
    hits = _dominating(C["nan-and-normalization"].run(ctx, CFG), "observation.state")
    assert {8, 13, 18} <= {f.details["dim"] for f in hits}


def _nyu(ctx):
    hits = _dominating(C["nan-and-normalization"].run(ctx, CFG), "action")
    assert 10 in {f.details["dim"] for f in hits}


def _mendelbeans(ctx):
    res = C["stats-drift"].run(ctx, CFG)
    assert res.verdict == Severity.FAIL, res.summary


EXPECT = {
    "robocoin_jitter_stats": _robocoin,
    "openarm_gripper": _openarm,
    "ucsd_parked": _ucsd,
    "dobbe_glitch": _dobbe,
    "austin_dropout": _austin,
    "nyu_wrap": _nyu,
    "mendelbeans_stale_stats": _mendelbeans,
}


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_adjudicated_verdict_holds(ctx_of, name):
    EXPECT[name](ctx_of(name))

"""Metamorphic invariance: no check's verdict may move when the data is
re-expressed without changing its quality (src/kinema/probes.py).

Every false positive the 2026-11-18/19 board runs surfaced was introduced by
a change and only found after it was public; most were representation
artifacts these probes expose: an absolute threshold meeting data at another
scale (RoboCOIN's f32 floor, parked dims), or an order/sample dependence.
This runs on every push, offline, over the synthetic clean dataset and every
synthetic defect, plus tests/edge_shapes.py: the real-data shapes that sit
near thresholds. test_zz_suite_stays_within_budget keeps it fast.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kinema import probes
from kinema.checks import ALL_CHECKS, CheckConfig
from kinema.dataset import load_dataset
from kinema.fixtures.synth import KNOWN_DEFECTS, make_dataset

CFG = CheckConfig()
_STARTED = time.monotonic()
# per-push CI budget for this whole module (about 13 s on a laptop)
_BUDGET_S = 45.0

# checks whose verdict legitimately depends on the representation itself
REPRESENTATION_BOUND = {
    # an exact float32-accumulation fingerprint of the stored stats: any
    # re-expression of the numbers breaks the bitwise match by construction
    "stats-f32-fingerprint",
}

from corpus import cases as corpus_cases  # noqa: E402

# synthetic fixtures, the edge shapes, and every adjudicated real-data case
FIXTURES = ["clean", *KNOWN_DEFECTS, "delta-actions", "edge-shapes",
            *(f"corpus:{c}" for c in corpus_cases())]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    cache: dict[str, object] = {}

    def get(name: str):
        if name not in cache:
            root = tmp_path_factory.mktemp(name.replace("/", "_").replace(":", "_")) / "ds"
            if name == "clean":
                make_dataset(root, episodes=24)
            elif name == "delta-actions":
                make_dataset(root, episodes=24, action_mode="delta")
            elif name == "edge-shapes":
                from edge_shapes import make_edge_dataset
                make_edge_dataset(root)
            elif name.startswith("corpus:"):
                from corpus import materialize
                materialize(root, name.split(":", 1)[1])
            else:
                # defects with layout prerequisites (synth.make_dataset)
                extra = {"stale-file-pointer": {"shards": 2},
                         "missing-video-file": {"cameras": ["observation.images.top"]}}.get(name, {})
                make_dataset(root, episodes=24, defects=[name], **extra)
            cache[name] = load_dataset(str(root))
        return cache[name]
    return get


def _signatures(ctx) -> dict[str, tuple]:
    return {c.id: probes.signature(c.run(ctx, CFG)) for c in ALL_CHECKS}


def _per_dim_offsets(ctx, seed: int) -> np.ndarray:
    d = ctx.state.shape[1]
    rng = np.random.default_rng(seed)
    sd = np.nanstd(ctx.state, axis=0)
    return rng.uniform(-3, 3, size=d) * np.where(sd > 0, sd, 1.0)


PROBES = {
    # degrees <-> radians, both directions of magnitude
    "unit-deg": lambda ctx: probes.affine(ctx, 180 / np.pi),
    "unit-milli": lambda ctx: probes.affine(ctx, 1e-3),  # mm -> m
    "unit-kilo": lambda ctx: probes.affine(ctx, 1e3),  # m -> mm
    # re-zeroed encoders: the same per-joint offset on command and measurement
    "rezero": lambda ctx: probes.affine(ctx, 1.0, _per_dim_offsets(ctx, 0)),
    "permute": lambda ctx: probes.permute_episodes(ctx, seed=0),
    "permute-2": lambda ctx: probes.permute_episodes(ctx, seed=1),
    "permute-reverse": lambda ctx: probes.reverse_episodes(ctx),
}


# real-data cases are larger: both scale directions, re-zeroing and the
# deterministic order reversal cover every bug class the probes have found
CORPUS_PROBES = {"unit-milli", "unit-kilo", "rezero", "permute-reverse"}


@pytest.mark.parametrize("probe", sorted(PROBES))
@pytest.mark.parametrize("fixture", FIXTURES)
def test_verdicts_survive_probe(loaded, fixture, probe):
    if fixture.startswith("corpus:") and probe not in CORPUS_PROBES:
        pytest.skip("corpus cases run the probes that find distinct bug classes (CI budget)")
    if fixture == "delta-actions" and probe == "rezero":
        pytest.skip("delta commands have no zero to move: an offset changes their meaning")
    ctx = loaded(fixture)
    if ctx.action is not None and ctx.state is not None and ctx.action.shape != ctx.state.shape \
            and probe == "rezero":
        pytest.skip("state and action widths differ: no per-joint pairing to re-zero")
    base = _signatures(ctx)
    moved = _signatures(PROBES[probe](ctx))
    exempt = set() if probe.startswith("permute") else REPRESENTATION_BOUND
    diffs = {cid: (base[cid], moved[cid]) for cid in base
             if cid not in exempt and base[cid] != moved[cid]}
    assert not diffs, f"{fixture} under {probe}: " + "; ".join(
        f"{cid}: {b[0]}{sorted(b[1])[:6]} -> {m[0]}{sorted(m[1])[:6]}" for cid, (b, m) in diffs.items())


def test_probes_change_the_numbers_but_not_the_rows(loaded):
    # guard the probes themselves: a no-op probe would pass every test above
    ctx = loaded("clean")
    deg = probes.affine(ctx, 180 / np.pi)
    assert not np.allclose(deg.state, ctx.state) and deg.state.shape == ctx.state.shape
    feat = ctx.feature_keys.get("observation.state", "observation.state")
    assert np.allclose(np.asarray(deg.stats_json[feat]["std"]),
                       np.asarray(ctx.stats_json[feat]["std"]) * 180 / np.pi)
    perm = probes.permute_episodes(ctx, seed=0)
    assert perm.episode_ids() != ctx.episode_ids()
    assert sorted(perm.episode_ids()) == sorted(ctx.episode_ids())
    for e in ctx.episode_ids()[:5]:
        assert np.array_equal(perm.state[perm.rows_for(e)], ctx.state[ctx.rows_for(e)])
    sub = probes.keep_episodes(ctx, ctx.episode_ids()[:5])
    assert sub.episode_ids() == ctx.episode_ids()[:5] and sub.truncated


def test_zz_suite_stays_within_budget():
    # runs last (file order): a fixture or probe that makes the suite slow
    # fails here instead of quietly taxing every push
    elapsed = time.monotonic() - _STARTED
    assert elapsed < _BUDGET_S, f"invariance suite took {elapsed:.0f}s (budget {_BUDGET_S:.0f}s)"

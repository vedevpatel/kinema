"""Unit tests for scripts/harm_study.py injectors and ridge baseline."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harm_study.py"


def _load():
    spec = importlib.util.spec_from_file_location("harm_study", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eps(n=5, frames=40, dims=2):
    rng = np.random.default_rng(7)
    return [(rng.standard_normal((frames, dims)).cumsum(0),
             rng.standard_normal((frames, dims))) for _ in range(n)]


def test_inject_duplicates_adds_copies():
    mod = _load()
    eps = _eps(n=10)
    out = mod.inject_duplicates(eps, np.random.default_rng(0))
    assert len(out) == 12  # 20% of 10
    orig = [(s.tobytes(), a.tobytes()) for s, a in eps]
    assert all((s.tobytes(), a.tobytes()) in orig for s, a in out)


def test_inject_scale_mismatch_halves_some_actions_only():
    mod = _load()
    eps = _eps(n=20)
    out = mod.inject_scale_mismatch(eps, np.random.default_rng(0))
    changed = sum(1 for (s, a), (s0, a0) in zip(out, eps)
                  if not np.allclose(a, a0))
    assert 0 < changed < 20
    for (s, a), (s0, a0) in zip(out, eps):
        assert np.array_equal(s, s0)  # states never touched
        assert np.allclose(a, a0) or np.allclose(a, a0 * 0.5)


def test_inject_idle_starts_prepends_still_frames():
    mod = _load()
    eps = _eps(n=3, frames=40)
    out = mod.inject_idle_starts(eps, np.random.default_rng(0), pad=30)
    for (s, a), (s0, a0) in zip(out, eps):
        assert len(s) == len(s0) + 30 and len(a) == len(a0) + 30
        assert np.all(s[:30] == s0[0])  # still: first state repeated
        assert np.all(a[:30] == 0)  # zero commands
        assert np.array_equal(s[30:], s0) and np.array_equal(a[30:], a0)


def test_ridge_recovers_linear_map():
    mod = _load()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((2000, 3))
    W_true = rng.standard_normal((3, 2))
    Y = X @ W_true
    W = mod.ridge_fit(X, Y)
    assert mod.ridge_mse(W, X, Y) < 1e-6


def test_harmful_defect_raises_mse_clean_baseline():
    """The harm signal itself: a linear world + halved actions trains worse."""
    mod = _load()
    rng = np.random.default_rng(2)
    W_true = rng.standard_normal((3, 2))
    eps = [(X := rng.standard_normal((200, 3)).cumsum(0), X @ W_true)
           for _ in range(8)]
    held = eps[-2:]
    train = eps[:-2]
    Xh, Yh = mod.flatten(held)
    base = mod.ridge_mse(mod.ridge_fit(*mod.flatten(train)), Xh, Yh)
    bad = mod.ridge_mse(
        mod.ridge_fit(*mod.flatten(mod.inject_scale_mismatch(
            train, np.random.default_rng(3)))), Xh, Yh)
    assert bad > base + 0.1

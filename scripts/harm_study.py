#!/usr/bin/env python3
"""Harm study, v0 (finding #9): do injected data defects move a TRAINING
outcome, not just a check verdict?

The detection harness (tests + leaderboard) proves checks fire on defects.
It says nothing about whether the defect hurts a policy. This script closes
that loop the cheap way: real pusht episodes, defect injectors that mirror
what specific checks catch, a CPU-trainable behavior-cloning baseline
(ridge regression state -> action; no GPU, no new deps), and the held-out
action-prediction MSE delta as the harm metric.

A defect that a check flags but that does not move held-out MSE is a
candidate for downgrading; a defect that moves MSE and is NOT flagged is a
coverage hole. The GPU-grade protocol (real BC policy, rollout success
rate) is documented in docs/harm-study.md - this script is the always-runs
floor, not the ceiling.

Run:  python scripts/harm_study.py [--episodes 120] [--json /tmp/harm.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kinema.dataset import load_dataset  # noqa: E402

PUSHT = "lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19"


def episode_arrays(ctx, episodes: int):
    """Per-episode (state, action) float arrays for the first N episodes."""
    eps = []
    for ep in ctx.episode_ids()[:episodes]:
        rows = ctx.rows_for(ep)
        eps.append((ctx.state[rows].astype(np.float64),
                    ctx.action[rows].astype(np.float64)))
    return eps


# --- defect injectors: each mirrors a check ---------------------------------

def inject_clean(eps, rng):
    return eps


def inject_duplicates(eps, rng):
    """20% of episodes copied - the duplicates check's target."""
    out = list(eps)
    k = max(1, len(eps) // 5)
    for i in rng.choice(len(eps), size=k, replace=False):
        out.append(eps[i])
    return out


def inject_scale_mismatch(eps, rng):
    """Actions halved in 25% of episodes, states untouched - the calibration
    delta-scale check's target (inconsistent action magnitudes)."""
    out = []
    for i, (s, a) in enumerate(eps):
        if rng.random() < 0.25:
            a = a * 0.5
        out.append((s, a))
    return out


def inject_idle_starts(eps, rng, pad=30):
    """30 still frames prepended to every episode - the idle-segments
    check's target (teaches waiting)."""
    out = []
    for s, a in eps:
        s_pad = np.repeat(s[:1], pad, axis=0)
        a_pad = np.zeros((pad, a.shape[1]))
        out.append((np.concatenate([s_pad, s]), np.concatenate([a_pad, a])))
    return out


DEFECTS = {
    "clean": inject_clean,
    "duplicates_20pct": inject_duplicates,
    "scale_mismatch_0.5x_25pct": inject_scale_mismatch,
    "idle_start_30f": inject_idle_starts,
}


def flatten(eps):
    s = np.concatenate([e[0] for e in eps])
    a = np.concatenate([e[1] for e in eps])
    n = min(len(s), len(a))
    return s[:n], a[:n]


def ridge_fit(X, Y, lam=1e-3):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ Y)


def ridge_mse(W, X, Y):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    return float(np.mean((Xb @ W - Y) ** 2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", metavar="PATH")
    args = ap.parse_args()

    ctx = load_dataset(PUSHT, max_episodes=args.episodes + args.holdout)
    eps = episode_arrays(ctx, args.episodes + args.holdout)
    train_eps, held_eps = eps[: args.episodes], eps[args.episodes :]
    Xh, Yh = flatten(held_eps)

    rows = []
    base = None
    for name, inject in DEFECTS.items():
        defective = inject(train_eps, np.random.default_rng(args.seed))
        Xt, Yt = flatten(defective)
        W = ridge_fit(Xt, Yt)
        mse = ridge_mse(W, Xh, Yh)
        if name == "clean":
            base = mse
        rows.append({"defect": name, "heldout_mse": mse})
        print(f"{name:28s} held-out MSE {mse:.5f}", flush=True)
    for r in rows:
        r["mse_delta_pct"] = 100.0 * (r["heldout_mse"] - base) / base if base else 0.0
    print("\nharm (vs clean): " + ", ".join(
        f"{r['defect']} {r['mse_delta_pct']:+.1f}%" for r in rows[1:]))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"dataset": PUSHT, "episodes": args.episodes,
             "holdout": args.holdout, "seed": args.seed, "results": rows},
            indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

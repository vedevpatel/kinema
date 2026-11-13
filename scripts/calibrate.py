#!/usr/bin/env python3
"""Threshold calibration: inject defects of known size into real datasets
and measure what each check catches.

    python scripts/calibrate.py ds1 ds2 ... [--max-episodes 50] [--markdown out.md]

Datasets are local paths or Hub ids (owner/name[@revision]). For every
dataset the checks first run on the untouched data (baseline), then once
per (defect, magnitude) on an in-memory copy with that defect injected. A
defect counts as detected when its target check reports a WARN/FAIL
finding the baseline run did not.

What this measures: sensitivity (how big a defect must be before it is
caught) and the baseline (what each check flags on data nobody tampered
with - real issues or false positives, worth reading either way). What it
does NOT measure: whether a defect of that size actually hurts a trained
policy. That needs training runs.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kinema.checks import ALL_CHECKS, CheckConfig, Severity  # noqa: E402
from kinema.dataset import load_dataset  # noqa: E402

CHECKS = {c.id: c for c in ALL_CHECKS}


def _clone(ctx):
    c = copy.copy(ctx)
    for k in ("state", "action", "timestamp", "frame_index"):
        v = getattr(ctx, k)
        setattr(c, k, None if v is None else v.copy())
    c.stats = copy.deepcopy(ctx.stats)
    c.stats_json = copy.deepcopy(ctx.stats_json)
    c.episode_stats = copy.deepcopy(ctx.episode_stats)
    return c


def _flags(ctx, check_id):
    r = CHECKS[check_id].run(ctx, CheckConfig())
    return {f.message for f in r.findings if f.severity >= Severity.WARN}


def _compared_joint(ctx):
    """A joint the calibration check actually compares, or None."""
    r = CHECKS["calibration-drift"].run(ctx, CheckConfig())
    joints = (r.details or {}).get("compared_joints") or []
    return int(joints[0]) if joints else None


def _target_joint(ctx):
    """A continuous arm dim: not a gripper, not a velocity command, not discrete.
    (The highest-variance dim is usually a binary gripper, which the checks
    deliberately treat differently - injecting there tests nothing.)"""
    from kinema.stats import action_gripper_dims, velocity_dims

    n = ctx.action.shape[1]
    named, inferred = action_gripper_dims(ctx)
    skip = named | inferred | velocity_dims(ctx.action_names, n)
    std = np.nanstd(ctx.action, axis=0)
    cands = [j for j in range(n) if j not in skip and std[j] > 1e-8
             and len(np.unique(ctx.action[:, j])) > 10]
    return max(cands, key=lambda j: std[j]) if cands else None


def _mid_episode(ctx, j=None):
    """A long-enough episode - where joint j moves the most, when given."""
    eps = [e for e in ctx.episode_ids() if len(ctx.rows_for(e)) >= 40]
    if not eps:
        return None
    if j is None:
        return eps[len(eps) // 2]
    return max(eps, key=lambda e: float(np.ptp(ctx.action[ctx.rows_for(e), j])))


# --- defects: (name, target check, magnitudes, inject(ctx, m) -> ctx | None) ---

def _corrupt_stats(ctx, fn):
    """Corrupt the stats where the dataset keeps its source of truth: v2.1
    aggregates per-episode stats (corrupt those), others read stats.json."""
    if ctx.stats_source != "meta/stats.json":
        hit = False
        for rec in ctx.episode_stats.values():
            if "action" in rec:
                fn(rec["action"])
                hit = True
        if not hit:
            return None
        from kinema.stats import aggregate
        agg = aggregate([r["action"] for r in ctx.episode_stats.values() if "action" in r])
        ctx.stats["action"] = {k: v.tolist() for k, v in agg.items()}
        return ctx
    st = ctx.stats_json.get("action")
    if not st or "std" not in st:
        return None
    st = {k: np.asarray(v, dtype=np.float64) for k, v in st.items()}
    fn(st)
    st = {k: v.tolist() for k, v in st.items()}
    ctx.stats_json["action"] = st
    ctx.stats["action"] = st
    return ctx


def _std_scale(ctx, m):
    def f(st):
        st["std"] = st["std"] * m
    return _corrupt_stats(ctx, f)


def _mean_shift(ctx, m):
    def f(st):
        st["mean"] = st["mean"] + m * st["std"]
    return _corrupt_stats(ctx, f)


def _calib_offset(ctx, m):
    j = _compared_joint(ctx)  # n/a where calibration cannot run (delta actions...)
    if j is None:
        return None
    ctx.action[:, j] += m * np.ptp(ctx.state[:, j])
    return ctx


def _calib_scale(ctx, m):
    j = _compared_joint(ctx)
    if j is None:
        return None
    ctx.action[:, j] += m * (ctx.state[:, j] - np.nanmedian(ctx.state[:, j]))
    return ctx


def _dead_follower(ctx, m):
    j = _compared_joint(ctx)
    if j is None:
        return None
    ep = _mid_episode(ctx, j)  # an episode where the joint is actually commanded
    if ep is None:
        return None
    rows = ctx.rows_for(ep)
    ctx.state[rows, j] = ctx.state[rows[0], j]
    return ctx


def _empty_episode(ctx, m):
    ep = _mid_episode(ctx)
    if ep is None:
        return None
    rows = ctx.rows_for(ep)
    ctx.action[rows] = ctx.action[rows[0]]
    if ctx.state is not None:
        ctx.state[rows] = ctx.state[rows[0]]
    return ctx


def _idle_start(ctx, m):
    ep = _mid_episode(ctx)
    if ep is None or not ctx.fps:
        return None
    rows = ctx.rows_for(ep)
    n = min(len(rows) - 2, int(m * ctx.fps))
    ctx.action[rows[:n]] = ctx.action[rows[0]]
    if ctx.state is not None:
        ctx.state[rows[:n]] = ctx.state[rows[0]]
    return ctx


def _spike(ctx, m):
    j = _target_joint(ctx)
    ep = _mid_episode(ctx)
    if ep is None or j is None:
        return None
    r = ctx.rows_for(ep)[len(ctx.rows_for(ep)) // 2]
    ctx.action[r, j] += m * np.nanstd(ctx.action[:, j])
    return ctx


def _saturation(ctx, m):
    j = _target_joint(ctx)
    if j is None:
        return None
    hi = np.nanpercentile(ctx.action[:, j], 100 - m)
    ctx.action[:, j] = np.minimum(ctx.action[:, j], hi)
    return ctx


def _copy(ctx, m):
    eps = [e for e in ctx.episode_ids() if len(ctx.rows_for(e)) >= max(40, int(m) + 5)]
    if len(eps) < 2:
        return None
    src, dst = eps[0], eps[-1]
    rs, rd = ctx.rows_for(src), ctx.rows_for(dst)
    n = min(len(rs), len(rd), int(m)) if m else min(len(rs), len(rd))
    ctx.action[rd[:n]] = ctx.action[rs[:n]]
    return ctx


def _jitter(ctx, m):
    if not ctx.fps:
        return None
    ep = _mid_episode(ctx)
    if ep is None:
        return None
    rows = ctx.rows_for(ep)
    rng = np.random.default_rng(0)
    ctx.timestamp[rows[1:-1]] += rng.uniform(-m, m, len(rows) - 2) / ctx.fps
    return ctx


def _gap(ctx, m):
    if not ctx.fps:
        return None
    ep = _mid_episode(ctx)
    if ep is None:
        return None
    rows = ctx.rows_for(ep)
    ctx.timestamp[rows[len(rows) // 2:]] += m / ctx.fps
    return ctx


DEFECTS = [
    ("action std stored x m", "stats-drift", [1.03, 1.1, 1.3, 2.0, 4.0], _std_scale),
    ("action mean shifted m std", "stats-drift", [0.05, 0.2, 0.5, 1.0], _mean_shift),
    ("calibration offset m x range", "calibration-drift", [0.02, 0.05, 0.1, 0.2], _calib_offset),
    ("calibration scale error m", "calibration-drift", [0.03, 0.1, 0.2, 0.4], _calib_scale),
    ("dead follower (1 episode)", "stuck-actuator", [1], _dead_follower),
    ("empty episode (1)", "stuck-actuator", [1], _empty_episode),
    ("idle start m seconds", "idle-segments", [1, 2, 4], _idle_start),
    ("one-frame spike m std", "outlier-episodes", [3, 10, 30], _spike),
    ("clipped top m %", "action-saturation", [2, 5, 10, 20], _saturation),
    ("copied m frames (0 = all)", "duplicate-and-leakage", [10, 30, 0], _copy),
    ("timestamp jitter m x period", "timestamp-grid", [0.05, 0.2, 0.5], _jitter),
    ("dropped frames (m gap)", "timestamp-grid", [1, 3], _gap),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--max-episodes", type=int, default=50)
    ap.add_argument("--markdown", default="")
    args = ap.parse_args()

    ctxs = {}
    for src in args.datasets:
        try:
            ctxs[src] = load_dataset(src, max_episodes=args.max_episodes)
        except Exception as e:  # noqa: BLE001
            print(f"skip {src}: {e}", file=sys.stderr)
    names = list(ctxs)

    # baseline: what each check flags on untouched data
    base = {src: {cid: _flags(ctx, cid) for cid in CHECKS if cid != "visual-quality"}
            for src, ctx in ctxs.items()}
    flagged = {cid: sum(1 for s in names if base[s][cid]) for cid in CHECKS if cid != "visual-quality"}

    lines = [f"# kinema calibration ({len(names)} datasets, up to {args.max_episodes} episodes each)", ""]
    lines += ["## Sensitivity: % of datasets where the injected defect is caught", "",
              "| defect | check | " + " | ".join(f"m={m}" for m in ("a", "b", "c", "d", "e")) + " |",
              "|---|---|---|---|---|---|---|"]
    for label, cid, mags, fn in DEFECTS:
        cells = []
        for m in mags:
            hit = tried = 0
            for src, ctx in ctxs.items():
                mod = fn(_clone(ctx), m)
                if mod is None:
                    continue
                tried += 1
                if _flags(mod, cid) - base[src][cid]:
                    hit += 1
            cells.append(f"{m}: {100 * hit / tried:.0f}% ({tried})" if tried else f"{m}: n/a")
        cells += [""] * (5 - len(cells))
        lines.append(f"| {label} | `{cid}` | " + " | ".join(cells) + " |")
    lines += ["", "## Baseline: datasets each check already flags (WARN/FAIL) untouched", "",
              "| check | flagged |", "|---|---|"]
    for cid, n in flagged.items():
        lines.append(f"| `{cid}` | {n}/{len(names)} |")
    lines += ["", "Baseline flags are real issues or false positives - read them before "
              "moving a threshold. Sensitivity says nothing about training impact."]
    out = "\n".join(lines)
    print(out)
    if args.markdown:
        Path(args.markdown).write_text(out + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

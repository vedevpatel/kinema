"""Rebuild the adjudicated corpus (tests/corpus.py) from pinned Hub revisions.

    python tests/data/corpus/build_corpus.py            # every case
    python tests/data/corpus/build_corpus.py nyu_wrap   # one case

Each case keeps only the episodes (and, where noted, dims) that carry the
phenomenon, at the share of frames it has in the real dataset, so the check
sees the same shape the board did. Verdicts were adjudicated against the raw
data on 2026-11-19 (see the case notes); tests/test_corpus.py asserts them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from kinema.dataset import _load_episode_meta, _load_episode_stats, load_dataset  # noqa: E402

CASES = {
    # must NOT flag: stats-drift. Channels at ~0.9 jitter by std 4e-6 and the
    # stored stats carry converter noise at that scale (FAILed at the old
    # 1e-6 flatness line). Partial sample + full catalogue, as on the board.
    # The board's 40-episode sample, cut to the dims that carry it (11-12)
    # plus neighbours and a few ordinary dims.
    "robocoin_jitter_stats": dict(repo="RoboCOIN/AI2_Alphabot_2_package_fruit", sample=40,
                                  keep_stats=True, partial=True, dims=[0, 1, 2, 10, 11, 12, 13]),
    # must NOT flag: action-saturation. RJ8/LJ8 are two-level grippers named
    # like joints; 98-100% of frames sit at exactly one limit.
    "openarm_gripper": dict(repo="ethanCSL/openarm_wipeboard_lerobot_real_v00", sample=4),
    # must NOT flag: dominating outliers. State dim 5 is parked at 0.90 and
    # moves in one episode.
    "ucsd_parked": dict(repo="lerobot/ucsd_pick_and_place_dataset", sample=250,
                        pick=dict(feature="state", dim=5, extra=24)),
    # MUST flag: dominating outliers on action dims. Odometry glitches in a
    # handful of the 5208 episodes (2414, 4123 among them).
    "dobbe_glitch": dict(repo="IPEC-COMMUNITY/dobbe_lerobot", episodes=[2414, 4123], spread=40),
    # MUST flag: dominating outliers on state dims 8/13/18. All-zero state
    # dropout frames scattered through most episodes.
    "austin_dropout": dict(repo="lerobot/austin_buds_dataset", sample=15),
    # MUST flag: dominating outliers on action dim 10. Angle wrap-arounds
    # (-2*pi jumps) in some episodes.
    "nyu_wrap": dict(repo="lerobot/nyu_franka_play_dataset", sample=250,
                     pick=dict(feature="action", dim=10, extra=30)),
    # MUST flag: stats-drift FAIL. Episode 51 reaches -53.5 on dim 0 while
    # stats.json says min -24.4 - stats never updated after it was recorded.
    "mendelbeans_stale_stats": dict(repo="MendelBeans/lerobot_dataset_put_pen_in_the_holder_20261118_215903",
                                    episodes=[51], spread=10, keep_stats=True, partial=True),
}


def _revision(repo: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def _far(x: np.ndarray) -> np.ndarray:
    q01, q50, q99 = np.percentile(x, [1, 50, 99])
    return np.abs(x - q50) > 20 * (q99 - q01)


def build(name: str, spec: dict) -> Path:
    rev = _revision(spec["repo"])
    src = f"{spec['repo']}@{rev}"
    if "episodes" in spec:
        ctx = load_dataset(src, max_episodes=spec["spread"])
        have = set(ctx.episode_ids())
        missing = [e for e in spec["episodes"] if e not in have]
        if missing:  # the phenomenon episodes must be in the extract
            full = load_dataset(src)
            keep = sorted(set(spec["episodes"]) | have)
            ctx = full
        else:
            keep = sorted(have)
    else:
        ctx = load_dataset(src, max_episodes=spec["sample"])
        keep = ctx.episode_ids()
    if "pick" in spec:
        p = spec["pick"]
        arr = ctx.state if p["feature"] == "state" else ctx.action
        far = _far(arr[:, p["dim"]])
        hit = sorted({int(e) for e in ctx.episode_index[far]})
        others = [e for e in ctx.episode_ids() if e not in hit]
        step = max(1, len(others) // p["extra"])
        keep = sorted(set(hit) | set(others[::step][:p["extra"]]))
    rows = np.concatenate([ctx.rows_for(e) for e in keep])
    out = {
        "state": ctx.state[rows].astype(np.float32), "action": ctx.action[rows].astype(np.float32),
        "episode_index": ctx.episode_index[rows].astype(np.int32),
        "frame_index": ctx.frame_index[rows].astype(np.int32),
        "fps": np.float64(ctx.fps or 30.0),
        "state_names": np.array([str(n) for n in (ctx.state_names or [f"s{i}" for i in range(ctx.state.shape[1])])]),
        "action_names": np.array([str(n) for n in (ctx.action_names or [f"a{i}" for i in range(ctx.action.shape[1])])]),
        "source": np.array(src),
    }
    if spec.get("keep_stats"):
        keys = ctx.feature_keys
        if not ctx.stats_json:
            out["no_stats_json"] = np.int8(1)  # stats come from per-episode stats only
        for short, canon in (("state", "observation.state"), ("action", "action")):
            st = ctx.stats_json.get(keys.get(canon, canon)) or {}
            if not st:
                continue
            for k in ("mean", "std", "min", "max"):
                out[f"stats_{short}_{k}"] = np.asarray(st[k], dtype=np.float64)
    if spec.get("partial"):
        cat = _load_episode_meta(ctx.root)
        out["catalogue_ids"] = np.array([e.index for e in cat], dtype=np.int64)
        out["catalogue_lengths"] = np.array([e.length for e in cat], dtype=np.int64)
        if spec.get("keep_stats"):
            eps_stats = _load_episode_stats(ctx.root, ctx.feature_keys)
            for short, canon in (("state", "observation.state"), ("action", "action")):
                feat = ctx.feature_keys.get(canon, canon)
                d = out[short].shape[1]
                for k in ("mean", "std", "min", "max"):
                    out[f"epstats_{short}_{k}"] = np.stack([
                        np.asarray(eps_stats.get(e.index, {}).get(feat, {}).get(k, np.full(d, np.nan)), dtype=np.float64)
                        for e in cat])
    if spec.get("dims"):  # the same dims of every array, statistic and name
        d = np.asarray(spec["dims"])
        for k in list(out):
            v = out[k]
            if k in ("state", "action") or k.startswith("epstats_"):
                out[k] = v[:, d]
            elif k.startswith("stats_") or k in ("state_names", "action_names"):
                out[k] = v[d]
    path = HERE / f"{name}.npz"
    np.savez_compressed(path, **out)
    print(f"{name}: {src} | {len(keep)} episodes, {len(rows)} frames | {path.stat().st_size / 1024:.0f} KB")
    return path


if __name__ == "__main__":
    for n in (sys.argv[1:] or CASES):
        build(n, CASES[n])

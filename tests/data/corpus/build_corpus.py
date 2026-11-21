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
    # (renamed on the Hub from ..._20261118_215903, which redirects here; the
    # revision is pinned - the dataset keeps growing)
    "mendelbeans_stale_stats": dict(repo="MendelBeans/lerobot_dataset_put_pen_in_the_holder",
                                    revision="b281c2c1f1c92d5180ca33ac1c36898a7f8c1ec3",
                                    episodes=[51], spread=10, keep_stats=True, partial=True),
    # The cases below load the exact data files the 2026-11-20 board sampled
    # (scripts/leaderboard.py download_sample) at the revision it read, then
    # keep `keep` episodes and the listed state / action dims.
    #
    # must NOT flag: dominating outliers. joint_qvel[0,1,5] (state 84/85/89;
    # base_qvel 253-255 is a byte-exact copy) reads ~1e-7 while the base
    # moves at ~0.25 m/s - uncorrelated with d(base_qpos)/dt - and its
    # spikes form a continuous heavy tail with as many frames at 10-20 bulk
    # widths as beyond 20. No frame_index column (the loader fills zeros):
    # the board called them reset transients; they sit mid-episode.
    "groot_basevel_tail": dict(
        repo="nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim",
        subpath="sim_behavior_r1_pro.task-0013_loading_the_car",
        revision="ea7ac0b68f87da62f1e726771bba0fe74300802f",
        files=[f"data/chunk-000/episode_{e:08d}.parquet" for e in (0, 5, 10, 15, 20, 26, 31, 36, 41)],
        keep=[0, 15, 26, 36], state_dims=[84, 85, 89], action_dims=[0]),
    # must NOT flag: dominating outliers. State dim 146 (robot_ori_sin[0] in
    # the R1Pro layout): single-frame spikes scattered mid-episode, a
    # continuous tail out of a 4e-5-wide bulk. The board and docs named these
    # reset transients. Source: the official BEHAVIOR-1K 2025 challenge demos
    # (MIT), task 45 - lerobot/behavior1k-task0045, where the board met it,
    # is a copy that declares no license (dims 140/146/150 bit-identical).
    "behavior1k_tilt_tail": dict(
        repo="behavior-1k/2025-challenge-demos",
        revision="33639692425296185f245aaa648530178c7b3d2b",
        files=[f"data/task-0045/episode_{e:08d}.parquet" for e in (450010, 450060, 450110, 450150, 450200, 450300)],
        meta=["meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"],
        keep=[450010, 450060, 450110, 450150, 450200, 450300], state_dims=[140, 146, 150], action_dims=[0]),
}


def _load_files(spec: dict):
    """Load the named data files (or the board's file spread, files="board")
    of a pinned revision, with the full meta/, as the board does."""
    from huggingface_hub import list_repo_tree, snapshot_download

    sub = spec.get("subpath", "")
    pre = f"{sub}/" if sub else ""
    files = spec["files"]
    if files == "board":  # download_sample's even spread of 40 files
        names = sorted(f.path[len(pre):] for f in list_repo_tree(
            spec["repo"], repo_type="dataset", recursive=True, revision=spec["revision"],
            path_in_repo=f"{pre}data") if hasattr(f, "size"))
        n = len(names)
        files = [names[i] for i in sorted({round(i * (n - 1) / 39) for i in range(40)} if n > 40 else range(n))]
    meta = spec.get("meta") or ["meta/**"]  # a repo with per-episode meta files names what it needs
    local = Path(snapshot_download(spec["repo"], repo_type="dataset", revision=spec["revision"],
                                   allow_patterns=[*(pre + m for m in meta), *(pre + f for f in files)]))
    root = local / sub if sub else local
    ctx = load_dataset(str(root), max_episodes=250)
    src = f"{spec['repo']}{'//' + sub if sub else ''}@{spec['revision']}"
    return ctx, src


def _revision(repo: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().dataset_info(repo).sha


def _far(x: np.ndarray) -> np.ndarray:
    q01, q50, q99 = np.percentile(x, [1, 50, 99])
    return np.abs(x - q50) > 20 * (q99 - q01)


def build(name: str, spec: dict) -> Path:
    src = f"{spec['repo']}@{spec.get('revision') or _revision(spec['repo'])}"
    if "files" in spec:
        ctx, src = _load_files(spec)
        keep = spec.get("keep") or ctx.episode_ids()
    elif "episodes" in spec:
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
    if ctx.present_columns is not None and "frame_index" not in ctx.present_columns:
        # the loader zero-fills a missing frame_index: keep each frame's real
        # position (for timestamps) and mark the column absent, as it is
        out["frame_index"] = np.concatenate([np.arange(len(ctx.rows_for(e))) for e in keep]).astype(np.int32)
        out["no_frame_index"] = np.int8(1)
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
    for short in ("state", "action"):  # separate cuts per feature
        if spec.get(f"{short}_dims"):
            d = np.asarray(spec[f"{short}_dims"])
            out[short] = out[short][:, d]
            out[f"{short}_names"] = out[f"{short}_names"][d]
    path = HERE / f"{name}.npz"
    np.savez_compressed(path, **out)
    print(f"{name}: {src} | {len(keep)} episodes, {len(rows)} frames | {path.stat().st_size / 1024:.0f} KB")
    return path


if __name__ == "__main__":
    for n in (sys.argv[1:] or CASES):
        build(n, CASES[n])

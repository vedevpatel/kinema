#!/usr/bin/env python3
"""Harm table v1: injected defects on lerobot/pusht, each scored by
what it does to a trained policy - held-out action MSE and rollout success -
against a clean arm trained the same way.

Arms: clean + duplicates_20pct, idle_start_30f, scale_mismatch_0.5x_25pct
(the harm study v0 recipes, imported from scripts/harm_study.py unchanged)
+ stuck_joint_30pct (one action dim frozen to its episode-start value in
30% of episodes) + action_saturation_30pct (actions clipped to the central
30% of each dim's range in every episode) + dropped_frames_10pct_25pct
(10% of interior frames deleted in 25% of episodes). Policy: LeRobot's diffusion-policy pusht example
(examples/training/train_policy.py at lerobot==0.5.1: DiffusionConfig
defaults, Adam lr 1e-4, 5000 steps), batch 32 for every arm. Outcomes:
held-out action MSE on the 24 episodes after the 120 training episodes (the
harm_study.py split) and success over 20 gym_pusht rollouts. 3 seeds.

Every GPU run is recorded in a manifest. A completed run is never retrained;
a run killed mid-way stays on the ledger as spent GPU time. Before an arm
launches, its projected cumulative GPU time (everything spent so far, plus
the estimated cost of the arm's remaining runs) must fit the ceiling
(15 GPU-hours) or the harness refuses it. docs/harm-table.json is rebuilt
from the manifest after every completed run.

Run:
  python scripts/harm_table.py --manifest PATH            # GPU, the full table
  python scripts/harm_table.py --manifest PATH --steps 500 --eval-episodes 2 \\
      --arms clean,stuck_joint_30pct --seeds 0            # GPU smoke
  python scripts/harm_table.py --dry-run [--out PATH]     # CPU, ridge BC

The GPU path needs `pip install "lerobot[pusht,transformers-dep]==0.5.1"`
(Python >= 3.12; transformers-dep pins the transformers release lerobot
0.5.1 imports at package load - a newer one breaks `import lerobot.envs`);
--dry-run needs only this repo.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import harm_study as hs  # noqa: E402

SCHEMA = "kinema.harm-table/v1"
MANIFEST_SCHEMA = "kinema.harm-manifest/v1"
DATASET_REPO, DATASET_REVISION = hs.PUSHT.split("@")
TRAIN_EPISODES = 120  # harm_study.py --episodes default
HOLDOUT_EPISODES = 24  # harm_study.py --holdout default
SEEDS = (0, 1, 2)
STEPS = 5000
BATCH_SIZE = 32  # identical for every arm, clean included
EVAL_EPISODES = 20
CEILING_GPU_HOURS = 15.0
LEROBOT_VERSION = "0.5.1"
POLICY_SOURCE = f"lerobot=={LEROBOT_VERSION} examples/training/train_policy.py (diffusion, pusht)"
EVAL_START_SEED = 100_000  # rollout env seeds: identical for every arm and seed (paired)
HELDOUT_NOISE_SEED = 1234  # diffusion sampling noise for held-out MSE: identical across arms
ESTIMATE_MARGIN = 1.10  # projected cost = worst observed comparable run x margin
MAX_TABLE_BYTES = 100_000


# --- the new recipe ----------------------------------------------------------

def inject_stuck_joint(eps, rng, fraction: float = 0.30):
    """One action dim frozen to its episode-start value in 30% of episodes
    (a dead motor during those recordings) - the stuck-actuator check's
    target. The dim is drawn once per dataset, the episodes without
    replacement; states are untouched."""
    if not eps:
        return []
    dim = int(rng.integers(eps[0][1].shape[1]))
    k = int(round(fraction * len(eps)))
    chosen = {int(i) for i in rng.choice(len(eps), size=k, replace=False)}
    out = []
    for i, (s, a) in enumerate(eps):
        if i in chosen:
            a = a.copy()
            a[:, dim] = a[0, dim]
        out.append((s, a))
    return out


def inject_action_saturation(eps, rng, low: float = 0.35, high: float = 0.65):
    """Actions clipped to the central 30% of each dim's range in EVERY
    episode - a controller saturating at its limits for the whole recording,
    so commands pile up at exactly the dataset min/max: the action-saturation
    check's target. Bounds are the train split's per-dim 35th/65th
    percentiles, computed before clipping; states are untouched. (Clipping
    only some episodes would leave the unclipped ones holding the global
    extremes, and the pile-up would sit off the limit the check measures.)"""
    if not eps:
        return []
    a_all = np.concatenate([a for _, a in eps])
    lo = np.quantile(a_all, low, axis=0)
    hi = np.quantile(a_all, high, axis=0)
    return [(s, np.clip(a, lo, hi)) for s, a in eps]


def action_saturation_self_check(eps, seed: int, low: float = 0.35, high: float = 0.65) -> list[str]:
    """Problems with the saturation recipe on these episodes ([] = pass):
    after injection at least one action dim must pin >=25% of its frames
    exactly at a clip bound - below that the recipe is not the defect the
    action-saturation check measures."""
    out = inject_action_saturation(eps, np.random.default_rng(seed), low, high)
    if not out:
        return ["no episodes"]
    a_all = np.concatenate([a for _, a in out])
    lo = np.quantile(np.concatenate([a for _, a in eps]), low, axis=0)
    hi = np.quantile(np.concatenate([a for _, a in eps]), high, axis=0)
    best = 0.0
    for j in range(a_all.shape[1]):
        tol = 1e-9 * max(1.0, abs(hi[j] - lo[j]))
        pinned = ((np.abs(a_all[:, j] - hi[j]) <= tol) | (np.abs(a_all[:, j] - lo[j]) <= tol)).mean()
        best = max(best, float(pinned))
    if best < 0.25:
        return [f"best dim pins only {best:.0%} of frames at a bound after clipping"]
    return []


def inject_dropped_frames(eps, rng, drop: float = 0.10, fraction: float = 0.25):
    """10% of interior frames deleted in 25% of episodes - the dropped-frames
    defect as the learner experiences it (missing transitions). The check
    spots the gap in frame_index metadata; the harm question is what the
    missing rows do to training, which is what this arm measures. First and
    last frames always survive; states and actions drop together."""
    out = []
    for s, a in eps:
        if len(s) > 2 and rng.random() < fraction:
            interior = np.arange(1, len(s) - 1)
            k = max(1, int(round(drop * len(interior))))
            keep = np.ones(len(s), dtype=bool)
            keep[rng.choice(interior, size=k, replace=False)] = False
            s, a = s[keep], a[keep]
        out.append((s, a))
    return out


def dropped_frames_self_check(eps, seed: int, drop: float = 0.10, fraction: float = 0.25) -> list[str]:
    """Problems with the dropped-frames recipe on these episodes ([] = pass):
    the same seed must drop the same frames, and some episode must lose
    frames (a recipe that never drops is not the defect)."""
    if not eps:
        return ["no episodes"]
    a = inject_dropped_frames(eps, np.random.default_rng(seed), drop, fraction)
    b = inject_dropped_frames(eps, np.random.default_rng(seed), drop, fraction)
    if [len(s) for s, _ in a] != [len(s) for s, _ in b]:
        return ["not deterministic for a fixed seed"]
    if [len(s) for s, _ in a] == [len(s) for s, _ in eps]:
        return ["no frames dropped"]
    if any(not np.array_equal(a[i][0][0], eps[i][0][0]) or not np.array_equal(a[i][0][-1], eps[i][0][-1])
           for i in range(len(eps)) if len(eps[i][0]) > 2):
        return ["first/last frame lost"]
    return []


def stuck_joint_self_check(eps, seed: int, fraction: float = 0.30) -> list[str]:
    """Problems with the stuck_joint recipe on these episodes ([] = pass):
    exactly one dim frozen to its start value, round(30%) of episodes,
    states untouched, the same seed gives the same output."""
    problems = []
    out = inject_stuck_joint(eps, np.random.default_rng(seed), fraction)
    again = inject_stuck_joint(eps, np.random.default_rng(seed), fraction)
    if len(out) != len(eps):
        return [f"episode count changed: {len(eps)} -> {len(out)}"]
    if any(not (np.array_equal(s, s2) and np.array_equal(a, a2)) for (s, a), (s2, a2) in zip(out, again)):
        problems.append("not seed-deterministic")
    dims: set[int] = set()
    changed = 0
    for (s, a), (s0, a0) in zip(out, eps):
        if not np.array_equal(s, s0):
            problems.append("a state was modified")
            break
        diff = [d for d in range(a0.shape[1]) if not np.array_equal(a[:, d], a0[:, d])]
        if not diff:
            continue
        changed += 1
        dims.update(diff)
        for d in diff:
            if not np.all(a[:, d] == a0[0, d]):
                problems.append(f"dim {d} not frozen to its episode-start value")
    if len(dims) > 1:
        problems.append(f"more than one dim frozen: {sorted(dims)}")
    k = int(round(fraction * len(eps)))
    if dims:
        d = next(iter(dims))
        frozen = sum(1 for _, a in out if np.all(a[:, d] == a[0, d]))
        already = sum(1 for _, a0 in eps if np.all(a0[:, d] == a0[0, d]))
        if not (changed <= k <= frozen <= k + already):
            problems.append(f"frozen in {frozen}/{len(eps)} episodes, expected {k} (~{fraction:.0%})")
    elif k:
        problems.append("no episode was frozen")
    return problems


@dataclass(frozen=True)
class Defect:
    check: str
    recipe: str
    params: dict[str, Any]
    inject: Callable


DEFECTS: dict[str, Defect] = {
    "duplicates_20pct": Defect(
        "duplicate-and-leakage", "copy 20% of training episodes (appended)",
        {"duplicate_fraction": 0.20}, hs.inject_duplicates),
    "idle_start_30f": Defect(
        "idle-segments", "prepend 30 still frames to every episode (first observation repeated, zero action)",
        {"pad_frames": 30, "pad_action": 0.0}, hs.inject_idle_starts),
    "scale_mismatch_0.5x_25pct": Defect(
        "calibration-drift", "actions x0.5 in ~25% of episodes (Bernoulli per episode), states untouched",
        {"action_scale": 0.5, "episode_probability": 0.25}, hs.inject_scale_mismatch),
    "stuck_joint_30pct": Defect(
        "stuck-actuator", "one action dim frozen to its episode-start value in 30% of episodes",
        {"frozen_dims": 1, "episode_fraction": 0.30, "frozen_value": "episode start"}, inject_stuck_joint),
    "action_saturation_30pct": Defect(
        "action-saturation", "actions clipped to the central 30% of each dim's range in every episode",
        {"clip_quantiles": [0.35, 0.65]}, inject_action_saturation),
    "dropped_frames_10pct_25pct": Defect(
        "dropped-frames", "10% of interior frames deleted in 25% of episodes (first/last kept)",
        {"drop_fraction": 0.10, "episode_fraction": 0.25}, inject_dropped_frames),
}
ARMS = ("clean", *DEFECTS)


def inject(arm: str, eps, seed: int):
    """The arm's training episodes; the defect rng is seeded per arm exactly
    as harm_study.py seeds it, so seed 0 reproduces the v0 numbers."""
    if arm == "clean":
        return list(eps)
    return DEFECTS[arm].inject(eps, np.random.default_rng(seed))


def harm_split():
    """The harm study split, loaded exactly as harm_study.py loads it:
    kinema's 144-episode sample of pusht (ids spread evenly over all 206),
    first 120 train, last 24 held out. Returns (train ids, held-out ids,
    per-episode (state, action) arrays in the same order)."""
    from kinema.dataset import load_dataset

    n = TRAIN_EPISODES + HOLDOUT_EPISODES
    ctx = load_dataset(hs.PUSHT, max_episodes=n)
    ids = [int(e) for e in ctx.episode_ids()[:n]]
    eps = hs.episode_arrays(ctx, n)
    if len(ids) != n or len(set(ids)) != n:
        raise RuntimeError(f"expected {n} distinct pusht episodes, got {len(set(ids))}")
    return ids[:TRAIN_EPISODES], ids[TRAIN_EPISODES:], eps


# --- run config, manifest, ceiling -------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    steps: int = STEPS
    eval_episodes: int = EVAL_EPISODES
    batch_size: int = BATCH_SIZE

    @property
    def key(self) -> str:
        return (f"diffusion/lerobot{LEROBOT_VERSION}/steps{self.steps}/bs{self.batch_size}"
                f"/eval{self.eval_episodes}/{DATASET_REVISION[:12]}/split{TRAIN_EPISODES}+{HOLDOUT_EPISODES}")


def run_key(cfg: RunConfig, arm: str, seed: int) -> str:
    return f"{cfg.key}/{arm}/seed{seed}"


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


class Manifest:
    """Ledger of every GPU run: completed runs (resumed, never retrained),
    the run in flight (heartbeat), and runs that died mid-way (their GPU time
    stays spent). Rewritten atomically on every change."""

    def __init__(self, path: Path, ceiling_gpu_hours: float = CEILING_GPU_HOURS):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text())
            if self.data.get("schema") != MANIFEST_SCHEMA:
                raise ValueError(f"{path}: not a {MANIFEST_SCHEMA} manifest")
        else:
            self.data = {"schema": MANIFEST_SCHEMA, "created": _utcnow(), "runs": {},
                         "abandoned": [], "overhead": []}
        self.data["ceiling_gpu_hours"] = ceiling_gpu_hours
        # a run still marked running was killed with its session: keep its
        # GPU time on the ledger, free the key for a fresh run
        for key, rec in list(self.data["runs"].items()):
            if rec.get("status") != "done":
                rec["status"] = "abandoned"
                self.data["abandoned"].append({"key": key, **rec})
                del self.data["runs"][key]
        self._last_save = 0.0
        self._lock = threading.RLock()  # the heartbeat thread saves too

    @property
    def ceiling_seconds(self) -> float:
        return float(self.data["ceiling_gpu_hours"]) * 3600

    def save(self) -> None:
        with self._lock:
            self.data["spent_gpu_hours"] = round(self.spent_seconds() / 3600, 4)
            _atomic_write(self.path, json.dumps(self.data, indent=1))
            self._last_save = time.monotonic()

    def spent_seconds(self) -> float:
        return (sum(r.get("gpu_seconds", 0.0) for r in self.data["runs"].values())
                + sum(r.get("gpu_seconds", 0.0) for r in self.data["abandoned"])
                + sum(r.get("gpu_seconds", 0.0) for r in self.data["overhead"]))

    def done(self, key: str) -> dict | None:
        rec = self.data["runs"].get(key)
        return rec if rec and rec.get("status") == "done" else None

    def start(self, key: str, cfg: RunConfig, arm: str, seed: int) -> None:
        with self._lock:
            self.data["runs"][key] = {"status": "running", "config": cfg.key, "arm": arm, "seed": seed,
                                      "steps": cfg.steps, "eval_episodes": cfg.eval_episodes,
                                      "batch_size": cfg.batch_size, "started": _utcnow(), "gpu_seconds": 0.0}
            self.save()

    def heartbeat(self, key: str, gpu_seconds: float, every: float = 30.0) -> None:
        with self._lock:
            rec = self.data["runs"].get(key)
            if rec is None or rec.get("status") != "running":
                return
            rec["gpu_seconds"] = gpu_seconds
            if time.monotonic() - self._last_save >= every:
                self.save()

    def finish(self, key: str, result: dict, gpu_seconds: float) -> None:
        with self._lock:
            rec = self.data["runs"][key]
            rec.update(result)
            rec.update(status="done", finished=_utcnow(), gpu_seconds=gpu_seconds)
            self.save()

    def add_overhead(self, what: str, gpu_seconds: float) -> None:
        with self._lock:
            self.data["overhead"].append({"what": what, "at": _utcnow(), "gpu_seconds": gpu_seconds})
            self.save()

    def estimate_seconds(self, cfg: RunConfig, prior_hours: float) -> float:
        """Projected GPU seconds for one run of `cfg`: the worst completed
        run of the same config; else the worst completed run of any config,
        rescaled (training by steps, rollouts by episodes); else the prior.
        Times the safety margin."""
        done = [r for r in self.data["runs"].values() if r.get("status") == "done"]
        same = [r["gpu_seconds"] for r in done if r.get("config") == cfg.key]
        if same:
            return max(same) * ESTIMATE_MARGIN
        scaled = []
        for r in done:
            parts = r.get("train_seconds", 0.0) + r.get("eval_seconds", 0.0)
            fixed = max(0.0, r["gpu_seconds"] - parts)
            scaled.append(fixed + r.get("train_seconds", 0.0) * cfg.steps / max(1, r["steps"])
                          + r.get("eval_seconds", 0.0) * cfg.eval_episodes / max(1, r["eval_episodes"]))
        if scaled:
            return max(scaled) * ESTIMATE_MARGIN
        return prior_hours * 3600


TrainFn = Callable[[str, int, RunConfig, Callable[[], None]], dict]


def run_plan(manifest: Manifest, cfg: RunConfig, arms: Sequence[str], seeds: Sequence[int],
             train_fn: TrainFn, *, prior_hours: float = 1.0, session_hours: float | None = None,
             clock: Callable[[], float] = time.monotonic, on_done: Callable[[], None] | None = None,
             log: Callable[[str], None] = print, heartbeat_seconds: float = 30.0) -> str:
    """Run every (arm, seed) not already done, arm by arm. Before an arm
    launches: spent + estimate x (its remaining runs) must fit the ceiling,
    or the arm is refused and nothing else starts ("refused"). Before each
    run the same holds for that run alone. A run that would overrun
    `session_hours` is left for the next session ("session-budget").
    Returns "complete", "refused" or "session-budget"."""
    session_start = clock()
    for arm in arms:
        todo = [s for s in seeds if not manifest.done(run_key(cfg, arm, s))]
        for s in seeds:
            if s not in todo:
                log(f"[resume] {arm} seed {s}: done, not retrained")
        if not todo:
            continue
        est = manifest.estimate_seconds(cfg, prior_hours)
        spent = manifest.spent_seconds()
        projected = spent + est * len(todo)
        if projected > manifest.ceiling_seconds:
            log(f"[ceiling] REFUSED arm {arm}: spent {spent / 3600:.2f} + {len(todo)} run(s) x "
                f"{est / 3600:.2f} = {projected / 3600:.2f} GPU-h > ceiling "
                f"{manifest.ceiling_seconds / 3600:.2f} GPU-h; nothing launched")
            return "refused"
        log(f"[ceiling] arm {arm}: projected {projected / 3600:.2f} / "
            f"{manifest.ceiling_seconds / 3600:.2f} GPU-h")
        for seed in todo:
            key = run_key(cfg, arm, seed)
            est = manifest.estimate_seconds(cfg, prior_hours)
            spent = manifest.spent_seconds()
            if spent + est > manifest.ceiling_seconds:
                log(f"[ceiling] REFUSED {arm} seed {seed}: spent {spent / 3600:.2f} + "
                    f"{est / 3600:.2f} GPU-h > ceiling; nothing launched")
                return "refused"
            if session_hours is not None and (clock() - session_start) + est > session_hours * 3600:
                log(f"[session] {arm} seed {seed} (~{est / 3600:.2f} GPU-h) would overrun the "
                    f"{session_hours:.2f} h session budget; resume in a new session")
                return "session-budget"
            manifest.start(key, cfg, arm, seed)
            t0 = clock()
            # a background heartbeat keeps the ledger current through every
            # phase, LeRobot's eval loop included: a run killed anywhere loses
            # at most `heartbeat_seconds` of spent time
            stop = threading.Event()

            def beat(key=key, t0=t0, stop=stop):
                while not stop.wait(heartbeat_seconds):
                    manifest.heartbeat(key, clock() - t0, every=0)

            beater = threading.Thread(target=beat, daemon=True)
            beater.start()
            try:
                result = train_fn(arm, seed, cfg, lambda: manifest.heartbeat(key, clock() - t0))
            except BaseException:
                stop.set()
                beater.join()
                manifest.heartbeat(key, clock() - t0, every=0)  # stays "running" -> abandoned, still spent
                raise
            stop.set()
            beater.join()
            manifest.finish(key, result, gpu_seconds=clock() - t0)
            log(f"[done] {arm} seed {seed}: held-out MSE {result['heldout_mse']:.4f}, success "
                f"{result['success_pct']}%, {(clock() - t0) / 60:.1f} min; ledger "
                f"{manifest.spent_seconds() / 3600:.2f} GPU-h")
            if on_done:
                on_done()
    return "complete"


# --- the table ----------------------------------------------------------------

def _mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs))


def confidence_label(n_seeds: int) -> str:
    return f"indicative ({n_seeds} seed{'s' if n_seeds != 1 else ''})"


def build_table(results: dict[tuple[str, int], dict], *, policy: dict, steps: int, eval_episodes: int | None,
                split: tuple[list[int], list[int]], compute: dict | None = None, date: str | None = None,
                dates: dict[str, str] | None = None) -> dict:
    """The schema-v1 document from per-(arm, seed) results ({"heldout_mse",
    "success_pct" or None}). A defect row uses the seeds where both it and
    clean finished; a defect with none has no row yet. `dates` overrides a
    defect row's measured date (table_from_manifest passes the arm's finish
    day); `date` remains the document's generation day and the row fallback."""
    date = date or dt.date.today().isoformat()
    dates = dates or {}
    rows = []
    for defect, spec in DEFECTS.items():
        seeds = sorted(s for (a, s) in results if a == defect and ("clean", s) in results)
        if not seeds:
            continue
        mse = [100.0 * (results[(defect, s)]["heldout_mse"] - results[("clean", s)]["heldout_mse"])
               / results[("clean", s)]["heldout_mse"] for s in seeds]
        succ_raw = [(results[(defect, s)].get("success_pct"), results[("clean", s)].get("success_pct"))
                    for s in seeds]
        succ = None if any(d is None or c is None for d, c in succ_raw) else [d - c for d, c in succ_raw]
        rows.append({
            "defect": defect,
            "check": spec.check,
            "recipe": {"description": spec.recipe, "params": copy.deepcopy(spec.params)},
            "policy": policy["name"],
            "dataset_revision": DATASET_REVISION,
            "steps": steps,
            "seeds": seeds,
            "mse_delta_pct": {"per_seed": [round(x, 3) for x in mse], "mean": round(_mean(mse), 3)},
            "success_delta_pts": ({"per_seed": [round(x, 3) for x in succ], "mean": round(_mean(succ), 3)}
                                  if succ is not None else {"per_seed": [None] * len(seeds), "mean": None}),
            "date": dates.get(defect, date),
            "confidence": confidence_label(len(seeds)),
        })
    runs = [{"arm": a, "seed": s, **{k: v for k, v in r.items() if k in _RUN_FIELDS}}
            for (a, s), r in sorted(results.items(), key=lambda kv: (ARMS.index(kv[0][0]), kv[0][1]))]
    all_seeds = sorted({s for _, s in results})
    return {
        "schema": SCHEMA,
        "generated": date,
        "dataset": {"repo_id": DATASET_REPO, "revision": DATASET_REVISION,
                    "episode_sample": (f"kinema load_dataset(max_episodes={TRAIN_EPISODES + HOLDOUT_EPISODES}): ids "
                                       f"spread evenly over the dataset; first {TRAIN_EPISODES} train, last "
                                       f"{HOLDOUT_EPISODES} held out (harm_study.py split)"),
                    "train_episodes": list(split[0]), "heldout_episodes": list(split[1])},
        "policy": policy,
        "steps": steps,
        "eval_episodes": eval_episodes,
        "seeds": all_seeds,
        "confidence": confidence_label(len(all_seeds)),
        "outcomes": {
            "mse_delta_pct": "held-out action MSE, defect arm vs clean arm, same seed, % of clean",
            "success_delta_pts": "rollout success rate, defect arm minus clean arm, same seed, percentage points",
        },
        "rows": rows,
        "runs": runs,
        "compute": compute or {},
    }


_RUN_FIELDS = ("heldout_mse", "success_pct", "avg_max_reward", "gpu_seconds", "gpu", "it_per_s")


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_table(doc: Any) -> list[str]:
    """Schema-v1 problems with a harm-table document ([] = valid)."""
    if not isinstance(doc, dict):
        return ["document is not an object"]
    errs = []
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    for k, typ in (("generated", str), ("dataset", dict), ("policy", dict), ("steps", int),
                   ("seeds", list), ("confidence", str), ("rows", list), ("runs", list), ("compute", dict)):
        if not isinstance(doc.get(k), typ):
            errs.append(f"top-level {k!r} missing or not {typ.__name__}")
    if errs:
        return errs
    if doc["dataset"].get("revision") != DATASET_REVISION:
        errs.append("dataset revision is not the pinned pusht revision")
    tr, he = doc["dataset"].get("train_episodes"), doc["dataset"].get("heldout_episodes")
    if not (isinstance(tr, list) and isinstance(he, list) and len(tr) == TRAIN_EPISODES
            and len(he) == HOLDOUT_EPISODES and all(isinstance(i, int) for i in tr + he)
            and len(set(tr) | set(he)) == TRAIN_EPISODES + HOLDOUT_EPISODES):
        errs.append(f"dataset split must be {TRAIN_EPISODES} train + {HOLDOUT_EPISODES} held-out distinct episode ids")
    has_rollouts = doc["policy"].get("rollouts", False)
    seen = set()
    for i, row in enumerate(doc["rows"]):
        where = f"rows[{i}]"
        if not isinstance(row, dict):
            errs.append(f"{where} is not an object")
            continue
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            errs.append(f"{where} missing {missing}")
            continue
        d = row["defect"]
        if d not in DEFECTS:
            errs.append(f"{where} unknown defect {d!r}")
            continue
        if d in seen:
            errs.append(f"{where} duplicate defect {d!r}")
        seen.add(d)
        if row["check"] != DEFECTS[d].check:
            errs.append(f"{where} check {row['check']!r} is not {DEFECTS[d].check!r}")
        if row["recipe"] != {"description": DEFECTS[d].recipe, "params": DEFECTS[d].params}:
            errs.append(f"{where} recipe does not match the {d} recipe")
        if row["dataset_revision"] != DATASET_REVISION or row["steps"] != doc["steps"]:
            errs.append(f"{where} revision/steps disagree with the table")
        seeds = row["seeds"]
        if not (isinstance(seeds, list) and seeds and all(isinstance(s, int) for s in seeds)):
            errs.append(f"{where} seeds must be a non-empty int list")
            continue
        if row["confidence"] != confidence_label(len(seeds)):
            errs.append(f"{where} confidence {row['confidence']!r} != {confidence_label(len(seeds))!r}")
        for metric, required in (("mse_delta_pct", True), ("success_delta_pts", has_rollouts)):
            m = row[metric]
            ps = m.get("per_seed") if isinstance(m, dict) else None
            if not isinstance(ps, list) or len(ps) != len(seeds):
                errs.append(f"{where} {metric}.per_seed must have one value per seed")
                continue
            if all(v is None for v in ps) and m.get("mean") is None and not required:
                continue
            if not all(_num(v) for v in ps) or not _num(m.get("mean")):
                errs.append(f"{where} {metric} has non-finite or missing values")
            elif abs(_mean(ps) - m["mean"]) > 1e-3 + 1e-6 * abs(m["mean"]):
                errs.append(f"{where} {metric}.mean is not the mean of per_seed")
    if len(json.dumps(doc).encode()) >= MAX_TABLE_BYTES:
        errs.append(f"document is >= {MAX_TABLE_BYTES} bytes")
    return errs


ROW_KEYS = ("defect", "check", "recipe", "policy", "dataset_revision", "steps", "seeds",
            "mse_delta_pct", "success_delta_pts", "date", "confidence")


def write_table(path: Path, doc: dict) -> None:
    errs = validate_table(doc)
    if errs:
        raise ValueError("harm table failed schema validation: " + "; ".join(errs))
    _atomic_write(path, json.dumps(doc, indent=1) + "\n")


def table_from_manifest(manifest: Manifest, cfg: RunConfig, split: tuple[list[int], list[int]]) -> dict:
    results = {(r["arm"], r["seed"]): r for r in manifest.data["runs"].values()
               if r.get("status") == "done" and r.get("config") == cfg.key}
    gpus = sorted({r.get("gpu", "?") for r in results.values()})
    # each row's date is the day the arm's measurement COMPLETED (its latest
    # finished run), not the day the table happened to be regenerated - a
    # regen must not re-stamp old measurements
    dates = {}
    for (arm, _seed), r in results.items():
        if r.get("finished"):
            d = r["finished"][:10]
            if arm not in dates or d > dates[arm]:
                dates[arm] = d
    return build_table(
        results,
        policy={"name": "diffusion", "source": POLICY_SOURCE, "rollouts": True,
                "batch_size": cfg.batch_size, "optimizer": "Adam lr=1e-4 (example)",
                "num_inference_steps": "config default (= num_train_timesteps, 100)",
                "normalization_stats": "per arm, from the arm's own training frames"},
        steps=cfg.steps, eval_episodes=cfg.eval_episodes, split=split, dates=dates,
        compute={"gpu": gpus, "table_runs_gpu_hours": round(sum(r["gpu_seconds"] for r in results.values()) / 3600, 3),
                 "ledger_gpu_hours": round(manifest.spent_seconds() / 3600, 3),
                 "ceiling_gpu_hours": manifest.data["ceiling_gpu_hours"]},
    )


# --- CPU dry-run: the harm_study.py ridge math --------------------------------

def dry_run(train_eps, held_eps, seeds: Sequence[int]) -> dict[tuple[str, int], dict]:
    Xh, Yh = hs.flatten(held_eps)
    out = {}
    for seed in seeds:
        for arm in ARMS:
            W = hs.ridge_fit(*hs.flatten(inject(arm, train_eps, seed)))
            out[(arm, seed)] = {"heldout_mse": hs.ridge_mse(W, Xh, Yh), "success_pct": None}
    return out


GATE0 = {  # harm study v0 numbers the dry-run must reproduce: (target %, abs tol, rel tol)
    "duplicates_20pct": (0.1, 1.0, None),
    "scale_mismatch_0.5x_25pct": (347.0, None, 0.20),
    "idle_start_30f": (881.0, None, 0.20),
}


def gate0_report(doc: dict, self_check: list[str]) -> bool:
    ok = True
    rows = {r["defect"]: r for r in doc["rows"]}
    for defect, (target, abs_tol, rel_tol) in GATE0.items():
        r = rows[defect]["mse_delta_pct"]
        tol = abs_tol if abs_tol is not None else rel_tol * abs(target)
        hit = abs(r["mean"] - target) <= tol
        ok &= hit
        print(f"[gate0] {defect:28s} mean {r['mean']:+.1f}% per-seed {r['per_seed']} "
              f"target {target:+.1f}% +/-{tol:.1f} -> {'PASS' if hit else 'FAIL'}")
    sj = rows["stuck_joint_30pct"]["mse_delta_pct"]
    finite = all(_num(v) for v in sj["per_seed"]) and _num(sj["mean"])
    for arm in ("action_saturation_30pct", "dropped_frames_10pct_25pct"):
        m = rows[arm]["mse_delta_pct"]
        arm_finite = all(_num(v) for v in m["per_seed"]) and _num(m["mean"])
        ok &= arm_finite
        print(f"[gate0] {arm:28s} mean {m['mean']:+.1f}% per-seed {m['per_seed']} finite={arm_finite}")
    ok &= finite and not self_check
    print(f"[gate0] stuck_joint_30pct            mean {sj['mean']:+.1f}% per-seed {sj['per_seed']} "
          f"finite={finite}; self-check {'PASS' if not self_check else 'FAIL: ' + '; '.join(self_check)}")
    print(f"[gate0] {'PASS' if ok else 'FAIL'}")
    return ok


def main_dry_run(args) -> int:
    train_ids, held_ids, eps = harm_split()
    train_eps, held_eps = eps[:TRAIN_EPISODES], eps[TRAIN_EPISODES:]
    problems = [p for s in args.seeds for p in stuck_joint_self_check(train_eps, s)]
    problems += [f"action_saturation: {p}" for s in args.seeds
                 for p in action_saturation_self_check(train_eps, s)]
    problems += [f"dropped_frames: {p}" for s in args.seeds
                 for p in dropped_frames_self_check(train_eps, s)]
    results = dry_run(train_eps, held_eps, args.seeds)
    for (arm, seed), r in sorted(results.items(), key=lambda kv: (kv[0][1], ARMS.index(kv[0][0]))):
        print(f"seed {seed} {arm:28s} held-out MSE {r['heldout_mse']:.5f}")
    doc = build_table(results, policy={"name": "ridge-bc (dry-run)", "source": "scripts/harm_study.py math",
                                       "rollouts": False}, steps=0, eval_episodes=None,
                     split=(train_ids, held_ids))
    errs = validate_table(doc)
    print(f"[schema] {'valid' if not errs else 'INVALID: ' + '; '.join(errs)}")
    if args.out:
        write_table(Path(args.out), doc)
        print(f"wrote {args.out}")
    ok = gate0_report(doc, problems) and not errs
    return 0 if ok else 1


# --- GPU path: the LeRobot diffusion pusht example ----------------------------

class PushtFrames:
    """Frames of the harm-split pusht episodes (train then held out, in
    split order), decoded once per process: state/action float32, image
    uint8 CHW, per-episode row bounds."""

    def __init__(self, state, action, image, bounds):
        self.state, self.action, self.image, self.bounds = state, action, image, bounds

    def episodes(self, lo: int, hi: int):
        """(state + frame-id column, action) float64 tuples - the shape the
        harm_study injectors take. The frame id rides along with the state,
        so a recipe that copies or repeats a state copies its image too."""
        out = []
        for start, end in self.bounds[lo:hi]:
            ids = np.arange(start, end, dtype=np.float64)[:, None]
            out.append((np.concatenate([self.state[start:end].astype(np.float64), ids], axis=1),
                        self.action[start:end].astype(np.float64)))
        return out


def load_pusht_frames(episode_ids: Sequence[int], ref_eps, cache_dir: Path) -> PushtFrames:
    """Decode `episode_ids` through LeRobotDataset; their state/action must
    match `ref_eps` (kinema's loader, what harm_study.py trains on)."""
    import hashlib

    tag = hashlib.sha1(",".join(map(str, episode_ids)).encode()).hexdigest()[:10]
    cache = cache_dir / f"pusht-{DATASET_REVISION[:12]}-{len(episode_ids)}ep-{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        return PushtFrames(z["state"], z["action"], z["image"], [tuple(b) for b in z["bounds"]])
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET_REPO, revision=DATASET_REVISION, episodes=sorted(episode_ids))
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False,
                                         num_workers=min(8, os.cpu_count() or 1))
    parts: dict[str, list] = {"ep": [], "fi": [], "state": [], "action": [], "image": []}
    for b in loader:
        parts["ep"].append(b["episode_index"].numpy().reshape(-1))
        parts["fi"].append(b["frame_index"].numpy().reshape(-1))
        parts["state"].append(b["observation.state"].numpy())
        parts["action"].append(b["action"].numpy())
        parts["image"].append((b["observation.image"] * 255).round().clamp(0, 255).to(torch.uint8).numpy())
    ep, fi = np.concatenate(parts["ep"]), np.concatenate(parts["fi"])
    order = np.lexsort((fi, ep))
    ep, fi = ep[order], fi[order]
    state = np.concatenate(parts["state"])[order].astype(np.float32)
    action = np.concatenate(parts["action"])[order].astype(np.float32)
    image = np.concatenate(parts["image"])[order]
    bounds = []
    for e in episode_ids:
        rows = np.flatnonzero(ep == e)
        if len(rows) == 0 or not np.array_equal(fi[rows], np.arange(len(rows))) or rows[-1] - rows[0] + 1 != len(rows):
            raise RuntimeError(f"episode {e}: frames missing or out of order")
        bounds.append((int(rows[0]), int(rows[-1]) + 1))
    # the same frames harm_study.py reads through kinema's loader
    for e, (s0, a0), (start, end) in zip(episode_ids, ref_eps, bounds, strict=True):
        if s0.shape != (end - start, 2) or not (np.allclose(s0, state[start:end], atol=1e-4)
                                                and np.allclose(a0, action[start:end], atol=1e-4)):
            raise RuntimeError(f"episode {e}: LeRobot frames disagree with kinema's loader on state/action")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, state=state, action=action, image=image, bounds=np.array(bounds))
    return PushtFrames(state, action, image, bounds)


def _windows(eps_lengths: Sequence[int], obs_delta: Sequence[int], act_delta: Sequence[int]):
    """LeRobotDataset delta_timestamps semantics over concatenated episodes:
    per sample, row indices of its observation and action frames (clamped
    to the episode) and the action padding mask."""
    obs_idx, act_idx, act_pad, cur = [], [], [], []
    off = 0
    od, ad = np.asarray(obs_delta), np.asarray(act_delta)
    for L in eps_lengths:
        t = np.arange(L)[:, None]
        obs_idx.append(np.clip(t + od, 0, L - 1) + off)
        raw = t + ad
        act_pad.append((raw < 0) | (raw >= L))
        act_idx.append(np.clip(raw, 0, L - 1) + off)
        cur.append(t[:, 0] + off)
        off += L
    return (np.concatenate(obs_idx), np.concatenate(act_idx), np.concatenate(act_pad), np.concatenate(cur))


def arm_arrays(eps):
    """Concatenate (state + frame id, action) episodes into per-row arrays."""
    s = np.concatenate([e[0] for e in eps])
    a = np.concatenate([e[1] for e in eps])
    return (s[:, :-1].astype(np.float32), a.astype(np.float32), np.rint(s[:, -1]).astype(np.int64),
            [len(e[0]) for e in eps])


def frame_channel_moments(image_u8, chunk: int = 1024):
    """Per-frame, per-channel mean, mean of squares, min, max in [0, 1]
    (chunked: float64 of every pusht frame at once is ~4 GB)."""
    parts: list[list[np.ndarray]] = [[], [], [], []]
    for i in range(0, len(image_u8), chunk):
        img = image_u8[i:i + chunk].astype(np.float64) / 255.0
        for acc, v in zip(parts, (img.mean(axis=(2, 3)), (img ** 2).mean(axis=(2, 3)),
                                  img.min(axis=(2, 3)), img.max(axis=(2, 3)))):
            acc.append(v)
    return tuple(np.concatenate(p) for p in parts)


def arm_stats(state, action, img_ids, frame_chan_moments):
    """Normalization stats from the arm's own training rows (what a
    dataset's stats.json would hold for this data): min/max/mean/std for
    state and action, per-channel mean/std in [0, 1] for the image."""
    def vec(x):
        return {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0)}

    m1, m2, mn, mx = frame_chan_moments
    w = np.bincount(img_ids, minlength=len(m1)).astype(np.float64)
    mean = w @ m1 / w.sum()
    std = np.sqrt(np.maximum(w @ m2 / w.sum() - mean ** 2, 0.0))
    used = np.unique(img_ids)
    img_min, img_max = mn[used].min(axis=0), mx[used].max(axis=0)
    shape = (3, 1, 1)
    return {
        "observation.state": vec(state),
        "action": vec(action),
        "observation.image": {"mean": mean.reshape(shape).astype(np.float32),
                              "std": std.reshape(shape).astype(np.float32),
                              "min": img_min.reshape(shape).astype(np.float32),
                              "max": img_max.reshape(shape).astype(np.float32)},
    }


def _probe_rollout(env, policy, env_pre, preprocessor, postprocessor, dev) -> dict:
    """Time one policy inference and one env step on a real observation, and
    report where the tensors live - the rollout's cost model, logged per run."""
    import torch
    from lerobot.envs.utils import preprocess_observation

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    obs, _ = env.reset(seed=[0] * env.num_envs)
    batch = preprocessor(env_pre(preprocess_observation(obs)))

    # the rollout's cost model: one denoising-UNet forward, and one of its
    # 2048-channel convs alone, per batch size and grad mode (ms)
    cfg = policy.config
    stacked = {"observation.state": torch.stack([batch["observation.state"]] * cfg.n_obs_steps, dim=1),
               "observation.images": torch.stack([torch.stack([batch[k] for k in cfg.image_features], dim=-4)]
                                                 * cfg.n_obs_steps, dim=1)}
    unet, conv = policy.diffusion.unet, policy.diffusion.unet.mid_modules[0].conv1.block[0]
    with torch.no_grad():
        cond1 = policy.diffusion._prepare_global_conditioning(stacked)[:1]
    timing = {}
    for bs in (2, 32, 512):
        x = torch.zeros((bs, cfg.horizon, cfg.action_feature.shape[0]), device=dev)
        ts = torch.full((bs,), 10, dtype=torch.long, device=dev)
        cond = cond1.expand(bs, -1).contiguous()
        xc = torch.zeros((bs, conv.in_channels, cfg.horizon // 4), device=dev)
        for mode, ctx in (("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)):
            for name, fn in (("unet", lambda: unet(x, ts, global_cond=cond)), ("conv", lambda: conv(xc))):
                with ctx():
                    fn()
                    sync()
                    t = time.monotonic()
                    for _ in range(2):
                        fn()
                    sync()
                timing[f"{name}/bs{bs}/{mode}"] = round((time.monotonic() - t) / 2 * 1000, 2)
    bench = torch.backends.cudnn.benchmark

    policy.reset()
    sync()
    t = time.monotonic()
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    sync()
    infer = time.monotonic() - t
    t = time.monotonic()
    env.step(action.numpy())
    step = time.monotonic() - t
    policy.reset()
    return {"inference_s": round(infer, 3), "env_step_s": round(step, 4), "forward_ms": timing,
            "cudnn_benchmark": bench, "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "cudnn_enabled": torch.backends.cudnn.enabled,
            "param_device": str(next(policy.parameters()).device),
            "obs_devices": sorted({str(v.device) for v in batch.values() if hasattr(v, "device")}),
            "torch_threads": torch.get_num_threads(), "n_envs": env.num_envs}


class GpuRunner:
    """train_fn for run_plan: one arm x seed of the LeRobot diffusion pusht
    example, then held-out MSE and rollout success."""

    def __init__(self, device: str, cache_dir: Path, split: tuple[list[int], list[int]], ref_eps):
        self.device = device
        self.cache_dir = cache_dir
        self.split, self.ref_eps = split, ref_eps
        self._frames: PushtFrames | None = None

    def frames(self) -> PushtFrames:
        if self._frames is None:
            self._frames = load_pusht_frames(self.split[0] + self.split[1], self.ref_eps, self.cache_dir)
            self._moments = frame_channel_moments(self._frames.image)
        return self._frames

    def __call__(self, arm: str, seed: int, cfg: RunConfig, heartbeat: Callable[[], None]) -> dict:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import torch
        from lerobot.configs.types import FeatureType
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.feature_utils import dataset_to_policy_features
        from lerobot.envs.configs import PushtEnv
        from lerobot.envs.factory import make_env, make_env_pre_post_processors
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.scripts.lerobot_eval import eval_policy
        from lerobot.utils.random_utils import set_seed

        # cuDNN autotuning, as LeRobot's own train and eval scripts enable it
        # (it picks conv algorithms, not math; TF32 stays off as in the
        # example). Measured: an A100 runs a batch-2 UNet forward in ~8 ms; a
        # T4 under torch 2.10 takes ~850 ms with or without autotuning, which
        # makes rollouts there take ~1 h per batch - use an A100 or better.
        torch.backends.cudnn.benchmark = True
        fr = self.frames()
        dev = torch.device(self.device)
        train_eps = inject(arm, fr.episodes(0, TRAIN_EPISODES), seed)
        state, action, img_ids, lengths = arm_arrays(train_eps)
        stats = arm_stats(state, action, img_ids, self._moments)

        # the example, verbatim where it can be: features from the dataset
        # metadata, DiffusionConfig defaults, Adam lr 1e-4
        set_seed(seed)
        meta = LeRobotDatasetMetadata(DATASET_REPO, revision=DATASET_REVISION)
        features = dataset_to_policy_features(meta.features)
        output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {k: ft for k, ft in features.items() if k not in output_features}
        pcfg = DiffusionConfig(input_features=input_features, output_features=output_features, device=self.device)
        policy = DiffusionPolicy(pcfg)
        policy.train()
        policy.to(dev)
        preprocessor, postprocessor = make_pre_post_processors(pcfg, dataset_stats=stats)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        img_key = next(iter(pcfg.image_features))

        obs_idx, act_idx, act_pad, _ = _windows(lengths, pcfg.observation_delta_indices, pcfg.action_delta_indices)
        T = {k: torch.as_tensor(v, device=dev) for k, v in dict(
            state=state, action=action, img_ids=img_ids, obs_idx=obs_idx, act_idx=act_idx, act_pad=act_pad).items()}
        images = torch.as_tensor(fr.image, device=dev)  # uint8, every train + held-out frame

        def obs_batch(st, ids, oi):
            return {img_key: images[ids[oi]].float() / 255.0, "observation.state": st[oi]}

        # shuffle=True, drop_last=True, as the example's DataLoader
        gen = torch.Generator().manual_seed(seed)
        n = len(obs_idx)
        step, t0 = 0, time.monotonic()
        while step < cfg.steps:
            perm = torch.randperm(n, generator=gen).to(dev)
            for i in range(0, n - cfg.batch_size + 1, cfg.batch_size):
                idx = perm[i:i + cfg.batch_size]
                batch = obs_batch(T["state"], T["img_ids"], T["obs_idx"][idx])
                batch["action"] = T["action"][T["act_idx"][idx]]
                batch["action_is_pad"] = T["act_pad"][idx]
                batch = preprocessor(batch)
                loss, _ = policy.forward(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 250 == 0 or step == cfg.steps:
                    print(f"  {arm} seed {seed} step {step}/{cfg.steps} loss {loss.item():.4f} "
                          f"{step / (time.monotonic() - t0):.2f} it/s", flush=True)
                    heartbeat()
                if step >= cfg.steps:
                    break
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_seconds = time.monotonic() - t0

        # held-out action MSE: the action the policy would execute at t
        # (first of its chunk), raw units, on clean held-out episodes
        t1 = time.monotonic()
        policy.eval()
        h_state, h_action, h_ids, h_len = arm_arrays(fr.episodes(TRAIN_EPISODES, TRAIN_EPISODES + HOLDOUT_EPISODES))
        h_obs, _, _, h_cur = _windows(h_len, pcfg.observation_delta_indices, pcfg.action_delta_indices)
        hs_t, hid_t = torch.as_tensor(h_state, device=dev), torch.as_tensor(h_ids, device=dev)
        noise_gen = torch.Generator().manual_seed(HELDOUT_NOISE_SEED)
        a_dim = pcfg.action_feature.shape[0]
        preds = []
        with torch.no_grad():
            for i in range(0, len(h_obs), 512):
                oi = torch.as_tensor(h_obs[i:i + 512], device=dev)
                b = dict(preprocessor(obs_batch(hs_t, hid_t, oi)))
                b["observation.images"] = torch.stack([b[k] for k in pcfg.image_features], dim=-4)
                noise = torch.randn((len(oi), pcfg.horizon, a_dim), generator=noise_gen).to(dev)
                preds.append(postprocessor(policy.diffusion.generate_actions(b, noise=noise))[:, 0].cpu().numpy())
        pred = np.concatenate(preds)
        heldout_mse = float(np.mean((pred.astype(np.float64) - h_action[h_cur].astype(np.float64)) ** 2))
        heldout_seconds = time.monotonic() - t1
        print(f"  {arm} seed {seed} held-out MSE {heldout_mse:.3f} in {heldout_seconds:.1f}s "
              f"({len(h_obs)} frames)", flush=True)
        heartbeat()

        # rollout success: LeRobot's own env factory and eval loop
        t2 = time.monotonic()
        env_cfg = PushtEnv()
        envs = make_env(env_cfg, n_envs=min(cfg.eval_episodes, 20))
        env = next(iter(next(iter(envs.values())).values()))
        env_pre, env_post = make_env_pre_post_processors(env_cfg, pcfg)
        probe = _probe_rollout(env, policy, env_pre, preprocessor, postprocessor, dev)
        print(f"  {arm} seed {seed} rollout probe: {probe}", flush=True)
        set_seed(seed)
        with torch.no_grad():
            info = eval_policy(env, policy, env_pre, env_post, preprocessor, postprocessor,
                               n_episodes=cfg.eval_episodes, start_seed=EVAL_START_SEED)
        env.close()
        eval_seconds = time.monotonic() - t2
        agg = info["aggregated"]
        print(f"  {arm} seed {seed} rollouts: success {agg['pc_success']:.1f}% over {cfg.eval_episodes} "
              f"in {eval_seconds:.1f}s", flush=True)
        return {
            "heldout_mse": heldout_mse,
            "success_pct": float(agg["pc_success"]),
            "avg_max_reward": float(agg["avg_max_reward"]),
            "train_seconds": train_seconds,
            "heldout_seconds": heldout_seconds,
            "eval_seconds": eval_seconds,
            "it_per_s": cfg.steps / train_seconds,
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type,
            "num_inference_steps": int(policy.diffusion.num_inference_steps),
            "rollout_probe": probe,
            "train_rows": int(len(state)),
            "train_samples": int(n),
        }


def main_gpu(args) -> int:
    if not args.manifest:
        raise SystemExit("--manifest is required for the GPU path")
    cfg = RunConfig(steps=args.steps, eval_episodes=args.eval_episodes)
    manifest = Manifest(Path(args.manifest), args.ceiling_gpu_hours)
    manifest.save()
    out = Path(args.out)
    print(f"[plan] {cfg.key}: arms {list(args.arms)} x seeds {list(args.seeds)}; ledger "
          f"{manifest.spent_seconds() / 3600:.2f} / {manifest.ceiling_seconds / 3600:.2f} GPU-h")

    t0 = time.monotonic()
    train_ids, held_ids, ref_eps = harm_split()
    split = (train_ids, held_ids)
    print(f"[split] train {len(train_ids)} episodes {train_ids[:5]}..., held out {held_ids}")

    def write():
        write_table(out, table_from_manifest(manifest, cfg, split))
        print(f"[table] wrote {out}", flush=True)

    runner = GpuRunner(args.device, Path(args.cache_dir).expanduser(), split, ref_eps)
    in_runs: list[float] = []

    def timed(arm, seed, cfg, heartbeat):
        t = time.monotonic()
        try:
            return runner(arm, seed, cfg, heartbeat)
        finally:
            in_runs.append(time.monotonic() - t)

    try:
        status = run_plan(manifest, cfg, args.arms, args.seeds, timed, prior_hours=args.prior_run_hours,
                          session_hours=args.session_hours, on_done=write)
    finally:
        # process time outside runs is GPU-runtime time too: on the ledger
        overhead = time.monotonic() - t0 - sum(in_runs)
        if overhead > 1:
            manifest.add_overhead("harness process time outside runs", overhead)
    if any(r.get("config") == cfg.key for r in manifest.data["runs"].values()):
        write()
    print(f"[status] {status}; ledger {manifest.spent_seconds() / 3600:.3f} / "
          f"{manifest.ceiling_seconds / 3600:.2f} GPU-h")
    return {"complete": 0, "refused": 3, "session-budget": 75}[status]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="CPU ridge-BC path (harm_study.py math); no GPU, no manifest")
    ap.add_argument("--manifest", help="GPU run ledger (JSON); resumed if present")
    ap.add_argument("--out", help="table path (GPU default docs/harm-table.json; dry-run writes only if given)")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    ap.add_argument("--arms", type=lambda s: s.split(","), default=list(ARMS))
    ap.add_argument("--ceiling-gpu-hours", type=float, default=CEILING_GPU_HOURS)
    ap.add_argument("--prior-run-hours", type=float, default=1.0,
                    help="projected cost of a run before any comparable run has finished")
    ap.add_argument("--session-hours", type=float, help="do not start a run that would end after this")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default="~/.cache/kinema-harm-table")
    args = ap.parse_args()
    bad = [a for a in args.arms if a not in ARMS]
    if bad or "clean" not in args.arms:
        raise SystemExit(f"--arms must include clean and come from {list(ARMS)}; got {bad or args.arms}")
    args.arms = [a for a in ARMS if a in args.arms]
    if args.dry_run:
        return main_dry_run(args)
    args.out = args.out or str(REPO / "docs" / "harm-table.json")
    return main_gpu(args)


if __name__ == "__main__":
    raise SystemExit(main())

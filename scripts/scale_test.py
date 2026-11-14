#!/usr/bin/env python3
"""Scale test (finding #11): run kinema check on the very large datasets and
record wall time, peak RSS and per-run outcome at growing episode counts.

The v0 leaderboard was tuned on <=5k-episode sets; DROID (76k episodes),
BitRobot and robocasa were never loaded. This script is the honest way to
close that: run it on a machine big enough to hold the working set.

Requirements (measured, not guessed):
  - RAM: a 500-episode DROID sample peaked at ~1.8 GB AFTER the loader fixes
    (batched row-group reads + memmap writeback flushing, e4439dc) - but was
    OOM-killed on a 2 GB sandbox at exit -9 before finishing. Run on >=8 GB
    to have headroom; 16 GB (GitHub's larger runner) is comfortable.
  - Disk: sampled Hub shards land in ~/.cache/huggingface; a 500-episode
    DROID sample is ~9 GB. Full BitRobot is much larger - keep 100 GB free.
  - Time: DROID 500-ep fetch+load dominates; expect 10+ min per run on
    datacenter bandwidth, much more on a laptop.
  - NOTE: --video is NOT part of this test yet - on DROID it fetches 823
    files even for a 4-episode sample (video chunks don't align with the
    episode sampler). Fix that before adding video to the scale gate.

Usage:
  python scripts/scale_test.py                       # the default ladder
  python scripts/scale_test.py --repo lerobot/droid_1.0.1 --episodes 50 100 250 500
  python scripts/scale_test.py --out /tmp/scaletest  # where JSON lands

Each run writes <tag>.json (the kinema report, when the run completes) and
<tag>.meta.json (exit code, wall time, peak RSS, user/sys CPU). A run killed
by the OOM killer shows exit -9; that IS the scale signal - the episode
count where it dies is the machine's ceiling.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_LADDER = [
    ("lerobot/droid_1.0.1", [50, 100, 250, 500, 1000]),
]


def run_one(repo: str, episodes: int, out: Path, frame_budget: int | None) -> dict:
    tag = f"{repo.split('/')[-1].replace('.', '_')}_{episodes}"
    env = dict(os.environ)
    if frame_budget:
        env["KINEMA_FRAME_BUDGET_BYTES"] = str(frame_budget)
    start = time.time()
    p = subprocess.Popen(
        [sys.executable, "-m", "kinema.cli", "check", repo,
         "--max-episodes", str(episodes), "--json", str(out / f"{tag}.json")],
        env=env,
    )
    pid, status, ru = os.wait4(p.pid, 0)
    meta = {
        "repo": repo,
        "max_episodes": episodes,
        "exit": os.waitstatus_to_exitcode(status),
        "wall_s": round(time.time() - start, 1),
        "peak_rss_mb": round(ru.ru_maxrss / 1024, 1),
        "user_s": round(ru.ru_utime, 1),
        "sys_s": round(ru.ru_stime, 1),
    }
    (out / f"{tag}.meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", action="append", help="Hub repo id (repeatable)")
    ap.add_argument("--episodes", type=int, nargs="+",
                    default=[50, 100, 250, 500, 1000])
    ap.add_argument("--out", default="/tmp/scaletest")
    ap.add_argument("--frame-budget-bytes", type=int, default=256 << 20,
                    help="KINEMA_FRAME_BUDGET_BYTES for each run (0 = unset)")
    args = ap.parse_args()

    ladder = [(r, args.episodes) for r in args.repo] if args.repo else DEFAULT_LADDER
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for repo, counts in ladder:
        for n in counts:
            print(f"=== {repo} max_episodes={n}", flush=True)
            meta = run_one(repo, n, out, args.frame_budget_bytes or None)
            results.append(meta)
            print(f"    exit {meta['exit']}  wall {meta['wall_s']}s  "
                  f"peak RSS {meta['peak_rss_mb']} MB", flush=True)
            if meta["exit"] == -9:
                print("    OOM-killed: stopping the ladder here - larger "
                      "counts will not fit on this machine", flush=True)
                break

    print("\nrepo  episodes  exit  wall_s  peak_rss_mb")
    for m in results:
        print(f"{m['repo']}  {m['max_episodes']}  {m['exit']}  "
              f"{m['wall_s']}  {m['peak_rss_mb']}")
    (out / "summary.json").write_text(json.dumps(results, indent=1))
    return 0 if all(m["exit"] == 0 for m in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

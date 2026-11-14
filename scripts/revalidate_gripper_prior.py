"""Revalidate the gripper-slot prior against name-labelled dims (finding #15).

The prior in ``stats.inferred_gripper_dims`` decides which unnamed dims are
grippers.  Its only honest error budget comes from datasets whose dims DO
carry names: strip the names, run the prior, and score its slot picks against
the names as ground truth.  That is what finding #15's "~7% of slot picks"
number means, and what this script re-measures after the both-ends rule.

Run:  python scripts/revalidate_gripper_prior.py [--max-episodes 60] [--json out.json]

Scoring, per dataset with at least one non-generic action dim name:
  - prediction: inferred_gripper_dims(action, names=None) (as if unlabelled)
  - ground truth: is_gripper(name) over non-generic names only
  - a pick on a labelled non-gripper is a false positive; a labelled gripper
    sitting in a slot the prior did not pick is a false negative; a labelled
    gripper in a NON-slot dim is reported apart (a layout miss, finding #16,
    not a slot-prior error).  Dims with generic/absent names are unlabelled
    and excluded from the score, as are datasets whose names are all generic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from kinema.dataset import load_dataset  # noqa: E402
from kinema.stats import (  # noqa: E402
    gripper_dims,
    inferred_gripper_dims,
    is_generic,
)

CATALOG = REPO / "scripts" / "calibration_datasets.txt"


def catalog_specs(path: Path = CATALOG) -> list[str]:
    specs = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            specs.append(line.split()[0])
    return specs


def slot_dims(n: int) -> set[int]:
    """Dims the prior can ever pick: last, and mid for even 12+ layouts."""
    slots = {n - 1}
    if n >= 12 and n % 2 == 0:
        slots.add(n // 2 - 1)
    return slots


def old_rule_picks(action, slots) -> set[int]:
    """The pre-#15 prior: slot + dwell at EITHER extreme (no both-ends rule)."""
    import numpy as np

    out = set()
    for j in sorted(slots):
        x = action[:, j]
        x = x[np.isfinite(x)]
        if len(x) < 20:
            continue
        lo, hi = np.percentile(x, [1, 99])
        rng = hi - lo
        if rng <= 1e-8:
            continue
        dwell = float(np.mean(x <= lo + 0.1 * rng) + np.mean(x >= hi - 0.1 * rng))
        if dwell >= 0.30:
            out.add(j)
    return out


def _tally(picks, truth, slots):
    return {
        "tp": sorted(picks & truth),
        "fp": sorted(picks - truth),
        "fn": sorted((truth & slots) - picks),
    }


def score_dataset(action, names) -> dict:
    """Score the prior's picks on one dataset's labelled dims."""
    n = action.shape[1]
    ok_names = names if names and len(names) == n else None
    labelled = {j for j in range(n) if ok_names and not is_generic([ok_names[j]])}
    truth = (gripper_dims(ok_names, n) if ok_names else set()) & labelled
    picks = inferred_gripper_dims(action, None)
    slots = slot_dims(n)
    old = old_rule_picks(action, slots)
    row = {
        "dims": n,
        "labelled": len(labelled),
        "grippers": sorted(truth),
        "picks": sorted(picks),
        "layout_miss": sorted(truth - slots),
    }
    row.update(_tally(picks, truth, slots))
    row["old_picks"] = sorted(old)
    row["old_fp"] = _tally(old, truth, slots)["fp"]
    row["old_fn"] = _tally(old, truth, slots)["fn"]
    return row


def aggregate(rows: list[dict]) -> dict:
    """Pool per-dataset scores into the precision/recall summary."""
    tp = sum(len(r["tp"]) for r in rows)
    fp = sum(len(r["fp"]) for r in rows)
    fn = sum(len(r["fn"]) for r in rows)
    layout = sum(len(r["layout_miss"]) for r in rows)
    picks = tp + fp
    old_picks = sum(len(r["old_picks"]) for r in rows)
    old_fp = sum(len(r["old_fp"]) for r in rows)
    old_fn = sum(len(r["old_fn"]) for r in rows)
    return {
        "datasets": len(rows),
        "slot_picks": picks,
        "fp": fp,
        "fp_rate": (fp / picks) if picks else 0.0,
        "precision": (tp / picks) if picks else 1.0,
        "recall": (tp / (tp + fn)) if tp + fn else 1.0,
        "fn": fn,
        "layout_misses": layout,
        "old_rule": {
            "slot_picks": old_picks,
            "fp": old_fp,
            "fp_rate": (old_fp / old_picks) if old_picks else 0.0,
            "fn": old_fn,
            "recall": ((old_picks - old_fp) / (old_picks - old_fp + old_fn))
                      if old_picks - old_fp + old_fn else 1.0,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-episodes", type=int, default=60)
    ap.add_argument("--json", metavar="PATH", help="write full results JSON")
    ap.add_argument("--resume", metavar="PATH",
                    help="JSONL of per-dataset results; completed specs are "
                    "skipped and new ones appended (survives interruption)")
    args = ap.parse_args()

    rows, skipped, done = [], [], set()
    resume = Path(args.resume) if args.resume else None
    if resume and resume.exists():
        for line in resume.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add(rec["spec"])
            (skipped if "error" in rec else rows).append(rec)
    out = open(resume, "a") if resume else None
    try:
        for spec in catalog_specs():
            if spec in done:
                continue
            try:
                ctx = load_dataset(spec, max_episodes=args.max_episodes)
            except Exception as e:  # noqa: BLE001 - report and move on
                rec = {"spec": spec, "error": str(e)[:200]}
                skipped.append(rec)
            else:
                if (ctx.action is None or not ctx.action_names
                        or is_generic(ctx.action_names)):
                    rec = {"spec": spec, "error": "no usable action names"}
                elif len(ctx.action_names) != ctx.action.shape[1]:
                    rec = {"spec": spec,
                           "error": f"names ({len(ctx.action_names)}) don't "
                                    f"match dims ({ctx.action.shape[1]})"}
                    skipped.append(rec)
                else:
                    rec = {"spec": spec}
                    rec.update(score_dataset(ctx.action, ctx.action_names))
                    rows.append(rec)
                del ctx
            if out:
                out.write(json.dumps(rec) + "\n")
                out.flush()
    finally:
        if out:
            out.close()

    summary = aggregate(rows)
    print(f"{'dataset':70s} dims picks truth  fp      fn      layout-miss")
    for r in rows:
        print(f"{r['spec'][:70]:70s} {r['dims']:4d} {str(r['picks']):7s} "
              f"{str(r['grippers']):7s} {str(r['fp']):7s} {str(r['fn']):7s} "
              f"{r['layout_miss']}")
    print()
    print(f"scored datasets: {summary['datasets']}  "
          f"slot picks: {summary['slot_picks']}  FP: {summary['fp']} "
          f"({summary['fp_rate']:.1%} of picks)  "
          f"precision {summary['precision']:.1%}  recall {summary['recall']:.1%}  "
          f"layout misses: {summary['layout_misses']}")
    o = summary["old_rule"]
    print(f"old either-end rule for comparison: picks {o['slot_picks']}  "
          f"FP {o['fp']} ({o['fp_rate']:.1%} of picks)  recall {o['recall']:.1%}")
    if skipped:
        print(f"skipped {len(skipped)} datasets:")
        for s in skipped:
            print(f"  {s['spec']}: {s['error']}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "rows": rows,
                                               "skipped": skipped}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

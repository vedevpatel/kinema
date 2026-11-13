"""Check 1 - stats-drift.

Every training run normalizes with the dataset's stored stats. If those
numbers drifted from the data (dataset edited after stats were computed,
stats copied from another dataset, unit change), normalization silently
feeds the policy wrong-scaled inputs. Recompute and compare.

Errors are measured in the units normalization cares about: mean shift in
standard deviations, std as a ratio, min/max as a fraction of the range.
(Relative error on a mean that sits near zero is meaningless.)

On a sampled subset the whole-dataset stats are not a valid reference - a
50-episode sample has its own mean and range. The subset is compared
against the re-aggregated per-episode stats of exactly the loaded
episodes instead, and stats.json is checked against all per-episode stats
from the (always complete) metadata.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY
from kinema.stats import aggregate, as_stat_dict, data_stats


def _errors(actual: dict, ref: dict) -> dict[str, np.ndarray]:
    """Per-stat, per-dim error in normalization-relevant units."""
    rng = np.maximum(ref["max"] - ref["min"], actual["max"] - actual["min"])
    rng_floor = np.maximum(rng, 1e-8)
    sigma = np.maximum.reduce([ref["std"], actual["std"], 1e-3 * rng, np.full_like(rng, 1e-8)])
    out = {
        "mean": np.abs(actual["mean"] - ref["mean"]) / sigma,
        "min": np.abs(actual["min"] - ref["min"]) / rng_floor,
        "max": np.abs(actual["max"] - ref["max"]) / rng_floor,
    }
    # std ratio; dims with (near-)zero variance on both sides are check 8's job
    # (a std of 1e-3 vs 1e-7 on a constant dim is not drift that matters)
    # Converters often floor std (e.g. at 1e-3) for dead dims; a floor over
    # a dim that does not move in the data is not drift that matters.
    scale = np.maximum(1.0, np.abs(ref["mean"]))
    both_flat = (np.maximum(ref["std"], actual["std"]) < 1e-3 * scale) | (
        (actual["std"] < 1e-4 * scale) & (ref["std"] <= 1e-2 * scale)
    )
    out["std"] = np.where(
        both_flat, 0.0, np.abs(actual["std"] - ref["std"]) / np.maximum(ref["std"], 1e-8)
    )
    return out


_UNIT = {"mean": "std devs", "std": "relative", "min": "of range", "max": "of range"}


def _fmt(stat: str, err: float) -> str:
    return f"{err:.2f} std devs" if stat == "mean" else f"{err:.1%}" + (
        " of the range" if stat in ("min", "max") else ""
    )


class StatsDriftCheck(Check):
    id = "stats-drift"
    title = "Stats drift (stored normalization stats vs actual data)"

    def _compare(self, res, feat, actual, ref, ref_label, cfg, what) -> float:
        """Append findings for actual-vs-ref; return worst error scaled to
        its own FAIL threshold (1.0 == at the FAIL line)."""
        tol = {
            "mean": (cfg.stats_mean_warn_sigma, cfg.stats_mean_fail_sigma),
            "std": (cfg.stats_std_warn, cfg.stats_std_fail),
            "min": (cfg.stats_range_warn, cfg.stats_range_fail),
            "max": (cfg.stats_range_warn, cfg.stats_range_fail),
        }
        worst = 0.0
        for stat in ("mean", "std", "min", "max"):
            if stat not in ref:
                res.findings.append(
                    Finding(self.id, Severity.WARN, f"{feat}: {ref_label} is missing '{stat}'")
                )
                continue
            if ref[stat].shape != actual[stat].shape:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{feat}.{stat}: shape {ref[stat].shape} in {ref_label} "
                        f"vs {actual[stat].shape} in {what}",
                    )
                )
                continue
        if any(ref.get(k) is None or ref[k].shape != actual[k].shape for k in ("mean", "std", "min", "max")):
            return worst
        errs = _errors(actual, ref)
        for stat, err in errs.items():
            warn, fail = tol[stat]
            dim = int(np.argmax(err))
            e = float(err[dim])
            worst = max(worst, e / fail)
            if e > fail:
                sev = Severity.FAIL
            elif e > warn:
                sev = Severity.WARN
            else:
                continue
            # a stored std that is too LARGE only shrinks inputs; too small
            # amplifies them (the dangerous direction). Cap the former at WARN.
            if stat == "std" and ref["std"][dim] > actual["std"][dim]:
                sev = min(sev, Severity.WARN)
            res.findings.append(
                Finding(
                    self.id,
                    sev,
                    f"{feat}.{stat}: {ref_label} is off by {_fmt(stat, e)} on dim {dim} "
                    f"(stored {ref[stat][dim]:.6g}, {what} {actual[stat][dim]:.6g}). "
                    "Normalization at training and inference will disagree.",
                    details={"feature": feat, "stat": stat, "dim": dim, "error": e,
                             "unit": _UNIT[stat], "reference": ref_label},
                )
            )
        return worst

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        worst = 0.0
        loaded = ctx.episode_ids()
        mode = "full"
        episode_stats_ok = False
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            stored = as_stat_dict(ctx.stats.get(feat))
            if not stored or "mean" not in stored:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{feat}: no normalization stats (neither meta/stats.json nor "
                        "per-episode stats) - policies that normalize will fail",
                    )
                )
                continue
            actual = data_stats(arr)
            if actual is None:
                continue

            if not ctx.truncated:
                worst = max(worst, self._compare(res, feat, actual, stored, ctx.stats_source, cfg, "data"))
            else:
                per_ep = [ctx.episode_stats[e][feat] for e in loaded
                          if feat in ctx.episode_stats.get(e, {})]
                sub_ref = aggregate(per_ep) if len(per_ep) == len(loaded) else None
                if sub_ref is not None:
                    mode = "per-episode"
                    before = len(res.findings)
                    worst = max(worst, self._compare(
                        res, feat, actual, sub_ref, "per-episode stats", cfg,
                        f"data ({len(loaded)} loaded episodes)"))
                    episode_stats_ok = not any(
                        f.severity >= Severity.WARN for f in res.findings[before:])
                    if ctx.stats_source == "meta/stats.json":
                        # training normalizes with stats.json, not these: broken
                        # per-episode stats only bite on re-aggregation / merges
                        for f in res.findings[before:]:
                            if f.severity == Severity.FAIL:
                                f.severity = Severity.WARN
                                f.message += (" (per-episode stats only; training uses "
                                              "meta/stats.json, but merges and re-aggregation "
                                              "will inherit the error)")
                else:
                    # no exact reference for a subset: only a value outside the
                    # stored range is conclusive (the stats cannot describe it)
                    mode = "containment"
                    if "min" in stored and "max" in stored and stored["min"].shape == actual["min"].shape:
                        rng = np.maximum(stored["max"] - stored["min"], 1e-8)
                        over = np.maximum(stored["min"] - actual["min"], actual["max"] - stored["max"]) / rng
                        dim = int(np.argmax(over))
                        e = float(over[dim])
                        worst = max(worst, e / cfg.stats_range_fail)
                        if e > cfg.stats_range_warn:
                            res.findings.append(Finding(
                                self.id,
                                Severity.FAIL if e > cfg.stats_range_fail else Severity.WARN,
                                f"{feat}: sampled data reaches {e:.1%} of the range outside the "
                                f"stored [min, max] on dim {dim} - the stored stats cannot "
                                "describe this data",
                                details={"feature": feat, "dim": dim, "outside_range_fraction": e},
                            ))

            # stats.json vs the per-episode stats it should be aggregated from:
            # conclusive on any download, since meta is always complete
            if feat in ctx.stats_json and ctx.stats_source == "meta/stats.json":
                all_ids = [e for e in ctx.episode_stats]
                per_all = [ctx.episode_stats[e][feat] for e in all_ids if feat in ctx.episode_stats[e]]
                agg_all = aggregate(per_all) if per_all else None
                js = as_stat_dict(ctx.stats_json.get(feat))
                # only conclusive when the per-episode stats themselves agree
                # with the loaded data; otherwise they are the broken side
                if (agg_all is not None and js and len(per_all) >= len(ctx.episode_stats)
                        and ctx.truncated and mode == "per-episode" and episode_stats_ok):
                    worst = max(worst, self._compare(
                        res, feat, agg_all, js, "meta/stats.json", cfg,
                        f"aggregated per-episode stats ({len(per_all)} episodes)"))

        if mode == "containment":
            res.findings.append(Finding(
                self.id, Severity.INFO,
                "sampled subset and no per-episode stats: only range containment is "
                "checked (mean/std of a sample are not comparable to whole-dataset stats)",
            ))
        if not any(f.severity >= Severity.WARN for f in res.findings):
            ref = {"full": f"{ctx.stats_source}", "per-episode": "per-episode stats of the loaded episodes",
                   "containment": "stored range"}[mode]
            res.summary = f"data matches {ref} (worst {worst:.0%} of the FAIL threshold)"
        else:
            res.summary = f"worst error {worst:.1f}x the FAIL threshold"
        return res

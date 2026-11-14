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
            # A stored std that is too SMALL amplifies values (always serious).
            # Too LARGE shrinks them: harmless-ish for state inputs (WARN at
            # most), but for actions the normalized targets shrink and the
            # loss on that dim drops by ratio^2 - at 3x the policy all but
            # ignores it.
            oversize_note = ""
            if stat == "std" and ref["std"][dim] > actual["std"][dim]:
                ratio = ref["std"][dim] / max(actual["std"][dim], 1e-12)
                if feat == ACTION_KEY and ratio >= cfg.stats_std_oversize_fail:
                    sev = Severity.FAIL
                    oversize_note = (f" Stored std is {ratio:.1f}x the real one: the loss on "
                                     f"this action dim is down-weighted ~{ratio**2:.0f}x.")
                else:
                    sev = min(sev, Severity.WARN)
            res.findings.append(
                Finding(
                    self.id,
                    sev,
                    f"{feat}.{stat}: {ref_label} is off by {_fmt(stat, e)} on dim {dim} "
                    f"(stored {ref[stat][dim]:.6g}, {what} {actual[stat][dim]:.6g}). "
                    "Normalization at training and inference will disagree." + oversize_note,
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
                    # a sample's mean/std differ from the whole dataset's, but
                    # not by multiples: gross corruption is still visible
                    if "std" in stored and stored["std"].shape == actual["std"].shape:
                        sd_s, sd_a = stored["std"], actual["std"]
                        live = (sd_s > 1e-6) & (sd_a > 1e-6)
                        ratio = np.where(live, np.maximum(sd_s, sd_a) / np.maximum(np.minimum(sd_s, sd_a), 1e-12), 1.0)
                        shift = np.where(live, np.abs(actual["mean"] - stored["mean"]) / np.maximum(sd_s, 1e-12), 0.0)
                        for kind, garr, lim, fmt in (("std", ratio, cfg.stats_sample_std_ratio, "{:.1f}x"),
                                                    ("mean", shift, cfg.stats_sample_mean_sigma, "{:.1f} std devs")):
                            d = int(np.argmax(garr))
                            if garr[d] > lim:
                                res.findings.append(Finding(
                                    self.id, Severity.WARN,
                                    f"{feat}.{kind}: the sampled data is off the stored stats by "
                                    f"{fmt.format(float(garr[d]))} on dim {d} - far more than sampling "
                                    "explains; the stored stats look wrong (sampled audit: load the "
                                    "full dataset to confirm)",
                                    details={"feature": feat, "stat": kind, "dim": d,
                                             "error": float(garr[d]), "reference": "sampled"},
                                ))
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

                    # sharper than the gross-ratio backstop when enough
                    # episodes are loaded (finding #19): the sampling
                    # distribution of the pooled mean/std, by bootstrap over
                    # episodes. The episode is the resampling unit - frames
                    # within one are autocorrelated, so a per-frame standard
                    # error would be far too tight. The stored stats are the
                    # population numbers; they should fall inside the
                    # sample's bootstrap interval unless they are wrong.
                    eps_ok = []
                    for ep_id in loaded:
                        ep_rows = ctx.rows_for(ep_id)
                        a = arr[ep_rows]
                        a = a[np.isfinite(a).all(axis=1)]
                        if len(a) >= 5:
                            eps_ok.append((a.mean(axis=0), a.std(axis=0), float(len(a))))
                    if (len(eps_ok) >= cfg.stats_sample_min_episodes
                            and "mean" in stored and "std" in stored
                            and stored["mean"].shape == actual["mean"].shape):
                        mode = "bootstrap"
                        em = np.stack([t[0] for t in eps_ok])
                        es = np.stack([t[1] for t in eps_ok])
                        ec = np.asarray([t[2] for t in eps_ok])
                        E = len(eps_ok)
                        brng = np.random.default_rng(0)
                        idx = brng.integers(0, E, size=(cfg.stats_sample_bootstrap, E))
                        W = ec[idx]  # (B, E)
                        sel_m = em[idx]  # (B, E, D)
                        wsum = W.sum(axis=1)[:, None]
                        pm = (W[:, :, None] * sel_m).sum(axis=1) / wsum
                        pv = (W[:, :, None] * (es[idx] ** 2 + (sel_m - pm[:, None, :]) ** 2)).sum(axis=1) / wsum
                        ps = np.sqrt(np.maximum(pv, 0.0))
                        qlo, qhi = cfg.stats_sample_alpha / 2, 1 - cfg.stats_sample_alpha / 2
                        m_lo, m_hi = np.quantile(pm, [qlo, qhi], axis=0)
                        s_lo, s_hi = np.quantile(ps, [qlo, qhi], axis=0)
                        for kind, st, lo, hi in (("mean", stored["mean"], m_lo, m_hi),
                                                 ("std", stored["std"], s_lo, s_hi)):
                            out = np.maximum(lo - st, st - hi) / np.maximum((hi - lo) / 2, 1e-12)
                            d = int(np.argmax(out))
                            if out[d] > 0:
                                res.findings.append(Finding(
                                    self.id, Severity.WARN,
                                    f"{feat}.{kind}: the stored {kind} is inconsistent with the "
                                    f"loaded sample on dim {d} (stored {st[d]:.6g}, outside the "
                                    f"sample's {100 * (1 - cfg.stats_sample_alpha):.1f}% bootstrap "
                                    f"interval [{lo[d]:.6g}, {hi[d]:.6g}] over {E} episodes) - the "
                                    "stored stats look wrong (sampled audit: load the full dataset "
                                    "to confirm)",
                                    details={"feature": feat, "stat": kind, "dim": d,
                                             "reference": "bootstrap", "episodes": E,
                                             "stored": float(st[d]),
                                             "interval": [float(lo[d]), float(hi[d])]},
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
                "sampled subset, too few loaded episodes for a bootstrap reference: "
                "only gross mean/std shifts and range containment are checked",
            ))
        elif mode == "bootstrap" and not any(
                f.severity >= Severity.WARN for f in res.findings):
            res.findings.append(Finding(
                self.id, Severity.INFO,
                "sampled subset and no per-episode stats: mean/std checked against "
                "the bootstrap sampling distribution of the loaded episodes, plus "
                "range containment (a full load is the conclusive audit)",
            ))
        if not any(f.severity >= Severity.WARN for f in res.findings):
            ref = {"full": f"{ctx.stats_source}", "per-episode": "per-episode stats of the loaded episodes",
                   "bootstrap": "the sample's bootstrap interval and stored range",
                   "containment": "stored range"}[mode]
            res.summary = f"data matches {ref} (worst {worst:.0%} of the FAIL threshold)"
        else:
            res.summary = f"worst error {worst:.1f}x the FAIL threshold"
        return res

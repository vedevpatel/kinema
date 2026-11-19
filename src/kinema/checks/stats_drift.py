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


# A dim is inert for normalization when it does not move IN THE DATA at its
# own magnitude: whatever the stored mean/std, a constant input normalizes
# to a constant. Judged from the data side only - the stored std is exactly
# the number under test, so it cannot decide flatness. Relative to |mean|
# because float32 resolution is relative (a constant channel at 4096 accumulates
# ~1e-9 of f32 noise, and its converter-written stored std can reach 0.5:
# the RoboCOIN case). The line is in float32 ulps (2^-23 of |mean|): 3e-5
# is ~250 ulps. 1e-6 (~8 ulps) was too tight - RoboCOIN Alphabot channels
# at ~0.9 jitter by std 4e-6 (~70 ulps, far below any encoder tick) and
# their stored stats carry the same converter noise: 4 board FAIL/WARNs at
# 1e-6, all clean at 3e-5. A joint at 1000 with std 0.1 (~1600 ulps of real
# motion) stays compared, as does one with std 0.9.
_FLAT_REL = 3e-5

# a sample spanning less than this share of the stored [min, max] has not seen
# the dataset's spread, so sampled mean/std gaps cannot be called wrong stats
_SAMPLE_MIN_COVER = 0.5


def _reach_note(stored: dict, actual: dict, d: int) -> str:
    """When the stored [min, max] reaches far beyond every loaded frame, say
    so: unloaded episodes then hold extreme values that set this dim's
    normalization (IPEC-COMMUNITY/dobbe_lerobot: stored z reaches -132 m, a
    40-episode sample stays within [-2.4, 1.7]) - nan-and-normalization
    measures it on a load that includes them."""
    if "min" not in stored or "max" not in stored:
        return ""
    span = float(actual["max"][d] - actual["min"][d])
    reach = max(float(actual["min"][d] - stored["min"][d]), float(stored["max"][d] - actual["max"][d]))
    if span <= 0 or not np.isfinite(reach) or reach < 20 * span:
        return ""
    return (f"; the stored range [{stored['min'][d]:.4g}, {stored['max'][d]:.4g}] reaches "
            f"{reach / span:.0f}x the loaded data's spread beyond it - a few unloaded "
            "episodes may hold extreme frames that dominate this dim's normalization "
            "(a full load measures it)")


def noise_floor(actual: dict) -> np.ndarray:
    """Per dim, the smallest spread that is not float32 accumulation noise:
    _FLAT_REL of the dim's own magnitude (max |value|). Scale-covariant - a
    unit change moves the floor with the data - where the old absolute
    floors (1e-8 sigma / range guards, max(1, |mean|) in the flat line) let
    millimetre vs metre decide the verdict (tests/test_invariance.py)."""
    mag = np.maximum(np.abs(actual["min"]), np.abs(actual["max"]))
    return np.where(np.isfinite(mag), _FLAT_REL * mag, 0.0)


_TINY = 1e-300  # division guard only; never a threshold


def flat_in_data(actual: dict) -> np.ndarray:
    return actual["std"] <= noise_floor(actual)


def _errors(actual: dict, ref: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-stat, per-dim error in normalization-relevant units.

    Returns (errors, floored_mean): floored_mean carries the UNfloored mean
    sigma-error on dims the constant-dim floor zeroed (0 elsewhere), so the
    caller can surface a would-have-warned dim as one INFO instead of hiding
    the floor's work.
    """
    rng = np.maximum(ref["max"] - ref["min"], actual["max"] - actual["min"])
    floor = np.maximum(noise_floor(actual), _TINY)
    rng_floor = np.maximum(rng, floor)
    sigma = np.maximum.reduce([ref["std"], actual["std"], 1e-3 * rng, floor])
    both_flat = flat_in_data(actual)
    raw_mean = np.abs(actual["mean"] - ref["mean"]) / sigma
    out = {
        # the mean comparison gets the same floor as std: on a dim that does
        # not move in the data, a stored-vs-recomputed mean difference is a
        # DC offset on a constant input - inert - and the sigma floor
        # (1e-3 * range) collapses to the stored std, which on a constant dim
        # is converter f32-accumulation noise (RoboCOIN: stored mean off by
        # 7.5e-6 on a channel whose true std is 0; a naive-f32 running sum
        # reproduces the stored mean to 13 significant digits)
        "mean": np.where(both_flat, 0.0, raw_mean),
        "min": np.abs(actual["min"] - ref["min"]) / rng_floor,
        "max": np.abs(actual["max"] - ref["max"]) / rng_floor,
    }
    out["std"] = np.where(
        # relative to the STORED std, not the noise floor: a stored std 10x
        # too small is a 10x error even when it sits under the floor
        both_flat, 0.0, np.abs(actual["std"] - ref["std"]) / np.maximum(ref["std"], _TINY)
    )
    return out, np.where(both_flat, raw_mean, 0.0)


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
        present = []
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
            present.append(stat)
        if not present:
            return worst
        # compare whatever IS stored: a mean/std-only entry still has a mean
        # and std worth checking. Absent stats borrow the data's own values,
        # which only shapes the shared sigma/range floors and yields no error.
        ref = {**{k: actual[k] for k in ("mean", "std", "min", "max")},
               **{k: ref[k] for k in present}}
        errs, floored_mean = _errors(actual, ref)
        errs = {k: v for k, v in errs.items() if k in present}
        n_floored = int((floored_mean > cfg.stats_mean_warn_sigma).sum())
        if n_floored:
            res.findings.append(Finding(
                self.id, Severity.INFO,
                f"{feat}: {n_floored} near-constant dim(s) whose stored stats disagree "
                "with the data at their own tiny scale - converter f32 noise, inert "
                "for normalization (not scored)",
                details={"feature": feat, "dims": n_floored}))
        for stat, err in errs.items():
            warn, fail = tol[stat]
            err = np.where(np.isfinite(err), err, -np.inf)
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
        # only episodes that actually hold frames: a catalogued episode with no
        # rows (dropped-frames' finding) must not enter the reference, or
        # correct per-episode stats get blamed for the missing data. With such
        # episodes the data is a subset of what stats.json describes.
        loaded = [e for e in ctx.episode_ids() if len(ctx.rows_for(e))]
        subset = ctx.truncated or len(loaded) < len(ctx.episode_ids())
        mode = "full"
        episode_stats_ok = False
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            stored = as_stat_dict(ctx.stats.get(feat))
            if not stored or ("mean" not in stored and "min" not in stored):
                # nothing to compare: absence is deployment-readiness's FAIL
                # (one condition, one severity, charged once)
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.INFO,
                        f"{feat}: no normalization stats (neither meta/stats.json nor "
                        "per-episode stats) - nothing to compare; deployment-readiness "
                        "reports the missing stats",
                    )
                )
                continue
            actual = data_stats(arr)
            if actual is None:
                continue
            source = (ctx.stats_sources or {}).get(feat, ctx.stats_source)

            if not subset:
                worst = max(worst, self._compare(res, feat, actual, stored, source, cfg, "data"))
            else:
                per_ep = [ctx.episode_stats[e][feat] for e in loaded
                          if feat in ctx.episode_stats.get(e, {})]
                sub_ref = aggregate(per_ep) if len(per_ep) == len(loaded) else None
                if sub_ref is not None:
                    mode = "per-episode"
                    before = len(res.findings)
                    worst = max(worst, self._compare(
                        res, feat, actual, sub_ref, "per-episode stats", cfg,
                        f"data ({len(loaded)} loaded episodes with frames)"))
                    episode_stats_ok = not any(
                        f.severity >= Severity.WARN for f in res.findings[before:])
                    if source == "meta/stats.json":
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
                        nf = noise_floor(actual)
                        live = (sd_s > nf) & (sd_a > nf) & (sd_a > 0)
                        ratio = np.where(live, np.maximum(sd_s, sd_a) / np.maximum(np.minimum(sd_s, sd_a), 1e-12), 1.0)
                        shift = np.where(live, np.abs(actual["mean"] - stored["mean"]) / np.maximum(sd_s, 1e-12), 0.0)
                        # ...but only in the direction sampling cannot explain.
                        # A sample NARROWER than the stored stats (smaller std,
                        # or any mean shift) is what a few episodes that missed
                        # the dataset's spread look like - GR00T sim: 3 of 200
                        # episodes hold a joint's excursions; 6-episode samples
                        # saw std 22x-278x under correct stored stats, even
                        # with a few frames spanning the stored range. That
                        # direction is INFO, unless the stored stats are
                        # impossible on their own terms: a std above the stored
                        # range, or a mean outside [min, max] by > 10% of it
                        # (GR00T task-0023: std 2.84 on a stored range of 0.32).
                        # Dims flat in the data are inert and never gate here:
                        # converter f32 noise can push a near-constant dim's
                        # stored std close to its range (GR00T dims 80-83: 0.88x).
                        imp = np.zeros_like(sd_s, dtype=bool)
                        if "min" in stored and "max" in stored and stored["min"].shape == sd_s.shape:
                            srng = stored["max"] - stored["min"]
                            m_s = stored["mean"]
                            with np.errstate(invalid="ignore"):
                                imp = ((sd_s > srng) | (m_s < stored["min"] - 0.1 * srng)
                                       | (m_s > stored["max"] + 0.1 * srng)) & (srng > 0) & ~flat_in_data(actual)
                        for kind, garr, lim, fmt in (("std", ratio, cfg.stats_sample_std_ratio, "{:.1f}x"),
                                                    ("mean", shift, cfg.stats_sample_mean_sigma, "{:.1f} std devs")):
                            # excused dims are judged apart: the worst EXCUSED
                            # dim must not hide a gated one behind it
                            narrow = ((kind == "mean") | (sd_a < sd_s)) & ~imp
                            for excused in (True, False):
                                masked = np.where(narrow == excused, garr, -np.inf)
                                d = int(np.argmax(masked))
                                if not masked[d] > lim:
                                    continue
                                if excused:
                                    res.findings.append(Finding(
                                        self.id, Severity.INFO,
                                        f"{feat}.{kind}: the sampled data is off the stored stats by "
                                        f"{fmt.format(float(garr[d]))} on dim {d}, in the direction a "
                                        f"{len(loaded)}-episode sample that missed part of the dataset's "
                                        "spread produces - not evidence the stored stats are wrong "
                                        "(load the full dataset to judge)" + _reach_note(stored, actual, d),
                                        details={"feature": feat, "stat": kind, "dim": d,
                                                 "error": float(garr[d]), "reference": "sampled",
                                                 "sampling_explains": True},
                                    ))
                                    continue
                                worst = max(worst, float(garr[d]) / lim)
                                why = (" - and the stored stats are impossible on their own terms "
                                       f"(stored std {sd_s[d]:.4g}, mean {stored['mean'][d]:.4g}, "
                                       f"range [{stored['min'][d]:.4g}, {stored['max'][d]:.4g}])"
                                       if imp[d] else " - far more than sampling explains")
                                res.findings.append(Finding(
                                    self.id, Severity.WARN,
                                    f"{feat}.{kind}: the sampled data is off the stored stats by "
                                    f"{fmt.format(float(garr[d]))} on dim {d}{why}; the stored stats "
                                    "look wrong (sampled audit: load the full dataset to confirm)",
                                    details={"feature": feat, "stat": kind, "dim": d,
                                             "error": float(garr[d]), "reference": "sampled",
                                             "stored_inconsistent": bool(imp[d])},
                                ))
                    if "min" in stored and "max" in stored and stored["min"].shape == actual["min"].shape:
                        rng = np.maximum(stored["max"] - stored["min"],
                                         np.maximum(noise_floor(actual), _TINY))
                        over = np.maximum(stored["min"] - actual["min"], actual["max"] - stored["max"]) / rng
                        over = np.where(np.isfinite(over), over, -np.inf)
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
                        # outside the interval is not enough on its own (tiny
                        # gaps over many dims tested at once fire on clean data);
                        # the stored value must ALSO sit beyond the interval by
                        # at least the full-load WARN tolerance, in the same
                        # units. Dims constant in the data are inert (f32 noise).
                        live_d = ~flat_in_data(actual)
                        sd_norm = np.maximum(np.maximum(stored["std"], actual["std"]), 1e-12)
                        cov_b = np.ones_like(stored["std"], dtype=float)
                        if "min" in stored and "max" in stored and stored["min"].shape == actual["min"].shape:
                            with np.errstate(invalid="ignore", divide="ignore"):
                                cb = ((actual["max"] - actual["min"])
                                      / np.maximum(stored["max"] - stored["min"], 1e-12))
                            cov_b = np.where(np.isfinite(cb), cb, 1.0)
                        for kind, st, lo, hi, warn_tol, fail_tol, denom in (
                                ("mean", stored["mean"], m_lo, m_hi, cfg.stats_mean_warn_sigma,
                                 cfg.stats_mean_fail_sigma, sd_norm),
                                ("std", stored["std"], s_lo, s_hi, cfg.stats_std_warn,
                                 cfg.stats_std_fail, np.maximum(stored["std"], 1e-12))):
                            gap = np.maximum(np.maximum(lo - st, st - hi), 0.0) / denom
                            gap = np.where(live_d & np.isfinite(gap), gap, 0.0)
                            # a stored value ABOVE the interval (std) or anywhere
                            # outside it (mean) is what a sample that missed the
                            # dataset's spread produces: excused, like the gross
                            # backstop, when the sample spans too little of the
                            # stored range (kuka: 40 episodes spanning 12% of a
                            # dim; correct stored std 0.00227 above [0.0010,
                            # 0.0020]; 300 episodes measure 0.0024)
                            excused = (cov_b < _SAMPLE_MIN_COVER) & ((kind == "mean") | (st > hi))
                            ex_gap = np.where(excused, gap, 0.0)
                            gap = np.where(excused, 0.0, gap)
                            de = int(np.argmax(ex_gap))
                            if ex_gap[de] >= warn_tol and ex_gap[de] > gap.max():
                                res.findings.append(Finding(
                                    self.id, Severity.INFO,
                                    f"{feat}.{kind}: the stored {kind} sits outside the loaded "
                                    f"sample's bootstrap interval on dim {de} (stored {st[de]:.6g}, "
                                    f"interval [{lo[de]:.6g}, {hi[de]:.6g}] over {E} episodes), but "
                                    f"the sample spans only {cov_b[de]:.1%} of the stored [min, max] - "
                                    "episodes it did not load can explain the gap (load the full "
                                    "dataset to judge)",
                                    details={"feature": feat, "stat": kind, "dim": de,
                                             "reference": "bootstrap", "episodes": E,
                                             "stored": float(st[de]), "gap": float(ex_gap[de]),
                                             "range_coverage": float(cov_b[de])},
                                ))
                            d = int(np.argmax(gap))
                            if gap[d] <= 0:
                                continue
                            material = gap[d] >= warn_tol
                            if material:
                                worst = max(worst, float(gap[d]) / fail_tol)
                            res.findings.append(Finding(
                                self.id, Severity.WARN if material else Severity.INFO,
                                f"{feat}.{kind}: the stored {kind} sits outside the loaded "
                                f"sample's {100 * (1 - cfg.stats_sample_alpha):.1f}% bootstrap "
                                f"interval on dim {d} (stored {st[d]:.6g}, interval "
                                f"[{lo[d]:.6g}, {hi[d]:.6g}] over {E} episodes"
                                + (f", {_fmt(kind, float(gap[d]))} beyond it) - the stored stats "
                                   "look wrong; a sampled audit cannot prove it, load the full "
                                   "dataset to confirm"
                                   if material else
                                   ") - a gap below the WARN tolerance: descriptive only"),
                                details={"feature": feat, "stat": kind, "dim": d,
                                         "reference": "bootstrap", "episodes": E,
                                         "stored": float(st[d]), "gap": float(gap[d]),
                                         "interval": [float(lo[d]), float(hi[d])]},
                            ))

            # stats.json vs the per-episode stats it should be aggregated from:
            # conclusive on any download, since meta is always complete
            if feat in ctx.stats_json and source == "meta/stats.json":
                all_ids = [e for e in ctx.episode_stats]
                per_all = [ctx.episode_stats[e][feat] for e in all_ids if feat in ctx.episode_stats[e]]
                agg_all = aggregate(per_all) if per_all else None
                js = as_stat_dict(ctx.stats_json.get(feat))
                # only conclusive when the per-episode stats themselves agree
                # with the loaded data; otherwise they are the broken side
                if (agg_all is not None and js and len(per_all) >= len(ctx.episode_stats)
                        and subset and mode == "per-episode" and episode_stats_ok):
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
            ref = {"full": "the stored stats", "per-episode": "per-episode stats of the loaded episodes",
                   "bootstrap": "the sample's bootstrap interval and stored range",
                   "containment": "stored range"}[mode]
            res.summary = f"data matches {ref} (worst {worst:.0%} of the FAIL threshold)"
        else:
            n_gated = sum(1 for f in res.findings if f.severity >= Severity.WARN)
            res.summary = (f"worst error {worst:.1f}x the FAIL threshold" if worst > 0
                           else f"{n_gated} gated finding(s) without a threshold ratio")
        return res

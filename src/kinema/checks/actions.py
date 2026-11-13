"""Checks 7 & 8 - action saturation and NaN/normalization hazards."""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY


class ActionSaturationCheck(Check):
    id = "action-saturation"
    title = "Action saturation (commands clipped at the limits)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        entry = ctx.stats.get(ACTION_KEY) or {}
        if "min" in entry and "max" in entry:
            lo = np.asarray(entry["min"], dtype=np.float64).ravel()
            hi = np.asarray(entry["max"], dtype=np.float64).ravel()
        else:
            lo, hi = action.min(axis=0), action.max(axis=0)
        rng = np.maximum(1e-8, hi - lo)
        delta = 0.01 * rng
        saturated = (np.abs(action - lo) < delta) | (np.abs(action - hi) < delta)
        frac = saturated.mean(axis=0)
        for j, f in enumerate(frac):
            if f > cfg.saturation_warn_fraction:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"action dim {j}: {f:.1%} of commands sit at the min/max limit - "
                        "clipped commands the policy will imitate",
                        details={"dim": j, "saturated_fraction": float(f)},
                    )
                )
        if not res.findings:
            res.summary = "no action dim saturated above 5%"
        else:
            res.summary = f"{len(res.findings)} dim(s) saturated"
        return res


class NanAndNormalizationCheck(Check):
    id = "nan-and-normalization"
    title = "NaN / Inf and normalization hazards"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            n_nan = int(np.isnan(arr).sum())
            n_inf = int(np.isinf(arr).sum())
            if n_nan or n_inf:
                bad_eps = sorted(
                    {
                        int(ctx.episode_index[r])
                        for r in np.nonzero(~np.isfinite(arr).all(axis=1))[0]
                    }  # noqa: C401
                )
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{feat}: {n_nan} NaN and {n_inf} Inf values "
                        f"(episodes {bad_eps[:10]}{'...' if len(bad_eps) > 10 else ''}) - "
                        "one NaN poisons the loss for the whole batch",
                        details={"nan": n_nan, "inf": n_inf, "episodes": bad_eps},
                    )
                )
            std = arr.std(axis=0)
            zero = np.nonzero(std == 0)[0]
            tiny = np.nonzero((std > 0) & (std < 1e-6))[0]
            for j in zero:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{feat} dim {j}: zero variance - std=0 breaks mean/std "
                        "normalization (division by zero) and the dim carries no signal",
                        details={"feature": feat, "dim": int(j)},
                    )
                )
            for j in tiny:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{feat} dim {j}: std {std[j]:.2e} - near-zero variance amplifies "
                        "noise after normalization",
                        details={"feature": feat, "dim": int(j), "std": float(std[j])},
                    )
                )
        if not res.findings:
            res.summary = "no NaN/Inf; all dims have usable variance"
        return res

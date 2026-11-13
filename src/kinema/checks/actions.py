"""Checks 7 & 8 - action saturation and NaN/normalization hazards."""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY
from kinema.stats import gripper_dims


class ActionSaturationCheck(Check):
    id = "action-saturation"
    title = "Action saturation (commands clipped at the limits)"

    # Clipping leaves a pile-up of commands at EXACTLY the limit value while
    # the robot is moving. Merely sitting near an extreme is normal (rest
    # poses, grippers fully open/shut), as are binary or constant dims.

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        action = ctx.action
        if action is None:
            return self.skip("action column missing")
        n = action.shape[1]
        grip = gripper_dims(ctx.action_names, n)
        finite = np.isfinite(action).all(axis=1)
        a = action[finite]
        if len(a) < 20:
            return self.skip("too few finite action frames")
        lo, hi = a.min(axis=0), a.max(axis=0)
        rng = hi - lo
        tol = 1e-6 * np.maximum(rng, 1e-12)

        # moving frames: some dim changed since the previous frame of the
        # same episode (idle stretches are idle-segments' job)
        eps = np.maximum(1e-8, cfg.idle_eps_frac * a.std(axis=0))
        moving = np.zeros(len(action), dtype=bool)
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 2:
                continue
            d = np.abs(np.diff(action[rows], axis=0)) >= eps
            moving[rows[1:]] = d.any(axis=1)
        moving = moving[finite]

        checked = 0
        for j in range(n):
            if j in grip or rng[j] <= 1e-8 or len(np.unique(a[:, j])) <= 10:
                continue  # gripper, constant or discrete-valued dim
            if moving.sum() < 20:
                continue
            checked += 1
            at_hi = np.abs(a[:, j] - hi[j]) <= tol[j]
            at_lo = np.abs(a[:, j] - lo[j]) <= tol[j]
            f_hi = float((at_hi & moving).sum() / moving.sum())
            f_lo = float((at_lo & moving).sum() / moving.sum())
            f = max(f_hi, f_lo)
            if f > cfg.saturation_warn_fraction:
                side = "max" if f_hi >= f_lo else "min"
                name = f" ({ctx.action_names[j]})" if ctx.action_names and len(ctx.action_names) == n else ""
                if f > 0.5:
                    # most of the data AT the limit is a mode, not a clipped tail:
                    # an unnamed gripper / binary-like command or a rest state
                    res.findings.append(Finding(
                        self.id, Severity.INFO,
                        f"action dim {j}{name}: {f:.0%} of commands sit exactly at its {side} - "
                        "a gripper or on/off command rather than clipping?",
                        details={"dim": j, "saturated_fraction": f, "side": side},
                    ))
                    continue
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"action dim {j}{name}: {f:.1%} of commands during motion are pinned at "
                        f"exactly its {side} ({(hi if side == 'max' else lo)[j]:.6g}) - clipped "
                        "commands the policy will imitate",
                        details={"dim": j, "saturated_fraction": f, "side": side},
                    )
                )
        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = (
                f"no continuous action dim pinned at a limit above "
                f"{cfg.saturation_warn_fraction:.0%} ({checked} dims checked)"
            )
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
                        Severity.INFO,
                        f"{feat} dim {j}: constant (std=0) - carries no signal; if it ever "
                        "varies at deployment, mean/std normalization (std + 1e-8) blows "
                        "it up. Usually padding; consider dropping it",
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

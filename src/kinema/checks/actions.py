"""Checks 7 & 8 - action saturation and NaN/normalization hazards."""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY
from kinema.stats import action_gripper_dims, is_generic, toggle_dims, velocity_dims


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
        named_grip, inferred_grip = action_gripper_dims(ctx)
        grip = named_grip  # inferred grippers are checked but only reported at INFO
        # on/off commands beyond the gripper slots (odd layouts the slot prior
        # misses) get the same downgrade: piling up at an extreme is their
        # normal operation, not clipping
        toggles = toggle_dims(ctx, exclude=named_grip | inferred_grip
                              | velocity_dims(ctx.action_names, n))
        # finite values are judged per dim: one all-NaN padding dim must not
        # discard every row (it used to skip the whole check)
        fin = np.isfinite(action)
        if fin.any(axis=0).sum() == 0 or fin.sum(axis=0).max() < 20:
            return self.skip("too few finite action frames")
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanmin(np.where(fin, action, np.nan), axis=0), np.nanmax(np.where(fin, action, np.nan), axis=0)
            sd = np.nanstd(np.where(fin, action, np.nan), axis=0)
        rng = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
        tol = 1e-6 * np.maximum(rng, 1e-12)

        # moving frames: some dim changed since the previous frame of the
        # same episode (idle stretches are idle-segments' job)
        eps = np.maximum(1e-8, cfg.idle_eps_frac * np.where(np.isfinite(sd), sd, 0.0))
        moving = np.zeros(len(action), dtype=bool)
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 2:
                continue
            step = np.abs(np.diff(action[rows], axis=0))
            moving[rows[1:]] = (np.isfinite(step) & (step >= eps)).any(axis=1)
        names_ok = bool(ctx.action_names) and len(ctx.action_names) == n

        checked = 0
        for j in range(n):
            col = action[:, j]
            ok = fin[:, j]
            x = col[ok]
            if j in grip or rng[j] <= 1e-8 or len(x) < 20 or len(np.unique(x)) <= 10:
                continue  # gripper, constant, mostly-missing or discrete-valued dim
            mv = moving & ok
            if mv.sum() < 20:
                continue
            checked += 1
            at_hi = ok & (np.abs(col - hi[j]) <= tol[j])
            at_lo = ok & (np.abs(col - lo[j]) <= tol[j])
            f_hi = float((at_hi & mv).sum() / mv.sum())
            f_lo = float((at_lo & mv).sum() / mv.sum())
            f = max(f_hi, f_lo)
            if f > cfg.saturation_warn_fraction:
                side = "max" if f_hi >= f_lo else "min"
                name = f" ({ctx.action_names[j]})" if names_ok else ""
                # most of the data AT the limit reads as a mode (an unnamed
                # gripper, an on/off command, a rest pose) - but only when the
                # dim could be one. A dim NAMED as a joint pinned 65% of the
                # time is worse clipping than 25%, never better.
                named_joint = names_ok and not is_generic([ctx.action_names[j]])
                if f > 0.5 and not named_joint:
                    # most of the data AT the limit is a mode, not a clipped tail:
                    # an unnamed gripper / binary-like command or a rest state
                    res.findings.append(Finding(
                        self.id, Severity.INFO,
                        f"action dim {j}{name}: {f:.0%} of commands sit exactly at its {side} - "
                        "a gripper or on/off command rather than clipping?",
                        details={"dim": j, "saturated_fraction": f, "side": side},
                    ))
                    continue
                likely_grip = j in inferred_grip or j in toggles
                note = (" [dim looks like an unnamed gripper, so not gated]" if j in inferred_grip
                        else " [two-level on/off command, so not gated]" if j in toggles else "")
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.INFO if likely_grip else Severity.WARN,
                        f"action dim {j}{name}: {f:.1%} of commands during motion are pinned at "
                        f"exactly its {side} ({(hi if side == 'max' else lo)[j]:.6g}) - clipped "
                        "commands the policy will imitate" + note,
                        details={"dim": j, "saturated_fraction": f, "side": side},
                    )
                )
        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = (
                f"no continuous action dim pinned at a limit above "
                f"{cfg.saturation_warn_fraction:.0%} ({checked} dims checked)"
            )
        else:
            n_gated = sum(1 for f in res.findings if f.severity >= Severity.WARN)
            res.summary = f"{n_gated} dim(s) saturated" + (
                f" (+{len(res.findings) - n_gated} informational)" if len(res.findings) > n_gated else "")
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
            with np.errstate(invalid="ignore"):
                std = np.nanstd(np.where(np.isfinite(arr), arr, np.nan), axis=0)
                mean = np.nanmean(np.where(np.isfinite(arr), arr, np.nan), axis=0)
            std = np.where(np.isfinite(std), std, 0.0)
            mean = np.where(np.isfinite(mean), mean, 0.0)
            zero = np.nonzero(std == 0)[0]
            # "near-zero" relative to the dim's own magnitude and to the
            # feature's typical spread - not an absolute 1e-6, which flagged
            # every dim of data simply recorded in small units
            live = std[std > 0]
            typical = float(np.median(live)) if len(live) else 0.0
            ref_scale = np.maximum(np.abs(mean), typical)
            tiny = np.nonzero((std > 0) & (std < 1e-6 * ref_scale))[0]
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
        else:
            n_bad = sum(1 for f in res.findings if f.severity == Severity.FAIL)
            n_tiny = sum(1 for f in res.findings if f.severity == Severity.WARN)
            n_const = sum(1 for f in res.findings if f.severity == Severity.INFO)
            res.summary = ", ".join(x for x in (
                f"{n_bad} feature(s) with NaN/Inf" if n_bad else "",
                f"{n_tiny} near-constant dim(s)" if n_tiny else "",
                f"{n_const} constant dim(s)" if n_const else "") if x)
        return res

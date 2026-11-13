"""Check 12 - deployment readiness.

The "low loss, failed robot" check. Training can succeed while the
dataset is undeployable: episodes shorter than the policy's action
chunk, stats that cannot produce the normalization the policy expects,
or a state/action layout the policy does not assume. Grounded in the
openpi #906 failure mode (normalization mismatch between training and
inference is the community's top diagnosis).
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY

# Documented default action horizons / chunk sizes. Conservative: used
# only to flag episodes too short to fill one chunk.
POLICY_DEFAULTS = {
    "act": {"chunk_size": 100, "quantile_norm": False},
    "diffusion": {"chunk_size": 16, "quantile_norm": False},
    "smolvla": {"chunk_size": 50, "quantile_norm": False},
    "pi0": {"chunk_size": 50, "quantile_norm": True},
    "pi0_fast": {"chunk_size": 50, "quantile_norm": True},
}


class DeploymentReadinessCheck(Check):
    id = "deployment-readiness"
    title = "Deployment readiness (would this dataset serve a policy?)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        policy = POLICY_DEFAULTS.get(cfg.policy.lower()) if cfg.policy else None
        if cfg.policy and policy is None:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    f"unknown policy {cfg.policy!r}; running policy-agnostic checks "
                    f"(known: {', '.join(sorted(POLICY_DEFAULTS))})",
                )
            )

        # 1. stats coverage: training consumes observation.state + action
        for feat in (STATE_KEY, ACTION_KEY):
            if feat not in ctx.stats:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{feat}: absent from meta/stats.json - the policy cannot "
                        "normalize inputs/outputs at inference",
                    )
                )

        # 2. quantile normalization support (openpi normalizes with q01/q99)
        if policy and policy["quantile_norm"]:
            for feat in (STATE_KEY, ACTION_KEY):
                entry = ctx.stats.get(feat) or {}
                if entry and not ("q01" in entry and "q99" in entry):
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"{cfg.policy} normalizes with q01/q99 quantiles but stats.json "
                            f"has none for {feat} - recompute stats with quantiles or the "
                            "openpi normalizer will estimate them itself",
                        )
                    )

        # 3. episode length vs action chunk
        if policy:
            chunk = policy["chunk_size"]
            short = [e.index for e in ctx.episodes if 0 < e.length < chunk]
            if short:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{len(short)}/{ctx.total_episodes} episodes are shorter than "
                        f"{cfg.policy}'s action chunk ({chunk} frames): {short[:10]}"
                        f"{'...' if len(short) > 10 else ''} - they train on padded or "
                        "truncated chunks",
                        details={"episodes": short, "chunk_size": chunk},
                    )
                )

        # 4. dims: info.json feature shapes vs the actual arrays
        feats = ctx.info.get("features") or {}
        for key, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            declared = (feats.get(key) or {}).get("shape")
            if declared and list(declared) != [arr.shape[1]]:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"{key}: info.json declares shape {declared} but the data has "
                        f"{arr.shape[1]} dims",
                    )
                )

        # 5. would inference see OOD inputs? normalize the data with the
        #    STORED stats (what a deployed policy uses) and look past 3 sigma
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            entry = ctx.stats.get(feat) or {}
            if "mean" not in entry or "std" not in entry:
                continue
            mean = np.asarray(entry["mean"], dtype=np.float64).ravel()
            std = np.asarray(entry["std"], dtype=np.float64).ravel()
            if mean.shape[0] != arr.shape[1]:
                continue
            safe = np.where(std > 1e-8, std, 1.0)
            z = np.abs((arr - mean) / safe)
            frac = float((z > 3).mean())
            if frac > 0.001:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{feat}: {frac:.2%} of values sit >3 sigma from the stored "
                        "normalization stats - at inference the policy sees inputs "
                        "outside the range it trained on (the openpi #906 failure mode)",
                        details={"feature": feat, "ood_fraction": frac},
                    )
                )

        # 6. delta_timestamps compatibility
        if ctx.fps:
            t = ctx.timestamp
            grid_err = float(np.abs(t - np.round(t * ctx.fps) / ctx.fps).max()) if len(t) else 0.0
            if grid_err > 0.5 / ctx.fps:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"timestamps up to {grid_err * 1000:.1f} ms off-grid - "
                        "delta_timestamps window sampling will misalign frames",
                    )
                )

        if not res.findings:
            res.summary = (
                f"ready for {cfg.policy}" if cfg.policy else "no deployment blockers found"
            )
        else:
            n_fail = sum(1 for f in res.findings if f.severity == Severity.FAIL)
            n_warn = sum(1 for f in res.findings if f.severity == Severity.WARN)
            res.summary = f"{n_fail} blocker(s), {n_warn} warning(s)"
            if cfg.policy:
                res.summary += f" for {cfg.policy}"
        return res

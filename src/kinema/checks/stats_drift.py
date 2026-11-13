"""Check 1 - stats-drift.

Every training run normalizes with meta/stats.json. If those numbers
drifted from the data (dataset edited after stats were computed, stats
copied from another dataset, unit change), normalization silently feeds
the policy wrong-scaled inputs. Recompute and compare.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY


class StatsDriftCheck(Check):
    id = "stats-drift"
    title = "Stats drift (meta/stats.json vs actual data)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        worst = 0.0
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None:
                continue
            entry = ctx.stats.get(feat)
            if not isinstance(entry, dict) or "mean" not in entry:
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"{feat}: no entry in meta/stats.json - "
                        "policies that normalize will fail or use wrong stats",
                    )
                )
                continue
            finite_rows = np.isfinite(arr).all(axis=1)
            data = arr[finite_rows]
            if len(data) == 0:
                continue
            actual = {
                "mean": data.mean(axis=0),
                "std": data.std(axis=0),
                "min": data.min(axis=0),
                "max": data.max(axis=0),
            }
            for stat_name, a in actual.items():
                stored = entry.get(stat_name)
                if stored is None:
                    res.findings.append(
                        Finding(self.id, Severity.WARN, f"{feat}: stats.json is missing '{stat_name}'")
                    )
                    continue
                b = np.asarray(stored, dtype=np.float64).ravel()
                if b.shape != a.shape:
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.FAIL,
                            f"{feat}.{stat_name}: shape {b.shape} in stats.json "
                            f"vs {a.shape} in data",
                        )
                    )
                    continue
                denom = np.abs(b)
                # near-zero stored values make relative error meaningless; skip
                # (zero-variance dims are check 8's job)
                ok = denom > 1e-8
                if not ok.any():
                    continue
                rel = np.abs(a[ok] - b[ok]) / denom[ok]
                max_rel = float(rel.max())
                dim = int(rel.argmax())
                worst = max(worst, max_rel)
                if max_rel > cfg.stats_fail_tol:
                    sev = Severity.FAIL
                elif max_rel > cfg.stats_rel_tol:
                    sev = Severity.WARN
                else:
                    continue
                res.findings.append(
                    Finding(
                        self.id,
                        sev,
                        f"{feat}.{stat_name}: off by {max_rel:.1%} on dim {dim} "
                        f"(stored {b[ok][dim]:.6g}, actual {a[ok][dim]:.6g}). "
                        "Normalization at training and inference will disagree.",
                        details={"feature": feat, "stat": stat_name, "dim": dim, "rel_err": max_rel},
                    )
                )
        if not res.findings:
            res.summary = "stats.json matches the data (mean/std/min/max within 1%)"
        else:
            res.summary = f"worst relative error {worst:.1%}"
        return res

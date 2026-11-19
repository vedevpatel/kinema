"""Check 19 - stats f32 accumulation fingerprint.

Some converters compute per-episode stats with naive float32 accumulation:
a running f32 sum for the mean, then a running f32 sum of squared
deviations for the std. On long episodes the rounding error is
deterministic, so the stored stats reproduce a naive f32 recomputation
bitwise while disagreeing with the float64 truth. On a constant channel
that fabricates a nonzero std and DC-offsets the mean (RoboCOIN: stored
std 7.5e-6 on a channel whose true std is 0, stored mean reproduced to 13
significant digits by a naive f32 running sum).

The defect is provenance, not drift: the data is fine, the stored stats
are accumulation-rounded. Training that normalizes from them feeds the
policy the converter's rounding error as if it were data statistics. This
check fires INFO only; stats-drift owns material disagreement, and its
constant-dim floor already covers the normalization consequence.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import ACTION_KEY, STATE_KEY

_SAMPLE_CAP = 5  # episodes per feature; one extra pass each, no new downloads
_MIN_FRAMES = 30  # accumulation error needs length to diverge from exact


def _naive_f32_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(mean, std) as a naive float32 converter computes them: running f32
    sum / n, then running f32 sum of squared deviations / n. np.cumsum in
    float32 reproduces sequential f32 accumulation bitwise (verified against
    the RoboCOIN fingerprint and random data)."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    n = np.float32(x32.shape[0])
    mean = np.asarray(np.cumsum(x32, axis=0, dtype=np.float32)[-1] / n, dtype=np.float32)
    dev = np.asarray(x32 - mean, dtype=np.float32)
    var = np.asarray(
        np.cumsum(np.asarray(dev * dev, dtype=np.float32), axis=0, dtype=np.float32)[-1] / n,
        dtype=np.float32)
    return mean.astype(np.float64), np.sqrt(var).astype(np.float64)


class StatsF32FingerprintCheck(Check):
    id = "stats-f32-fingerprint"
    title = "Stats f32 accumulation fingerprint"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(check_id=self.id, title=self.title)
        if not ctx.episode_stats:
            return self.skip("no per-episode stats")
        loaded = [e for e in ctx.episode_ids() if len(ctx.rows_for(e))]
        if not loaded:
            return self.skip("no episodes with frames loaded")
        picks = [loaded[i] for i in np.linspace(0, len(loaded) - 1,
                                                min(_SAMPLE_CAP, len(loaded))).astype(int)]
        picked: dict[str, list[int]] = {}
        dims: dict[str, list[int]] = {}
        for feat, arr in ((STATE_KEY, ctx.state), (ACTION_KEY, ctx.action)):
            if arr is None or not len(arr) or arr.dtype.kind != "f":
                continue
            hit_eps: list[int] = []
            hit_dims: set[int] = set()
            for ep in picks:
                stored = ctx.episode_stats.get(ep, {}).get(feat)
                if not stored or "mean" not in stored or "std" not in stored:
                    continue
                rows = ctx.rows_for(ep)
                if len(rows) < _MIN_FRAMES:
                    continue
                x = arr[rows]
                if not np.isfinite(x).all():
                    continue
                naive_mean, naive_std = _naive_f32_stats(x)
                exact_mean = x.mean(axis=0)
                stored_mean = np.asarray(stored["mean"], dtype=np.float64)
                stored_std = np.asarray(stored["std"], dtype=np.float64)
                if stored_mean.shape != naive_mean.shape or stored_std.shape != naive_std.shape:
                    continue
                material = np.abs(stored_mean - exact_mean) > np.maximum(
                    1e-9 * np.abs(exact_mean), 1e-12)
                match = (stored_mean == naive_mean) & (stored_std == naive_std)
                hit = match & material
                if hit.any():
                    hit_eps.append(ep)
                    hit_dims.update(np.flatnonzero(hit).tolist())
            if hit_eps:
                picked[feat] = hit_eps
                dims[feat] = sorted(hit_dims)
        total = sum(len(v) for v in picked.values())
        needed = 1 if len(picks) < 2 else 2
        if total >= needed:
            which = ", ".join(
                f"{feat} dims {ds[:5]}{'...' if len(ds) > 5 else ''}"
                for feat, ds in dims.items())
            res.findings.append(Finding(
                self.id, Severity.INFO,
                f"per-episode stats reproduce naive float32 accumulation "
                f"bitwise ({total} sampled episode(s), {which}): the "
                "converter's stats are accumulation-rounded, not data-exact; "
                "constant channels get a fabricated std and a DC-offset mean",
                details={"episodes": {f: e for f, e in picked.items()},
                         "dims": {f: d for f, d in dims.items()}}))
            res.summary = f"per-episode stats carry naive float32 accumulation ({total} episode(s))"
        else:
            res.summary = (f"no float32-accumulation fingerprint in {len(picks)} sampled "
                           "episode(s)")
        return res

"""Checks 2 & 3 - timestamp grid and dropped frames.

Hand-rolled dataset conversions routinely produce timestamps that sit
off the 1/fps grid, non-monotonic runs, and frame_index gaps. Any
loader that samples with delta_timestamps then returns misaligned
windows, and some crash outright.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity


class TimestampGridCheck(Check):
    id = "timestamp-grid"
    title = "Timestamp grid (on-grid, monotonic, low jitter)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if not ctx.fps:
            return self.skip("fps missing from meta/info.json")
        fps = ctx.fps
        worst_frac = 0.0
        worst_ep = None
        flagged = 0
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            t = ctx.timestamp[rows]
            grid = np.round(t * fps) / fps
            off = np.abs(t - grid)
            max_off = float(off.max())
            frac = max_off * fps  # as a fraction of the frame period
            d = np.diff(t)
            nonmono = int((d <= 0).sum())
            if frac > worst_frac:
                worst_frac, worst_ep = frac, ep
            if nonmono:
                flagged += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"episode {ep}: {nonmono} non-monotonic timestamp(s) "
                        "(equal or decreasing) - temporal window sampling breaks here",
                        episode_index=ep,
                    )
                )
            elif frac > cfg.grid_warn_frac:
                flagged += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"episode {ep}: timestamps up to {max_off * 1000:.1f} ms off the "
                        f"{fps:g} fps grid ({frac:.0%} of the frame period); a loader needs "
                        f"tolerance_s >= {max_off:.4f}",
                        episode_index=ep,
                    )
                )
        if not res.findings:
            res.summary = f"all episodes on the {fps:g} fps grid and monotonic"
        else:
            res.summary = (
                f"{flagged}/{ctx.total_episodes} episodes flagged; "
                f"worst off-grid {worst_frac:.0%} of frame period (episode {worst_ep})"
            )
        return res


class DroppedFramesCheck(Check):
    id = "dropped-frames"
    title = "Dropped / duplicated frames"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        n_gap = n_meta_mismatch = n_dup = 0
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            fi = ctx.frame_index[rows]
            meta = ctx.meta_for(ep)
            if len(fi) > 1:
                gaps = int((np.diff(fi) > 1).sum())
                dups = int((np.diff(fi) == 0).sum())
            else:
                gaps = dups = 0
            if gaps:
                n_gap += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"episode {ep}: {gaps} gap(s) in frame_index - dropped frames",
                        episode_index=ep,
                    )
                )
            if dups:
                n_dup += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"episode {ep}: {dups} duplicated frame_index value(s)",
                        episode_index=ep,
                    )
                )
            if meta is not None and meta.length and meta.length != len(rows):
                n_meta_mismatch += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.FAIL,
                        f"episode {ep}: metadata says {meta.length} frames but the data "
                        f"holds {len(rows)} rows - episode metadata disagrees with the data",
                        episode_index=ep,
                        details={"meta_length": meta.length, "rows": int(len(rows))},
                    )
                )
        if cfg.video:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    "video frame-count cross-check not implemented in v0 "
                    "(parquet side verified)",
                )
            )
        else:
            res.findings.append(
                Finding(
                    self.id,
                    Severity.INFO,
                    "video frame counts not checked (pass --video to include MP4s)",
                )
            )
        if n_gap == n_meta_mismatch == n_dup == 0:
            res.summary = "frame_index contiguous and episode lengths match the data"
        else:
            res.summary = (
                f"{n_gap} episodes with gaps, {n_dup} with duplicate indices, "
                f"{n_meta_mismatch} with metadata length mismatches"
            )
        return res

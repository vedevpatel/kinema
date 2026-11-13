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
    title = "Timestamp grid (spacing, monotonic, start at 0)"

    # Measured on frame-to-frame spacing, not distance from an absolute
    # grid: a constant offset leaves spacing intact, while any jitter makes
    # the absolute distance saturate at half a period and say nothing.
    # LeRobot compares spacing against 1/fps within tolerance_s (default
    # 1e-4 s), and looks video frames up BY timestamp - so timestamps that
    # do not start at 0 (e.g. wall-clock seconds) break video decoding.

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if not ctx.fps:
            return self.skip("fps missing from meta/info.json")
        fps = ctx.fps
        period = 1.0 / fps
        has_video = any(
            isinstance(v, dict) and v.get("dtype") == "video"
            for v in (ctx.info.get("features") or {}).values()
        )
        worst_jit, worst_ep = 0.0, None
        flagged = 0
        offset_eps: list[tuple[int, float]] = []
        for ep in ctx.episode_ids():
            rows = ctx.rows_for(ep)
            if len(rows) < 3:
                continue
            t = ctx.timestamp[rows]
            d = np.diff(t)
            nonmono = int((d <= 0).sum())
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
                continue
            if abs(t[0]) > 0.5 * period:
                offset_eps.append((ep, float(t[0])))
            jit = float(np.abs(d - period).max()) * fps  # fraction of a period
            gaps = int((d > 1.5 * period).sum())
            if jit > worst_jit:
                worst_jit, worst_ep = jit, ep
            if jit > cfg.grid_warn_frac:
                flagged += 1
                res.findings.append(
                    Finding(
                        self.id,
                        Severity.WARN,
                        f"episode {ep}: frame spacing deviates up to {jit:.0%} of the "
                        f"{fps:g} fps period"
                        + (f", {gaps} dropped-frame gap(s)" if gaps else "")
                        + f"; LeRobot's default tolerance_s (1e-4 s) rejects this - "
                        f"needs tolerance_s >= {jit * period:.4f}",
                        episode_index=ep,
                        details={"jitter_fraction": jit, "gaps": gaps},
                    )
                )
        if offset_eps:
            ex = ", ".join(f"{e} ({t0:.4g}s)" for e, t0 in offset_eps[:5])
            res.findings.append(
                Finding(
                    self.id,
                    Severity.FAIL if has_video else Severity.INFO,
                    f"{len(offset_eps)} episode(s) whose timestamps do not start at 0: {ex}"
                    + (" - video frames are looked up by timestamp, so decoding will miss"
                       if has_video else " - harmless without videos (spacing is what matters)"),
                    details={"episodes": [e for e, _ in offset_eps]},
                )
            )
        if not any(f.severity >= Severity.WARN for f in res.findings):
            res.summary = f"evenly spaced at {fps:g} fps and monotonic"
        else:
            res.summary = (
                f"{flagged}/{ctx.total_episodes} episodes flagged; worst spacing deviation "
                f"{worst_jit:.0%} of a period (episode {worst_ep})"
                + (f"; {len(offset_eps)} not starting at 0" if offset_eps else "")
            )
        return res


class DroppedFramesCheck(Check):
    id = "dropped-frames"
    title = "Dropped / duplicated frames"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        n_gap = n_meta_mismatch = n_dup = 0
        if ctx.unreadable_files:
            ex = "; ".join(f"{name} ({err})" for name, err in ctx.unreadable_files[:3])
            res.findings.append(
                Finding(
                    self.id,
                    Severity.FAIL,
                    f"{len(ctx.unreadable_files)} data file(s) cannot be read: {ex} - "
                    "loading this dataset will crash; their episodes were not audited",
                    details={"files": [n for n, _ in ctx.unreadable_files]},
                )
            )
        if ctx.duplicate_episode_ids:
            dup = ctx.duplicate_episode_ids
            ex = ", ".join(f"{e} (x{n})" for e, n in list(dup.items())[:8])
            res.findings.append(
                Finding(
                    self.id,
                    Severity.FAIL,
                    f"{len(dup)} episode index(es) catalogued more than once in the metadata: "
                    f"{ex}{'...' if len(dup) > 8 else ''} - typical of merged recording sessions "
                    "that each restarted at 0; frames of different episodes share one index",
                    details={"duplicates": {str(k): v for k, v in dup.items()}},
                )
            )
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
        # video segment length vs frames in the data, from metadata alone
        # (v3 records every episode's [from, to) per camera). A segment
        # shorter than the data means frame lookups past its end fail.
        n_vid_bad = 0
        if ctx.video_segments and ctx.fps:
            short: dict[str, list] = {}
            long_: dict[str, list] = {}
            for ep in ctx.episode_ids():
                n_rows = len(ctx.rows_for(ep))
                for cam, (_, _, t0, t1) in ctx.video_segments.get(ep, {}).items():
                    n_vid = int(round((t1 - t0) * ctx.fps))
                    if n_vid < n_rows - 1:
                        short.setdefault(cam, []).append((ep, n_vid, n_rows))
                    elif n_vid > n_rows + 1:
                        long_.setdefault(cam, []).append((ep, n_vid, n_rows))
            for cam, bad in short.items():
                n_vid_bad += len(bad)
                ep, nv, nr = bad[0]
                res.findings.append(Finding(
                    self.id, Severity.FAIL,
                    f"camera {cam}: {len(bad)} episode(s) whose video segment is shorter than "
                    f"the data (e.g. episode {ep}: {nv} video frames, {nr} rows) - frame "
                    "lookups past the segment end fail",
                    details={"camera": cam, "episodes": [b[0] for b in bad]},
                ))
            for cam, bad in long_.items():
                n_vid_bad += len(bad)
                ep, nv, nr = bad[0]
                res.findings.append(Finding(
                    self.id, Severity.WARN,
                    f"camera {cam}: {len(bad)} episode(s) whose video segment is longer than "
                    f"the data (e.g. episode {ep}: {nv} video frames, {nr} rows) - images and "
                    "states may be misaligned",
                    details={"camera": cam, "episodes": [b[0] for b in bad]},
                ))
        elif not ctx.video_segments:
            res.findings.append(Finding(
                self.id, Severity.INFO,
                "no per-episode video segments in the metadata (v2.x or no videos): "
                "video frame counts not cross-checked",
            ))
        if n_gap == n_meta_mismatch == n_dup == n_vid_bad == 0 and not ctx.duplicate_episode_ids \
                and not ctx.unreadable_files:
            res.summary = "frame_index contiguous; episode lengths" + (
                " and video segments" if ctx.video_segments else "") + " match the data"
        else:
            res.summary = (
                f"{n_gap} episodes with gaps, {n_dup} with duplicate indices, "
                f"{n_meta_mismatch} with metadata length mismatches"
            )
        return res

"""Metadata checks: embedded images, camera coverage, task strings, reward/done.

These read the catalogues, not the frames: an embedded image column, a
camera whose video is missing for some episodes, an empty task string or a
reward that never changes are all invisible to the numeric checks but each
breaks a real training path (bloated shards, a modality that silently drops
out, language conditioning on an empty string, RL labels that carry no
signal).
"""

from __future__ import annotations

import json

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity

_EMPTY_TASK_WARN_SHARE = 0.05  # WARN when >= 5% of episodes lack a task string
_MISSING_VIDEO_WARN_SHARE = 0.10  # WARN when a camera misses >= 10% of episodes


def _image_features(ctx) -> dict[str, dict]:
    """Declared features stored as embedded images (frames inside the parquet
    shards), as opposed to video chunks."""
    return {n: f for n, f in (ctx.info.get("features") or {}).items()
            if isinstance(f, dict) and f.get("dtype") == "image"}


def _video_features(ctx) -> list[str]:
    return [n for n, f in (ctx.info.get("features") or {}).items()
            if isinstance(f, dict) and f.get("dtype") == "video"]


class EmbeddedImagesCheck(Check):
    id = "embedded-images"
    title = "Embedded images in the parquet shards"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        imgs = _image_features(ctx)
        if not imgs:
            res.summary = "no image features declared"
            return res
        names = ", ".join(sorted(imgs))
        res.findings.append(Finding(
            self.id, Severity.INFO,
            f"{len(imgs)} image feature(s) embedded in the parquet shards ({names}): "
            "shards are big and every frame decodes on read - the video layout "
            "(dtype \"video\", MP4 chunks) streams far better for training",
            details={"features": sorted(imgs)},
        ))
        res.summary = f"{len(imgs)} embedded image feature(s)"
        return res


class CameraCoverageCheck(Check):
    id = "camera-coverage"
    title = "Per-episode camera coverage"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        cams = _video_features(ctx)
        if not cams:
            return self.skip("no video cameras declared")
        wanted = set(ctx.episode_ids())

        # camera -> set of episodes with video, from the metadata where the
        # format records it, else from the files on disk
        have: dict[str, set[int]] = {}
        if ctx.video_segments:
            for ep, per_cam in ctx.video_segments.items():
                if ep in wanted:
                    for cam in per_cam:
                        have.setdefault(cam, set()).add(ep)
        else:
            for p in sorted(ctx.root.glob("videos/*/*/episode_*.mp4")):
                try:
                    ep = int(p.stem.split("_")[-1])
                except ValueError:
                    continue
                if ep in wanted:
                    have.setdefault(p.parent.name, set()).add(ep)
            if not have:
                return self.skip("video files not downloaded (pass --video) "
                                 "and the metadata records no segments")

        for cam in cams:
            missing = sorted(wanted - have.get(cam, set()))
            if not missing:
                continue
            share = len(missing) / max(len(wanted), 1)
            sev = Severity.WARN if share >= _MISSING_VIDEO_WARN_SHARE else Severity.INFO
            ex = ", ".join(map(str, missing[:8])) + (" ..." if len(missing) > 8 else "")
            res.findings.append(Finding(
                self.id, sev,
                f"camera {cam}: no video for {len(missing)}/{len(wanted)} episodes ({ex}) - "
                "training code that assumes the modality is always there reads a gap",
                details={"camera": cam, "episodes": missing},
            ))
        res.summary = (f"{len(cams)} camera(s) over {len(wanted)} episodes; "
                       f"{sum(len(f.details['episodes']) for f in res.findings)} missing segments")
        return res


def _task_table(ctx) -> dict[int, str]:
    """task_index -> task string from meta/tasks (parquet v2.1+/v3, jsonl v2.0)."""
    out: dict[int, str] = {}
    pq_path = ctx.root / "meta" / "tasks.parquet"
    if pq_path.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(pq_path)
            names = t.column_names
            idx_col = "task_index" if "task_index" in names else names[0]
            task_col = "task" if "task" in names else names[-1]
            for i, s in zip(t.column(idx_col).to_pylist(), t.column(task_col).to_pylist()):
                out[int(i)] = str(s)
        except Exception:  # noqa: BLE001 - unreadable table: other checks report it
            pass
        return out
    jl_path = ctx.root / "meta" / "tasks.jsonl"
    if jl_path.exists():
        for line in jl_path.read_text().splitlines():
            try:
                row = json.loads(line)
                out[int(row["task_index"])] = str(row["task"])
            except (KeyError, ValueError, TypeError):
                continue
    return out


class TaskStringsCheck(Check):
    id = "task-strings"
    title = "Task strings"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        empty = [e.index for e in ctx.episodes
                 if not e.tasks or all(not str(t).strip() for t in e.tasks)]
        if empty:
            share = len(empty) / max(len(ctx.episodes), 1)
            sev = Severity.WARN if share >= _EMPTY_TASK_WARN_SHARE else Severity.INFO
            ex = ", ".join(map(str, empty[:8])) + (" ..." if len(empty) > 8 else "")
            res.findings.append(Finding(
                self.id, sev,
                f"{len(empty)}/{len(ctx.episodes)} episodes have an empty task string ({ex}) - "
                "language-conditioned policies train on \"\" for them",
                details={"episodes": empty},
            ))

        table = _task_table(ctx)
        if table:
            # episode metadata names a task the table does not hold
            known = set(table.values())
            unknown = sorted({e.index for e in ctx.episodes
                              for t in e.tasks if str(t).strip() and str(t) not in known})
            if unknown:
                ex = ", ".join(map(str, unknown[:8])) + (" ..." if len(unknown) > 8 else "")
                res.findings.append(Finding(
                    self.id, Severity.INFO,
                    f"{len(unknown)} episodes name a task string with no entry in meta/tasks "
                    f"({ex}) - the episode metadata and the task table disagree",
                    details={"episodes": unknown},
                ))
            # frames carry a task_index the table does not define
            if ctx.task_index is not None:
                used = set(np.unique(np.asarray(ctx.task_index)[np.isfinite(ctx.task_index)]).astype(int))
                dangling = sorted(used - set(table))
                if dangling:
                    res.findings.append(Finding(
                        self.id, Severity.WARN,
                        f"frames reference task_index values with no entry in meta/tasks: "
                        f"{dangling[:8]}{' ...' if len(dangling) > 8 else ''} - "
                        "their task string is unrecoverable",
                        details={"task_indices": dangling},
                    ))
        res.summary = (f"{len(ctx.episodes)} episodes, {len(table)} tasks in meta/tasks; "
                       f"{len(res.findings)} finding(s)")
        return res


class RewardDoneCheck(Check):
    id = "reward-done"
    title = "Reward / done fields"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if ctx.reward is None and ctx.done is None:
            return self.skip("no reward/done fields (position-control datasets rarely have them)")

        if ctx.reward is not None:
            r = np.asarray(ctx.reward, dtype=np.float64)
            finite = np.isfinite(r)
            if not finite.any():
                res.findings.append(Finding(
                    self.id, Severity.WARN,
                    "reward is NaN on every frame - the field is present but carries nothing",
                ))
            else:
                bad = int((~finite).sum())
                if bad:
                    res.findings.append(Finding(
                        self.id, Severity.WARN,
                        f"reward is NaN on {bad}/{len(r)} frames",
                    ))
                if np.ptp(r[finite]) < 1e-8:
                    res.findings.append(Finding(
                        self.id, Severity.WARN,
                        f"reward is constant ({float(r[finite][0]):g}) on every frame - "
                        "no signal for RL or reward-conditioned training",
                    ))

        if ctx.done is not None:
            d = np.asarray(ctx.done, dtype=bool)
            if not d.any():
                res.findings.append(Finding(
                    self.id, Severity.WARN,
                    "done is never True on any frame - episodes never terminate; "
                    "anything segmenting on done falls off the end",
                ))
            else:
                # a done flag with frames after it inside the same episode
                continued = []
                for ep in ctx.episode_ids():
                    rows = ctx.rows_for(ep)
                    if len(rows) < 2:
                        continue
                    flags = d[rows]
                    if flags[:-1].any():
                        continued.append(ep)
                if continued:
                    ex = ", ".join(map(str, continued[:8])) + (" ..." if len(continued) > 8 else "")
                    res.findings.append(Finding(
                        self.id, Severity.WARN,
                        f"done is True before the last frame in {len(continued)} episodes ({ex}) - "
                        "the trajectory continues after termination",
                        details={"episodes": continued},
                    ))
        res.summary = ("reward and done present"
                       if ctx.reward is not None and ctx.done is not None
                       else f"reward {'present' if ctx.reward is not None else 'absent'}, "
                            f"done {'present' if ctx.done is not None else 'absent'}")
        return res

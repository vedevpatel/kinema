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
import pyarrow.parquet as pq

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity
from kinema.dataset import DatasetError, _load_episode_meta

_EMPTY_TASK_WARN_SHARE = 0.05  # WARN when >= 5% of episodes lack a task string
_MISSING_VIDEO_WARN_SHARE = 0.10  # WARN when a camera misses >= 10% of episodes
# video-file-pointers: metadata naming video files the repo does not hold
_MISSING_VIDEO_FILE_WARN_SHARE = 0.10  # WARN at >= 10% of episodes losing a camera
_MISSING_VIDEO_FILE_FAIL_SHARE = 0.50  # FAIL at half the dataset


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
            if ctx.video_partial:
                # a sampled audit fetches only a subset of the episode
                # videos, and v2.1 metadata records no segments to consult
                # instead: what is on disk says nothing about the episodes
                # that were not fetched
                return self.skip("episode videos only partially fetched - "
                                 "camera coverage needs a full video audit")
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


_POINTER_LIST_CAP = 10  # findings at most; the summary always carries the full count


def _shard_pointers(ctx) -> dict[int, tuple[int, int]] | None:
    """episode -> (chunk_index, file_index) from meta/episodes, or None when
    the metadata records no shard locations (early v3 exports)."""
    ep_dir = ctx.root / "meta" / "episodes"
    if not ep_dir.is_dir():
        return None
    need = ("episode_index", "data/chunk_index", "data/file_index")
    out: dict[int, tuple[int, int]] = {}
    for f in sorted(ep_dir.glob("**/*.parquet")):
        if any(c not in pq.read_schema(f).names for c in need):
            return None
        t = pq.read_table(f, columns=list(need))
        for ep, c, fi in zip(
            t.column("episode_index").to_pylist(),
            t.column("data/chunk_index").to_pylist(),
            t.column("data/file_index").to_pylist(),
        ):
            out[int(ep)] = (int(c), int(fi))
    return out


def _data_rel(template: str, chunk: int, file_: int) -> str:
    try:
        return template.format(chunk_index=chunk, file_index=file_)
    except (KeyError, IndexError, ValueError):
        return f"data/chunk-{chunk:03d}/file-{file_:03d}.parquet"


class EpisodeFilePointersCheck(Check):
    id = "episode-file-pointers"
    title = "Episode file pointers (v3 shard metadata)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if not str(ctx.format_version).startswith("v3"):
            return self.skip("v2.x layout: one parquet per episode, no shard pointers to check")
        decl = _shard_pointers(ctx)
        if decl is None:
            return self.skip("meta/episodes records no shard locations "
                             "(data/chunk_index, data/file_index)")
        template = (ctx.info or {}).get("data_path") or \
            "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"

        # the episode roster of every declared shard that exists locally; a
        # shard not downloaded (sampled audit) or unreadable is outside
        # coverage - the loader reports unreadable files itself
        local: dict[str, set[int]] = {}
        for c, fi in set(decl.values()):
            rel = _data_rel(template, c, fi)
            p = ctx.root / rel
            if not p.exists():
                continue
            try:
                local[rel] = {int(e) for e in
                              pq.read_table(p, columns=["episode_index"])
                              .column("episode_index").to_pylist()}
            except Exception:  # noqa: BLE001
                continue
        if not local:
            return self.skip("no declared data shards available locally to check against")
        actual = {e: rel for rel, eps in local.items() for e in eps}

        broken: list[tuple[int, str, str | None]] = []
        judged = 0
        for ep, (c, fi) in sorted(decl.items()):
            rel = _data_rel(template, c, fi)
            if rel not in local:
                continue  # shard not local: outside this run's coverage
            judged += 1
            if ep not in local[rel]:
                broken.append((ep, rel, actual.get(ep)))

        for ep, want, got in broken[:_POINTER_LIST_CAP]:
            res.findings.append(Finding(
                self.id, Severity.WARN,
                f"episode {ep}: metadata points at {want} but its rows are not there"
                + (f" - they live in {got}" if got else "")
                + "; LeRobot loads only the metadata-named file, so the episode "
                "silently drops out of training",
                episode_index=ep,
                details={"declared": want, "actual": got},
            ))
        if judged == len(decl):
            coverage = f"{judged} episodes, {len(local)} shard(s)"
        else:
            coverage = (f"checked {judged} of {len(decl)} catalogued episodes "
                        f"against {len(local)} local shard(s)")
        if broken:
            res.summary = (
                f"{len(broken)} episode(s) point at a shard that does not contain their "
                "rows - the v2.1->v3.0 converter rollover bug "
                "(https://github.com/huggingface/lerobot/issues/2401): those episodes "
                "silently vanish from LeRobot training loads. Re-convert or re-export "
                f"with a fixed lerobot to repair the metadata ({coverage})"
            )
        else:
            res.summary = f"every shard pointer matches its file's contents ({coverage})"
        res.details = {"broken": [ep for ep, _, _ in broken], "judged": judged,
                       "catalogued": len(decl), "local_shards": len(local)}
        return res


class VideoFilePointersCheck(Check):
    """Metadata referencing video files the repo does not contain.

    The video sibling of episode-file-pointers: LeRobot fetches the
    metadata-named MP4 for an episode's camera, so when the file was never
    uploaded (an interrupted export, files pruned by hand) the episode's
    rows load but its video does not decode - and the failure surfaces far
    from the cause. The referenced paths come from the v3 segment columns or
    the v2.x video_path template; they are checked against the repo's full
    file listing when the caller holds one (board audits), else the files
    on disk - a partial video download is not evidence about files it never
    fetched."""

    id = "video-file-pointers"
    title = "Video file pointers (metadata vs repo contents)"

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        cams = _video_features(ctx)
        if not cams:
            return self.skip("no video cameras declared")
        template = (ctx.info or {}).get("video_path")
        if not template:
            return self.skip("no video_path template in meta/info.json")
        chunk_size = int((ctx.info or {}).get("chunks_size") or 1000)

        if ctx.repo_files is not None:
            existing = {p for p in ctx.repo_files if p.startswith("videos/")}
            basis = "the repo file listing"
        elif ctx.video_partial:
            return self.skip("episode videos only partially fetched - pointer "
                             "existence needs the full repo file listing")
        else:
            if not (ctx.root / "videos").is_dir():
                # a local copy holding only data/ + meta/ (a common partial
                # download) is not evidence about videos it never fetched
                return self.skip("no episode videos fetched - fetch the videos/ "
                                 "tree (--video for Hub repos) to verify pointers")
            existing = {p.relative_to(ctx.root).as_posix()
                        for p in ctx.root.glob("videos/**/*") if p.is_file()}
            basis = "the files on disk"

        def render(cam: str, ep: int, ch: int, fi: int) -> str | None:
            try:
                # both naming generations, as the loader's video targeting does
                return template.format(video_key=cam, episode_index=ep,
                                       episode_chunk=ep // chunk_size,
                                       chunk_index=ch, file_index=fi)
            except (KeyError, IndexError, ValueError):
                return None

        # (camera, referenced path) -> the episodes pointing at it
        pointed: dict[tuple[str, str], set[int]] = {}
        if ctx.video_segments:
            for ep, per_cam in ctx.video_segments.items():
                for cam, (ch, fi, _from, _to) in per_cam.items():
                    path = render(cam, ep, ch, fi)
                    if path is None:
                        return self.skip("video_path template did not render")
                    pointed.setdefault((cam, path), set()).add(ep)
        elif "episode_index" in template:
            try:
                episodes = _load_episode_meta(ctx.root)
            except DatasetError:
                return self.skip("no episode metadata to render pointers from")
            for e in episodes:
                for cam in cams:
                    path = render(cam, e.index, e.index // chunk_size, e.index)
                    if path is None:
                        return self.skip("video_path template did not render")
                    pointed.setdefault((cam, path), set()).add(e.index)
        else:
            return self.skip("v3 layout without segment metadata: no pointers to check")
        if not pointed:
            return self.skip("no video pointers in the metadata")

        catalogued = len({e for eps in pointed.values() for e in eps})
        missing = sorted(k for k in pointed if k[1] not in existing)
        if not missing:
            res.summary = (f"all {len(pointed)} metadata-referenced video files exist "
                           f"({basis})")
            res.details = {"referenced": len(pointed), "missing": 0}
            return res

        missing_by_cam: dict[str, list[str]] = {}
        for cam, path in missing:
            missing_by_cam.setdefault(cam, []).append(path)
        for cam in sorted(missing_by_cam)[:_POINTER_LIST_CAP]:
            paths = missing_by_cam[cam]
            affected = sorted({e for p in paths for e in pointed[(cam, p)]})
            share = len(affected) / max(catalogued, 1)
            sev = (Severity.FAIL if share >= _MISSING_VIDEO_FILE_FAIL_SHARE
                   else Severity.WARN if share >= _MISSING_VIDEO_FILE_WARN_SHARE
                   else Severity.INFO)
            res.findings.append(Finding(
                self.id, sev,
                f"camera {cam}: the metadata references {len(paths)} video file(s) the repo "
                f"does not contain ({len(affected)}/{catalogued} episodes affected, "
                f"e.g. {paths[0]}) - LeRobot fetches the metadata-named path, so those "
                "episodes have no video to decode",
                details={"camera": cam, "missing_files": paths[:20], "episodes": affected},
            ))
        total_eps = sorted({e for key in missing for e in pointed[key]})
        cam_bits = ", ".join(
            f"{cam} {len({e for p in ps for e in pointed[(cam, p)]})}/{catalogued}"
            for cam, ps in sorted(missing_by_cam.items())
        )
        res.summary = (
            f"{len(missing)} video file(s) referenced by the metadata are missing from the "
            f"repo - {len(total_eps)}/{catalogued} episodes lose a camera's video ({cam_bits}); "
            "re-export or restore the files "
            f"(checked {len(pointed)} referenced files against {basis})"
        )
        res.details = {"referenced": len(pointed), "missing": len(missing),
                       "episodes_affected": total_eps, "basis": basis}
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
                # classify each episode's done pattern. A contiguous terminal
                # suffix longer than one frame is a padding convention (pusht
                # marks its last two frames), not a continuation; a True with
                # frames after it inside the same episode is the real defect.
                padding: list[tuple[int, int]] = []
                continued = []
                all_true: list[int] = []
                for ep in ctx.episode_ids():
                    rows = ctx.rows_for(ep)
                    if len(rows) < 2:
                        continue
                    flags = d[rows]
                    if not flags.any():
                        continue
                    first = int(flags.argmax())
                    if flags[first:].all():
                        suffix = len(flags) - first
                        if first == 0:
                            all_true.append(ep)
                        elif 1 < suffix <= 2:
                            # the one verified padding convention (pusht's
                            # final two frames); a longer terminal run means
                            # the trajectory really does continue post-done
                            padding.append((ep, suffix))
                        elif suffix > 2:
                            continued.append(ep)
                        # a single final-frame True is the normal convention
                    else:
                        continued.append(ep)
                if all_true:
                    ex = ", ".join(map(str, all_true[:8])) + (" ..." if len(all_true) > 8 else "")
                    res.findings.append(Finding(
                        self.id, Severity.WARN,
                        f"done is True on every frame in {len(all_true)} episodes ({ex}) - "
                        "episodes terminate at birth; anything segmenting on done gets nothing",
                        details={"episodes": all_true},
                    ))
                if padding:
                    eps = [ep for ep, _ in padding]
                    ks = sorted({k for _, k in padding})
                    ex = ", ".join(map(str, eps[:8])) + (" ..." if len(eps) > 8 else "")
                    res.findings.append(Finding(
                        self.id, Severity.INFO,
                        f"done is True on the final {'/'.join(map(str, ks))} frame(s) in "
                        f"{len(eps)} episodes ({ex}) - a terminal-flag padding convention; "
                        "segmentation on the first done still works",
                        details={"episodes": eps},
                    ))
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

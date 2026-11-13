"""Check 11 - visual quality (blur / lighting).

HF's RobotData GIGO study put bad visual quality at ~30% of training
loss on Open X-Embodiment.
Optional: needs the `av` extra and --video (videos are big).

Blur is variance-of-Laplacian (no cv2 dependency: a 4-neighbor
convolution done with numpy slicing). Lighting is the 95th-percentile
grayscale: an underexposed frame never gets bright anywhere.

Both video layouts are supported: v2.1 keeps one MP4 per episode and
camera; v3.0 packs many episodes into shared MP4s, and meta/episodes
records each episode's [from_timestamp, to_timestamp) segment per camera.
Frames are sampled evenly inside each episode's segment by seeking.
Findings are aggregated per camera - a camera that is blurry in most
episodes is one problem, not fifty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity

BLUR_WARN = 30.0  # variance of Laplacian below this reads as blurry
# underexposure: the 95th-percentile brightness (0-255) stays below this.
# Mean brightness misreads dark-but-well-lit scenes (a black sim table with
# bright objects averages ~40 while its highlights reach ~150).
DARK_WARN = 80.0
SAMPLES_PER_EPISODE = 10
CAMERA_WARN_SHARE = 0.25  # a camera flagged in >= 25% of sampled episodes


@dataclass
class _Segment:
    episode: int
    camera: str
    path: Path
    start: float  # seconds into the file
    end: float | None  # None: to the end of the file (v2.1 per-episode files)


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap.var())


def _segments(ctx) -> list[_Segment]:
    root: Path = ctx.root
    wanted = set(ctx.episode_ids())
    out: list[_Segment] = []
    # v2.1: videos/chunk-*/<camera>/episode_XXXXXX.mp4
    for p in sorted(root.glob("videos/*/*/episode_*.mp4")):
        try:
            ep = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        if ep in wanted:
            out.append(_Segment(ep, p.parent.name, p, 0.0, None))
    if out:
        return out
    # v3.0: shared files, segments recorded in meta/episodes (loaded by the loader)
    template = ctx.info.get("video_path") or ""
    if not template:
        return out
    for ep in sorted(wanted):
        for cam, (chunk, file_idx, start, end) in ctx.video_segments.get(ep, {}).items():
            try:
                rel = template.format(video_key=cam, chunk_index=chunk, file_index=file_idx)
            except (KeyError, IndexError, ValueError):
                continue
            path = root / rel
            if path.exists():
                out.append(_Segment(ep, cam, path, start, end))
    return out


def _sample_file(av, path: Path, segs: list[_Segment]) -> dict[int, list[np.ndarray]]:
    """Grayscale frames sampled evenly inside each segment of one file."""
    frames: dict[int, list[np.ndarray]] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        for seg in sorted(segs, key=lambda s: s.start):
            end = seg.end if seg.end is not None else duration
            if end is None:  # unknown length: decode the head sequentially
                got = []
                for i, fr in enumerate(container.decode(stream)):
                    if i % 5 == 0:
                        got.append(fr.to_ndarray(format="gray"))
                    if len(got) >= SAMPLES_PER_EPISODE:
                        break
                frames[seg.episode] = got
                container.seek(0)
                continue
            span = max(end - seg.start, 1e-6)
            targets = [seg.start + (k + 0.5) * span / SAMPLES_PER_EPISODE
                       for k in range(SAMPLES_PER_EPISODE)]
            got = []
            for ts in targets:
                container.seek(int(ts / stream.time_base), stream=stream, backward=True)
                for fr in container.decode(stream):
                    if fr.time is None or fr.time + 1e-6 >= ts:
                        got.append(fr.to_ndarray(format="gray"))
                        break
            frames[seg.episode] = got
    return frames


class VisualQualityCheck(Check):
    id = "visual-quality"
    title = "Visual quality (blur, lighting)"
    requires_video = True

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if not cfg.video:
            return self.skip("pass --video to run (downloads MP4s; needs the video extra)")
        try:
            import av
        except ImportError:
            return self.skip("PyAV not installed - pip install the video extra")

        segs = _segments(ctx)
        if not segs:
            return self.skip("no video files found for the loaded episodes")

        by_file: dict[Path, list[_Segment]] = {}
        for s in segs:
            by_file.setdefault(s.path, []).append(s)

        # camera -> episode -> (median blur, mean brightness)
        scores: dict[str, dict[int, tuple[float, float]]] = {}
        for path, file_segs in by_file.items():
            cam = file_segs[0].camera
            try:
                sampled = _sample_file(av, path, file_segs)
            except Exception as e:  # noqa: BLE001 - one bad file must not sink the run
                res.findings.append(Finding(
                    self.id, Severity.INFO,
                    f"{path.relative_to(ctx.root)}: could not decode ({type(e).__name__}: {e})"))
                continue
            for ep, imgs in sampled.items():
                if imgs:
                    scores.setdefault(cam, {})[ep] = (
                        float(np.median([_laplacian_variance(g) for g in imgs])),
                        float(np.median([np.percentile(g, 95) for g in imgs])),
                    )

        if not scores:
            return self.skip("no frames could be decoded")
        n_eps = len({ep for cam in scores.values() for ep in cam})
        for cam, eps in sorted(scores.items()):
            blur = {ep: b for ep, (b, _) in eps.items() if b < BLUR_WARN}
            dark = {ep: g for ep, (_, g) in eps.items() if g < DARK_WARN}
            for kind, bad, thresh, unit in (
                ("blurry", blur, BLUR_WARN, "blur score"),
                ("underexposed", dark, DARK_WARN, "95th-percentile brightness"),
            ):
                if not bad:
                    continue
                share = len(bad) / len(eps)
                worst = sorted(bad, key=lambda ep: bad[ep])[:8]
                sev = Severity.WARN if share >= CAMERA_WARN_SHARE else Severity.INFO
                res.findings.append(Finding(
                    self.id, sev,
                    f"camera {cam}: {kind} in {len(bad)}/{len(eps)} episodes "
                    f"({unit} < {thresh:g}; worst {worst})",
                    details={"camera": cam, "kind": kind, "episodes": sorted(bad),
                             "values": {str(e): v for e, v in bad.items()}},
                ))
        med = {cam: float(np.median([b for b, _ in eps.values()])) for cam, eps in scores.items()}
        res.details = {"median_blur_per_camera": med,
                       "median_p95_brightness_per_camera": {
                           cam: float(np.median([g for _, g in eps.values()]))
                           for cam, eps in scores.items()}}
        res.summary = (
            f"{n_eps} episodes x {len(scores)} camera(s) sampled; median blur "
            + ", ".join(f"{c} {v:.0f}" for c, v in sorted(med.items()))
        )
        return res

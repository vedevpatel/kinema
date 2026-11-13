"""Check 11 - visual quality (blur / lighting).

HF's RobotData GIGO study put bad visual quality at ~30% of training
loss on Open X-Embodiment.
Optional: needs the `av` extra and --video (videos are big).

Blur is variance-of-Laplacian (no cv2 dependency: a 4-neighbor
convolution done with numpy slicing). Lighting is mean grayscale.
"""

from __future__ import annotations

import numpy as np

from kinema.checks.base import Check, CheckConfig, CheckResult, Finding, Severity

BLUR_WARN = 30.0  # variance of Laplacian below this reads as blurry
DARK_WARN = 50.0  # mean grayscale (0-255) below this reads as underlit
SAMPLES_PER_EPISODE = 10


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap.var())


class VisualQualityCheck(Check):
    id = "visual-quality"
    title = "Visual quality (blur, lighting)"
    requires_video = True

    def run(self, ctx, cfg: CheckConfig) -> CheckResult:
        res = CheckResult(self.id, self.title)
        if not cfg.video:
            return self.skip("pass --video to run (downloads MP4s; needs the video extra)")
        try:
            import av  # noqa: F401
        except ImportError:
            return self.skip("PyAV not installed - pip install the video extra")

        # v2.1 layout: videos/chunk-*/<camera>/episode_XXXXXX.mp4 (one file per episode)
        per_ep = sorted(ctx.root.glob("videos/*/*/episode_*.mp4"))
        if not per_ep:
            return self.skip(
                "no per-episode MP4s found (v3 shared-file video attribution "
                "is not implemented in v0)"
            )

        cameras = sorted({p.parent.name for p in per_ep})
        by_ep: dict[int, list] = {}
        for p in per_ep:
            try:
                ep = int(p.stem.split("_")[-1])
            except ValueError:
                continue
            by_ep.setdefault(ep, []).append(p)

        n_blur = n_dark = 0
        checked = 0
        for ep, files in sorted(by_ep.items()):
            if ctx.episode_ids() and ep not in set(ctx.episode_ids()):
                continue
            checked += 1
            for f in files:
                blur_vals, gray_vals = [], []
                try:
                    container = av.open(str(f))
                    stream = container.streams.video[0]
                    n = stream.frames or SAMPLES_PER_EPISODE * 4
                    want = max(1, n // SAMPLES_PER_EPISODE)
                    for i, frame in enumerate(container.decode(stream)):
                        if i % want:
                            continue
                        img = frame.to_ndarray(format="gray")
                        blur_vals.append(_laplacian_variance(img))
                        gray_vals.append(float(img.mean()))
                        if len(blur_vals) >= SAMPLES_PER_EPISODE:
                            break
                    container.close()
                except Exception as e:  # noqa: BLE001 - one bad file must not sink the run
                    res.findings.append(
                        Finding(self.id, Severity.INFO, f"{f.name}: could not decode ({e})")
                    )
                    continue
                if not blur_vals:
                    continue
                med_blur = float(np.median(blur_vals))
                mean_gray = float(np.mean(gray_vals))
                if med_blur < BLUR_WARN:
                    n_blur += 1
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"episode {ep} ({f.parent.name}): median blur score "
                            f"{med_blur:.1f} < {BLUR_WARN:g} - frames are soft/out of focus",
                            episode_index=ep,
                            details={"camera": f.parent.name, "blur": med_blur},
                        )
                    )
                if mean_gray < DARK_WARN:
                    n_dark += 1
                    res.findings.append(
                        Finding(
                            self.id,
                            Severity.WARN,
                            f"episode {ep} ({f.parent.name}): mean brightness "
                            f"{mean_gray:.0f}/255 - underlit",
                            episode_index=ep,
                            details={"camera": f.parent.name, "brightness": mean_gray},
                        )
                    )
        if checked == 0:
            return self.skip("no video files matched the requested episodes")
        res.summary = (
            f"{checked} episodes x {len(cameras)} camera(s) sampled; "
            f"{n_blur} blurry, {n_dark} dark"
        )
        return res

# torchcodec backend comparison: method (2026-11-22)

This directory reads three frozen evidence directories and writes none of them:
[`audits/2026-11-22/`](../../audits/2026-11-22/) (phase 1),
[`loader-verification/2026-11-22/`](../../loader-verification/2026-11-22/) (phase 2) and
[`shifted-queries/2026-11-22/`](../../shifted-queries/2026-11-22/) (phase 3).
Phase 3 ran its shifted-query plans through LeRobot's pyav decoder path only,
because torchcodec, the default backend, could not load. This phase replays those
same plans through the torchcodec path and compares each batch with pyav.

It is a **direct-decoder** experiment. `LeRobotDataset` is not constructed,
patched or subclassed, and no LeRobot, torchcodec or dataset file is changed.

## Why torchcodec did not load

- Phase 3 recorded `RuntimeError: Could not load libtorchcodec`. Each FFmpeg
  major it tried (7, 6, 5, 4) failed with `libavutil.so.<N>: cannot open shared
  object file`.
- Reproduced here: `ldd site-packages/torchcodec/libtorchcodec_core7.so` lists
  all seven FFmpeg 7 sonames as `not found`. `ldconfig -p` has no libav/libsw
  library, and no `ffmpeg` is installed.
- Versions are not the cause. The official torchcodec README table pairs
  `0.5` with torch `2.7` and Python `>=3.9,<=3.13`. Installed are torchcodec 0.5,
  torch 2.7.1+cpu and Python 3.11.17, and LeRobot 0.3.3/0.4.0 require
  `torchcodec>=0.2.1,<0.6.0`. The v0.5.0 README supports FFmpeg majors 4–7.
- pyav worked because PyAV 15.1.0 bundles FFmpeg 7.1.x under renamed sonames in
  `site-packages/av.libs/`, which torchcodec does not look for.

## Environment (distinct, not a replay)

[`environment_spec.json`](environment_spec.json) defines `tc-v2` (LeRobot 0.3.3)
and `tc-v3` (LeRobot 0.4.0). Each is the **unchanged** phase-2 venv: no Python
package is added, removed or upgraded, and the harness checks the installed set
against the phase-2 freeze. FFmpeg 7 shared libraries are added on
`LD_LIBRARY_PATH`. [`setup_ffmpeg_aliases.py`](setup_ffmpeg_aliases.py) verifies
the sha256 of PyAV's bundled `libav*`/`libsw*` libraries, then creates
`/tmp/kinema-tc-ffmpeg-<env>/` with seven symlinks named with the sonames
`libtorchcodec_core7.so` asks for. The library path is
`<alias dir>:<venv>/site-packages/av.libs`. Undo it with `rm -rf`.

So both backends run the same FFmpeg 7.1.1 binaries in one process, and decoder
build differences cannot explain backend differences. The README's recommended
route (conda-forge FFmpeg via micromamba) was not used: this session's permission
policy refused downloading and executing the micromamba binary. Ubuntu 26.04
ships FFmpeg 8.0.1, which torchcodec 0.5 does not support. torchcodec's own
pinned S3 FFmpeg builds are link-time stubs with decoders disabled.

## How each backend picks a frame (source-cited)

Paths are under `site-packages/` unless prefixed with the torchcodec tag.

| Step | pyav path (phase 3) | torchcodec path (this phase) |
| --- | --- | --- |
| Dispatch | `lerobot/datasets/video_utils.py::decode_video_frames(..., "pyav")` → `decode_video_frames_torchvision` | `decode_video_frames(..., "torchcodec")` → `decode_video_frames_torchcodec` |
| Open | torchvision `VideoReader`, `seek(min(q), keyframes_only=True)`, read until `pts >= max(q)` | `VideoDecoder(path, seek_mode="approximate")`; 0.3.3 opens a new decoder per call; 0.4.0 caches one per path (`VideoDecoderCache`, via fsspec) |
| Query → frame | nearest loaded pts | `round(ts * metadata.average_fps)` → `get_frames_at(indices)` |
| `average_fps` | n/a | approximate mode does no scan, so `_core/_metadata.py::average_fps` returns `average_fps_from_header` = `av_q2d(r_frame_rate)` (v0.5.0 `SingleStreamDecoder.cpp`) |
| Negative index | n/a | `decoders/_video_decoder.py::get_frames_at`: `index + num_frames`, so it wraps to the end of the video |
| Index check | n/a | `validateFrameIndex`: `0 <= index < nb_frames` (header) or `RuntimeError: Invalid frame index` |
| Index → frame | n/a | `getPts` (approximate): `std::round((index / r_frame_rate) / time_base)`; `getNextFrameInternal` returns the first decoded frame with `pts >= cursor`; past the last frame: "no more frames left to decode" |
| Tolerance | `min_q \|f32(q) − f32(pts)\| < tolerance_s` over the loaded window, as an `AssertionError` | the same `torch.cdist(...).min(1)` check over the batch's returned frames, so a query can take another query's frame if it is closer |
| Colour | PyAV `frame.to_rgb()` (torchvision `io/video_reader.py`) | swscale `SWS_BILINEAR` → RGB24 if `width % 32 == 0`, else filtergraph `scale=W:H:sws_flags=bilinear` (v0.5.0 `CpuDeviceInterface.cpp`) |

The installed torchcodec Python files are byte-identical to tag v0.5.0
(commit `93cc0ade`). The compiled `.so` files cannot be compared with the C++
source, so [`predict.py`](predict.py) applies [`tcmodel.py`](tcmodel.py)'s
restatement of the rules above to demuxed packet pts and header fields. It
writes per-query predictions before execution, and the report compares them with
what the binary did. Predictions never decide a verdict.

## Experiment

[`tcprobe.py`](tcprobe.py) first checks the preregistration hashes, phase 3's own
preregistration, every frozen phase-1/2/3 artifact hash, every input against the
phase-3 plan record, LeRobot source hashes, the package set and the torchcodec
environment. Then, for every batch of `shifted-queries/2026-11-22/plan/<case>.json`
in file order:

1. upstream `decode_video_frames(path, queries, 1e-4, "torchcodec")`;
2. an independent torchcodec replay with a fresh decoder, one `get_frames_at([index])`
   per query. It must reproduce the upstream outcome and, on success, its output
   tensor, or the batch is a `harness_discrepancy`;
3. a pyav control: upstream `decode_video_frames(..., "pyav")` plus a replay of its
   VideoReader window, recorded separately from phase 3's frozen pyav evidence;
4. a per-query comparison of returned pts in time-base ticks and of uint8 pixels.
   When pixels differ, the torchcodec frame is also compared with pyav frames at
   window offsets −2..+2.

After all batches, one `backend=None` call records what the default selects.
[`selection_check.py`](selection_check.py) records `get_safe_default_codec()`,
the torchcodec import and a default-backend decode on a synthetic video, with and
without the FFmpeg aliases.

Outcome classes, bucket and case rules, and comparison rules are frozen in
[`expectations.json`](expectations.json). A tolerance `AssertionError`, torchcodec
`Invalid frame index` and end-of-stream are query-caused rejections and are
reported by mechanism. Dependency errors, other errors and harness discrepancies
are not evidence. Buckets (`loader_derived/in_range`, `loader_derived/padded`,
`nominal_grid/in_range`, `nominal_grid/outside_episode`) are never pooled, and
`nominal_grid` stays a labeled comparator, as in phase 3.

## Contact with pinned inputs before the freeze

Recorded so the preregistration can be judged on what was known:

- PyAV header reads (codec, r_frame_rate, nb_frames, time base) for the phase-2
  input videos, and `predict.py`'s demux of packet pts for the three cases. No
  frame was decoded.
- One environment smoke test decoded indices 11 and 12 of
  `r71_c1_f1/.../observation.images.primary/episode_000000.mp4` with torchcodec
  (pts 0.7333 s and 0.8000 s) while checking that the aliases load.
- `tcprobe.py --identity-only` for each case: hashing and environment checks, no
  decode.
- Synthetic-video tests ([`test_tcprobe_env.py`](test_tcprobe_env.py)) on videos
  generated in a temporary directory.

## Reproduce

```sh
D=torchcodec-backend/2026-11-22
# Inputs: phase-2 files under /tmp/kinema-phase2-inputs (see loader-verification/2026-11-22/METHOD.md).
python3 $D/setup_ffmpeg_aliases.py tc-v2
python3 $D/setup_ffmpeg_aliases.py tc-v3
V2="LD_LIBRARY_PATH=/tmp/kinema-tc-ffmpeg-v2:/tmp/kinema-phase2-v2/lib/python3.11/site-packages/av.libs"
V3="LD_LIBRARY_PATH=/tmp/kinema-tc-ffmpeg-v3:/tmp/kinema-phase2-v3/lib/python3.11/site-packages/av.libs"
# Before the freeze (outputs are hashed in preregistration.json):
/tmp/kinema-phase2-v3/bin/python -I -B $D/predict.py r1:c1:f0
/tmp/kinema-phase2-v2/bin/python -I -B $D/predict.py r71:c1:f1
/tmp/kinema-phase2-v2/bin/python -I -B $D/predict.py r94:c1:f1
# Execution:
env $V3 /tmp/kinema-phase2-v3/bin/python -I -B $D/tcprobe.py r1:c1:f0
env $V2 /tmp/kinema-phase2-v2/bin/python -I -B $D/tcprobe.py r71:c1:f1
env $V2 /tmp/kinema-phase2-v2/bin/python -I -B $D/tcprobe.py r94:c1:f1
for e in v2 v3; do
  /tmp/kinema-phase2-$e/bin/python -I -B $D/selection_check.py tc-$e no_ffmpeg
  env LD_LIBRARY_PATH=/tmp/kinema-tc-ffmpeg-$e:/tmp/kinema-phase2-$e/lib/python3.11/site-packages/av.libs \
    /tmp/kinema-phase2-$e/bin/python -I -B $D/selection_check.py tc-$e ffmpeg_aliases
done
python $D/make_report.py   # needs numpy only
python -m pytest tests/test_torchcodec_backend.py -q
env $V2 /tmp/kinema-phase2-v2/bin/python -I -B -m unittest discover -s $D -p 'test_tcprobe_env.py'
```

# torchcodec backend comparison (2026-11-22)

This report replays the frozen [phase-3](../../shifted-queries/2026-11-22/report.md) shifted-query plans through LeRobot's **torchcodec** decoder path and compares each batch with **pyav**. It does not revise the phase-1 audit, phase 2 or phase 3. Their files are unchanged and their hashes are re-checked: ok. Method: [METHOD.md](METHOD.md). Environment: [environment_spec.json](environment_spec.json).

Preregistration SHA-256 `3e6acadf6c9f3a30e477cd0fb31ec85467593d2c8572139ec2978549369c1114` (committed before any batch ran). Expectations SHA-256 `ce7fa5893ff3e379f7b29689f901439e3599a4d32c21a42a966bfd556499f25c`.

**All results are direct-decoder results.** `LeRobotDataset` was not constructed or patched, and the full-loader consequence stays unresolved for every case. The environment is distinct from phases 2 and 3: the same venvs and packages, with FFmpeg 7.1.1 shared libraries added on the library path.

## Results, kept separate by phase, backend and release

| Case | Release | Phase 3, pyav (frozen) | pyav control (this run) | torchcodec (this run) | torchcodec rejection mechanism | Full loader |
| --- | --- | --- | --- | --- | --- | --- |
| `r1:c1:f0` (primary) | 0.4.0 | refuted | refuted | **refuted** | – | unresolved |
| `r71:c1:f1` (primary) | 0.3.3 | refuted | refuted | **refuted** | – | unresolved |
| `r94:c1:f1` (comparator) | 0.3.3 | confirmed | confirmed | **confirmed** | index_error | unresolved |

Each verdict covers loader-derived queries only (in-range and padded together, as in phase 3) for the named release, backend, episode and anchors. Verdicts are not pooled and do not judge the original findings as a whole.

## Default-backend selection and import

| Env | LeRobot | FFmpeg on path | `get_safe_default_codec()` | import torchcodec | `decode_video_frames(..., backend=None)` | explicit pyav |
| --- | --- | --- | --- | --- | --- | --- |
| tc-v2 | 0.3.3 | no | `torchcodec` | dependency_error | dependency_error (RuntimeError) | ok |
| tc-v2 | 0.3.3 | yes | `torchcodec` | ok | ok | ok |
| tc-v3 | 0.4.0 | no | `torchcodec` | dependency_error | dependency_error (RuntimeError) | ok |
| tc-v3 | 0.4.0 | yes | `torchcodec` | ok | ok | ok |

These rows use a synthetic video. `get_safe_default_codec()` checks only `importlib.util.find_spec("torchcodec")`, so in both releases the default is torchcodec whether or not its libraries load. `LeRobotDataset.__init__` and `DatasetConfig.video_backend` use that default (source lines checked in each release). No pyav fallback happens when the import fails; the error propagates.

## Per-case results

### `r1:c1:f0` — lerobot/FineART@`15746b76ddc21ffd38d73888fefe2e7a1496385b`, episode 40455

Original assertion (phase 1, verbatim): 1 episode(s) off the 30 fps grid (episodes 40455): frame spacing deviates up to 67% of a period; LeRobot's default tolerance_s (1e-4 s) rejects these - needs tolerance_s >= 0.0222

Identity: frozen_sources_ok ok, inputs_rehashed_ok ok, packages_match_phase2_freeze ok, torchcodec_environment_ok ok, upstream_sources_ok ok. FFmpeg reported by torchcodec: 7.1.1.

torchcodec metadata (first camera, approximate mode): average_fps 90 (= average_fps_from_header), num_frames 26947 (from header; num_frames_from_content None), 640x360 h264.

| Bucket | Backend | Batches | Queries | Outcomes | Verdict | Max distance, ok (s) | Min distance, failing query (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | torchcodec | 68 | 188 | ok 68 | contradicts_claim | 1.53e-05 | – |
| loader_derived/in_range | pyav_control | 68 | 188 | ok 68 | contradicts_claim | 1.53e-05 | – |
| loader_derived/in_range | phase3_pyav | 68 | 188 | ok 68 | contradicts_claim | 1.53e-05 | – |
| loader_derived/padded | torchcodec | 12 | 52 | ok 12 | contradicts_claim | 0 | – |
| loader_derived/padded | pyav_control | 12 | 52 | ok 12 | contradicts_claim | 0 | – |
| loader_derived/padded | phase3_pyav | 12 | 52 | ok 12 | contradicts_claim | 0 | – |
| nominal_grid/in_range | torchcodec | 108 | 108 | ok 108 | contradicts_claim | 3.05e-05 | – |
| nominal_grid/in_range | pyav_control | 108 | 108 | ok 108 | contradicts_claim | 3.05e-05 | – |
| nominal_grid/in_range | phase3_pyav | 108 | 108 | ok 108 | contradicts_claim | 3.05e-05 | – |
| nominal_grid/outside_episode | torchcodec | 36 | 36 | tolerance_error 16, index_error 20 | supports_claim (tolerance_error, index_error) | – | 299 |
| nominal_grid/outside_episode | pyav_control | 36 | 36 | tolerance_error 36 | supports_claim (tolerance_error) | – | 0.0111 |
| nominal_grid/outside_episode | phase3_pyav | 36 | 36 | tolerance_error 36 | supports_claim (tolerance_error) | – | 0.0111 |

`nominal_grid` rows are the labeled comparator (`raw + k/fps`, not a LeRobot query); their verdict strings are bucket-rule outputs and are not evidence for or against the claim.

Backend agreement (torchcodec vs the same-process pyav control):

- `loader_derived/in_range`: same outcome in 68/68 batches (torchcodec=ok / pyav=ok: 68). Returned pts where both returned frames: 188 same, 0 different. Pixels there: 188 equal, 0 different.
- `loader_derived/padded`: same outcome in 12/12 batches (torchcodec=ok / pyav=ok: 12). Returned pts where both returned frames: 52 same, 0 different. Pixels there: 52 equal, 0 different.
- `nominal_grid/in_range`: same outcome in 108/108 batches (torchcodec=ok / pyav=ok: 108). Returned pts where both returned frames: 108 same, 0 different. Pixels there: 108 equal, 0 different.
- `nominal_grid/outside_episode`: same outcome in 16/36 batches (torchcodec=index_error / pyav=tolerance_error: 20; torchcodec=tolerance_error / pyav=tolerance_error: 16). Returned pts where both returned frames: 0 same, 0 different. Pixels there: 0 equal, 0 different. Frames each backend selected in batches then rejected by tolerance: 0 same pts, 16 different.

pyav control vs frozen phase 3: 0 outcome changes in 224 batches, 0 returned-pts changes in 384 queries. Replays reproduced upstream in 224/224 torchcodec batches and 224/224 pyav batches.

Source model vs binary: 0 batch-outcome mismatches in 224, 0 index or returned-pts mismatches in 384 queries.

torchcodec rejections by mechanism:

- `nominal_grid/outside_episode: tolerance_error`: 16 batches; 16 contain a negative index wrapped to the end of the video, 0 an index at or above nb_frames (26947). Example (observation.images.cam_high, first, raw_plus_-4_over_fps): queries ['-0.1333333'] → raw index [-12] → index [26935] → returned pts ['299.2778'].
- `nominal_grid/outside_episode: index_error`: 20 batches; 0 contain a negative index wrapped to the end of the video, 20 an index at or above nb_frames (26947). Example (observation.images.cam_high, last, raw_plus_1_over_fps): queries ['299.4222'] → raw index [26948] → index [26948] → returned pts ['–'].

Default backend after all batches: `get_safe_default_codec()` = `torchcodec`, outcome ok, equal to explicit torchcodec: True.

Conclusion (scoped): torchcodec shifted loader-derived lookup is **refuted** for LeRobot 0.4.0 decode_video_frames(backend='torchcodec', tolerance_s=1e-4) with torchcodec 0.5 and FFmpeg 7.1.1 (PyAV-bundled), direct decoder, frozen phase-3 loader-derived query sets, episode 40455, anchors first/middle/worst_grid/last, 4 camera(s). Full loader: unresolved: LeRobotDataset construction fails in 0.4.0 and 0.5.1 on source dtype 'language' (phase 2); not bypassed.

Selection note: every stream has r_frame_rate 90, matching the 90 Hz rows. At every probed loader-derived query, `round(ts * 90)` named the frame whose pts equals the stored row timestamp, which is the same frame pyav's nearest-pts rule chose. That includes the worst-grid anchor (row 1458). The tolerance claim is therefore refuted on this path for both backends. This says nothing about phase 3's separate timeline finding (info.json declares 30 fps, so delta index 1 spans 1/90 s).

Comparator note (`nominal_grid/outside_episode`, not loader queries): the backends reject these queries by different mechanisms. Queries before 0 s give a negative index, which `VideoDecoder.get_frames_at` wraps to the end of the video, and tolerance then rejects them at about 299.4 s distance. Queries past the last frame raise `Invalid frame index`. pyav returns the nearest window frame and rejects at 0.0111 s. LeRobot's clamped loader queries never produce these values.

Remaining gaps (unresolved):

- Full loader: LeRobotDataset construction fails on source dtype 'language' (phase 2), so neither backend was exercised through __getitem__.
- LeRobot 0.5.1's torchcodec path was not run.
- Only episode 40455, four anchors and the default policy configs were probed.
- Phase 3's timeline finding (declared 30 fps vs 90 Hz rows and video, so delta windows span a third of the intended time) is untouched, and its effect on training is not measured.
- torchcodec's filtergraph colour path (width not a multiple of 32) and other FFmpeg 7 builds were not exercised.

Evidence: `evidence/r1_c1_f0_torchcodec.json`, `predictions/r1_c1_f0.json`; plan `shifted-queries/2026-11-22/plan/r1_c1_f0.json` (unchanged).

### `r71:c1:f1` — azsdsadasdsadasd/droid_filter_lerobot@`7415f67330af40953ea52aa61a74788c75e1a8d6`, episode 0

Original assertion (phase 1, verbatim): 23 episode(s) whose timestamps do not start at 0: 0 (0.7333s), 2335 (1s), 11676 (1.067s), 14011 (0.5333s), 16346 (1.733s) - video frames are looked up by timestamp, so decoding will miss

Identity: frozen_sources_ok ok, inputs_rehashed_ok ok, packages_match_phase2_freeze ok, torchcodec_environment_ok ok, upstream_sources_ok ok. FFmpeg reported by torchcodec: 7.1.1.

torchcodec metadata (first camera, approximate mode): average_fps 15 (= average_fps_from_header), num_frames 213 (from header; num_frames_from_content None), 256x160 h264.

| Bucket | Backend | Batches | Queries | Outcomes | Verdict | Max distance, ok (s) | Min distance, failing query (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | torchcodec | 51 | 141 | ok 51 | contradicts_claim | 0 | – |
| loader_derived/in_range | pyav_control | 51 | 141 | ok 51 | contradicts_claim | 0 | – |
| loader_derived/in_range | phase3_pyav | 51 | 141 | ok 51 | contradicts_claim | 0 | – |
| loader_derived/padded | torchcodec | 9 | 39 | ok 9 | contradicts_claim | 0 | – |
| loader_derived/padded | pyav_control | 9 | 39 | ok 9 | contradicts_claim | 0 | – |
| loader_derived/padded | phase3_pyav | 9 | 39 | ok 9 | contradicts_claim | 0 | – |
| nominal_grid/in_range | torchcodec | 81 | 81 | ok 81 | contradicts_claim | 9.54e-07 | – |
| nominal_grid/in_range | pyav_control | 81 | 81 | ok 81 | contradicts_claim | 9.54e-07 | – |
| nominal_grid/in_range | phase3_pyav | 81 | 81 | ok 81 | contradicts_claim | 9.54e-07 | – |
| nominal_grid/outside_episode | torchcodec | 27 | 27 | ok 27 | contradicts_claim | 9.54e-07 | – |
| nominal_grid/outside_episode | pyav_control | 27 | 27 | ok 27 | contradicts_claim | 9.54e-07 | – |
| nominal_grid/outside_episode | phase3_pyav | 27 | 27 | ok 27 | contradicts_claim | 9.54e-07 | – |

`nominal_grid` rows are the labeled comparator (`raw + k/fps`, not a LeRobot query); their verdict strings are bucket-rule outputs and are not evidence for or against the claim.

Backend agreement (torchcodec vs the same-process pyav control):

- `loader_derived/in_range`: same outcome in 51/51 batches (torchcodec=ok / pyav=ok: 51). Returned pts where both returned frames: 141 same, 0 different. Pixels there: 141 equal, 0 different.
- `loader_derived/padded`: same outcome in 9/9 batches (torchcodec=ok / pyav=ok: 9). Returned pts where both returned frames: 39 same, 0 different. Pixels there: 39 equal, 0 different.
- `nominal_grid/in_range`: same outcome in 81/81 batches (torchcodec=ok / pyav=ok: 81). Returned pts where both returned frames: 81 same, 0 different. Pixels there: 81 equal, 0 different.
- `nominal_grid/outside_episode`: same outcome in 27/27 batches (torchcodec=ok / pyav=ok: 27). Returned pts where both returned frames: 27 same, 0 different. Pixels there: 27 equal, 0 different.

pyav control vs frozen phase 3: 0 outcome changes in 168 batches, 0 returned-pts changes in 288 queries. Replays reproduced upstream in 168/168 torchcodec batches and 168/168 pyav batches.

Source model vs binary: 0 batch-outcome mismatches in 168, 0 index or returned-pts mismatches in 288 queries.

Default backend after all batches: `get_safe_default_codec()` = `torchcodec`, outcome ok, equal to explicit torchcodec: True.

Conclusion (scoped): torchcodec shifted loader-derived lookup is **refuted** for LeRobot 0.3.3 decode_video_frames(backend='torchcodec', tolerance_s=1e-4) with torchcodec 0.5 and FFmpeg 7.1.1 (PyAV-bundled), direct decoder, frozen phase-3 loader-derived query sets, episode 0, anchors first/middle/worst_grid/last, 3 camera(s). Full loader: unresolved: LeRobotDataset construction raises KeyError('stats') (phase 2); not bypassed.

Selection note: r_frame_rate 15 and nb_frames 213. Every loader-derived and nominal query mapped to an in-range index, and both backends returned the same frame with identical pixels. Whether the frame at 0.7333 s shows the same instant as row 0 (the video starts at 0 s, the rows at 0.7333 s) is a semantic-alignment question that neither backend tests.

Remaining gaps (unresolved):

- Full loader: LeRobotDataset construction raises KeyError('stats') (phase 2).
- The other 22 listed episodes were not probed.
- Whether the frame decoded at 0.7333 s shows the instant recorded in row 0 is not testable from timestamps or pixels alone.
- Only LeRobot 0.3.3 was run for this v2.1 dataset.

Evidence: `evidence/r71_c1_f1_torchcodec.json`, `predictions/r71_c1_f1.json`; plan `shifted-queries/2026-11-22/plan/r71_c1_f1.json` (unchanged).

### `r94:c1:f1` — Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot@`41194807df07267235098a020ae4204f02262a93`, episode 0

Original assertion (phase 1, verbatim): 40 episode(s) whose timestamps do not start at 0: 0 (1.784e+09s), 10 (1.784e+09s), 20 (1.784e+09s), 30 (1.784e+09s), 40 (1.784e+09s) - video frames are looked up by timestamp, so decoding will miss

Identity: frozen_sources_ok ok, inputs_rehashed_ok ok, packages_match_phase2_freeze ok, torchcodec_environment_ok ok, upstream_sources_ok ok. FFmpeg reported by torchcodec: 7.1.1.

torchcodec metadata (first camera, approximate mode): average_fps 30 (= average_fps_from_header), num_frames 437 (from header; num_frames_from_content None), 640x360 h264.

| Bucket | Backend | Batches | Queries | Outcomes | Verdict | Max distance, ok (s) | Min distance, failing query (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | torchcodec | 68 | 188 | index_error 68 | supports_claim (index_error) | – | – |
| loader_derived/in_range | pyav_control | 68 | 188 | tolerance_error 68 | supports_claim (tolerance_error) | – | 1.78e+09 |
| loader_derived/in_range | phase3_pyav | 68 | 188 | tolerance_error 68 | supports_claim (tolerance_error) | – | 1.78e+09 |
| loader_derived/padded | torchcodec | 12 | 52 | index_error 12 | supports_claim (index_error) | – | – |
| loader_derived/padded | pyav_control | 12 | 52 | tolerance_error 12 | supports_claim (tolerance_error) | – | 1.78e+09 |
| loader_derived/padded | phase3_pyav | 12 | 52 | tolerance_error 12 | supports_claim (tolerance_error) | – | 1.78e+09 |
| nominal_grid/outside_episode | torchcodec | 144 | 144 | index_error 144 | supports_claim (index_error) | – | – |
| nominal_grid/outside_episode | pyav_control | 144 | 144 | tolerance_error 144 | supports_claim (tolerance_error) | – | 1.78e+09 |
| nominal_grid/outside_episode | phase3_pyav | 144 | 144 | tolerance_error 144 | supports_claim (tolerance_error) | – | 1.78e+09 |

`nominal_grid` rows are the labeled comparator (`raw + k/fps`, not a LeRobot query); their verdict strings are bucket-rule outputs and are not evidence for or against the claim.

Backend agreement (torchcodec vs the same-process pyav control):

- `loader_derived/in_range`: same outcome in 0/68 batches (torchcodec=index_error / pyav=tolerance_error: 68). Returned pts where both returned frames: 0 same, 0 different. Pixels there: 0 equal, 0 different.
- `loader_derived/padded`: same outcome in 0/12 batches (torchcodec=index_error / pyav=tolerance_error: 12). Returned pts where both returned frames: 0 same, 0 different. Pixels there: 0 equal, 0 different.
- `nominal_grid/outside_episode`: same outcome in 0/144 batches (torchcodec=index_error / pyav=tolerance_error: 144). Returned pts where both returned frames: 0 same, 0 different. Pixels there: 0 equal, 0 different.

pyav control vs frozen phase 3: 0 outcome changes in 224 batches, 0 returned-pts changes in 384 queries. Replays reproduced upstream in 224/224 torchcodec batches and 224/224 pyav batches.

Source model vs binary: 0 batch-outcome mismatches in 224, 0 index or returned-pts mismatches in 384 queries.

torchcodec rejections by mechanism:

- `loader_derived/in_range: index_error`: 68 batches; 0 contain a negative index wrapped to the end of the video, 68 an index at or above nb_frames (437). Example (observation.images.front, first, no_delta): queries ['1.784332e+09'] → raw index [53529945600] → index [53529945600] → returned pts ['–'].
- `loader_derived/padded: index_error`: 12 batches; 0 contain a negative index wrapped to the end of the video, 12 an index at or above nb_frames (437). Example (observation.images.front, first, diffusion): queries ['1.784332e+09', '1.784332e+09'] → raw index [53529945600, 53529945600] → index [53529945600, 53529945600] → returned pts ['–', '–'].
- `nominal_grid/outside_episode: index_error`: 144 batches; 0 contain a negative index wrapped to the end of the video, 144 an index at or above nb_frames (437). Example (observation.images.front, first, raw_plus_-4_over_fps): queries ['1.784332e+09'] → raw index [53529945596] → index [53529945596] → returned pts ['–'].

Default backend after all batches: `get_safe_default_codec()` = `torchcodec`, outcome index_error (RuntimeError).

Conclusion (scoped): torchcodec shifted loader-derived lookup is **confirmed** via index_error for LeRobot 0.3.3 decode_video_frames(backend='torchcodec', tolerance_s=1e-4) with torchcodec 0.5 and FFmpeg 7.1.1 (PyAV-bundled), direct decoder, frozen phase-3 loader-derived query sets, episode 0, anchors first/middle/worst_grid/last, 4 camera(s). Full loader: unresolved: in 0.3.3 LeRobotDataset construction raises check_timestamps_sync ValueError (phase 2), so __getitem__ is never reached.

Mechanism note: the lookup is rejected under both backends, but differently. torchcodec raises `RuntimeError: Invalid frame index=53529945600 ... must be less than 437` from `validateFrameIndex` before LeRobot's tolerance assertion runs. pyav raises the tolerance `AssertionError`. The comparator's role is unchanged, and the exception class a caller would see differs by backend. No frames are returned, so no pixel comparison exists.

Remaining gaps (unresolved):

- Comparator only: episode 0 and LeRobot 0.3.3; the other 39 listed episodes were not probed.
- Full loader: construction raises check_timestamps_sync ValueError in 0.3.3 (phase 2), before any lookup.
- How callers react to the torchcodec RuntimeError instead of the pyav AssertionError is not tested.
- Epoch-scale timestamps under 0.4.0/v3.0 are not tested.

Evidence: `evidence/r94_c1_f1_torchcodec.json`, `predictions/r94_c1_f1.json`; plan `shifted-queries/2026-11-22/plan/r94_c1_f1.json` (unchanged).

## Adversarial review

- **Changed inputs.** Every input was re-hashed against the phase-3 plan record before any batch ran. Phase 3 had checked those records against the phase-2 ledger and the Hub snapshot revision. Every frozen phase-1/2/3 artifact used here was re-hashed too (`frozen_sources_ok`). FineART data and videos still have no frozen-audit LFS hash, so their identity rests on the phase-2 ledger and the snapshot revision, as in phase 3.
- **Changed dependencies.** The Python package set equals the phase-2 freeze in both venvs, and the LeRobot source hashes equal phase 3's. The only addition is FFmpeg 7.1.1 shared libraries on `LD_LIBRARY_PATH`, taken from PyAV's own wheel by soname aliases, which makes this a distinct environment. The pyav control in that environment reproduced every phase-3 outcome and every returned pts (0 drift in 1056 queries), so at these queries the added library path did not change the pyav results. The pre-freeze smoke test wrote 8 bytecode caches into the v2 venv (`evidence/venv_writes.json`); no source or package file changed.
- **Unverifiable binary.** The installed torchcodec `.so` files cannot be matched to the v0.5.0 C++ source. Their behaviour matched the source model's preregistered prediction on all 616 batches and 1056 queries (index, returned pts and outcome). That supports the source reading on these streams. A post-hoc demux (`evidence/stream_facts_posthoc.json`) shows every probed stream is H.264 High profile with B-frames (packet pts out of decode order), so presentation-order reordering was exercised. Untested: variable frame rate, a missing `nb_frames` header, seek-mode `exact`, and a header `r_frame_rate` that differs from the true frame spacing (the synthetic tests show torchcodec then indexes with the guessed rate).
- **Query drift.** The plans are phase 3's preregistered files, byte-identical (hash-checked). The tests assert that every evidence batch equals its plan batch, field by field, including every query value. No query was regenerated, added, dropped or chosen after seeing an outcome, and the predictions were committed before execution.
- **Harness fidelity.** Independent replays reproduced the upstream outcome, and on success the output tensor, in every batch for both backends. The torchcodec replay uses a fresh decoder, whereas 0.4.0 upstream uses its per-path cache. Their agreement shows no cache-order effect at these queries.
- **Pixels.** Every query where both backends returned frames had the same pts and bit-identical uint8 pixels, so the neighbour-frame investigation never triggered. Three conditions produced this: the same FFmpeg binaries, H.264 yuv420p input, and widths that are multiples of 32 (640 and 256), so torchcodec used swscale and not its filtergraph fallback. A different FFmpeg build, pixel format or width could produce pixel differences. Those are not tested here, and frame equality says nothing about semantic alignment.
- **Mechanism versus claim.** The r94 confirmation under torchcodec rests on index validation, not on the tolerance check. It is reported as `confirmed via index_error`, not as a tolerance confirmation. The FineART and droid refutations rest on all loader-derived batches decoding within tolerance under both backends.
- **Float32 margin.** The loader-derived success distances are unchanged across backends (FineART up to 1.53e-5 s, droid 0), because both backends returned the same frames. The phase-3 caveat stands: longer timelines or large v3 `from_timestamp` offsets would shrink the margin, and that is untested.
- **Broader-than-evidence conclusions.** Each verdict names its release, backend, query class, episode and anchors. Untested: LeRobot 0.5.1, other episodes, non-default configs, GPU decoding, DataLoader workers, the full loader and training. Default-backend selection was verified only in 0.3.3 and 0.4.0. Nothing here is aggregated into an accuracy figure or used to set a severity cutoff.

## Limits

- Direct-decoder results only. They never establish LeRobotDataset, DataLoader or training behaviour.
- Decoding is not semantic frame-to-row alignment. FineART's declared 30 fps vs observed 90 Hz and droid's row/video alignment question stay separate and untested here.
- One episode per case, four anchors, default policy configs from phase 3. Untested episodes are not validated.
- Releases 0.3.3 (droid, r94) and 0.4.0 (FineART) only; 0.5.1 and other torchcodec/FFmpeg builds are untested. The FFmpeg used is PyAV 15.1.0's bundled 7.1.x build; a system or conda FFmpeg 7 may decode pixels differently.
- No severity cutoff, product certification or aggregate accuracy figure is derived.

## Reproduce

See [METHOD.md](METHOD.md#reproduce). Inputs are not committed. Evidence files record input hashes and the outcome of every batch for both backends.

# Shifted-timestamp video lookup (2026-11-22)

This report tests the shifted-query consequence that [phase 2](../../loader-verification/2026-11-22/report.md) left open for FineART (`r1:c1:f0`). It runs the same query class on droid (`r71:c1:f1`) and uses `r94:c1:f1` as an epoch-scale comparator. It does not revise the frozen [phase-1 audit](../../audits/2026-11-22/audit.md) or the phase-2 report; the new results are listed separately below. Method: [METHOD.md](METHOD.md). Severity proposal (not implemented): [severity-proposal.md](severity-proposal.md).

Frozen audit SHA-256 `d1b13333442049e09b942d7aca0ad7b9824ca24dd9dff71c6732160593b66f40`. Phase-2 expectations SHA-256 `ee6d4f27e12e184d310fee484bcb7c162247e36a9f83424645f7ce9ce51a1b91`. Preregistration SHA-256 `63778776ba2b4c02f64cfa5fe91c2e3af23b2f6344b2916e826e0aca454ab4bc`. The preregistration was committed before any probe ran on pinned inputs.

**All results are direct-decoder results.** `LeRobotDataset` was not constructed or patched for any case, and the full-loader consequence stays unresolved for all three.

## Results, kept separate from the original reports

| Case | Phase 1 | Phase 2 (unchanged) | Shifted loader-derived lookup (new) | Loader / backend | Full loader |
| --- | --- | --- | --- | --- | --- |
| `r1:c1:f0` (primary) | unresolved (WARN) | unresolved | **refuted** | 0.4.0 / pyav | unresolved |
| `r71:c1:f1` (primary) | unresolved (FAIL) | refuted | **refuted** | 0.3.3 / pyav | unresolved |
| `r94:c1:f1` (comparator) | unresolved (FAIL) | confirmed | **confirmed** | 0.3.3 / pyav | unresolved |

"Refuted" and "confirmed" apply only to the executed query class, release, backend, episode and anchors named in each case. They are not verdicts on the original findings as a whole.

## What the loader queries

These come from the installed source, with hashes in `expectations.json` (paths under `site-packages/lerobot/`):

- `datasets/factory.py::resolve_delta_timestamps` sets `delta_timestamps = [i / fps for i in cfg.observation_delta_indices]` for every `observation.*` key, including video keys. `LeRobotDataset` defaults to `delta_timestamps=None`, which queries only the current timestamp.
- `get_delta_indices` gives `round(d * fps)` row offsets, and `_get_query_indices` clamps them to the episode and sets `<key>_is_pad`. The video queries are then the **stored timestamps of the clamped rows** (`_get_query_timestamps`), converted to float32 by `hf_transform_to_torch`. Release 0.4.0 also adds `from_timestamp`.
- Phase 2's `raw + k/30` is therefore not a LeRobot query. It is kept below as a labeled `nominal_grid` comparator.
- Release 0.3.3 calls `check_timestamps_sync` at construction; 0.4.0 does not call it anywhere. In both releases the pyav decoder requires `|float32(q) - float32(pts)| < 1e-4` for every query in a batch.
- The default backend is torchcodec whenever its module spec exists. libtorchcodec failed to load in both environments, so only pyav was exercised.

## Per-case results

### `r1:c1:f0` — lerobot/FineART@`15746b76ddc21ffd38d73888fefe2e7a1496385b`, episode 40455

Original assertion (phase 1, verbatim): 1 episode(s) off the 30 fps grid (episodes 40455): frame spacing deviates up to 67% of a period; LeRobot's default tolerance_s (1e-4 s) rejects these - needs tolerance_s >= 0.0222

Identity: inputs re-hashed against the phase-2 ledger and Hub snapshot revision: ok. Upstream sources match the preregistered hashes: ok. Packages differing from the phase-2 freeze: 0. The frozen audit holds LFS hashes for 0/7 inputs; the rest are checked against the phase-2 ledger only.

Data: 26946 rows, declared fps 30, stored `float32` (info `float32`), first/last 0/299.389 s. Video (first camera): 26947 frames at 90 fps, pts 0–299.4 s. Anchors: {'first': 0, 'last': 26945, 'middle': 13473, 'worst_grid': 1458}.

Conformance: delta rows equal upstream `get_delta_indices` and pass `check_delta_timestamps` for every config. The float32 query timestamps match upstream `hf_transform_to_torch` (0 mismatches in 80 batches). No 0.4.0 `load_hf_dataset` replay was run (it would hit the language dtype).

| Bucket | Batches | Queries | ok | tolerance error | other | Max distance, ok (s) | Min distance, failing query (s) | Bucket rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | 68 | 188 | 68 | 0 | 0 | 1.53e-05 | – | all within tolerance |
| loader_derived/padded | 12 | 52 | 12 | 0 | 0 | 0 | – | all within tolerance |
| nominal_grid/in_range | 108 | 108 | 108 | 0 | 0 | 3.05e-05 | – | all within tolerance |
| nominal_grid/outside_episode | 36 | 36 | 0 | 36 | 0 | – | 0.0111 | tolerance failure observed |

Padding: every padded query reads the boundary row (first or last) and decodes that boundary frame; `is_pad` marks it: yes. Effective versus intended offsets at the middle anchor (tdmpc, s): 0/0, 0.01111/0.03333, 0.02222/0.06667, 0.03334/0.1, 0.04445/0.1333, 0.05556/0.1667.

Source note: LeRobot 0.4.0 has no check_timestamps_sync call: spacing is never validated against tolerance_s, so the decoder is the only place tolerance_s can reject a timestamp (datasets/lerobot_dataset.py, sha256 2fc7bbd5...).

Timeline note: info.json declares 30 fps while rows and all four videos are 90 Hz. resolve_delta_timestamps(i/30) maps to row offset i, which spans i/90 s: about one third of the window each config intends.

Failing batches (36; full query, returned-pts and distance lists are in `report.json`): nominal_grid/outside_episode: 36 (anchors first, last; configs raw_plus_-1_over_fps, raw_plus_-2_over_fps, raw_plus_-3_over_fps, raw_plus_-4_over_fps, raw_plus_1_over_fps, raw_plus_2_over_fps, raw_plus_3_over_fps, raw_plus_4_over_fps, raw_plus_5_over_fps).

Conclusion (scoped): shifted loader-derived lookup is **refuted** for LeRobot 0.4.0 decode_video_frames(backend='pyav', tolerance_s=1e-4), direct decoder, loader-derived query sets for default diffusion/vqbet/tdmpc/smolvla configs plus no-delta control, episode 40455, anchors first/middle/worst_grid/last, 4 camera(s). Full loader: unresolved (LeRobotDataset construction fails in 0.4.0 and 0.5.1 on source dtype 'language' (phase 2). Not bypassed, so __getitem__ was never exercised.) Default torchcodec backend: unresolved (dependency failure).

Remaining gaps:

- The full-loader query path is still blocked by the language dtype; this result does not establish that training loads.
- Default torchcodec backend: libtorchcodec failed to load (dependency), so it is untested. torchcodec maps timestamps to frame indices using average_fps, a different rule.
- Only episode 40455, four anchors, default configs. Not tested in 0.5.1.
- The 1/3 window compression's effect on a trained policy is not measured.

Evidence: `plan/r1_c1_f0.json`, `evidence/r1_c1_f0_probe.json`, `evidence/r1_c1_f0_video_facts.json`

### `r71:c1:f1` — azsdsadasdsadasd/droid_filter_lerobot@`7415f67330af40953ea52aa61a74788c75e1a8d6`, episode 0

Original assertion (phase 1, verbatim): 23 episode(s) whose timestamps do not start at 0: 0 (0.7333s), 2335 (1s), 11676 (1.067s), 14011 (0.5333s), 16346 (1.733s) - video frames are looked up by timestamp, so decoding will miss

Identity: inputs re-hashed against the phase-2 ledger and Hub snapshot revision: ok. Upstream sources match the preregistered hashes: ok. Packages differing from the phase-2 freeze: 0. The frozen audit holds LFS hashes for 1/5 inputs; the rest are checked against the phase-2 ledger only.

Data: 192 rows, declared fps 15, stored `float32` (info `float32`), first/last 0.733333/13.4667 s. Video (first camera): 213 frames at 15 fps, pts 0–14.1333 s. Anchors: {'first': 0, 'last': 191, 'middle': 96, 'worst_grid': 116}.

Conformance: delta rows equal upstream `get_delta_indices` and pass `check_delta_timestamps` for every config. The float32 query timestamps match upstream `hf_transform_to_torch` (0 mismatches in 60 batches). The 0.3.3 `load_dataset` replay gave 0 mismatches.

| Bucket | Batches | Queries | ok | tolerance error | other | Max distance, ok (s) | Min distance, failing query (s) | Bucket rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | 51 | 141 | 51 | 0 | 0 | 0 | – | all within tolerance |
| loader_derived/padded | 9 | 39 | 9 | 0 | 0 | 0 | – | all within tolerance |
| nominal_grid/in_range | 81 | 81 | 81 | 0 | 0 | 9.54e-07 | – | all within tolerance |
| nominal_grid/outside_episode | 27 | 27 | 27 | 0 | 0 | 9.54e-07 | – | all within tolerance |

Padding: every padded query reads the boundary row (first or last) and decodes that boundary frame; `is_pad` marks it: yes. Effective versus intended offsets at the middle anchor (tdmpc, s): 0/0, 0.06667/0.06667, 0.1333/0.1333, 0.2/0.2, 0.2667/0.2667, 0.3333/0.3333.

Source note: 0.3.3 runs check_timestamps_sync at construction, but this check is on spacing, and droid spacing is on the 1/15 s grid. Phase 2's construction failure happened earlier, on episode stats.

Timeline note: Rows span 0.7333 to 13.4667 s; the videos span 0 to 14.1333 s (213 frames for 192 rows). Lookups succeed because the video timeline covers the row timestamps. Whether the frame at 0.7333 s shows the same instant as row 0 cannot be determined from timestamps.

Conclusion (scoped): shifted loader-derived lookup is **refuted** for LeRobot 0.3.3 decode_video_frames(backend='pyav', tolerance_s=1e-4), direct decoder, loader-derived query sets for default diffusion/vqbet/tdmpc/smolvla configs plus no-delta control, episode 0, anchors first/middle/worst_grid/last, 3 camera(s). Full loader: unresolved (LeRobotDataset construction raises KeyError('stats') on episode metadata (phase 2). Not bypassed.) Default torchcodec backend: unresolved (dependency failure).

Remaining gaps:

- The other 22 listed episodes were not probed.
- Image-content alignment between rows and video frames is untested.
- Full loader blocked by KeyError('stats'); default torchcodec backend untested (dependency).

Evidence: `plan/r71_c1_f1.json`, `evidence/r71_c1_f1_probe.json`, `evidence/r71_c1_f1_video_facts.json`

### `r94:c1:f1` — Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot@`41194807df07267235098a020ae4204f02262a93`, episode 0

Original assertion (phase 1, verbatim): 40 episode(s) whose timestamps do not start at 0: 0 (1.784e+09s), 10 (1.784e+09s), 20 (1.784e+09s), 30 (1.784e+09s), 40 (1.784e+09s) - video frames are looked up by timestamp, so decoding will miss

Identity: inputs re-hashed against the phase-2 ledger and Hub snapshot revision: ok. Upstream sources match the preregistered hashes: ok. Packages differing from the phase-2 freeze: 0. The frozen audit holds LFS hashes for 1/6 inputs; the rest are checked against the phase-2 ledger only.

Data: 437 rows, declared fps 30, stored `float64` (info `float64`), first/last 1.78433e+09/1.78433e+09 s. Video (first camera): 437 frames at 30 fps, pts 0–14.5333 s. Anchors: {'first': 0, 'last': 436, 'middle': 218, 'worst_grid': 13}.

Conformance: delta rows equal upstream `get_delta_indices` and pass `check_delta_timestamps` for every config. The float32 query timestamps match upstream `hf_transform_to_torch` (0 mismatches in 80 batches). The 0.3.3 `load_dataset` replay gave 0 mismatches.

| Bucket | Batches | Queries | ok | tolerance error | other | Max distance, ok (s) | Min distance, failing query (s) | Bucket rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loader_derived/in_range | 68 | 188 | 0 | 68 | 0 | – | 1.78e+09 | tolerance failure observed |
| loader_derived/padded | 12 | 52 | 0 | 12 | 0 | – | 1.78e+09 | tolerance failure observed |
| nominal_grid/outside_episode | 144 | 144 | 0 | 144 | 0 | – | 1.78e+09 | tolerance failure observed |

Padding: every padded query reads the boundary row (first or last) and decodes that boundary frame; `is_pad` marks it: yes. Effective versus intended offsets at the middle anchor (tdmpc, s): 0/0, 0/0.03333, 0/0.06667, 0/0.1, 0/0.1333, 0/0.1667.

Source note: hf_transform_to_torch converts the float64 epoch timestamps to float32. Every value in the episode rounds to 1784331520.0 (the float32 spacing at this magnitude is 128 s), so all delta windows collapse to one query value.

Timeline note: Video pts run 0 to 14.53 s; queries are about 1.784e9 s. Because the episode's float32 time range also collapses to one value, every nominal query is classed as outside the episode, which is an artifact of the preregistered rule.

Failing batches (224; full query, returned-pts and distance lists are in `report.json`): loader_derived/in_range: 68 (anchors first, last, middle, worst_grid; configs diffusion, no_delta, smolvla, tdmpc, vqbet); loader_derived/padded: 12 (anchors first, last; configs diffusion, tdmpc, vqbet); nominal_grid/outside_episode: 144 (anchors first, last, middle, worst_grid; configs raw_plus_-1_over_fps, raw_plus_-2_over_fps, raw_plus_-3_over_fps, raw_plus_-4_over_fps, raw_plus_1_over_fps, raw_plus_2_over_fps, raw_plus_3_over_fps, raw_plus_4_over_fps, raw_plus_5_over_fps).

Conclusion (scoped): shifted loader-derived lookup is **confirmed** for LeRobot 0.3.3 decode_video_frames(backend='pyav', tolerance_s=1e-4), direct decoder, loader-derived query sets for default diffusion/vqbet/tdmpc/smolvla configs plus no-delta control, episode 0, anchors first/middle/worst_grid/last, 4 camera(s). Full loader: unresolved (In 0.3.3, LeRobotDataset construction raises check_timestamps_sync ValueError (phase 2), so __getitem__ is never reached.) Default torchcodec backend: unresolved (dependency failure).

Remaining gaps:

- Comparator only: 0.3.3 and episode 0 only; the other 39 listed episodes were not probed.
- Epoch-scale timestamps under 0.4.0/v3.0 are not tested.

Evidence: `plan/r94_c1_f1.json`, `evidence/r94_c1_f1_probe.json`, `evidence/r94_c1_f1_video_facts.json`

## Adversarial review

- **Cherry-picked queries.** The offsets come from a rule: every default config in the release whose `observation_delta_indices` is not None, plus a no-delta control. `plan.py` rescans all policy configs and refuses a mismatch. The anchors are fixed, every camera is probed, and the complete query set was committed in the preregistration commit before execution. The 4 anchors are not every frame, and for droid and r94 the worst-grid anchor deviates only slightly.
- **Padding behavior.** Clamping and `is_pad` follow `_get_query_indices` in both releases. Padded queries re-decode the boundary frame; the loader never queries outside the episode. The failing FineART `nominal_grid/outside_episode` queries (before 0 s or past the last pts) correspond to no loader query, and they are not counted as evidence for the claim. `is_pad` itself was not observed through `LeRobotDataset`. With `episodes=[...]`, the 0.4.0 index semantics (global `dataset_from_index` against the loaded table) were not exercised.
- **Revision drift.** Each input was re-hashed against the phase-2 ledger, its cache snapshot path names the pinned 40-character revision, and the frozen LFS hash was checked where one exists. FineART data and videos have no frozen-audit LFS hash, so their identity rests on the phase-2 ledger and the snapshot revision. The upstream source files and the package set match the phase-2 records.
- **Harness fidelity.** For each batch, an independent VideoReader replay reproduced the upstream pass or fail, and its frames were tensor-equal to the upstream output on every success (0 discrepancies). The float32 query timestamps match upstream `hf_transform_to_torch`.
- **Float32 margin.** FineART successes reach distances of 1.53e-5 s (loader-derived) and 3.05e-5 s (nominal), on the order of float32 resolution near 300 s (about 3.05e-5 s) and within a third of `tolerance_s`. Longer timelines, such as large v3 `from_timestamp` offsets, would shrink that margin. That was not tested.
- **Broader-than-evidence conclusions.** Every conclusion names its release, backend, query class, episode and anchors. Decoder success is not reported as loader or training success. Nothing here is aggregated into an accuracy figure.
- **Prediction check.** The preregistered predictions said loader-derived queries would decode for FineART and droid and fail for r94, and they did. They did not anticipate the FineART `nominal_grid` failures outside the episode range, or r94's empty `nominal_grid/in_range` bucket, which comes from the float32 range collapse. Both are reported as observed.

## Reproduce

See [METHOD.md](METHOD.md#reproduce). Inputs are not committed. The evidence files record their SHA-256 hashes and the outcome of every batch.

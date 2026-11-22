# Phase 2: pinned loader consequence verification (2026-11-22)

This report measures eleven unresolved assertions from the [frozen audit](../../audits/2026-11-22/audit.md). It does not certify kinema or revise the phase-1 verdicts.

Frozen audit SHA-256: `d1b13333442049e09b942d7aca0ad7b9824ca24dd9dff71c6732160593b66f40`. Preregistered expectations SHA-256: `ee6d4f27e12e184d310fee484bcb7c162247e36a9f83424645f7ce9ce51a1b91`.

Scoped case results: **4 confirmed, 3 refuted, 4 unresolved**. These are neither product accuracy nor a training outcome measure.

LeRobot 0.3.3 tested v2.0/v2.1; 0.4.0 tested v3.0. FineART also received a 0.5.1 schema-compatibility probe. These versions follow their declared dataset-format support; full package versions and source hashes are in `evidence/environments.json`.

| Case | Check | Consequence tested | Result |
| --- | --- | --- | --- |
| `r67:c17:f0` | deployment-readiness | LeRobot 0.3.3 v2.0 episode-0 iteration | **unresolved** |
| `r79:c17:f0` | deployment-readiness | LeRobot 0.3.3 v2.0 dataset construction | **unresolved** |
| `r256:c2:f0` | dropped-frames | LeRobot 0.3.3 v2.0 episode-0 item and video access | **refuted** |
| `r75:c12:f4` | episode-file-pointers | LeRobot 0.4.0 v3.0 episode-2035 selection from metadata-named file | **confirmed** |
| `r36:c12:f1` | episode-file-pointers | LeRobot 0.4.0 v3.0 episode-129 selection from metadata-named file | **confirmed** |
| `r94:c1:f1` | timestamp-grid | LeRobot 0.3.3 v2.0 episode-0 construction and direct pyav video lookup | **confirmed** |
| `r1:c1:f0` | timestamp-grid | LeRobot 0.4.0 v3.0 direct pyav video lookup at episode-40455 timestamps | **refuted** |
| `r261:c2:f4` | dropped-frames | LeRobot 0.4.0 v3.0 episode-59 item and hand_right lookup | **confirmed** |
| `r51:c17:f1` | deployment-readiness | LeRobot 0.3.3 v2.0 episode-0 iteration | **unresolved** |
| `r71:c1:f1` | timestamp-grid | LeRobot 0.3.3 v2.1 direct pyav video lookup on episode 0 | **refuted** |
| `r15:c17:f0` | deployment-readiness | LeRobot 0.3.3 v2.0 episode-0 iteration | **unresolved** |

## Per-case observations

### `r67:c17:f0` — unresolved

Source: `jesbu1/libero_90_lerobot@fa5ab6b7cdcab51fdfd88c8f216c00426508ecde`. Original assertion: observation.state: no normalization stats (meta/stats.json or per-episode stats) - the policy cannot normalize inputs/outputs at inference

Observed: 98/98 episode rows selected; first, middle and last item access succeeded. Source has no stats file, but declares state/actions and no observation.state.

Reason: A loader returning raw frames does not test the stated inference-policy normalization consequence. No policy configuration or checkpoint is specified; the named feature is absent.

Remaining gap: Test a specified policy/checkpoint and its feature mapping and normalization path.

Evidence: `evidence/r67_c17_f0.json`, `evidence/r67_c17_f0_loader.json`

### `r79:c17:f0` — unresolved

Source: `easyminnn/exumi-lerobot@334bf5334477e613c7fef5774247257e86a7a95b`. Original assertion: observation.state: no normalization stats (meta/stats.json or per-episode stats) - the policy cannot normalize inputs/outputs at inference

Observed: Pinned info.json omits features; construction raises KeyError(features) before episode access. No stats file is present.

Reason: The observed schema failure is separate from the asserted missing-stats inference consequence. No normalization path was reached.

Remaining gap: Identify a supported feature schema and specific policy/checkpoint before an inference test.

Evidence: `evidence/r79_c17_f0.json`, `evidence/r79_c17_f0_loader.json`

### `r256:c2:f0` — refuted

Source: `nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PnPCupToDrawerClose@09c6de8af50168090e7e9cc01e1ec3bce788de24`. Original assertion: frame_index column missing from the frame data - a default LeRobot feature that loaders and tools expect; gap/duplicate-index checks not evaluated (metadata length and video evidence are still checked)

Observed: Pinned parquet omits frame_index; all 229 rows were selected, and first, middle and last item/video access succeeded. Returned feature keys omit frame_index.

Reason: The default dataset path tested does not require frame_index to index or access these frames.

Remaining gap: Other downstream tools that explicitly require frame_index were not tested.

Evidence: `evidence/r256_c2_f0.json`, `evidence/r256_c2_f0_loader.json`

### `r75:c12:f4` — confirmed

Source: `BitRobot/berkeley-frodobots-lerobot-7k@096912cad3512e29d95dd809af7cb9fc020e4de7`. Original assertion: episode 2035: metadata points at data/chunk-000/file-006.parquet but its rows are not there; LeRobot loads only the metadata-named file, so the episode silently drops out of training

Observed: Pinned catalog names file-006 and length 71; selected loader table had 751774 rows from that shard but zero rows labeled episode 2035.

Reason: The normal episode-selection path does not expose the catalogued episode from the metadata-named file.

Remaining gap: No training run; cannot establish whether a trainer silently omits it, errors, or uses another representation.

Evidence: `evidence/r75_c12_f4.json`, `evidence/r75_c12_f4_loader.json`

### `r36:c12:f1` — confirmed

Source: `VLABench/vlabench_primitive_ft_lerobot_video@9846a2f6bead3873251dc4fe3079359d57326b7c`. Original assertion: episode 129: metadata points at data/chunk-000/file-101.parquet but its rows are not there; LeRobot loads only the metadata-named file, so the episode silently drops out of training

Observed: Pinned catalog names file-101 and length 181; selected loader table had 85 rows from that shard but zero rows labeled episode 129.

Reason: The normal episode-selection path does not expose the catalogued episode from the metadata-named file.

Remaining gap: No training run; cannot establish whether a trainer silently omits it or errors.

Evidence: `evidence/r36_c12_f1.json`, `evidence/r36_c12_f1_loader.json`

### `r94:c1:f1` — confirmed

Source: `Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot@41194807df07267235098a020ae4204f02262a93`. Original assertion: 40 episode(s) whose timestamps do not start at 0: 0 (1.784e+09s), 10 (1.784e+09s), 20 (1.784e+09s), 30 (1.784e+09s), 40 (1.784e+09s) - video frames are looked up by timestamp, so decoding will miss

Observed: Default dataset construction rejects timestamp spacing after float32 conversion of epoch-scale values. Direct decode_video_frames at pinned first/middle/last raw timestamps failed for all four cameras (12/12); video times were about 0–15 s versus raw timestamps about 1.784e9 s.

Reason: The asserted video lookup miss is reproduced on the affected pinned episode with the relevant decoder, although dataset construction fails earlier.

Remaining gap: The other 39 named episodes were not probed; the first load failure is timestamp validation rather than a video lookup error.

Evidence: `evidence/r94_c1_f1.json`, `evidence/r94_c1_f1_loader.json`, `evidence/r94_c1_f1_decoder.json`

### `r1:c1:f0` — refuted

Source: `lerobot/FineART@15746b76ddc21ffd38d73888fefe2e7a1496385b`. Original assertion: 1 episode(s) off the 30 fps grid (episodes 40455): frame spacing deviates up to 67% of a period; LeRobot's default tolerance_s (1e-4 s) rejects these - needs tolerance_s >= 0.0222

Observed: Pinned episode has 26946 rows, about 1/90-s spacing, and maximum 30-fps grid deviation 0.022225 s. Direct decoder returned frames within default 1e-4-s tolerance for first/middle/worst-grid/last positions across all four cameras (16/16). LeRobotDataset construction in 0.4.0 and 0.5.1 fails earlier on source dtype language.

Reason: Off-30-fps spacing did not make video lookup reject the tested frames; the loader does not reach timestamp access because of an independent schema incompatibility.

Remaining gap: Full dataset iteration with this custom language schema remains unsupported by the tested releases; no inference about other loaders.

Evidence: `evidence/r1_c1_f0.json`, `evidence/r1_c1_f0_loader_0.4.0.json`, `evidence/r1_c1_f0_loader_0.5.1.json`, `evidence/r1_c1_f0_decoder.json`

### `r261:c2:f4` — confirmed

Source: `Traly/AgiBotWorld-Alpha-lerobot//task_392@b38fd11acfe54f06b8b73b217f1b7c5c89b2c552`. Original assertion: camera observation.images.hand_right: 1 episode(s) whose video segment is shorter than the data (e.g. episode 59: 3241 video frames, 9903 rows) - frame lookups past the segment end fail

Observed: Pinned catalog declares 3241 frames and hand_right video ending at 544.0667 s; selected shard contains 9903 rows labeled episode 59. First/middle hand_right lookups succeeded; final row timestamp 109.1667 s shifts to 545.2 s and hand_right lookup fails. Full item access at the final row first fails on head camera.

Reason: The specific hand_right frame access fails after the declared segment end on the affected pinned episode.

Remaining gap: Other cameras and training behavior beyond first/middle/last were not exhaustively tested; duplicate episode labels may be the underlying source issue.

Evidence: `evidence/r261_c2_f4.json`, `evidence/r261_c2_f4_loader.json`, `evidence/r261_c2_f4_hand_right.json`

### `r51:c17:f1` — unresolved

Source: `VLABench/vlabench_primitive_ft_lerobot@460a2a4dc0bf29bef92b3e195a0f4e87f2d026a6`. Original assertion: action: no normalization stats (meta/stats.json or per-episode stats) - the policy cannot normalize inputs/outputs at inference

Observed: 93/93 episode rows selected; first, middle and last item access succeeded. Source has no stats file; feature is actions, not action.

Reason: The asserted action inference normalization cannot be tested without a named action feature mapping and policy checkpoint.

Remaining gap: Test a specified policy/checkpoint and feature mapping.

Evidence: `evidence/r51_c17_f1.json`, `evidence/r51_c17_f1_loader.json`

### `r71:c1:f1` — refuted

Source: `azsdsadasdsadasd/droid_filter_lerobot@7415f67330af40953ea52aa61a74788c75e1a8d6`. Original assertion: 23 episode(s) whose timestamps do not start at 0: 0 (0.7333s), 2335 (1s), 11676 (1.067s), 14011 (0.5333s), 16346 (1.733s) - video frames are looked up by timestamp, so decoding will miss

Observed: Pinned first timestamp is 0.733333 s. Direct decoder returned frames at first/middle/last raw timestamps for all three cameras (9/9). Dataset construction separately raises KeyError(stats) in episode metadata.

Reason: A nonzero start did not make the tested timestamp-to-video lookups miss.

Remaining gap: The other 22 named episodes and complete dataset iteration remain untested; episode-stats schema blocks construction.

Evidence: `evidence/r71_c1_f1.json`, `evidence/r71_c1_f1_loader.json`, `evidence/r71_c1_f1_decoder.json`

### `r15:c17:f0` — unresolved

Source: `VLABench/vlabench_primitive_pretrain_lerobot@aaa042a8ffb56c931ad738bd220c0c944d8479e7`. Original assertion: observation.state: no normalization stats (meta/stats.json or per-episode stats) - the policy cannot normalize inputs/outputs at inference

Observed: 81/81 episode rows selected; first, middle and last item access succeeded. Source has no stats file; feature is state, not observation.state.

Reason: Raw frame loading cannot settle the stated inference normalization failure without a policy/checkpoint and feature mapping.

Remaining gap: Test a specified policy/checkpoint and feature mapping.

Evidence: `evidence/r15_c17_f0.json`, `evidence/r15_c17_f0_loader.json`

## Interpretation limits

The four deployment-readiness claims remain unresolved at the asserted inference-policy consequence. Three sources load raw episode items but use different feature names; one source fails earlier because `features` is absent. A loader opening data cannot establish whether a particular pretrained policy can normalize at inference. No policy checkpoint or inference configuration was specified.

The two file-pointer probes confirm that the affected episode has zero rows in the selected loader table. They do not observe a trainer, so the words ‘silently drops out of training’ remain untested. Cybernetic-Labs video lookup fails at the raw epoch-scale timestamps, but ordinary dataset construction rejects timestamp spacing first. The droid and FineART video lookup claims are contradicted by direct decoder probes, while their full dataset construction has separate schema blockers.

## Adversarial review

- **Selection bias:** All eleven unresolved findings in the four named checks were included; no case removed because of unavailable per-dataset pages or difficult loaders. The previously confirmed r413:c2:f2 is excluded by the frozen unresolved filter.
- **Shared assumptions:** The verifier imports no kinema detector code. It uses upstream LeRobot releases, raw parquet/MP4 and the frozen audit only to identify assertions and pins. A direct decoder probe cannot replace full dataset iteration.
- **Revision drift:** Every recovered path was fetched at a 40-character commit and SHA-256 hashed. Frozen info/stats and available LFS hashes were checked. No latest-branch data were substituted.
- **Incomplete access:** Frame probes use first/middle/last and claim-specific positions, not every frame. BitRobot and VLABench pointer cases inspect the selected loader table but do not run training. FineART and droid cannot complete dataset construction for independent schema reasons.
- **Weak labels:** No policy checkpoint is specified for four inference-normalization claims. Those remain unresolved despite successful raw frame loading in three cases. No training or real-world miss-rate claim is made.

## Reproduce

```sh
# Install the declared LeRobot releases in separate temporary Python environments (see evidence/environments.json).
# Recover only the listed metadata and affected episode files at their immutable revisions:
python loader-verification/2026-11-22/prepare.py CASE_ID --metadata
python loader-verification/2026-11-22/prepare.py CASE_ID --path RELATIVE_PATH
python loader-verification/2026-11-22/run_loader.py CASE_ID
python loader-verification/2026-11-22/make_report.py
```

The exact prepared paths, content hashes, operations, observed outputs and errors are retained in `report.json` and `evidence/`. Local episode and video bytes are not committed.

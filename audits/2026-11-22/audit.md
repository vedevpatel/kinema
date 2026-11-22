# Kinema assurance phase 1: frozen finding audit (2026-11-22)

This independently audits a sample of published assertions. It does **not** certify kinema, its scores, or the board.

Frozen board commit: `f701ff4bdd7be514644fa2769aa1bb22e979e4d6`; uncompressed board SHA-256: `7345742fcec55f5f494a25da65cc86d4ad90f507de51914546007a0cea603824`; board engine: `f9a1edd38b66`; main engine at freeze: `01fb98a42eb7`. The stored board engine differs from main, and no current detector output is used as truth.

Board generated `2026-11-21 17:46 UTC`; seed `kinema-assurance-phase1-2026-11-22-v1`. The frame contains 1839 WARN/FAIL finding instances on 203 datasets out of 414 board rows. Of 6834 total findings, 4995 were excluded ({'INFO': 4995}). Only INFO, PASS and SKIP findings excluded; no WARN/FAIL rows excluded for absent dataset pages.

Selection: two per check by fixed hash, covering both severities where available and preferring distinct datasets; remaining slots draw from FAIL findings. This deliberately oversamples rare checks and FAILs. The selection used only the frozen board JSON. No evidence or page availability influenced it.

Results: **9 confirmed, 1 false alarm, 30 unresolved** of 40 selected; unresolved/total = **30/40**. Cases span 39 datasets. No aggregate accuracy or real-world miss rate is inferred. Precision is not reported because most cases remain unresolved.

`audit.json` has every finding, pinned revision, commands, raw observations and reason. `board.json.gz` is the frozen selection source; `provenance.json` records resolved source commits and metadata hashes. `artifact_hashes.json` records 146 pinned paths with Hub LFS SHA-256 or Git blob IDs where exposed. `observations.json` preserves raw-source probes. See `RUBRIC.md` for the preregistered labels.

## By check

| Check | Confirmed | False alarm | Unresolved | Total |
| --- | ---: | ---: | ---: | ---: |
| action-saturation | 0 | 0 | 2 | 2 |
| calibration-drift | 0 | 0 | 2 | 2 |
| deployment-readiness | 0 | 0 | 4 | 4 |
| dropped-frames | 1 | 0 | 2 | 3 |
| duplicate-and-leakage | 0 | 0 | 2 | 2 |
| episode-file-pointers | 0 | 0 | 2 | 2 |
| idle-segments | 0 | 0 | 2 | 2 |
| nan-and-normalization | 0 | 0 | 2 | 2 |
| outlier-episodes | 0 | 0 | 2 | 2 |
| reward-done | 2 | 0 | 0 | 2 |
| rlds-mirror-drift | 0 | 0 | 2 | 2 |
| rlds-shard-inventory | 0 | 0 | 2 | 2 |
| stats-drift | 2 | 0 | 0 | 2 |
| stuck-actuator | 0 | 1 | 1 | 2 |
| task-strings | 2 | 0 | 0 | 2 |
| timestamp-grid | 0 | 0 | 3 | 3 |
| video-file-pointers | 2 | 0 | 0 | 2 |
| visual-quality | 0 | 0 | 2 | 2 |

## By dataset

| Dataset | Confirmed | False alarm | Unresolved | Total |
| --- | ---: | ---: | ---: | ---: |
| BitRobot/berkeley-frodobots-lerobot-7k | 0 | 0 | 1 | 1 |
| Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot | 0 | 0 | 1 | 1 |
| IPEC-COMMUNITY/bridge_orig_lerobot | 1 | 0 | 0 | 1 |
| MendelBeans/lerobot_dataset_put_pen_in_the_holder_20261118_215903 | 1 | 0 | 0 | 1 |
| OpenGraphLabs-Research/lerobot-simulation-koch-arm-to-the-shelf-01 | 1 | 0 | 0 | 1 |
| OpenMOSS-Team/robodojo_lerobot_v3.0 | 1 | 0 | 0 | 1 |
| RoboCOIN/Agilex_Cobot_Magic_heat_sandwich | 0 | 0 | 1 | 1 |
| RoboCOIN/Agilex_Cobot_Magic_storage_towel | 0 | 0 | 1 | 1 |
| RoboCOIN/Galbot_G1_bag_table_trash_1 | 0 | 1 | 0 | 1 |
| ShareLab-SII/thinking_droid_lerobot_output_qwen3vl | 1 | 0 | 0 | 1 |
| Traly/AgiBotWorld-Alpha-lerobot//task_392 | 0 | 0 | 1 | 1 |
| VLABench/vlabench_primitive_ft_lerobot | 0 | 0 | 1 | 1 |
| VLABench/vlabench_primitive_ft_lerobot_video | 0 | 0 | 1 | 1 |
| VLABench/vlabench_primitive_pretrain_lerobot | 0 | 0 | 1 | 1 |
| azsdsadasdsadasd/droid_filter_lerobot | 0 | 0 | 1 | 1 |
| easyminnn/exumi-lerobot | 0 | 0 | 1 | 1 |
| ethanCSL/robotwin_stack_blocks_two_lerobot_source_10 | 0 | 0 | 1 | 1 |
| jesbu1/bridge_v2_lerobot | 1 | 0 | 0 | 1 |
| jesbu1/libero_90_lerobot | 0 | 0 | 1 | 1 |
| lerobot/FineART | 0 | 0 | 1 | 1 |
| lerobot/behavior1k-task0045 | 0 | 0 | 1 | 1 |
| lerobot/fmb | 0 | 0 | 1 | 1 |
| lerobot/high_quality_folding | 0 | 0 | 1 | 1 |
| lerobot/iamlab_cmu_pickup_insert | 1 | 0 | 0 | 1 |
| lerobot/libero_plus | 0 | 0 | 1 | 1 |
| lerobot/stanford_robocook | 1 | 0 | 1 | 2 |
| lerobot/xarm_lift_medium_replay | 0 | 0 | 1 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PnPBottleToCabinetClose | 0 | 0 | 1 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PnPCupToDrawerClose | 0 | 0 | 1 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA | 0 | 0 | 1 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0007_picking_up_toys | 1 | 0 | 0 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0021_collecting_childrens_toys | 0 | 0 | 1 | 1 |
| nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0048_canning_food | 0 | 0 | 1 | 1 |
| rlds/bridge_data_msr | 0 | 0 | 1 | 1 |
| rlds/language_table | 0 | 0 | 1 | 1 |
| rlds/language_table_sim | 0 | 0 | 1 | 1 |
| rlds/vima_converted_externally_to_rlds | 0 | 0 | 1 | 1 |
| sunLry/robotwin-23task-lerobot | 0 | 0 | 1 | 1 |
| ygtxr1997/bridge_orig_lerobot | 0 | 0 | 1 | 1 |

## Separate miss exercise

The structural regression pack has 5 known-broken fixtures and 3 clean controls. Its test run: 47 passed; the adjudicated corpus: 11 passed. The FastUMI missing-video case and its v2.1 clean sibling are matched controls. `misses.json` lists each fixture's pinned revision and expected defects.

All structural cases were selected, trimmed and checked during detector development. The adjudicated corpus is also a regression guard for known cases. Neither is held out.

Reviewed tests/data/structural, tests/data/corpus and network regression cases in tests/test_real_datasets.py; each contains development-known cases or labels derived from prior kinema work. No genuinely independent, pinned broken and matched clean holdout was established in this phase.

No real-world miss rate is inferred from these fixtures.

## Adversarial review

- **Selection bias:** Two-per-check and FAIL fill overrepresent rare checks and FAIL findings; distinct-dataset preference changes prevalence. The two bridge task-string rows appear to be mirrors, so 39 dataset IDs are not 39 independent sources. No population precision estimate is claimed.
- **Shared assumptions:** Raw probes use source info.json path templates and published episode IDs to locate targets, but never kinema detection helpers. Metadata can be stale: the OpenGraph case was cross-checked against raw parquet. Heuristic arithmetic and source stats alone are not treated as proof of a defect.
- **Revision drift:** All 35 selected Hub dataset IDs resolved to the exact row commit; source metadata and available object IDs are frozen. Four RLDS cases have only non-immutable GCS stamps and stay unresolved. Main's engine 01fb98a42eb7 differs from the board's f9a1edd38b66; no main rerun adjudicates a stored row.
- **Weak labels:** The structural regression pack and corpus influenced development and cannot estimate misses. Most heuristic warnings lack video or task context. The one false alarm rejects the motor interpretation of a static quaternion component while preserving the static signal.
- **Page eligibility:** Sampling reads all WARN/FAIL findings in frozen data.json, not per-dataset pages. Missing pages cannot silently remove rows; the test runs the sampler with no page directory.

## Proposed next step

Open a separate detector-fix PR for the quaternion-as-actuator false alarm after preserving this evidence. For the assurance layer, pin complete GCS object generations, run the relevant LeRobot loader on pointer and timestamp cases, inspect selected videos and trajectories, and acquire genuinely held-out broken and matched clean examples with independent labels.

## False alarms

- `r216:c5:f1` stuck-actuator (RoboCOIN/Galbot_G1_bag_table_trash_1@e8a2977c4adb): Pinned feature schema names dimension 34 base_orient_quat_w, a base-orientation quaternion component rather than an actuator. Episode 279 raw values stay 0.99999887..1.0 while adjacent quat_z stays near zero. The static signal exists, but calling it a dead or disabled motor is false.

## Unresolved cases

- `r60:c6:f0` action-saturation (lerobot/xarm_lift_medium_replay@1da06eafab3a): Pinned revision and metadata recovered, but raw commands and actuator limits were not inspected; an exact maximum would not itself establish clipping.
- `r387:c6:f4` action-saturation (nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0021_collecting_childrens_toys@ea7ac0b68f87): Nested source revision recovered, but the selected action frames, physical limits and motion context were not independently examined.
- `r257:c3:f2` calibration-drift (nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA@09c6de8af501): No independent synchronized command/state analysis or action-semantics check; the reported offset cannot establish calibration failure.
- `r254:c3:f11` calibration-drift (nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PnPBottleToCabinetClose@09c6de8af501): No independent still-frame or gripper-semantics inspection; inferred unnamed gripper and offset interpretation remain unsupported.
- `r67:c17:f0` deployment-readiness (jesbu1/libero_90_lerobot@fa5ab6b7cdca): Complete pinned meta listing has no stats file, but raw feature names are state/actions rather than observation.state/action and the relevant LeRobot loader and normalization path were not exercised.
- `r79:c17:f0` deployment-readiness (easyminnn/exumi-lerobot@334bf5334477): Complete pinned meta listing has no stats file; info.json lacks a features map while parquet has observation.state. Loader behavior and whether inference can compute or supply stats were not tested.
- `r256:c2:f0` dropped-frames (nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim//LeRobot/gr1_unified.PnPCupToDrawerClose@09c6de8af501): Pinned episode-0 parquet schema lacks frame_index, but default LeRobot loader behavior and the claimed impact on tools were not exercised.
- `r121:c8:f0` duplicate-and-leakage (sunLry/robotwin-23task-lerobot@f69d7e56e291): Pinned revision recovered, but the paired action arrays, task context and video were not inspected; a shared scripted opening could be legitimate.
- `r69:c8:f0` duplicate-and-leakage (lerobot/libero_plus@f3f49f426d75): Pinned revision recovered, but both complete action trajectories and other modalities were not compared; identical actions alone would not establish duplicated recordings.
- `r75:c12:f4` episode-file-pointers (BitRobot/berkeley-frodobots-lerobot-7k@096912cad351): Pinned metadata points episode 2035 to file-006 and that file contains zero episode-2035 rows. The statement that the relevant LeRobot loader silently drops it was not tested; a separate unpacked representation may hold its rows.
- `r36:c12:f1` episode-file-pointers (VLABench/vlabench_primitive_ft_lerobot_video@9846a2f6bead): Pinned metadata points episode 129 to file-101 and that file contains zero episode-129 rows. The claimed LeRobot loader consequence was not tested.
- `r220:c4:f0` idle-segments (RoboCOIN/Agilex_Cobot_Magic_storage_towel@c39c1f754c5d): Pinned revision recovered; no independent action trajectory or video inspection establishes that the opening pause exists or is a data defect.
- `r225:c4:f0` idle-segments (RoboCOIN/Agilex_Cobot_Magic_heat_sandwich@f71cba34e53d): Pinned revision recovered; no independent trajectory or task/video inspection distinguishes intentional start/end pauses from a defect.
- `r390:c7:f8` nan-and-normalization (nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0048_canning_food@ea7ac0b68f87): Pinned source stats report dimension 117 near 1e-12 std, but raw frame distribution and actual normalization behavior were not checked; near constancy alone is not a defect.
- `r393:c7:f14` nan-and-normalization (lerobot/behavior1k-task0045@27458fab90ba): Pinned source stats report dimension 123 near 1e-12 std, but raw frame distribution and actual normalization behavior were not checked; near constancy alone is not a defect.
- `r153:c9:f6` outlier-episodes (lerobot/fmb@76c86f111d32): Pinned revision recovered, but duration/velocity metrics and task/video context were not independently evaluated; outlier arithmetic alone cannot establish bad data.
- `r127:c9:f25` outlier-episodes (lerobot/stanford_robocook@49337d017189): Pinned revision recovered, but the episode velocity trace and context were not independently inspected.
- `r288:c4:f0` rlds-mirror-drift (rlds/language_table_sim@gcs:3f7cd120): GCS board stamp is not an immutable object generation; the two historical bucket listings and shard bytes cannot be reconstructed from it.
- `r291:c4:f0` rlds-mirror-drift (rlds/language_table@gcs:dd64678c): GCS board stamp is not an immutable object generation; historical mirror completeness and byte counts cannot be established.
- `r366:c2:f0` rlds-shard-inventory (rlds/vima_converted_externally_to_rlds@gcs:ed19c7b5): GCS board stamp lacks immutable object generations and no complete historical shard listing is available.
- `r324:c2:f0` rlds-shard-inventory (rlds/bridge_data_msr@gcs:d1439b0e): GCS board stamp lacks immutable object generations and no complete historical shard listing is available.
- `r26:c5:f1` stuck-actuator (lerobot/high_quality_folding@c9eb858d4b84): Pinned revision recovered, but joint-10 trace, task and video were not inspected; a static joint in one episode need not be defective.
- `r94:c1:f1` timestamp-grid (Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot@41194807df07): Pinned episode-0 raw timestamp starts at 1784331549.834715 rather than 0, but all 40 listed episodes and the relevant LeRobot video lookup were not tested.
- `r1:c1:f0` timestamp-grid (lerobot/FineART@15746b76ddc2): Pinned revision recovered, but episode 40455 frame times and the LeRobot tolerance failure were not independently reproduced.
- `r48:c16:f0` visual-quality (ygtxr1997/bridge_orig_lerobot@5e1c02c29f21): Pinned revision recovered, but the named video frames were not decoded or visually inspected; no independent exposure judgment.
- `r377:c16:f1` visual-quality (ethanCSL/robotwin_stack_blocks_two_lerobot_source_10@2098d9e38351): Pinned revision recovered, but the named video frames were not decoded or visually inspected; saturation fraction alone cannot prove unusable exposure.
- `r261:c2:f4` dropped-frames (Traly/AgiBotWorld-Alpha-lerobot//task_392@b38fd11acfe5): Pinned nested revision recovered, but episode-59 video segment and data row counts were not independently inspected.
- `r51:c17:f1` deployment-readiness (VLABench/vlabench_primitive_ft_lerobot@460a2a4dc0bf): Complete pinned meta listing lacks stats, but raw feature is actions rather than action and the LeRobot loader/normalization path was not exercised.
- `r71:c1:f1` timestamp-grid (azsdsadasdsadasd/droid_filter_lerobot@7415f67330af): Pinned episode-0 raw timestamp starts at 0.733333 s rather than 0, but all 23 listed episodes and the relevant LeRobot video lookup were not tested.
- `r15:c17:f0` deployment-readiness (VLABench/vlabench_primitive_pretrain_lerobot@aaa042a8ffb5): Complete pinned meta listing lacks stats, but raw feature is state rather than observation.state and the LeRobot loader/normalization path was not exercised.

## Confirmed cases

- `r413:c2:f2` dropped-frames (MendelBeans/lerobot_dataset_put_pen_in_the_holder_20261118_215903@0c0e6528c6b3): Complete pinned metadata and data parquet listings show two episode-50 catalogue rows of length 358 and 360, with 358 and 360 corresponding raw rows (718 under the same episode ID). The published 358-versus-718 conflict is real; the underlying issue is an episode-ID collision.
- `r379:c15:f0` reward-done (lerobot/iamlab_cmu_pickup_insert@5806e1046ce1): Complete pinned data listing has one parquet; all 146241 raw next.reward values are 0, matching declared frame count. This confirms the recorded reward channel contains no varying reward signal, without judging other training uses.
- `r127:c15:f0` reward-done (lerobot/stanford_robocook@49337d017189): Complete pinned data listing has one parquet; all 112980 raw next.reward values are 0, matching declared frame count. This confirms the recorded reward channel contains no varying reward signal, without judging other training uses.
- `r177:c0:f3` stats-drift (OpenGraphLabs-Research/lerobot-simulation-koch-arm-to-the-shelf-01@e059e7da2d11): At the pinned revision, meta/stats.json dim 3 matches raw episode 0 (std 0.00323964), while raw episode 1 has mean -0.295504, outside stored range [-0.006419, 0.017326]. Stored normalization stats are inconsistent with raw data. The board's exact 200-episode aggregate was not independently recomputed.
- `r388:c0:f1` stats-drift (nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0007_picking_up_toys@ea7ac0b68f87): Pinned meta/stats.json dim 118 claims mean -96.8734 outside its own [-100.2373, -99.7850] range and std 2.9952 greater than half that range (0.2262). Raw episode 0 has 7712 values, mean -99.9488 and std 0.0703. Source stats are inconsistent with raw data.
- `r25:c14:f0` task-strings (jesbu1/bridge_v2_lerobot@3899a5d24770): Pinned meta/episodes.jsonl lists all nine named episodes with tasks equal to a one-element empty string list. This confirms absent language text in those episode records.
- `r4:c14:f0` task-strings (IPEC-COMMUNITY/bridge_orig_lerobot@0e9d76d07e9d): Pinned meta/episodes.jsonl lists all nine named episodes with tasks equal to a one-element empty string list. This confirms absent language text in those episode records; this source appears correlated with the jesbu1 mirror.
- `r202:c13:f1` video-file-pointers (ShareLab-SII/thinking_droid_lerobot_output_qwen3vl@9ab04fbc1211): Complete pinned root listing contains data and meta but no videos directory. info.json declares this camera and a videos/ path; meta/episodes.jsonl has 92233 episodes. The reported camera video files are absent from the source tree.
- `r12:c13:f1` video-file-pointers (OpenMOSS-Team/robodojo_lerobot_v3.0@d3670ddbe2b6): Complete pinned camera listing plus all 3500 metadata episode rows independently yields 83 referenced MP4 files absent and 1677 affected episodes, matching the finding.

## Reproduction

```sh
python3 scripts/assurance_sample.py docs/data.json audits/2026-11-22
.venv/bin/python scripts/assurance_probe.py audits/2026-11-22
.venv/bin/python scripts/assurance_observe.py audits/2026-11-22
.venv/bin/python scripts/assurance_artifacts.py audits/2026-11-22
python3 scripts/assurance_report.py audits/2026-11-22
```

The Hub and GCS can change availability; pinned Hub commits and the frozen observations preserve what this audit saw. The GCS rows are unresolved because the board stamps are not immutable object generations.

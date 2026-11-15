# tiered-v2: WARN weights re-anchored on harm table v1.1

Score model `tiered-v2` replaces `tiered-v1.1`. Only WARN-weight
memberships change. The band edges (100/10/0), FAIL cost (25), per-check
billing and the correlated-check waiver all stay as they were.

## (a) Source numbers: harm table v1.1

`docs/harm-table.json`: diffusion policy, `lerobot/pusht`, batch 32, 20k
optimizer steps, seeds 0/1/2, 50 eval episodes. Outcome: held-out action MSE
delta vs the clean arm under the same seed, as a % of clean. Rollout success
stays at the floor for every arm (clean 2%/4%/2%), so it cannot separate
defects in this protocol, and held-out MSE is the outcome metric.

| defect | check | MSE delta, mean | per seed (0/1/2) | v1, 5k steps (mean) |
| --- | --- | --- | --- | --- |
| scale_mismatch_0.5x_25pct | calibration-drift | **+1475.3%** | +2174.6 / +693.8 / +1557.4 | +665.1% |
| stuck_joint_30pct | stuck-actuator | **+697.8%** | +504.6 / +1003.7 / +585.2 | +544.6% |
| idle_start_30f | idle-segments | **+75.1%** | +129.2 / +82.1 / +13.8 | +331.7% |
| duplicates_20pct | duplicate-and-leakage | **-3.6%** | -2.7 / +0.9 / -8.9 | +3.9% |

## (b) Weight and tier changes

Tiers: A >= 100% (weight 10), B >= 10% (6), C < 10% (2).

| check | tiered-v1.1 | tiered-v2 | reason |
| --- | --- | --- | --- |
| calibration-drift | 10, A, measured (+347%, ridge BC v0) | 10, A, measured | Scale mismatch is the top killer, and it gets worse with training (+665% at 5k, +1475% at 20k). |
| stuck-actuator | 6, placeholder | **10, A, measured** | A solid #2 at both budgets (+545%, then +698%), and every seed is above 500%. |
| idle-segments | 10, A, measured (+881%, ridge BC v0) | **6, B, measured** | The harm shrinks with training (+332% at 5k, +75% at 20k). The diffusion policy learns past the padding. |
| duplicate-and-leakage | 2, C, measured (+0.1%, ridge BC v0) | 2, C, measured | Benign at both budgets (+3.9%, then -3.6%). |

Net change: stuck-actuator +4, idle-segments -4. calibration-drift and
duplicate-and-leakage keep their weights; only their evidence changes.

## (c) Score impact (local rescore simulation)

`scripts/leaderboard.py --rescore --out docs` ran on the 2026-11-15 board
(194 audited datasets, all stored under `tiered-v1.1`) and was then
reverted. Nothing from the simulation is committed.

- **43 of 194 scores change.** 32 go down and 11 go up, and every one moves by
  exactly 4 points. The board mean goes from 89.31 to 88.88.
- Down (-4, 32 datasets): a stuck-actuator WARN and no idle-segments WARN.
  26 of these have stuck-actuator as their only WARN among the four
  re-weighted checks.
- Up (+4, 11 datasets): an idle-segments WARN and no charged
  stuck-actuator. 8 have idle-segments alone, and 3 have stuck-actuator
  waived under idle-segments (see the open item below).
- Unchanged (13 datasets): both checks WARN and both are charged, so
  +4 - 4 = 0.
- The board has no score bands, so "crossing a threshold" here means moving
  across a 10-point decade. 12 datasets do that:
  - Down: FedorX8/bridge_v2_lerobot,
    enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot-past,
    lerobot/abc_130k_v3_val, lerobot/berkeley_rpt and
    rpwang/bimanual_assembly_insert_force_lerobot_mix854 (91 -> 87);
    sunLry/robotwin-23task-lerobot (92 -> 88);
    lerobot/taco_play (81 -> 77); jesbu1/bridge_v2_lerobot (73 -> 69).
  - Up: dadm022/lerobot_rgb_vla and lerobot/svla_so100_sorting (87 -> 91);
    lerobot/droid_1.0.1 (79 -> 83); IPEC-COMMUNITY/droid_lerobot (59 -> 63).
  - No dataset reaches or leaves the 0 floor.

**Open item (not changed here): the waiver now discounts the heavier
check.** The edge is (idle-segments -> stuck-actuator). Under v1.1 it waived
a 6 under a 10. Under v2 it waives a 10 under a 6. As a result, a dataset
whose dead episodes trip both checks scores 94, while a dataset with
stuck-actuator alone scores 90. Examples: IPEC-COMMUNITY/berkeley_mvp_lerobot
goes 90 -> 94 and lerobot/svla_so100_sorting goes 87 -> 91. Possible fixes
(each one is a separate decision): reverse the edge, or charge
max(upstream, downstream) when the waiver fires.

All 43 changed datasets:

| dataset | v1.1 | v2 | delta | WARNs on re-weighted checks |
| --- | --- | --- | --- | --- |
| IPEC-COMMUNITY/berkeley_mvp_lerobot | 90 | 94 | +4 | idle-segments, stuck-actuator (waived) |
| IPEC-COMMUNITY/droid_lerobot | 59 | 63 | +4 | calibration-drift, idle-segments |
| Project-IRA/TPSoSe2026_Dataset_Full_Merged_Final_LeRobot_SO101_V1 | 90 | 94 | +4 | idle-segments |
| dadm022/lerobot_rgb_vla | 87 | 91 | +4 | idle-segments |
| lerobot/FineART | 90 | 94 | +4 | idle-segments |
| lerobot/droid_1.0.1 | 79 | 83 | +4 | idle-segments, stuck-actuator (waived) |
| lerobot/robomme | 72 | 76 | +4 | idle-segments |
| lerobot/svla_so100_pickplace | 81 | 85 | +4 | idle-segments |
| lerobot/svla_so100_sorting | 87 | 91 | +4 | idle-segments, stuck-actuator (waived) |
| saaduddinM/rbo_oxe_base_droid_lerobot | 70 | 74 | +4 | calibration-drift, idle-segments |
| yinchenghust/robomme_lerobot | 90 | 94 | +4 | idle-segments |
| BitRobot/HIW-500-LeRobot | 85 | 81 | -4 | stuck-actuator |
| BitRobot/berkeley-frodobots-lerobot-7k | 85 | 81 | -4 | stuck-actuator |
| Cybernetic-Labs/openarm-packingbench-v2-groot-lerobot | 66 | 62 | -4 | stuck-actuator |
| FedorX8/bridge_v2_lerobot | 91 | 87 | -4 | stuck-actuator |
| IPEC-COMMUNITY/berkeley_fanuc_manipulation_lerobot | 94 | 90 | -4 | stuck-actuator |
| IPEC-COMMUNITY/iamlab_cmu_pickup_insert_lerobot | 94 | 90 | -4 | stuck-actuator |
| IPEC-COMMUNITY/jaco_play_lerobot | 78 | 74 | -4 | calibration-drift, stuck-actuator |
| IPEC-COMMUNITY/taco_play_lerobot | 79 | 75 | -4 | calibration-drift, duplicate-and-leakage, stuck-actuator |
| OpenGraphLabs-Research/lerobot-simulation-koch-arm-to-the-shelf-01 | 29 | 25 | -4 | duplicate-and-leakage, stuck-actuator |
| VLABench/vlabench_primitive_ft_lerobot_video | 86 | 82 | -4 | stuck-actuator |
| Voxel51/HIW-500-LeRobot-48ep | 88 | 84 | -4 | stuck-actuator |
| albertwilcox/bimesa-57-lerobot | 94 | 90 | -4 | stuck-actuator |
| enactic/openarm-2-cell-pick_up_cube_mujoco-lerobot-past | 91 | 87 | -4 | stuck-actuator |
| jesbu1/bridge_v2_lerobot | 73 | 69 | -4 | stuck-actuator |
| lerobot/abc_130k_v3_train | 94 | 90 | -4 | stuck-actuator |
| lerobot/abc_130k_v3_val | 91 | 87 | -4 | stuck-actuator |
| lerobot/aloha_sim_insertion_scripted | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_sim_insertion_scripted_image | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_static_coffee_new | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_static_pingpong_test | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_static_towel | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_static_vinh_cup_left | 94 | 90 | -4 | stuck-actuator |
| lerobot/aloha_static_ziploc_slide | 94 | 90 | -4 | stuck-actuator |
| lerobot/berkeley_fanuc_manipulation | 94 | 90 | -4 | stuck-actuator |
| lerobot/berkeley_rpt | 91 | 87 | -4 | stuck-actuator |
| lerobot/jaco_play | 88 | 84 | -4 | stuck-actuator |
| lerobot/robocasa_target_human_unified | 88 | 84 | -4 | stuck-actuator |
| lerobot/robotwin_unified | 86 | 82 | -4 | duplicate-and-leakage, stuck-actuator |
| lerobot/taco_play | 81 | 77 | -4 | calibration-drift, stuck-actuator |
| lerobot/ucsd_pick_and_place_dataset | 88 | 84 | -4 | stuck-actuator |
| rpwang/bimanual_assembly_insert_force_lerobot_mix854 | 91 | 87 | -4 | stuck-actuator |
| sunLry/robotwin-23task-lerobot | 92 | 88 | -4 | duplicate-and-leakage, stuck-actuator |

## (d) Finding #34: calibration-drift pairing gap (engine, later release)

calibration-drift pairs state dims with action dims. When the two have
different widths, pairing has a gap: for example BitRobot has 29 state dims
and 23 action dims, and robocasa has 16 and 12. The fix belongs in
`src/kinema/checks/`, which is part of the engine hash. It therefore ships
in a later engine release, not on this branch, and this branch keeps the
engine hash at `cec4945aa518`. tiered-v2 already charges scale drift at
tier A (weight 10) wherever the check fires. The gap affects detection
coverage, not the weight.

## (e) Rollout

1. Ved reviews and merges `tiered-v2`.
2. The next scheduled nightly rescores the board. Every normal
   `scripts/leaderboard.py` run rescores the entries it resumes, using the
   stored per-check verdicts and findings (the same path as `--rescore`).
   Nothing is re-audited, nothing is downloaded, and no workflow is
   dispatched by hand.
3. That run rewrites `score_model` to `tiered-v2` in `data.json` and,
   because scores move, in every `docs/dataset/*.json` and dataset page
   (`rewrite_dataset_pages`, which keeps each page's audit stamp). It does
   not change the audit stamp or the engine. A report JSON stored outside
   the board (for example from a user's own `kinema audit --json`) keeps
   its old `score_model` until it is rescored or re-audited.
4. Stored results keep their engine hash (`cec4945aa518`), because
   `report.py` is not engine-hashed. No result is marked stale.

# Adjudicated corpus

Small frozen extracts of real datasets (`*.npz`), each with the verdict a
check must or must not give, asserted offline on every push by
`tests/test_corpus.py` and run through the invariance probes. Rebuild with
`python tests/data/corpus/build_corpus.py [case ...]` (needs the Hub); every
case is pinned to an exact revision.

**Licenses.** A case is kept only if its source dataset's card declares a
license that permits redistribution; each extract is a derivative excerpt
redistributed under that license, with attribution below.
`tests/test_corpus.py` fails if a case is missing from this table or its
license is not one of those. `lerobot/behavior1k-task0045` and
`hggforget/lerobot_2026-10-26_12-09-32` declare no license: the first case
was rebuilt from its licensed upstream; the second was removed - its
state-side dropout is covered by `austin_dropout`, its action-side half by a
synthetic test in `tests/test_dominating_outliers.py`. Removing a file from
`main` does not remove it from earlier commits.

| case | source (attribution) | revision | license | what it pins |
| --- | --- | --- | --- | --- |
| `austin_dropout` | [lerobot/austin_buds_dataset](https://huggingface.co/datasets/lerobot/austin_buds_dataset) | `d9f9289c349f` | mit | MUST flag: all-zero state dropout frames mid-episode (state dims 8/13/18) |
| `behavior1k_tilt_tail` | [behavior-1k/2025-challenge-demos](https://huggingface.co/datasets/behavior-1k/2025-challenge-demos) | `336396924252` | mit | must NOT flag: single-frame spikes continuing a 4e-5-wide bulk's tail (state dim 146). Task 45 of the official challenge demos; `lerobot/behavior1k-task0045`, where the board met it, is a copy with no declared license (the frozen arrays are bit-identical from either source) |
| `dobbe_glitch` | [IPEC-COMMUNITY/dobbe_lerobot](https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot) | `538cbb8366c6` | apache-2.0 | MUST flag: odometry glitches in a few episodes dominate the action dims' normalization |
| `groot_basevel_tail` | [nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0013_loading_the_car](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim) | `ea7ac0b68f87` | cc-by-4.0 | must NOT flag: joint_qvel/base_qvel spikes forming a continuous heavy tail |
| `mendelbeans_stale_stats` | [MendelBeans/lerobot_dataset_put_pen_in_the_holder](https://huggingface.co/datasets/MendelBeans/lerobot_dataset_put_pen_in_the_holder) | `b281c2c1f1c9` | apache-2.0 | MUST flag: stats-drift FAIL - episode 51 reaches -53.5 while stats.json says min -24.4 (renamed on the Hub from `..._20261118_215903`, which redirects here) |
| `nyu_wrap` | [lerobot/nyu_franka_play_dataset](https://huggingface.co/datasets/lerobot/nyu_franka_play_dataset) | `3b2367d09641` | mit | MUST flag: angle wrap-arounds (-2*pi jumps) on action dim 10 |
| `openarm_gripper` | [ethanCSL/openarm_wipeboard_lerobot_real_v00](https://huggingface.co/datasets/ethanCSL/openarm_wipeboard_lerobot_real_v00) | `9d4b70910ab5` | apache-2.0 | must NOT flag: two-level grippers named like joints (action-saturation) |
| `robocoin_jitter_stats` | [RoboCOIN/AI2_Alphabot_2_package_fruit](https://huggingface.co/datasets/RoboCOIN/AI2_Alphabot_2_package_fruit) | `4ca82dfaf518` | apache-2.0 | must NOT flag: converter f32 noise on near-constant channels (stats-drift) |
| `ucsd_parked` | [lerobot/ucsd_pick_and_place_dataset](https://huggingface.co/datasets/lerobot/ucsd_pick_and_place_dataset) | `5af7ee806291` | mit | must NOT flag: a parked joint's one move (dominating outliers) |

`groot_basevel_tail` is an excerpt of NVIDIA's PhysicalAI-Robotics-GR00T-X-Embodiment-Sim,
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The
excerpt selects 4 episodes, 3 state dims and 1 action dim of one task, stored
as float32.

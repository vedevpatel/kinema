# Normalization and deployment-readiness verification: results (2026-11-23)

The six frozen findings are re-examined at their pinned revisions against the
training stacks those datasets are published for. The method, stats-source
trace and selection rules were committed before any probe ran (`METHOD.md`,
`expectations.json`, `preregistration.json`). Machine-readable results are in
`report.json`. The new labels, with pointers to the original cases, are in
`adjudications.json`. The frozen directories `audits/2026-11-22/` and
`loader-verification/2026-11-22/` are unchanged, and so are their adjudications.

The case label is the label of the consequence assertion: A4 for
deployment-readiness, B2 for nan-and-normalization. The factual assertions (A1,
A2, B1) are listed beside it. Each label is scoped to the named stage,
configuration, revision and operation. No label generalizes beyond them.

| Case | Phase 1 / 2 | Fact | Consequence | Label |
| --- | --- | --- | --- | --- |
| `r15:c17:f0` VLABench pretrain `aaa042a8` | unresolved / unresolved | stats absent: **confirmed**; label binds to column `state` | openpi fork: `normalizes_finite`; LeRobot 0.3.3: `not_consumed`; 0.4.0: `not_reached` | **refuted** |
| `r51:c17:f1` VLABench ft `460a2a4d` | unresolved / unresolved | stats absent: **confirmed**; label binds to column `actions` | openpi fork: `normalizes_finite`; LeRobot 0.3.3 and 0.4.0: `not_reached` | **refuted** |
| `r67:c17:f0` LIBERO-90 `fa5ab6b7` | unresolved / unresolved | stats absent: **confirmed**; label binds to column `state` | openpi upstream: `normalizes_finite`; LeRobot 0.3.3: `not_consumed`; 0.4.0: `not_reached` | **refuted** |
| `r79:c17:f0` exUMI `334bf533` | unresolved / unresolved | stats absent: **confirmed**; label is the literal parquet column `observation.state` | GR00T n1.6.1, LeRobot 0.3.3, LeRobot 0.4.0: all `not_reached` | **unresolved** |
| `r390:c7:f8` GR00T BEHAVIOR task-0048 `ea7ac0b6` | unresolved | near-zero std: **confirmed** | GR00T-N1.6-BEHAVIOR1k: `not_consumed`; LeRobot 0.3.3: `not_reached` | **refuted** (for that configuration, at feature selection) |
| `r393:c7:f14` LeRobot behavior1k-task0045 `27458fab` | unresolved | near-zero std: **confirmed** | LeRobot 0.4.0: MEAN_STD and MIN_MAX `not_amplified`, QUANTILES `amplified`, IDENTITY `not_consumed` | **partially resolved** |

## Missing-stats cases (r15, r51, r67, r79)

**A1, stats absent (confirmed for all four).** A fresh `meta/` listing at each
pinned revision has no `meta/stats.json` and no `meta/episodes_stats.jsonl`.
The listings have 3, 3, 3 and 28 files (`evidence/inputs.json`), and they agree
with phase 1.

**A2, feature binding (confirmed).** kinema's alias table binds `state` to
`observation.state` and `actions` to `action` (source pins `kinema.alias.*` at
the board engine). The FAIL text therefore names a canonical label, and the
datasets' columns are `state` (r15, r67) and `actions` (r51). For exUMI,
`observation.state` is the literal parquet column, but `info.json` declares no
features.

**A3, where the stack gets stats.** The FAIL text implies that the policy's
normalization stats come from the dataset files. That premise holds for one
stack and fails for the other two:
- **openpi, upstream and fork (refuted).** The training data config reads norm
  stats from `assets/<asset_id>` and never from `meta/stats.json`. The training
  loader refuses to run without those assets. We executed that guard: `ValueError:
  Normalization stats not found. Make sure to run scripts/compute_norm_stats.py`.
  At inference, the stats come from `checkpoint_dir/assets/<asset_id>`.
- **LeRobot 0.3.3 (confirmed).** `make_policy` passes `ds_meta.stats`, which is
  `None` for these v2.0 datasets.
- **GR00T (refuted by source only).** GR00T generates `meta/stats.json` from the
  parquet data when it is absent. That generation step was not reached here.

**A4, normalization consequence, per configuration:**

- **openpi (primary for r15, r51, r67).** The stack's own pinned LeRobot built
  episode 0, and `meta.stats` was `None`. The configs' repack and data transforms
  turned `state`/`actions` into openpi's `state`/`actions`. openpi's
  `RunningStats` produced statistics over the episode-0 frames (81, 93 and 98
  frames). `normalize.save` then `checkpoints.load_norm_stats` (the inference
  loader) followed by `Normalize` gave finite outputs under every config listed
  (6, 8 and 3 configs).
  - Released stats, independent of our computation, were also finite on the
    case's arrays. r51: `VLABench/pi0-primitive-10task` (max |z| 1.87 state,
    4.01 actions), `VLABench/pi0-fast-primitive-10task` and
    `VLABench/pi0-fast-ft-primitive-10task-deltachunk`. r15:
    `VLABench/pifast-pretrain-vlabench-primitive-aligned` and
    `VLABench/pi0-fast-primitive-deltachunk-w-two-bridge`, which have
    byte-identical stats files. r67: the three public openpi LIBERO checkpoints.
    Those were trained on `physical-intelligence/libero`, so for r67 they show
    loader mechanics and provenance only.
  - Two VLABench pi05 checkpoints and one PI0 checkpoint have no executable
    config in the pinned fork. They are provenance only: each carries its own
    `norm_stats.json`.
- **LeRobot 0.3.3 (secondary).** `dataset_to_policy_features` keeps only
  `observation*` keys as STATE and `action*` keys as ACTION, so `state` is never
  passed to a policy (`not_consumed`). `actions` is typed ACTION, but
  `make_policy` fails for both ACT and diffusion before any normalizer exists.
  ACT's `action_feature` requires the name `action`, and diffusion's
  `robot_state_feature` requires `observation.state`. Both raise
  `AttributeError: 'NoneType' object has no attribute 'shape'`. So **for LeRobot
  the blocker is the feature names, not missing stats**, and kinema does not
  currently report it.
- **LeRobot 0.4.0 (gate).** v2.0 datasets raise `NotImplementedError` in
  `BackwardCompatibilityError`, because 0.4.0 offers conversion only from v2.1.
  Source inspection also shows that 0.4.0's normalizer passes a feature without
  stats through unnormalized rather than failing. That path was not executed on
  these datasets.
- **r79 (unresolved).** GR00T's `generate_stats` raises `KeyError: 'features'`
  and writes nothing. `LeRobotEpisodeLoader` asserts that `meta/stats.json` is
  missing. `modality.json` has only `action` and `video`, with no state modality,
  so GR00T has no state key to normalize without a user-written config, which we
  did not invent. LeRobot 0.3.3 and 0.4.0 both stop at `KeyError: 'features'`.
  Every failure happens before normalization, so none of them confirms the claim.

**What the refutations rest on.** The finite S6/S7 outputs are close to
tautological: episode-0 frames normalized with stats computed from the same
frames. The refutation rests on the executed stats-source path (S2 guard, S3
`meta.stats is None`, S5 compute, S7 loader). The released VLABench stats
applied to the case's own data are its non-circular check. The original FAIL
says "the policy cannot normalize ... at inference". Under the stacks these
datasets are published for, the policy normalizes with stats it computes and
ships itself. That is a statement about these configurations only. It does not
mean every policy carries its own stats, or that missing dataset stats never
matter: LeRobot 0.3.3 would pass `None` if construction got that far.

## Near-constant cases (r390, r393)

**B1, near-zero std (confirmed for both).** Values are pooled over the frozen
sample of 8 files per case, read in full for the flagged dim:

| | board std | stored std | sampled std (pooled; per-file range) | sampled values |
| --- | --- | --- | --- | --- |
| r390 dim 117 | 1.16e-12 | 1.02e-12 | 8.09e-13; 5.1e-13 .. 1.06e-12 | 160,402 rows / 8 episodes; 148,622 exact zeros; 11,781 distinct; range [-9.2e-11, 4.1e-11] |
| r393 dim 123 | 8.57e-13 | 1.63e-12 | 1.21e-12; 3.1e-13 .. 1.61e-12 | 331,864 rows / 36 episodes; 318,399 exact zeros; 11,332 distinct; range [-5.5e-11, 4.0e-11] |

The dims are near-zero but not constant. Most rows are exactly 0, and the rest
are of order 1e-11. The board's std differs from the stored std because the
detector computes std from its own loaded rows (source pin
`kinema.nan.std_from_loaded_rows`). Both dims sit in `joint_qeffort`
(`state[112:140]`; r390 from its pinned `modality.json`, r393 by inference from
the sibling layout and a sin/cos structure check, max error 3.6e-8). Most of
that slice is similarly tiny: the median stored std of the other 27 dims is
2.8e-11 for r390 and 3.4e-11 for r393. The frozen control-dim rule therefore
picked near-zero siblings (dims 119 and 132), not ordinary channels.

**B2, amplification.**

- **r390, released GR00T-N1.6-BEHAVIOR1k (primary): `not_consumed`.** The
  processor's 21 state keys (82 dims) exclude `joint_qeffort`. The checkpoint's
  `statistics.json` carries no stats for that slice. GR00T's own loader on
  episode 0 returned 21 state groups without it, and `apply_state` with the
  checkpoint stats produced finite output in [-1, 1]. LeRobot 0.3.3 stops at
  construction because the GR00T-format v2.1 dataset has no
  `meta/episodes_stats.jsonl`.
  - Hypothetical control (G6; the released config does not select the dim):
    GR00T's min/max function on this dim with the dataset's own min/max leaves it
    at 0, because `np.isclose(max, min)` masks it. That holds even for synthetic
    shifts. It decides nothing.
- **r393, LeRobot 0.4.0 (primary).** This uses the real `NormalizerProcessorStep`
  with the stats `lerobot-train` selects (`dataset.meta.stats`). Inputs are cast
  to float32 as the loader does; equality was checked on 4 rows. Results on the
  sampled values of dim 123:

| STATE mode (shipped policies) | normalized std | max \|z\| | gain | outcome |
| --- | --- | --- | --- | --- |
| MEAN_STD (act, groot, pi0, smolvla) | 1.2e-4 | 0.0055 | 1e8 (eps 1e-8 dominates the 1.6e-12 std) | not amplified |
| MIN_MAX (diffusion, sac, vqbet) | 0.0050 | 0.25 | 4e9 | not amplified (half the 0.01 threshold) |
| QUANTILES (pi05) | 0.68 | 30.7 | 5.6e11 (q99 - q01 = 3.6e-12 is nonzero, so no eps substitution) | **amplified** |
| IDENTITY (tdmpc) | unchanged | - | 1 | not consumed |

No output was non-finite in any mode. This sample therefore gives no support for
the causes.json wording "Inf/NaN batches". That conclusion covers these inputs
only, and it does not mean tiny std can never produce non-finite values.

**B3, labelled synthetic deployment shift.** Each input is the stored mean plus a
shift δ (descriptive only):

| Mode | δ = 1e-9 | δ = 1e-6 | δ = 1e-3 |
| --- | --- | --- | --- |
| MEAN_STD | 0.1 | 100 | 1e5 |
| MIN_MAX | 4.1 | 4.1e3 | 4.1e6 |
| QUANTILES | 561 | 5.6e5 | 5.6e8 |

GR00T min/max (hypothetical) gives 0 at every shift.

**Post-hoc exploration (D9, not used for any label).** Under QUANTILES, the
sampled full rows reach |z| up to 2.3e9 on dims kinema did *not* flag. Dim 86,
for example, has std 3.6e-5 but q99 - q01 = 4.2e-12. For quantile-normalizing
policies, the q-range rather than the std predicts amplification
(`evidence/posthoc_feature_scan.json`).

## Normalization selection observed from executable configs

- **upstream openpi:** `pi0_libero` z-score, `pi05_libero` and `pi0_fast_libero`
  quantiles. `pi0_fast_libero` came through `create_base_config`, see D4.
- **VLABench fork:** every PI0 and PI0-FAST config is z-score except
  `pifast_ft_vlabench_primitive_aligned_q99`. The fork's base config never sets
  quantiles.
- **LeRobot released comparators:** `lerobot/smolvla_vlabench`,
  `lerobot/pi0_libero_finetuned_v044` and `lerobot/pi05_libero_finetuned_v044`
  all ship q01/q99 tensors but normalize with MEAN_STD, including the pi05
  checkpoint, even though LeRobot's pi05 default is QUANTILES. The VLABench PI0
  `norm_stats.json` files also contain quantiles that their configs do not use.
  So stats files that contain quantiles do not show which normalization a policy
  uses.

## Deviations from the preregistration (`deviations.json`)

Every changed preregistered file has its before and after hash recorded, and the
hash guard accepts only those. D1 to D5 fix harness bugs or close an
incompleteness in the frozen list. None of them changes an input, a threshold or
a rule.

| ID | Type | Change |
| --- | --- | --- |
| D1 | harness bug | the upstream LeRobot rejects `LEROBOT_HOME`, so the probe sets `HF_LEROBOT_HOME` for upstream |
| D2 | harness bug | the hash guard honors recorded deviations |
| D3 | harness bug | the fork's `RunningStats` takes 2-D input; the probe reshapes as both compute scripts do |
| D4 | offline isolation | FAST configs build a Hub tokenizer inside `create()`; the probe falls back to `create_base_config`, the function that sets `use_quantile_norm`, labelled `selection_scope` |
| D5 | incomplete frozen list | the frozen org-scan rule selected two VLABench checkpoints that the per-case list omitted; they were added by the frozen mechanical mapping |
| D6 | new script | `probe_openpi_fast_supplement.py`: FAST checkpoint stats on the 7-dim FAST-pipeline arrays (the fork's z-score does not slice stats to width) |
| D7, D8 | environment | the phase-2 LeRobot envs lacked policy-import packages; clones with identical packages plus import-only additions within declared bounds; lerobot sources byte-identical (`source_pins_posthoc.json`) |
| D9 | new script, post hoc | `posthoc_feature_scan.py` |

## Adversarial review

- **Circular controls.** C1 (openpi `Normalize(None)` is the identity) and G6
  (GR00T min/max on a dim the config never selects) are labelled controls and
  never decide a label. The S6/S7 finite checks are near-tautological, as noted
  above. The independent checks are the released checkpoint stats in S8 and
  the stats-source execution.
- **Cherry-picked policies.** The configurations were fixed by frozen rules:
  publisher or card-named stacks, every fork config for the dataset family, every
  STATE mode in a shipped LeRobot config, and first-party checkpoints only. The
  org scan added checkpoints that the author's own list had missed (D5).
  Third-party checkpoints, including many BEHAVIOR and VLABench fine-tunes, were
  excluded by rule.
- **Unreached paths.**
  - No trained policy and no weights were run.
  - openpi stats come from episode 0 only.
  - Full `create()` did not run for the FAST configs.
  - The VLABench pi05 checkpoints have no executable config.
  - LeRobot policies other than ACT and diffusion did not run for the v2.0 cases.
  - The r79 normalization stage was never reached.
  - For r390, the LeRobot path was not reached, and the normalizer's behavior on
    dim 117 was not exercised by the released config.
  - The r393 layout is inferred, and its GR00T path was not tested.
- **Breadth of conclusions.**
  - The refutations cover the named configurations at the pinned revisions only.
  - r393's amplification depends on the mode and is confirmed only for
    QUANTILES.
  - Whether amplification harms training or deployment was not tested.
  - B1 covers 8 sampled files per case, not every row.

## Proposed later corrections (not made in this PR)

1. **deployment-readiness "no normalization stats" FAIL.** Stop asserting
   inference failure. openpi computes its own stats and ships them in checkpoint
   assets. GR00T generates stats and ships `statistics.json`. LeRobot
   checkpoints carry processor stats. Absent dataset stats matter to pipelines
   that read `meta/stats.json` at training (LeRobot 0.3.x `make_policy`; LeRobot
   0.4.x passes the feature through silently). Consider WARN or INFO with
   policy-aware wording, and update the `missing-stats` cause, whose text says
   "inference without the same stats produces garbage actions".
2. **Report the column actually read.** Keep the canonical label, but name the
   dataset column (`state`, `actions`). Add a separate finding when LeRobot
   would not select it: STATE requires `observation*` and the exact name
   `observation.state`; ACTION requires the exact name `action`. That is the real
   LeRobot blocker for r15, r51 and r67.
3. **policies.json.** `pi0` has `quantile_norm: true`, but openpi PI0 is z-score
   and LeRobot pi0 is MEAN_STD. The VLABench fork uses z-score for PI0-FAST. The
   released LeRobot pi05 LIBERO checkpoint uses MEAN_STD. Quantile use is
   config-specific, so it should be overridable per run rather than fixed per
   policy name.
4. **nan-and-normalization near-zero rule.**
   - Make it mode-aware. Under MEAN_STD with an epsilon at or above the std, the
     channel is attenuated, not amplified. Under QUANTILES, a tiny nonzero
     q99 - q01 amplifies in-sample values; this check misses that, since it uses
     std (D9). MIN_MAX stays bounded in-sample but is sensitive to out-of-range
     values. GR00T masks `isclose` ranges.
   - Base the finding on stored stats when they exist. The board std came from
     sampled rows.
   - Update the `zero-variance-dims` cause, which says "Inf/NaN batches". That
     was not observed with these inputs.
5. **policy-dependent feature selection.** Released configs may never consume a
   flagged dim, as with GR00T BEHAVIOR's 82 of 256 state dims. That is worth a
   note in the explanation, without changing severity on its own.

## Remaining gaps

- No released checkpoint trained on LIBERO-90 or exUMI was found. No r393
  first-party policy was found.
- Inference with a trained model and the training effects of amplification were
  not tested.
- The r79 normalization stage needs a GR00T modality config for exUMI's state.
  None is published, so the stage was not reached.
- The VLABench pi05 checkpoints have no pinned executable config.
- The r393 state layout is inferred and has no published names.

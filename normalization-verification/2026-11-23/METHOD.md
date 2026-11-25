# Normalization and deployment-readiness verification: method (2026-11-23)

This directory is new. It reads, and never writes to, the frozen records in
[`audits/2026-11-22/`](../../audits/2026-11-22/) (phase 1) and
[`loader-verification/2026-11-22/`](../../loader-verification/2026-11-22/) (phase 2).
Their adjudications remain as they are. This directory records new evidence and
revised adjudications, each pointing to its original case.

The six cases are frozen findings on the 2026-11-21 board (engine `f9a1edd38b66`):

| Case | Check | Dataset @ revision | Assertion (verbatim) |
| --- | --- | --- | --- |
| `r15:c17:f0` | deployment-readiness FAIL | VLABench/vlabench_primitive_pretrain_lerobot @ `aaa042a8` | observation.state: no normalization stats (meta/stats.json or per-episode stats) - the policy cannot normalize inputs/outputs at inference |
| `r51:c17:f1` | deployment-readiness FAIL | VLABench/vlabench_primitive_ft_lerobot @ `460a2a4d` | action: (same text) |
| `r67:c17:f0` | deployment-readiness FAIL | jesbu1/libero_90_lerobot @ `fa5ab6b7` | observation.state: (same text) |
| `r79:c17:f0` | deployment-readiness FAIL | easyminnn/exumi-lerobot @ `334bf533` | observation.state: (same text) |
| `r390:c7:f8` | nan-and-normalization WARN | nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim//sim_behavior_r1_pro.task-0048_canning_food @ `ea7ac0b6` | observation.state dim 117: std 1.16e-12 - near-zero variance amplifies noise after normalization |
| `r393:c7:f14` | nan-and-normalization WARN | lerobot/behavior1k-task0045 @ `27458fab` | observation.state dim 123: std 8.57e-13 - near-zero variance amplifies noise after normalization |

`expectations.json` copies each finding from `audits/2026-11-22/sample.json`.
It also copies the phase-1 and phase-2 labels and observations. A test checks
both copies against the frozen files.

## Engine pin

With Python 3.11, the engine hash `f9a1edd38b66` matches five commits from
2026-11-20 (`0e4040c2` .. `02fbc570`). Their engine files are identical.
`deployment.py`, `actions.py`, `dataset.py` and `policies.json` have the same
git blob IDs at those commits and on current main (`7e18468a`). Main has changed
only `causes.json`, and only in its timestamp-grid entries. Source pins for the
detector are read from `02fbc570` with `git show`. No kinema code is imported or
re-run as evidence.

## Claim decomposition

The FAIL text combines several assertions, and each one is tested at its own
stage:

- **A1 stats absent** (metadata): there is no `meta/stats.json` and no
  per-episode stats at the revision.
- **A2 feature binding** (detector input): which dataset column the label names.
  kinema binds `state` to `observation.state` and `actions` to `action`.
- **A3 stats source** (per configuration): where the policy stack actually gets
  normalization statistics. This premise is implied by the FAIL text.
- **A4 normalization** (per configuration): what happens when the stack
  normalizes the dataset key. The outcome is `normalizes_finite`,
  `fails_missing_stats`, `not_consumed` or `not_reached`. A failure before
  normalization is `not_reached` and never confirms the claim.

The WARN text is split the same way:

- **B1**: is the dim's variance near zero, both in the stored stats and in a
  disclosed raw sample?
- **B2**: per configuration and normalization mode, does normalization pass the
  dim through, and is its in-sample variation amplified? The outcome is
  `not_consumed`, `nonfinite`, `amplified` (normalized std >= 0.1),
  `not_amplified` (<= 0.01), `intermediate` or `not_reached`.
- **B3**: labelled synthetic deployment shifts. These are descriptive only and
  never decide a verdict.

The labels and the rule that combines per-configuration outcomes into an overall
label are frozen in `expectations.json["labels"]`. The rules leave the verdict
open. The overall label follows the primary configuration when that outcome is
decisive. A secondary configuration that reaches a different decisive outcome
makes the label `partially_resolved`.

## Execution paths and where each one obtains stats (source inspection)

Line-level pins with file hashes are in `evidence/source_pins.json`, produced
by `source_pins.py`.

| Stack (pin) | Training stats | Inference stats | Normalization selection |
| --- | --- | --- | --- |
| openpi upstream `215abfb2` | `scripts/compute_norm_stats.py`: `RunningStats` over the **transformed data** (`state`, `actions` after repack and data transforms), saved to `assets/<asset_id>/`. The training loader raises if that file is missing. `meta/stats.json` is never read. | `policy_config.create_trained_policy` loads `checkpoint_dir/assets/<asset_id>` | `use_quantile_norm = model_type != PI0`: PI0 uses z-score (eps 1e-6); PI05 and PI0_FAST use q01/q99 (eps 1e-6) |
| VLABench openpi fork `4483d1da` | same compute and assets flow | same checkpoint-assets loader | `create_base_config` does **not** set `use_quantile_norm`, so the `DataConfig` default (False) applies, z-score for PI0 and PI0_FAST alike. One config (`..._aligned_q99`) opts into quantiles. There is no PI05 config. |
| LeRobot 0.3.3 | `make_policy` passes `ds_meta.stats`. For v2.0 this is `meta/stats.json`, or `None` when absent. For v2.1 it is the aggregate of `meta/episodes_stats.jsonl`. Without stats, normalizer buffers start at inf and normalization asserts. | buffers stored in the checkpoint weights | per-policy `normalization_mapping`. `dataset_to_policy_features` keeps only `observation*` keys as STATE and `action*` keys as ACTION. |
| LeRobot 0.4.0 | `lerobot-train` passes `dataset.meta.stats` to `NormalizerProcessorStep`. A feature without stats **passes through unnormalized**. | processor safetensors in the checkpoint | MEAN_STD (eps 1e-8). MIN_MAX and QUANTILES substitute eps only when the range is exactly 0. STATE defaults: MEAN_STD (act, groot, pi0, smolvla), MIN_MAX (diffusion, sac, vqbet), QUANTILES (pi05), IDENTITY (tdmpc). Rejects v2.x datasets without conversion. |
| Isaac-GR00T n1.6.1 `5dc80c4a` | training calls `generate_stats()` (needs `info.json["features"]`) and then the loader, which asserts `meta/stats.json` exists | `statistics.json` in the checkpoint processor | per-embodiment state keys. Min/max to [-1, 1], with `np.isclose(max, min)` channels left at 0 and outputs clipped. |

Quantile entries in a stats file do not show that a policy uses them. The
selection above is observed from the executable configuration (S1 in
`probe_openpi.py`), and it is not inferred from file contents.

## Configuration selection (frozen)

- **r67 (LIBERO-90)** uses openpi upstream `pi0_libero`, `pi05_libero` and
  `pi0_fast_libero`. The card says "to be used for π₀ fine-tuning", and the
  features equal openpi's LIBERO conversion script field for field. The released
  LIBERO checkpoints were trained on `physical-intelligence/libero`, not
  LIBERO-90. They are used only for provenance and inference-loader mechanics.
- **r15 and r51 (VLABench)** use the VLABench openpi fork. VLABench documents it
  as its training codebase. Every fork config whose `repo_id` matches the case's
  dataset family is included. Released checkpoints come from the VLABench
  organisation scan, which matches them by norm-stats asset ID. A checkpoint maps
  to a config only through the mechanical model-type, asset and aligned rule;
  without a match it is used for provenance only.
- **r79 (exUMI)** uses Isaac-GR00T n1.6.1. The card describes a GR00T-flavoured
  dataset for an Isaac-GR00T style loader. No released checkpoint was found.
- **r390 (GR00T BEHAVIOR)** uses `nvidia/GR00T-N1.6-BEHAVIOR1k`. This is the
  publisher's released policy, and its card lists this dataset collection.
- **r393 (LeRobot BEHAVIOR v3.0)** uses LeRobot 0.4.0, with every STATE mode
  that a shipped policy config uses. No first-party policy trained on it was
  found.
- **Secondary paths**: LeRobot 0.3.3 for every v2.x case, and the 0.4.0 v2.x
  compatibility gate for the missing-stats cases. Third-party checkpoints are
  not used.

## Bounded experiments

No training, no weights and no full-dataset downloads. The deployment cases
reuse phase 2's hashed meta/ and episode-0 parquet. For r390 and r393, the
loaders' meta/ files are fetched (r393's 200 `meta/sim/*.json` files are not).
Data files follow the frozen sampling rule: the first file plus seeded-hash
order, 8 files, a 450 MB cap, and a disclosed per-file report. Every probe runs
with `python -I`, offline, with private homes and private input copies.

| Script | Environment | What it executes |
| --- | --- | --- |
| `prepare.py` | phase-2 v2 | acquisition and hashes; fresh meta listings; checkpoint small files |
| `probe_channels.py` | phase-2 v2 | raw flagged and control dims per sampled file; sin/cos layout check |
| `probe_openpi.py upstream` / `fork` | env-openpi / env-openpi-vlabench | S1-S8 and C1 (see the script docstring) |
| `probe_lerobot.py 0.3.3` / `0.4.0` | phase-2 v2 / v3 | L1-L5; the v2.x gate; M1-M3 per STATE mode with synthetic shift |
| `probe_gr00t.py` | env-gr00t | G1-G6 |

Labelling rules: the openpi S5-S7 statistics come from episode 0 only. The
LeRobot 0.3.3 forward pass uses two real frames with the real `ds_meta`. Empty
or None stats inputs are reported as mechanism controls (C1). G6 is
hypothetical, because the released config does not select the flagged dim.

## Known limits, stated before running

See `expectations.json["known_limits_before_running"]`.

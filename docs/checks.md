# The checks, one by one

What each check flags, how it decides, its knobs (all `CheckConfig` fields,
overridable with `kinema check --config tuning.json`), and the limits worth
knowing before you trust or dismiss a finding. The one-line versions live in
the README table; this is the long form.

Severity ladder: INFO (worth knowing), WARN (worth fixing), FAIL (blocks
training outright). Exit code is 1 on any FAIL; `--fail-on warn` also gates
on WARN. A SKIP means the check could not run on this dataset - the reason
says why, and it is not a verdict on the data.

## Health score

Score model `tiered-v1` (reports, report JSON `score_model`, and the
leaderboard all name it). Start at 100; each FAIL check costs 25; each WARN
check costs its weight below; floor at 0. Billing is per check, never per
finding: a check that lists 40 flagged episodes costs the same as one that
lists 5, and making a check more detailed never moves a score. Limit: one
defect that trips several checks still pays each of them (idle padding often
also trips stuck-actuator). The score is derived from stored per-check verdicts, so
any stored result can be rescored without re-running checks. Results with no
`score_model` field were scored by `legacy-v0` (hand-tuned weights); scores
are not comparable across models.

A FAIL is a "data is not what it claims" blocker; its cost is trust, not
measured training damage, so it is not harm-scaled. WARN weights come in
three tiers anchored on the harm study v0 (`scripts/harm_study.py`: ridge
behavior cloning, state-only, held-out action MSE delta vs clean, pusht):

| tier | meaning | held-out MSE delta | weight |
| --- | --- | --- | --- |
| A | training killer | >= 100% | 10 |
| B | skews learning | 10-100% | 6 |
| C | review list | < 10% | 2 |

The band edges are provisional - three measurements define them - and get
re-anchored once when the full harm table lands. Each check's weight carries
its basis in report JSON (`weight`, `weight_basis`):

| check | weight | basis |
| --- | --- | --- |
| idle-segments | 10 | measured: +881% (tier A) |
| calibration-drift | 10 | measured: +347% (tier A) |
| duplicate-and-leakage | 2 | measured: +0.1% (tier C) |
| stats-drift, nan-and-normalization, deployment-readiness, timestamp-grid | 10 | placeholder |
| dropped-frames | 8 | placeholder |
| stuck-actuator, action-saturation, visual-quality | 6 | placeholder |
| outlier-episodes | 3 | placeholder |
| any other check (task-strings, reward-done, ...) | 8 | placeholder |

Placeholder weights are the hand-tuned legacy values, kept until the harm
study measures them. Limit: every measured number is one linear-BC MSE
delta, not rollout success; tiers are re-checked against success-rate
deltas when the GPU protocol runs.

## stats-drift

Stored normalization stats (`meta/stats.json`, or per-episode stats on v2.1)
disagree with the data: mean shift, std ratio, min/max excursions beyond the
stored range. Every training run normalizes with those numbers, so drift
means wrong-scaled inputs at training *and* inference.

Sampled loads (`--max-episodes`) are compared against the re-aggregated
per-episode stats of exactly the loaded episodes - never against
whole-dataset stats - and `stats.json` is checked against the aggregation of
all per-episode stats, the way LeRobot builds it. A sampled load without
per-episode stats gets a bootstrap reference once enough episodes are
loaded (>= 8): the sampling distribution of the pooled mean/std, resampling
episodes (frames within one are autocorrelated, so the episode is the
unit), and stored stats outside the 99.9% interval are flagged. With too
few episodes only the gross backstop applies (std off >= 3x, mean off
>= 2 sigma); a full load stays the conclusive audit.

Knobs: `stats_mean_warn_sigma` (0.10), `stats_mean_fail_sigma` (0.50),
`stats_std_warn` (0.05), `stats_std_fail` (0.25), `stats_std_oversize_fail`
(3.0), `stats_range_warn` (0.01), `stats_range_fail` (0.10),
`stats_sample_std_ratio` (3.0), `stats_sample_mean_sigma` (2.0).

## timestamp-grid

Timestamps off the 1/fps sampling grid, and non-monotonic runs.
`delta_timestamps` sampling returns misaligned windows when the grid slips.

Knobs: `grid_warn_frac` (0.10, as a fraction of the frame period).

## dropped-frames

`frame_index` gaps, episode lengths in the metadata vs rows in the data, and
v3 video segments vs data. A metadata lie here blocks training outright, so
mismatches are FAIL.

No knobs.

## calibration-drift

On frames where a joint holds still, command and measured position should
agree (the 17-degree leader/follower offset from LeRobot PR #3762 survived
recording AND training). Per comparable joint the still-frame offset is fit
as a line over position: the intercept is a constant calibration offset, the
slope a scale (gear-ratio/units) error, and a large offset the line does not
explain is backlash/load/contact - reported INFO, with the line-vs-step fit
that separates a scale error from backlash.

A dim is only compared when the action is a position command for the same
joint (names do not contradict it, action tracks state, similar range);
every other dim is classified and the skip says why. Grippers get their own
free-end test (a gripper commanded shut on an object never arrives, so only
the unblocked end says anything). Dims that look like unnamed grippers or
on/off commands - including toggles the slot prior misses, such as odd-sized
bimanual layouts - are reported but never gated.

Delta commands (most Open X-Embodiment conversions) get a scale-agreement
verdict: the measured per-frame step of the paired state dim is regressed
on the command, and a slope far from 1 (with R^2 >= 0.5) means action and
state disagree on scale - a conversion-units bug, or commands the robot
cannot follow (on lerobot/xarm_lift_medium the measured steps run at
~0.07x the commanded ones, R^2 up to 0.98: flagged). Absolute EE-pose
actions still cannot be compared without the robot's kinematics; those
dims classify as "different space" and sit the check out.

Limit (score weight): a WARN costs 10, tier A, measured - action scale 0.5x
in 25% of episodes (the delta-scale defect) raised held-out action MSE by
347% in the harm study v0 (ridge BC). Constant offsets on absolute commands
share the weight without a measurement of their own.

Knobs: `calib_warn_range_frac` (0.05), `calib_fail_range_frac` (0.15),
`calib_min_corr` (0.90), `calib_scale_warn` (0.05), `calib_scale_fail`
(0.15), `calib_scale_min_r2` (0.60), `calib_scale_line_vs_step` (1.5),
`calib_gripper_warn` (0.15).

## idle-segments

Episodes that start idle, or idle more than 30% of the time in runs longer
than 2 seconds: the command is held AND the robot is still. Idle stretches
teach the policy to do nothing - the classic "robot stays still at
inference" postmortem. A still tail after the task completes is INFO.

Limit: without recorded state, detection falls back to commands alone.
Velocity-named dims count as moving unless parked at their modal level (a
held drive command is constant motion, not idling), which removes the
classic false positive; unnamed velocity commands can still read as idle.

Limit (score weight): a WARN costs 10, tier A, measured - 30 idle frames
prepended raised held-out action MSE by 881% in the harm study v0 (ridge BC).

Knobs: `idle_eps_frac` (0.01), `idle_min_seconds` (2.0),
`idle_warn_fraction` (0.30), `idle_state_frac` (0.02).

## stuck-actuator

Four patterns: episodes where nothing moves at all (aborted/empty
recordings, WARN); a joint that moves in most episodes but not in a few
(dead or disabled motor during those recordings, WARN - a joint still in
nearly every episode is an unused DOF, INFO); a gripper that never actuates
in pick/place episodes; and commanded-but-not-followed - the leader moves,
the follower's measured position does not (dead follower motor; action-only
checks cannot see this).

Grippers and detected on/off (two-level) commands idle legitimately, so they
are reported but not gated; the toggle detection is data-driven, so odd
layouts the gripper slot prior misses (e.g. a 7-dim bimanual pair) get the
same treatment.

Limit: the gripper slot prior still advises from distribution shape
alone; it now requires the dim to VISIT both ends (the quieter end holds
>=8% of the dwell), which removes the parked-at-one-end false positives,
but a genuinely one-ended gripper (shut for the whole dataset) is no
longer recognised either. Misses stay safe: its findings keep their
severity instead of being downgraded.

Knobs: `stuck_range_frac` (0.005), `stuck_warn_share` (0.25),
`stuck_unused_share` (0.90), `follow_min_command` (0.10),
`follow_ratio` (0.10).

## action-saturation

Commands pinned at EXACTLY a limit while the robot is moving: clipping
leaves a pile-up at the limit value. Sitting near an extreme is normal (rest
poses, grippers fully open/shut), as are binary or constant dims, and most
data AT the limit is a mode rather than a clipped tail (INFO). Unnamed
grippers and detected on/off commands are reported but not gated.

Knobs: `saturation_warn_fraction` (0.05).

## nan-and-normalization

NaN/Inf anywhere (FAIL - one NaN poisons the loss for the whole batch),
constant dims (INFO - usually padding; if they ever vary at deployment,
mean/std normalization blows them up), and near-zero-variance dims (WARN -
normalization amplifies noise).

No knobs.

## duplicate-and-leakage

- exact: identical action trajectories (WARN).
- partial: identical 8-frame moving windows shared by 2-4 episodes. Windows,
  not frames: quantized commands repeat values by chance, a run of 8 moving
  frames does not. A window in more than 4 episodes is a shared motion, not
  a copy. Where the shared stretch sits and who shares it decides: a
  near-complete copy (>= 80% of the shorter episode) is a WARN; an aligned
  opening shared by ONLY two episodes (>= 10 windows) is a copied beginning -
  a scripted start would repeat across the dataset - also a WARN; an aligned
  opening shared more widely is a scripted start, INFO; a shared stretch
  anywhere else is a splice or botched merge, WARN.
- near: downsampled trajectories within 0.05 std per dim (z-scored, so no
  single large-unit dim decides). Exact up to 100k episodes with
  memory-bounded blocks (64 MB per block); above that an evenly-spaced
  sample with an INFO note saying so.
- leakage: any of the above spanning two splits declared in info.json -
  eval would be measured on training data (WARN).

Limit (score weight): a WARN costs 2, tier C, measured - 20% duplicate
episodes moved held-out action MSE by +0.1% in the harm study v0 (ridge BC):
easy to detect, nearly harmless to linear BC. That reading is not yet
confirmed against a high-capacity policy, and split leakage inflates
evaluation rather than hurting training, a harm this weight does not price.

No knobs.

## outlier-episodes

Per-episode metrics (duration, jerk, peak velocity, velocity-spike, static
fraction, end-pose deviation) flagged by IQR, the method from LeRobot PR
#3762. Only the harmful direction is flagged. Velocity and jerk are measured
per dim in units of that dim's std, without grippers, on/off toggles or
velocity commands (recognised by name, or structurally: a command that
tracks some state dim's per-frame delta); a one-frame spike in a velocity
command still counts, against the dim's own step scale. Duration and end
pose are compared within a task. An episode is a WARN when independent kinds
of evidence agree, or one metric is far out (Tukey k=3, >= 3x the median;
end pose must also clear half the task's end-pose range). static_fraction
never earns a single-metric WARN on its own - that is idle-segments' job.

Knobs: `outlier_iqr_k` (1.5).

## embedded-images

Image features stored inside the parquet shards instead of MP4 video chunks:
bloated shards, and every frame decodes on read. Reads the catalogues, not
the frames.

No knobs.

## camera-coverage

Episodes with no video for a declared camera - a modality silently dropping
out for those episodes. Uses v3 segment metadata, or the v2.1 files with
`--video`.

No knobs.

## task-strings

Empty task strings, frames pointing at a `task_index` the task table does
not define, episode tasks with no table entry. Language conditioning on `""`
trains a task-agnostic policy.

No knobs.

## reward-done

Constant or NaN reward, `done` never True, `done` True mid-episode with
frames after it. RL and reward-conditioned training get labels with no
signal.

No knobs.

## visual-quality

Blur (variance of Laplacian, frames decimated with an integer stride to ~VGA
first, so one threshold works at any resolution), underexposure (95th-
percentile brightness) and overexposure (pixels clipped at the codec's
brightest level). Per camera, sampled inside each episode's segment; v2.1
per-episode and v3 shared MP4s both work.

Needs `--video` and the `video` extra (`pip install -e .[video]`); the
nightly board skips it because the MP4s run to hundreds of MB per camera.

## deployment-readiness

Would this dataset serve the policy you are about to train (`--policy`)?
Episodes long enough to fill the policy's action chunk, stats covering what
its normalizer consumes, q01/q99 quantiles when the policy normalizes that
way, info.json shapes vs the actual arrays, and what the policy conditions
on: camera input for vision policies (WARN when a state-only dataset names
a vision policy), task strings for language-conditioned ones, and an INFO
note when a camera is smaller than the resolution the policy is usually
trained at.

The quantile check reads the raw `meta/stats.json` first, because that is
the file a stats-consuming pipeline reads: a v2.1 dataset whose stats.json
ships q01/q99 is fine even though the per-episode aggregate stats keep only
mean/std/min/max/count. Only when stats.json lacks quantiles does the WARN
fire. Note that openpi's documented flow recomputes its own
`norm_stats.json` from the data (`scripts/compute_norm_stats.py`) and never
reads `meta/stats.json`, so for a pure openpi run the WARN only matters
when another stage consumes the file.

What each policy expects is declared in `src/kinema/policies.json` - chunk
size, normalization, camera/language needs, typical resolution, and aliases
(`pi0.5` resolves to `pi05`). Adding or updating a policy is a JSON edit.
An unknown `--policy` still runs the policy-agnostic checks and says so.
The MCP copilot can override any declaration per call (`policy_overrides`),
so a custom training config - chunk 10, mean/std norm - is diagnosed
against its own assumptions; the declared defaults are untouched.

Limit: the declarations are the documented defaults; a policy trained with a
custom config may disagree with them, which is what `policy_overrides` and
the JSON file are for.

No `CheckConfig` knobs; runtime flags `--policy` and `--video`.

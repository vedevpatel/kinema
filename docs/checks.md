# The checks, one by one

What each check flags, how it decides, its knobs (all `CheckConfig` fields,
overridable with `kinema check --config tuning.json`), and the limits worth
knowing before you trust or dismiss a finding. The one-line versions live in
the README table; this is the long form.

Severity ladder: INFO (worth knowing), WARN (worth fixing), FAIL (blocks
training outright). Exit code is 1 on any FAIL; `--fail-on warn` also gates
on WARN. A SKIP means the check could not run on this dataset - the reason
says why, and it is not a verdict on the data.

## stats-drift

Stored normalization stats (`meta/stats.json`, or per-episode stats on v2.1)
disagree with the data: mean shift, std ratio, min/max excursions beyond the
stored range. Every training run normalizes with those numbers, so drift
means wrong-scaled inputs at training *and* inference.

Sampled loads (`--max-episodes`) are compared against the re-aggregated
per-episode stats of exactly the loaded episodes - never against
whole-dataset stats - and `stats.json` is checked against the aggregation of
all per-episode stats, the way LeRobot builds it. A sampled load without
per-episode stats can only prove gross corruption (std off >= 3x, mean off
>= 2 sigma); subtler drift needs the stats or a full load.

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

Limit: delta or EE-pose actions cannot be compared without the robot's
kinematics (most Open X-Embodiment conversions); those dims classify as
"delta command" / "different space" and sit the check out.

Knobs: `calib_warn_range_frac` (0.05), `calib_fail_range_frac` (0.15),
`calib_min_corr` (0.90), `calib_scale_warn` (0.05), `calib_scale_fail`
(0.15), `calib_scale_min_r2` (0.60), `calib_scale_line_vs_step` (1.5),
`calib_gripper_warn` (0.15).

## idle-segments

Episodes that start idle, or idle more than 30% of the time in runs longer
than 2 seconds: the command is held AND the robot is still. Idle stretches
teach the policy to do nothing - the classic "robot stays still at
inference" postmortem. A still tail after the task completes is INFO.

Limit: without recorded state, detection falls back to commands alone,
which is the old false-positive-prone behaviour - a held command while the
arm is under load reads as idle.

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

Limit: the gripper slot prior itself picks a non-gripper about 7% of the
time; on a misdetected joint a genuinely stuck motor only shows as INFO.

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

What each policy expects is declared in `src/kinema/policies.json` - chunk
size, normalization, camera/language needs, typical resolution. Adding or
updating a policy is a JSON edit. An unknown `--policy` still runs the
policy-agnostic checks and says so.

Limit: the declarations are the documented defaults; a policy trained with a
custom config may disagree with them, which is what `--config` and the JSON
file are for.

No `CheckConfig` knobs; runtime flags `--policy` and `--video`.

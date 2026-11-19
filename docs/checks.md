# The checks, one by one

What each check flags, how it decides, its configurable knobs (`CheckConfig` fields,
overridable with `kinema check --config tuning.json`), and the limits worth
knowing before you trust or dismiss a finding. The one-line versions live in
the README table; this is the long form.

Severity ladder: INFO (worth knowing), WARN (worth fixing), FAIL (blocks
training outright). Exit code is 1 on any FAIL; `--fail-on warn` also gates
on WARN. A SKIP means the check could not run on this dataset - the reason
says why, and it is not a verdict on the data.

## Health score

Score model `tiered-v3` (reports, report JSON `score_model`, and the
leaderboard all name it). Start at 100; each FAIL check costs 25; each WARN
check costs its weight below, unless a correlated-check waiver (below) waives
it; floor at 0. Billing is per check, never per finding: a check that lists
40 flagged episodes costs the same as one that lists 5, and making a check
more detailed never moves a score. The score is derived from stored per-check
verdicts and stored findings, so any stored result can be rescored without
re-running checks. `tiered-v3` is `tiered-v2` (same tiers, FAIL cost and
waiver) with WARN weights re-anchored on harm table v2, which measures 8
defects; `tiered-v2` anchored 4 of them on harm table v1.1. Weights are a
score-model decision, not automatic updates from the table: a new table
needs a new model version. Results with no `score_model` field were scored by
`legacy-v0` (hand-tuned weights); scores are not comparable across models.

Correlated checks: one defect can trip two checks. A declared waiver edge
(upstream -> downstream) waives the downstream check's WARN weight - the
upstream check then carries the LARGER of the two weights, so the pair
never pays less than the heavier defect alone - when the
upstream check is charged (WARN or FAIL), the downstream check is WARN, both
checks expose non-empty flagged episode sets in the report, and the Jaccard
overlap of those sets is >= 0.5 (inclusive): half or more of all flagged
episodes are shared, so the episodes say one defect. Otherwise both checks
pay. One edge ships:

| upstream | downstream | mechanism |
| --- | --- | --- |
| idle-segments | stuck-actuator | an episode that never moves is one idle run end to end and also a "no joint moves at all" dead recording - one dead-episode defect trips both |

FAILs are never waived (an upstream FAIL still counts as charged). Both
verdicts and all findings still show; only the charge changes. Report JSON
carries, per check, `charge` (the WARN weight after any waiver: a waived
downstream pays 0 and the upstream pays the larger of the two weights)
and, when waived, `waived_by` (the upstream
check id) and `waiver_evidence` (`shared_episodes`, `overlap`); leaderboard
dataset pages annotate the waived row. Each check's flagged episode set is
the union of its WARN/FAIL findings' `episode_index` and `details.episodes` (INFO notes are not evidence). Limit:
the waiver reads stored episode sets, so a check that reports none can never
be waived; and the gate only removes a double charge where the episodes
coincide - idle padding on episodes whose arm then works, plus dead joints
elsewhere, still pay both checks.

A FAIL is a "data is not what it claims" blocker; its cost is trust, not
measured training damage, so it is not harm-scaled. WARN weights come in
three tiers anchored on the harm table v2 (`scripts/harm_table.py`:
diffusion policy, pusht, 20k training steps, 3-5 paired seeds, 50 eval
episodes, held-out action MSE delta vs clean):

| tier | meaning | held-out MSE delta | weight |
| --- | --- | --- | --- |
| A | training killer | >= 100% | 10 |
| B | skews learning | 10-100% | 6 |
| C | review list | < 10% | 2 |

The band edges were re-anchored once for tiered-v2 and stay 100/10/0;
re-anchoring moves memberships, not edges. A measured weight is capped below
its tier when the injected defect is far stronger than what usually trips
the check, so tier A would overstate a typical WARN until a milder arm is
measured. Each check's weight carries
its basis in report JSON (`weight`, `weight_basis`):

| check | weight | basis |
| --- | --- | --- |
| calibration-drift | 10 | measured: +1475.3% (tier A) |
| stuck-actuator | 10 | measured: +697.8% (tier A) |
| action-saturation | 6 | measured: +1183.2% (tier A, capped at B: the arm clips every episode to the central 30% of each dim; the check WARNs at 5% of commands pinned) |
| outlier-episodes | 6 | measured: +107.7% (tier A, capped at B: the arm bursts +8-std actions into 5% of episodes; a typical WARN is milder) |
| idle-segments | 6 | measured: +75.1% (tier B; 3 seeds, wide spread) |
| stats-drift | 6 | measured: +16.9% (tier B) |
| dropped-frames | 2 | measured: +3.3% (tier C; seeds straddle zero) |
| duplicate-and-leakage | 2 | measured: +1.5% (tier C; 5 seeds straddle zero) |
| nan-and-normalization, deployment-readiness, timestamp-grid | 10 | placeholder |
| visual-quality | 6 | placeholder |
| any other check (task-strings, reward-done, ...) | 8 | placeholder |

The outlier-burst and stale-stats arms ran their defect seeds on an L4 and
their same-seed clean runs on an A100. Against the L4 clean runs instead
(same accelerator, different seeds) they measure +120% and +24%: neither
moves tier, so the accelerator confound does not change a weight. A tier C
weight does not mean a defect is benign: its seeds straddle zero, so these
runs cannot tell small harm from none.

Placeholder weights are the hand-tuned legacy values, kept where the harm
table has no arm. Limit: every measured number is a held-out MSE delta,
not rollout success; success floors for every arm in the v1.1 protocol
(clean 2%/4%/2% at 20k steps), so MSE is the outcome metric and no
success-rate re-check is pending.

## Harm table

The GPU table is `docs/harm-table.json`, rendered as `harm-table.html`, with raw
runs and recipes from `scripts/harm_table.py`. The CPU ridge baseline is separate:
see [harm-study.md](harm-study.md). Current evidence has eight defect arms plus
clean, diffusion policy at 20k steps, batch 32, 50 rollout episodes, and 3 seeds
per defect except duplicates (5). The dataset is pusht revision
`7628202a2180972f291ba1bc6723834921e72c19`, spread sample 144 episodes,
120 train and 24 held out. Exact episode IDs are in the JSON.

Arithmetic was corrected on 2026-11-18 from unchanged raw runs using the
repository's same-seed builder. The outlier burst mean is +107.722% and stale
stats mean +16.893% held-out MSE, replacing +120.178%/+23.730% values previously
computed against mean clean L4 runs at different seeds. Those newer defect seeds
0-2 used L4 while same-seed clean baselines used A100. Accelerator and stochastic
effects remain confounded. `accelerator_pairs` records this for each row; neither
comparison is an isolated causal estimate. Duplicate and dropped-frame deltas
straddle zero, so these runs do not establish benign behavior. Rollout success
is near the floor and cannot rank the defects.

The listed runs sum to 16.118 GPU-hours. The 11.424-hour ledger is partial, not
total compute. The published compute record reports an 18-hour ceiling; the
harness's default ceiling is 15 hours and can be overridden per run. Completed
manifest runs resume without retraining. This correction does not authorize or
launch more training. Scoring retains its historical v1.1 weights, separately
from the newer table. See [scale-test.md](scale-test.md) for loader scale checks.

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
loaded (>= 20; a percentile bootstrap over fewer is too narrow - 8 LIBERO
episodes put a correct stored std 31% outside the interval): the sampling
distribution of the pooled mean/std, resampling episodes (frames within one
are autocorrelated, so the episode is the unit). A stored value outside the
99.9% interval WARNs only when it also sits beyond the interval by at least
the full-load WARN tolerance (0.10 std devs for the mean, 5% for the std) -
tiny gaps across many dims fire on clean data, so a smaller gap is an INFO
note - and, when the stored value lies where a sample that missed the
dataset's spread would put it (a stored std above the interval, or a mean
outside it), only when the sample spans at least half of the stored
[min, max]; otherwise the gap is an INFO naming the coverage. Sampled
findings never FAIL: a deterministic spread sample cannot prove the stats
wrong, only flag them. With too few episodes only the gross backstop applies
(std off >= 3x, mean off >= 2 sigma), and only in the direction sampling
cannot explain: a sample with MORE spread than the stored std, or stored
stats impossible on their own terms (a std above the stored range, a mean
outside [min, max] by more than 10% of it). A sample narrower than the
stored stats is what a few episodes that missed the rare excursions look
like, so that gap is an INFO. Each kind reports its worst gated dim and its
worst excused dim separately, so an excused dim never hides a gated one. A full load stays the conclusive
audit.

Catalogued episodes with no frames are not part of the reference: the data
is then compared like a subset, so correct stats are never blamed for rows
that are missing (dropped-frames reports those). A dim that does not move in
the data at its own magnitude (std <= 3e-5 x max(1, |mean|), ~250 float32 ulps) is inert for
normalization whatever its stored mean/std - judged from the data alone, so
converter float32 noise in the stored stats of a constant channel (stored
std ~0.5 on a channel constant at 4096) is an INFO, while an offset joint
with real motion stays compared.

Knobs: `stats_mean_warn_sigma` (0.10), `stats_mean_fail_sigma` (0.50),
`stats_std_warn` (0.05), `stats_std_fail` (0.25), `stats_std_oversize_fail`
(3.0), `stats_range_warn` (0.01), `stats_range_fail` (0.10),
`stats_sample_std_ratio` (3.0), `stats_sample_mean_sigma` (2.0),
`stats_sample_min_episodes`, `stats_sample_bootstrap`, `stats_sample_alpha`.

## timestamp-grid

Timestamps off the 1/fps sampling grid, and non-monotonic runs. A frame table
with no `timestamp` column at all is one dataset-level FAIL (LeRobot reads it
for `delta_timestamps` windows and video lookup); a missing `frame_index`
column is one dropped-frames WARN - never invented per-episode defects.
`delta_timestamps` sampling returns misaligned windows when the grid slips.
Off-grid episodes are one aggregated finding listing them all; non-monotonic
runs stay per-episode FAILs.

Knobs: `grid_warn_frac` (0.10, as a fraction of the frame period).

## dropped-frames

`frame_index` gaps, episode lengths in the metadata vs rows in the data, and
v3 video segments vs data. A metadata lie here blocks training outright, so
mismatches are FAIL. Data files that cannot be read - corrupt pages, or a
file the Hub itself cannot serve - are a FAIL naming the files; the rest of
the dataset is still audited.

Frames that number episodes by position (0..N-1) while `meta/episodes`
catalogues other ids (`lerobot/behavior1k-task0045`: 450010, 450020, ...)
are matched by position - only when the parquet footer statistics prove it:
every frame index within 0..N-1, none overlapping the catalogue, and the
catalogue strictly increasing in file order (and in `dataset_from_index`).
LeRobot's default positional lookup trains on such a dataset, so it is a
WARN, not a refusal: selecting episodes by index and any tool keyed on
`episode_index` disagree with it. A wrong match would still show, as
per-episode length mismatches.

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

Delta commands can be normalized or transformed by the controller. Regressing
state steps on commands measures an action/state ratio, but a slope far from 1
does not establish a conversion error without a common-unit contract. Such
ratios remain INFO evidence, not scored warnings. Absolute-position calibration
checks are separate. Absolute EE-pose actions cannot be compared without the
robot's kinematics and sit the check out.

Limit (score weight): a WARN costs 10, tier A, measured - action scale 0.5x
in 25% of episodes (an injected action-scale mismatch) raised held-out action MSE by
1475.3% in harm table v1.1 (diffusion policy, 20k steps). Constant offsets on absolute commands
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

Limit (score weight): a WARN costs 6, tier B, measured - 30 idle frames
prepended raised held-out action MSE by 75.1% on average in harm table v1.1
(diffusion policy, 20k steps). The three seeds disagree by ~10x (+129%, +82%,
+14%; a paired 95% interval spans roughly -69% to +219%), so the tier is a
point estimate, not a settled one - it could be C or A at more seeds.

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
severity instead of being downgraded. A genuinely bimodal joint can
still be picked as a gripper; a real stuck or clipped joint misread as
one is reported but not gated, so it surfaces as INFO, not WARN.

Limit (score weight): a WARN costs 10, tier A, measured - one joint frozen
in 30% of episodes raised held-out action MSE by 697.8% in harm table v1.1
(diffusion policy, 20k steps). It is waived when idle-segments is also
charged and the two checks' flagged episode sets overlap at the gate (Jaccard >= 0.5) - the same dead episodes counted
once, at the larger weight; see Health score.

Knobs: `stuck_range_frac` (0.005), `stuck_warn_share` (0.25),
`stuck_unused_share` (0.90), `follow_min_command` (0.10),
`follow_ratio` (0.10).

## action-saturation

Commands pinned at EXACTLY a limit while the robot is moving: clipping
leaves a pile-up at the limit value. Sitting near an extreme is normal (rest
poses, grippers fully open/shut), as are binary or constant dims, and most
data AT the limit is a mode rather than a clipped tail (INFO) - unless the
dim is named as a joint, where long pinning is worse clipping, not better.
A command with >= 90% of all frames at exactly its min or max and >= 5% at
each is two-level (a gripper or switch, whatever its name - OpenArm's
`RJ8.pos`) and is INFO. Unnamed grippers and detected on/off commands are
reported but not gated.

Knobs: `saturation_warn_fraction` (0.05).

## nan-and-normalization

NaN/Inf anywhere (FAIL - one NaN poisons the loss for the whole batch),
constant dims (INFO - usually padding; if they ever vary at deployment,
mean/std normalization blows them up), near-zero-variance dims (WARN -
normalization amplifies noise), and dims whose normalization a few extreme
frames dominate (WARN): frames more than 20 bulk widths (q01..q99) from the
median, at most 1% of the dim's frames, that inflate its std at least 3x.
Mean/std normalization then squashes every other frame -
`IPEC-COMMUNITY/dobbe_lerobot`'s odometry glitches (z = -132 m in a handful
of its 5208 episodes) inflate the action dims' std up to 21x;
`lerobot/austin_buds_dataset`'s all-zero state dropout frames inflate
rotation entries 15-60x. The finding names the episodes. A dim flat in its
bulk (a parked joint that moves rarely) is not judged: its rare moves are its
signal. On a sampled load the frames must be in the loaded episodes;
stats-drift's sampled notes add a hint when the stored range reaches far
beyond everything loaded.

Validated before shipping: no finding on any of the 36 pinned calibration
datasets, and 1 of 40 randomly drawn board datasets - a true positive
(austin_buds' dropout frames).

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
- near: downsampled trajectories within 0.05 std RMS over all dims and grid
  points (z-scored, so no
  single large-unit dim decides). Exact up to 100k episodes with
  memory-bounded blocks (64 MB per block); above that an evenly-spaced
  sample with an INFO note saying so.
- leakage: any of the above spanning two splits declared in info.json -
  eval would be measured on training data (WARN). Split specs are
  "start:end" ranges, comma-joined for multi-range layouts
  ("14:29,50:115"); a spec that cannot be parsed is dropped whole and
  named in an INFO - its episodes go unlabelled, so leakage cannot be
  checked for them.

Limit (score weight): a WARN costs 2, tier C, measured - 20% duplicate
episodes moved held-out action MSE by -3.6% in harm table v1.1 (diffusion
policy, 20k steps): easy to detect, but seed deltas straddle zero and do not establish benign behavior. Split
leakage inflates evaluation rather than hurting training, a harm this weight
does not price.

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
of evidence agree, or one metric is far out (Tukey k=3, >= 6x the median;
end pose must also clear half the task's end-pose range). static_fraction
never earns a single-metric WARN and does not count toward agreement either -
a partly-still episode is idle-segments' job (on a dataset whose median
stillness is ~5%, "4x the median" is a pause, not a glitch; on real teleop,
spike+static pairs were a paused operator).

The far-out bar is 6x the median, not 3x: measured on the calibration halves
(2026-11-15), known-good datasets' far-out episodes cluster at 3.3-4.1x
(wide-outcome tasks end where the object lands), so 3x WARNed on 8 of 14
untouched quick datasets; an injected one-frame 30-std spike lands at
10-198x, so 6x keeps the spike floor with >2x of headroom. What remains
flagged on known-good data is real (a dead episode, two sim physics
glitches).

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

## episode-file-pointers

v3 episode metadata pointing at a parquet shard that does not contain the
episode's rows - the v2.1-to-v3.0 converter's file-rollover bug
([huggingface/lerobot#2401](https://github.com/huggingface/lerobot/issues/2401)):
the boundary episode is written into the new shard but its
`data/chunk_index` / `data/file_index` still name the previous one. LeRobot
loads only the metadata-named file, so the episode silently drops out of
training while the dataset looks complete on disk. kinema's own loader reads
the shards directly, so no other check sees it: this check compares each
catalogued episode's declared shard against the shard's actual episode
roster (one column read per shard).

Every catalogued episode whose declared shard is available locally is
checked. One WARN per broken episode (the first 10 are listed, with the
shard its rows actually live in when that shard is local). A single broken
pointer is a converter defect: re-convert or re-export with a fixed lerobot.
Found live in official exports (allenai/MolmoAct2-RT-1-Dataset,
unitreerobotics/G1_Dex3_ToastedBread_Dataset, LightwheelAI/Lightwheel-Tasks-X7S).

Known limits: v2.x layouts skip (one parquet per episode, nothing to point
wrong); a sampled audit judges only episodes whose declared shard was
downloaded - the summary reports the coverage; a shard the loader could not
read is outside coverage and reported by the loader itself.

No knobs.

## rlds-episode-counts

RLDS (Open X) datasets on GCS: dataset_info.json declares per-split shard
episode counts in shardLengths; this check compares the declaration against
the shards the bucket actually holds. FAIL when a declared split has the
wrong number of shard files or a split's shards exist with no dataset_info
entry; WARN on zero-episode declared shards. RLDS checks run in the
leaderboard's rlds lane only - they audit GCS metadata, never download
payloads.

## rlds-feature-spec

features.json sanity for RLDS datasets: FAIL when the featuresDict tree is
missing or unparseable (the legacy `{"type", "content"}` schema is read too);
SKIP for non-trajectory TFDS datasets with no `steps` sequence (VQA, prompt/
answer sets: the RLDS trajectory spec does not apply); WARN when steps lack
action or observation features; INFO when no image features are declared.
The counts, inventory, step-structure and mirror checks SKIP when a bucket
holds several shard generations and the metadata does not select one.

## rlds-shard-inventory

The RLDS analog of episode-file-pointers: shard names carry the pointer
(<split>.tfrecord-NNNNN-of-MMMMM) and the bucket listing is the existence
check. FAIL on missing (non-contiguous) shards, a filename-declared shard
total that disagrees with the bucket, or zero-byte shards; WARN on a single
shard over 2 GiB (one corrupt object kills the whole split) and on shard
files with unparseable names.

## rlds-step-structure

Sample-mode-only (off in the nightly board run): fetches the first shard's
first episode - after reading the 12-byte record header and skipping
episodes over the 64 MiB sample cap - and decodes it with kinema's
pure-Python TFRecord / tf.Example reader. FAIL when the record does not
decode or the is_first / is_last framing is broken; WARN when a steps/*
feature's value count is not a multiple of the step count.

## video-file-pointers

Episode metadata pointing at a video file the repo does not contain - the
video sibling of episode-file-pointers. LeRobot fetches the metadata-named
MP4 for an episode's camera; when that object was never uploaded (an export
interrupted mid-upload, files pruned by hand), the episode's rows load but
its video does not decode, and the failure surfaces far from the cause.

v3 pointers come from the segment columns in meta/episodes; v2.x paths are
rendered from `video_path` per episode and camera. The referenced paths are
checked against the repo's full file listing on Hub audits (no video
download needed) or against the files on disk for a local dataset - a
partial video download is not evidence about files it never fetched, so
that case skips. One finding per affected camera: INFO under 10% of
catalogued episodes, WARN at 10%, FAIL at 50% - half the dataset losing a
camera is a different defect class than a stray file. Found live in
OpenMOSS-Team/robodojo_lerobot_v3.0 (100 of 593 referenced files missing,
half the episodes affected; the twin export robodojo-lerobot-v3.0 is
complete).

Known limits: v3 exports without segment columns skip (no pointers to
resolve); episodes with no segment recorded at all are camera-coverage's
case, not this check's.

No knobs.

## task-strings

Empty task strings, frames pointing at a `task_index` the task table does
not define, episode tasks with no table entry. Language conditioning on `""`
trains a task-agnostic policy.

No knobs.

## reward-done

Constant or NaN reward, `done` never True, `done` True mid-episode with
frames after it, `done` True on every frame. Exactly two final `done` frames
(pusht's convention) are an INFO padding note. Three or more final `done`
frames still WARN. RL and reward-conditioned training get labels with no signal.

No knobs.

## visual-quality

Blur (variance of Laplacian, frames decimated with an integer stride to ~VGA
first, so one threshold works at any resolution), underexposure (95th-
percentile brightness) and overexposure (pixels clipped at the codec's
brightest level). Dark frames shrink the Laplacian's variance, so an episode
that is both dark and blurry is attributed to the lighting and counted once,
never double-reported. Per camera, sampled inside each episode's segment;
v2.1 per-episode and v3 shared MP4s both work.

Needs `--video` and the `video` extra (`pip install -e .[video]`). The nightly
board runs it on a capped spread of video shards (24 files, 400 MB). A clipped
region that sits in the same place in nearly every frame (a sim's white canvas,
a lamp in shot) is reported as a static saturated background (INFO), not as
overexposure.

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
## stats-f32-fingerprint

Were the stored per-episode stats computed from the data, or are they a
converter's float32 accumulation error stored as if it were statistics?
Some converters compute stats with a naive f32 running sum; on long
episodes the rounding is deterministic, so the stored mean and std
reproduce a naive f32 recomputation bitwise while disagreeing with the
float64 truth. On a constant channel this fabricates a nonzero std and
DC-offsets the mean: a pipeline normalizing from these stats feeds the
policy the converter's rounding error. The data itself is fine, so the
finding is an INFO (stats-drift owns material disagreement, and its
constant-dim floor covers the normalization consequence). Detection
recomputes the naive f32 stats on a sample of episodes and requires a
bitwise match on both mean and std plus a material float64 disagreement,
so honest stats never match.

## rlds-mirror-drift

The board RLDS lane compares two data-bearing GCS copies. It reports version,
episode, shard and byte differences, with bucket identity preserved. It never
substitutes a healthy mirror for an incomplete selected copy. Ambiguous shard
generations skip comparison. This is not part of the LeRobot CLI registry.

# kinema

[![PyPI](https://img.shields.io/pypi/v/kinema)](https://pypi.org/project/kinema/)
[![CI](https://github.com/vedevpatel/kinema/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/vedevpatel/kinema/actions/workflows/ci.yml?query=branch%3Amain)

Quality checks for [LeRobot](https://github.com/huggingface/lerobot) robot-learning datasets.

Your fine-tune failed and the loss looked fine. The problem is almost always in the data:
normalization stats that drifted from the actual values, calibration offset between the
leader and follower arm, episodes that are mostly the robot sitting still, timestamps off
the sampling grid. kinema finds those before you burn the GPU hours.

```bash
pip install kinema                         # or latest main: pip install "git+https://github.com/vedevpatel/kinema"

kinema check lerobot/pusht                 # straight off the Hugging Face Hub
kinema check ./my-dataset --policy pi0     # local dataset, deployment checks for pi0
kinema check hf://owner/name --json out.json --markdown report.md
kinema check owner/name@<commit> --video  # pinned revision; + blur/exposure from the MP4s
kinema check lerobot/droid_1.0.1 --max-episodes 50   # spread sample; downloads only the shards it reads
```

No `lerobot` dependency. Installs in seconds: `pyarrow`, `numpy`, `huggingface-hub`, `rich`.
Works on LeRobot **v3.0 and v2.1** layouts, local directories, `.zip`s, and Hub repo ids.

## What it looks like

Snapshot from `kinema check lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19 --policy smolvla`:

```
lerobot/pusht@7628202a2180972f291ba1bc6723834921e72c19  health score 90/100
format v3.0 | 206 episodes | 25650 frames | fps 10.0

check                   verdict summary
stats-drift             PASS    data matches meta/stats.json (worst 0% of the FAIL threshold)
timestamp-grid          PASS    evenly spaced at 10 fps and monotonic
dropped-frames          PASS    frame_index contiguous; episode lengths and video segments match the data
calibration-drift       PASS    2 joint(s) + 0 gripper(s) agree; worst offset -0.0% of range (joint 1)
idle-segments           PASS    no episode starts idle or idles >= 30%
stuck-actuator          PASS    every joint moves wherever the dataset uses it
action-saturation       PASS    no continuous action dim pinned at a limit above 5% (2 dims checked)
nan-and-normalization   PASS    no NaN/Inf; all dims have usable variance
duplicate-and-leakage   PASS    no exact, partial or near duplicates
outlier-episodes        INFO    0 episode(s) to review, 25 mild, of 206; worst: [15, 97, 107, 116, 159, 3, 4, 5, 10, 16]
embedded-images         PASS    no image features declared
camera-coverage         PASS    1 camera(s) over 206 episodes; 0 missing segments
episode-file-pointers   PASS    every shard pointer matches its file's contents (206 episodes, 1 shard(s))
video-file-pointers     PASS    all 1 metadata-referenced video files exist (the repo file listing)
task-strings            PASS    206 episodes, 1 tasks in meta/tasks; 0 finding(s)
reward-done             INFO    reward and done present
visual-quality          SKIP    pass --video to run (downloads MP4s; needs the video extra)
deployment-readiness    WARN    0 blocker(s), 1 warning(s) for smolvla
stats-f32-fingerprint   INFO

verdict: WARN; score model tiered-v3
```

Exit code is `1` on any FAIL (`--fail-on warn` to also gate on warnings), so it drops
straight into CI as a pre-training gate. Exit code `2` means the audit could not run at
all: the dataset failed to load, or `--config` / an argument was invalid.

## The checks

The long form of each check - method, severity logic, config knobs, known limits -
lives in [docs/checks.md](docs/checks.md).

| check | what it catches | why it kills fine-tunes |
|---|---|---|
| `stats-drift` | stored stats disagree with the data: mean shift > 0.1 std, std off > 5%, min/max off > 1% of range. Sampled loads are checked against the per-episode stats of exactly the loaded episodes, and `stats.json` against all per-episode stats; v2.1 uses aggregated `episodes_stats.jsonl` like LeRobot does | every run normalizes with those numbers; drift = wrong-scaled inputs at training *and* inference |
| `timestamp-grid` | timestamps off the 1/fps grid, non-monotonic runs | `delta_timestamps` sampling returns misaligned windows |
| `dropped-frames` | `frame_index` gaps, metadata length vs data mismatch | corrupt episodes; the metadata lie blocks training outright |
| `calibration-drift` | still-frame action-state disagreement per joint: a constant offset (leader/follower drift) or a scale error that grows across the range (a line that beats a two-level step, so backlash is not called scale). Only on dims where action is a position command for the same joint (names agree, action tracks state); grippers get a free-end test (the open end, where nothing blocks them); delta commands get an informational scale ratio | leader/follower drift (the 17-degree case from LeRobot PR #3762) survives recording *and* training |
| `idle-segments` | episodes that start idle, or idle >30% of the time in runs >2s (command held AND robot still); a still tail after the task is INFO | teaches the policy to do nothing - the classic "robot stays still at inference" postmortem |
| `stuck-actuator` | episodes where nothing moves; a joint that moves in most episodes but not in a few; gripper never firing on pick/place | the policy learns to ignore that joint |
| `action-saturation` | >5% of commands during motion pinned at exactly the min/max of a continuous, non-gripper dim | clipped commands the policy will imitate |
| `nan-and-normalization` | NaN/Inf, zero-variance dims, std < 1e-6 | one NaN poisons the loss for the whole batch |
| `duplicate-and-leakage` | exact-copy episodes, partial copies (an opening only two episodes share is a copied beginning, not a scripted start), near-duplicates (exact scan up to 100k episodes with memory-bounded blocks, an even sample with a note above that) | silent metric inflation |
| `outlier-episodes` | IQR outliers on the harmful side only: duration and end pose within a task, jerk/velocity per dim in std units without grippers or velocity commands (recognised by name or by tracking the state's delta, not its level - a spike beyond a velocity dim's own step scale still counts). WARN when two independent signals agree (stillness does not count - that is idle-segments' job) or one is far out (Tukey k=3, >=6x median - tuned using regression data, not a pristine holdout - and for end pose >= half the end-pose range) | a handful of broken episodes dominate the gradient |
| `embedded-images` | image features stored inside the parquet shards instead of MP4 video chunks | bloated shards; every frame decodes on read |
| `camera-coverage` | episodes with no video for a declared camera (v3 segment metadata, or the v2.1 files with `--video`) | a modality silently drops out for those episodes |
| `episode-file-pointers` | v3 episode metadata naming a shard that does not hold the episode's rows (the [lerobot#2401](https://github.com/huggingface/lerobot/issues/2401) converter rollover bug) | LeRobot loads only the metadata-named file - the episode silently vanishes from training |
| `video-file-pointers` | episode metadata naming a video file the repo does not contain (v3 segment columns or the v2.x `video_path` template, checked against the repo file listing) | the episode's rows load but its video does not decode - the failure surfaces far from the cause |
| `task-strings` | empty task strings, frames pointing at a `task_index` the task table does not define, episode tasks with no table entry | language conditioning trains on `""` or an unrecoverable task |
| `reward-done` | constant or NaN reward, `done` never True, `done` True mid-episode with frames after it | RL / reward-conditioned training gets labels with no signal |
| `visual-quality` | blur (variance of Laplacian, frames decimated to ~VGA first so one threshold works at any resolution), underexposure (95th-percentile brightness) and overexposure (pixels clipped at the codec's brightest level), per camera, sampled inside each episode's segment (v2.1 per-episode and v3 shared MP4s) | low-quality examples measurably hurt training - 20% low-quality data raised training loss ~30% in [HF's GIGO study](https://huggingface.co/blog/robotdata/gigo). Needs `--video` and the `video` extra |
| `deployment-readiness` | episode length vs the policy's action chunk, stats coverage, quantile-norm support, declared shapes, and what the policy conditions on (camera input, task language, typical resolution - declared per policy in `src/kinema/policies.json`) | the openpi #906 failure mode: low loss, failed robot |

The table above is the LeRobot `kinema check` registry. It also includes
`stats-f32-fingerprint`, an INFO-only exact converter-rounding provenance check.

The leaderboard's separate RLDS lane runs these metadata checks on GCS objects,
not through `kinema check` or the MCP LeRobot loader. Nontrajectory feature trees
and ambiguous generations are not forced into trajectory FAIL verdicts.

| RLDS check | evidence |
|---|---|
| `rlds-episode-counts` | declared split counts vs the selected shard generation |
| `rlds-feature-spec` | known current or legacy trajectory feature structure |
| `rlds-shard-inventory` | missing or zero-byte shards in an unambiguous generation |
| `rlds-step-structure` | optional sampled TFRecord episode framing, off by default |
| `rlds-mirror-drift` | disclosed differences between data-bearing bucket copies |

## Deployment readiness

Point it at the policy you're about to train:

```bash
kinema check ./so100-pickplace --policy pi0
```

It verifies the dataset can actually serve that policy: episodes long enough to fill an
action chunk, stats.json covering what the normalizer consumes, `q01/q99` quantiles when
the policy normalizes that way (openpi does), that declared shapes and timestamps
match what the policy's data loader will assume, and that the dataset carries what the
policy conditions on - camera input for vision policies, task strings for
language-conditioned ones, and a note when a camera is smaller than the resolution the
policy is usually trained at.

What each policy expects is declared in `src/kinema/policies.json` (chunk size,
normalization, camera/language needs, typical resolution; aliases like `pi0.5`
resolve to their family entry). Adding or updating a policy is a JSON edit, no
code change; an unknown `--policy` still gets the policy-agnostic checks.

## The fine-tune copilot (MCP)

`kinema mcp` starts an MCP server over stdio - the "why did my fine-tune fail"
layer on top of the same checks. Point your MCP client at it:

```json
{
  "mcpServers": {
    "kinema": {
      "command": "uvx",
      "args": ["kinema", "mcp"]
    }
  }
}
```

(For unreleased changes on main, use
`"args": ["--from", "git+https://github.com/vedevpatel/kinema", "kinema", "mcp"]`. A
`pip install -e .` checkout works too: `"command": "kinema", "args": ["mcp"]`.)

Four tools:

- `diagnose_finetune(dataset, policy, symptom, policy_overrides?, max_episodes?)` -
  ranks the data causes that explain a failed run, each with check evidence, a
  docs citation and a next step. Every check runs whatever the symptom says;
  the symptom only decides which causes rank first. Policies come from `policies.json`;
  cause clusters cover normalization, idle/static segments, metadata integrity,
  calibration and clipping, video and cameras, conditioning (language, timestamps)
  and data quality (duplicates, leakage, outliers, unreadable files, reward/done).
  RLDS keywords are recognized, but RLDS execution is a separate board lane, not
  the LeRobot MCP loader. Training with a custom
  chunk size or mean/std normalization? Pass `policy_overrides`
  (`{"chunk_size": 10, "quantile_norm": false}`) and the diagnosis computes
  against *your* setup, not the declared defaults.
- `run_checks(dataset, checks?, ...)` - the raw engine over MCP: a capped summary
  (verdict, score, top findings) sized for client output limits, with the full
  JSON report written to a file (`<dataset>.kinema-report-<timestamp>.json` next
  to a local dataset, else `~/.cache/kinema/reports/`; an explicit `report_path`
  must be absolute, must not pass through a symbolic link - system-owned ones such
  as macOS's `/tmp` are resolved - and only replaces an earlier kinema report).
- `similar_datasets(dataset? or report_path?, k?)` - ranks evaluated board reports
  (`report_path` here is an absolute path to a kinema JSON report to read)
  by shared WARN/FAIL evidence and shape. Unevaluated reports and the query itself
  are excluded. This is retrieval, not a guarantee that a match is suitable.
- `list_checks()` - the check catalog from the registry, with docs links.

`dataset` is an absolute local path (directory or `.zip`) or a Hub repo id
(`owner/name`). Relative paths are rejected: the server runs in whatever
directory your client launched it from, not yours.

The answers show their work instead of claiming your data is fine: every answer
enumerates which checks ran, which skipped and why, whether the load was sampled,
which action dims sat out, any findings outside the v0 clusters (verbatim), and
the blind spots no check covers (unit conventions, joint order, delta-vs-absolute
at inference, control-loop fps). Cause entries live in `src/kinema/causes.json` -
adding a failure-mode cluster or a policy is a JSON edit, and a schema test keeps
the map honest against the engine.

## Tuning the thresholds

Many numeric thresholds live in `CheckConfig` and can be overridden with a JSON
file of field names to values. Other rules are module constants, including
outlier agreement and relative-distance gates, metadata shares, and RLDS limits;
those require a code change. For configurable fields:

```bash
kinema check ./my-dataset --config tuning.json
```

```json
{
  "stats_std_fail": 0.4,
  "idle_min_seconds": 5,
  "calib_fail_range_frac": 0.2
}
```

Unknown keys and non-numeric values are rejected, so a typo fails loudly instead
of silently tuning nothing. `kinema check --help` lists the rest of the flags.

Resident frame memory is capped separately: `KINEMA_FRAME_BUDGET_BYTES` (default
1 GiB). Above it the frame arrays are memory-mapped to disk instead of held in
RAM, so a full load of a DROID-sized dataset cannot OOM a smaller machine.

## The leaderboard

The most-downloaded LeRobot datasets on Hugging Face (`--limit 200`: the top
lane), plus a capped block of seats per flagship publisher whose LeRobot-format
datasets the search does not surface (`FLAGSHIP_ORG_SEATS`, `FLAGSHIP_SEEDS` and
nested-repo seats in `scripts/leaderboard.py`), a probation lane of the newest
uploads, and an RLDS lane for the Open X-Embodiment datasets on Google Cloud
Storage (metadata checks only). The board audits nightly; the dataset list
itself is refreshed weekly by the `leaderboard-discover` workflow into
`docs/worklist.json`, and a row is pruned only after it is missing from three
fresh discoveries in a row:
**[vedevpatel.github.io/kinema](https://vedevpatel.github.io/kinema/)**

A new WARN or FAIL reaches the board only if it survives the robustness gate:
it is re-run with the data rescaled, episodes reordered and (on sampled loads) the sample halved,
and a verdict that moves publishes as INFO; each check publishes at most 10 new
WARN/FAIL verdicts a night ([docs/checks.md](docs/checks.md#robustness-gate)).

Board audits a sample of up to 250 episodes per dataset (`--max-episodes 250`),
spread evenly across what the download budget reaches - v2.x datasets keep one
file per episode and are capped at 40 files, so they audit fewer - to keep a
nightly run feasible; a local `kinema check` scans the full dataset. Board audits also fetch a
24-file spread of the audited episodes' video shards, hard-capped at 400 MB,
so visual-quality runs on the board; a dataset whose spread exceeds the cap
falls back to the middle shard that still fits the budget, and skips only
when no shard can be budgeted (the fallback and the skip are logged in the workflow run).
camera-coverage reads per-episode video segments from the metadata where the
format records them (v3); on v2.1 datasets it skips on the board rather than
count videos a sampled audit did not fetch as missing - a full local
`kinema check` still measures coverage from disk.
Sampled and full scans can score differently because coverage and applicable
checks differ. A sample-vs-full stats comparison is not a sound explanation for
a score difference: the checker uses exact per-episode stats where available.
Inspect the per-check evidence, engine stamp and video coverage before comparing
scores. Score examples are snapshots, not current promises.

## Try it on the demo dataset

```bash
kinema synth /tmp/demo --defects stats-drift,idle-segments,calibration-drift
kinema check /tmp/demo
```

## Roadmap

- Auto-fix suggestions with safe rewrites (stats recompute, idle trimming).

## Development

```bash
pip install -e .[dev]
python -m pytest                      # unit tests (synthetic defects + false-positive regressions)
ruff check src scripts tests          # lint
mypy src/kinema                       # type check
KINEMA_E2E=1 python -m pytest        # + real Hub datasets (pinned revisions) and the sensitivity guard
python scripts/calibrate.py --quick  # how big a defect must be before each check catches it
```

### The CI guards

The nightly e2e job runs on pinned real datasets: known-healthy data must not
FAIL, known defects must still be caught, and the sensitivity guard
(`tests/test_sensitivity.py`) injects defects of known size. Its **floors** are
detection rates a little under the measured ones - a drop means a threshold
lost its teeth. Its **ceilings** are the measured baseline: how many untouched
datasets each check already flags.

The pinned list has `tune` and `eval` regression groups, not a pristine held-out
evaluation. Guard floors and ceilings used measurements from the eval group, and
related corpora can overlap. `pusht-subtask` is grouped with `pusht` under tune to
avoid that known family split. Report these guards as regression evidence, not an
unbiased generalization estimate. Failed pinned loads make calibration incomplete
rather than publishing a smaller success-looking sample.

These suites need the Hub. If huggingface.co itself is unreachable they skip with a
visible reason (`tests/netcheck.py`) instead of failing CI on someone else's outage.

A ceiling failure is not automatically a false positive: it can also be a real
issue a change newly surfaces (a new check firing, a sharper threshold). When
one fails:

1. Reproduce locally: `KINEMA_E2E=1 python -m pytest tests/test_sensitivity.py -q`.
2. Read the new finding on the flagged dataset. If it is a false positive, fix
   the check - do not raise the ceiling to make it pass.
3. If the finding is real and the new count is the intended behavior, raise the
   number in `CEILINGS` in the same commit, with the message saying what the
   finding was and why it stays. The same rule applies in reverse to floors:
   lower one only with a reason in the commit message.

The full pinned list (~700 MB) runs weekly in the `calibration` workflow and on
demand from the Actions tab; the report lands as a run artifact.

Apache-2.0. Named for *kinema*, "motion" - the thing your dataset is supposed to contain.

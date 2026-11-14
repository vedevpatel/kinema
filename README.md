# kinema

Quality checks for [LeRobot](https://github.com/huggingface/lerobot) robot-learning datasets.

Your fine-tune failed and the loss looked fine. The problem is almost always in the data:
normalization stats that drifted from the actual values, calibration offset between the
leader and follower arm, episodes that are mostly the robot sitting still, timestamps off
the sampling grid. kinema finds those before you burn the GPU hours.

```bash
# not on PyPI yet - install from the repo
pip install "git+https://github.com/vedevpatel/kinema"
# or from source: git clone https://github.com/vedevpatel/kinema && pip install -e kinema

kinema check lerobot/pusht                 # straight off the Hugging Face Hub
kinema check ./my-dataset --policy pi0     # local dataset, deployment checks for pi0
kinema check hf://owner/name --json out.json --markdown report.md
kinema check owner/name@<commit> --video  # pinned revision; + blur/exposure from the MP4s
kinema check lerobot/droid_1.0.1 --max-episodes 50   # spread sample; downloads only the shards it reads
```

No `lerobot` dependency. Installs in seconds: `pyarrow`, `numpy`, `huggingface-hub`, `rich`.
Works on LeRobot **v3.0 and v2.1** layouts, local directories, `.zip`s, and Hub repo ids.

## What it looks like

`kinema check lerobot/pusht --policy smolvla`:

```
lerobot/pusht  health score 79/100
format v3.0 | 206 episodes | 25650 frames | fps 10.0

 check                 verdict  summary
 stats-drift           PASS     data matches meta/stats.json (worst 0% of the FAIL threshold)
 timestamp-grid        PASS     evenly spaced at 10 fps and monotonic
 dropped-frames        PASS     frame_index contiguous; episode lengths and video segments match the data
 calibration-drift     PASS     2 joint(s) + 0 gripper(s) agree; worst offset -0.0% of range (joint 1)
 idle-segments         PASS     no episode starts idle or idles >= 30%
 stuck-actuator        PASS     every joint moves wherever the dataset uses it
 action-saturation     PASS     no continuous action dim pinned at a limit above 5% (2 dims checked)
 nan-and-normalization PASS     no NaN/Inf; all dims have usable variance
 duplicate-and-leakage PASS     no exact, partial or near duplicates
 outlier-episodes      WARN     2 episode(s) to review, 23 mild, of 206; worst: [107, 159, 15, 97, ...]
 embedded-images       PASS     no image features declared
 camera-coverage       PASS     1 camera(s) over 206 episodes; 0 missing segments
 task-strings          PASS     206 episodes, 1 tasks in meta/tasks; 0 finding(s)
 reward-done           WARN     done is True before the last frame in 206 episodes - the trajectory continues after termination
 visual-quality        SKIP     pass --video to run (downloads MP4s; needs the video extra)
 deployment-readiness  WARN     0 blocker(s), 1 warning(s) for smolvla

verdict: WARN (0 fail, 4 warn, 23 info)
```

Exit code is `1` on any FAIL (`--fail-on warn` to also gate on warnings), so it drops
straight into CI as a pre-training gate.

## The checks

The long form of each check - method, severity logic, config knobs, known limits -
lives in [docs/checks.md](docs/checks.md).

| check | what it catches | why it kills fine-tunes |
|---|---|---|
| `stats-drift` | stored stats disagree with the data: mean shift > 0.1 std, std off > 5%, min/max off > 1% of range. Sampled loads are checked against the per-episode stats of exactly the loaded episodes, and `stats.json` against all per-episode stats; v2.1 uses aggregated `episodes_stats.jsonl` like LeRobot does | every run normalizes with those numbers; drift = wrong-scaled inputs at training *and* inference |
| `timestamp-grid` | timestamps off the 1/fps grid, non-monotonic runs | `delta_timestamps` sampling returns misaligned windows |
| `dropped-frames` | `frame_index` gaps, metadata length vs data mismatch | corrupt episodes; the metadata lie blocks training outright |
| `calibration-drift` | mean action-state offset on still frames, per joint, independent of position. Only on dims where action is a position command for the same joint (names agree, action tracks state); grippers excluded | leader/follower drift (the 17-degree case from LeRobot PR #3762) survives recording *and* training |
| `idle-segments` | episodes that start idle, or idle >30% of the time in runs >2s (command held AND robot still); a still tail after the task is INFO | teaches the policy to do nothing - the classic "robot stays still at inference" postmortem |
| `stuck-actuator` | episodes where nothing moves; a joint that moves in most episodes but not in a few; gripper never firing on pick/place | the policy learns to ignore that joint |
| `action-saturation` | >5% of commands during motion pinned at exactly the min/max of a continuous, non-gripper dim | clipped commands the policy will imitate |
| `nan-and-normalization` | NaN/Inf, zero-variance dims, std < 1e-6 | one NaN poisons the loss for the whole batch |
| `duplicate-and-leakage` | exact-copy episodes, partial copies (an opening only two episodes share is a copied beginning, not a scripted start), near-duplicates (exact scan up to 100k episodes with memory-bounded blocks, an even sample with a note above that) | silent metric inflation |
| `outlier-episodes` | IQR outliers on the harmful side only: duration and end pose within a task, jerk/velocity per dim in std units without grippers or velocity commands (recognised by name or by tracking the state's delta, not its level - a spike beyond a velocity dim's own step scale still counts). WARN when two independent signals agree or one is far out (Tukey k=3, >=3x median, and for end pose >= half the end-pose range) | a handful of broken episodes dominate the gradient |
| `embedded-images` | image features stored inside the parquet shards instead of MP4 video chunks | bloated shards; every frame decodes on read |
| `camera-coverage` | episodes with no video for a declared camera (v3 segment metadata, or the v2.1 files with `--video`) | a modality silently drops out for those episodes |
| `task-strings` | empty task strings, frames pointing at a `task_index` the task table does not define, episode tasks with no table entry | language conditioning trains on `""` or an unrecoverable task |
| `reward-done` | constant or NaN reward, `done` never True, `done` True mid-episode with frames after it | RL / reward-conditioned training gets labels with no signal |
| `visual-quality` | blur (variance of Laplacian, frames decimated to ~VGA first so one threshold works at any resolution), underexposure (95th-percentile brightness) and overexposure (pixels clipped at the codec's brightest level), per camera, sampled inside each episode's segment (v2.1 per-episode and v3 shared MP4s) | bad visuals cost ~30% of training loss ([HF GIGO](https://huggingface.co/blog/robotdata/gigo)). Needs `--video` and the `video` extra |
| `deployment-readiness` | episode length vs the policy's action chunk, stats coverage, quantile-norm support, declared shapes, and what the policy conditions on (camera input, task language, typical resolution - declared per policy in `src/kinema/policies.json`) | the openpi #906 failure mode: low loss, failed robot |

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
      "args": ["--from", "git+https://github.com/vedevpatel/kinema", "kinema", "mcp"]
    }
  }
}
```

(The plain `uvx kinema mcp` form works once the package is on PyPI - a planned
milestone, not done yet. Until then a `pip install -e .` checkout works too:
`"command": "kinema", "args": ["mcp"]`.)

Three tools:

- `diagnose_finetune(dataset, policy, symptom, policy_overrides?, max_episodes?)` -
  ranks the data causes that explain a failed run, each with check evidence, a
  docs citation and a next step. Every check runs whatever the symptom says;
  the symptom only decides which causes rank first. Scope is deliberately narrow: the pi0 family
  (`pi0`, `pi05`/`pi0.5`, `pi0_fast`) and two failure-mode clusters -
  normalization/stats mismatch and idle/static segments. Training with a custom
  chunk size or mean/std normalization? Pass `policy_overrides`
  (`{"chunk_size": 10, "quantile_norm": false}`) and the diagnosis computes
  against *your* setup, not the declared defaults.
- `run_checks(dataset, checks?, ...)` - the raw engine over MCP: a capped summary
  (verdict, score, top findings) sized for client output limits, with the full
  JSON report written to a file (`<dataset>.kinema-report-<timestamp>.json` next
  to a local dataset, else `~/.cache/kinema/reports/`; an explicit `report_path`
  must be absolute and only replaces an earlier kinema report).
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

Every threshold lives in `CheckConfig` with a conservative default. Override any
of them without editing code - a JSON file of field names to values:

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

The most-downloaded LeRobot datasets on Hugging Face, audited nightly:
**[vedevpatel.github.io/kinema](https://vedevpatel.github.io/kinema/)**

Board audits a sample of 50 episodes per dataset (`--max-episodes 50`), spread
evenly across the dataset, to keep a nightly run of 200 datasets feasible; a
local `kinema check` scans the full dataset.
Sampled and full scans can score differently - for example pusht scores 89 on the
board and 79 on a full local scan, because a 50-episode sample's recomputed stats
are compared against stats.json values computed over all 206 episodes.

## Try it on the demo dataset

```bash
kinema synth /tmp/demo --defects stats-drift,idle-segments,calibration-drift
kinema check /tmp/demo
```

## Roadmap

- Auto-fix suggestions with safe rewrites (stats recompute, idle trimming).
- Video checks on the leaderboard: v3 shared-MP4 attribution works locally (`--video`); the nightly board skips it because the MP4s run to hundreds of MB per camera.

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

The pinned calibration list splits into disjoint `tune` and `eval` halves
(`scripts/calibration_datasets.txt`): thresholds, floors and ceilings are derived
from the tune half, and the sensitivity guard runs on the eval half - the numbers
that protect the checks were never tuned on the data that guards them.

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

# kinema

Quality checks for [LeRobot](https://github.com/huggingface/lerobot) robot-learning datasets.

Your fine-tune failed and the loss looked fine. The problem is almost always in the data:
normalization stats that drifted from the actual values, calibration offset between the
leader and follower arm, episodes that are mostly the robot sitting still, timestamps off
the sampling grid. kinema finds those before you burn the GPU hours.

```bash
pip install kinema
kinema check lerobot/pusht                 # straight off the Hugging Face Hub
kinema check ./my-dataset --policy pi0     # local dataset, deployment checks for pi0
kinema check hf://owner/name --json out.json --markdown report.md
```

No `lerobot` dependency. Installs in seconds: `pyarrow`, `numpy`, `huggingface-hub`, `rich`.
Works on LeRobot **v3.0 and v2.1** layouts, local directories, `.zip`s, and Hub repo ids.

## What it looks like

`kinema check lerobot/pusht --policy smolvla`:

```
lerobot/pusht  health score 84/100
format v3.0 | 206 episodes | 25650 frames | fps 10.0

 check                 verdict  summary
 stats-drift           PASS     data matches meta/stats.json (worst 0% of the FAIL threshold)
 timestamp-grid        PASS     all episodes on the 10 fps grid and monotonic
 dropped-frames        INFO     frame_index contiguous and episode lengths match the data
 calibration-drift     PASS     max joint offset 0.0% of range (joint 1) over 2 comparable joints
 idle-segments         PASS     no episode starts idle or idles >= 30%
 stuck-actuator        PASS     every joint moves wherever the dataset uses it
 action-saturation     PASS     no continuous action dim pinned at a limit above 5% (2 dims checked)
 nan-and-normalization PASS     no NaN/Inf; all dims have usable variance
 duplicate-and-leakage PASS     no exact or near duplicates
 outlier-episodes      WARN     3 episode(s) to review, 22 mild, of 206; worst: [107, 159, 50, ...]
 visual-quality        SKIP     pass --video to run (downloads MP4s; needs the video extra)
 deployment-readiness  WARN     1/206 episodes shorter than smolvla's action chunk

verdict: WARN (0 fail, 4 warn, 23 info)
```

Exit code is `1` on any FAIL (`--fail-on warn` to also gate on warnings), so it drops
straight into CI as a pre-training gate.

## The checks

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
| `duplicate-and-leakage` | exact-copy episodes, near-duplicates | silent metric inflation |
| `outlier-episodes` | IQR outliers on the harmful side only: duration and end pose within a task, jerk/velocity per dim in std units without grippers. WARN when two independent signals agree or one is far out (Tukey k=3, >=3x median) | a handful of broken episodes dominate the gradient |
| `visual-quality` | blur (variance of Laplacian), lighting, per camera | bad visuals cost ~30% of training loss ([HF GIGO](https://huggingface.co/blog/robotdata/gigo)). Needs `--video` and the `video` extra |
| `deployment-readiness` | episode length vs action chunk, stats coverage, quantile-norm support, declared shapes, timestamp grid | the openpi #906 failure mode: low loss, failed robot |

## Deployment readiness

Point it at the policy you're about to train:

```bash
kinema check ./so100-pickplace --policy pi0
```

It verifies the dataset can actually serve that policy: episodes long enough to fill an
action chunk, stats.json covering what the normalizer consumes, `q01/q99` quantiles when
the policy normalizes that way (openpi does), and that declared shapes and timestamps
match what the policy's data loader will assume.

## The leaderboard

The most-downloaded LeRobot datasets on Hugging Face, audited nightly:
**[vedevpatel.github.io/kinema](https://vedevpatel.github.io/kinema/)**

Board audits load the first 50 episodes per dataset (`--max-episodes 50`) to keep a
nightly run of 200 datasets feasible; a local `kinema check` scans the full dataset.
Sampled and full scans can score differently - for example pusht scores 67 on the
board and 84 on a full local scan, because a 50-episode sample's recomputed stats
are compared against stats.json values computed over all 206 episodes.

## Try it on the demo dataset

```bash
kinema synth /tmp/demo --defects stats-drift,idle-segments,calibration-drift
kinema check /tmp/demo
```

## Roadmap

- **Fine-tune failure copilot (MCP)**: dataset + training config + symptom in, ranked
  probable causes with evidence out. The checks above are the engine; the copilot is the
  diagnosis layer.
- Auto-fix suggestions with safe rewrites (stats recompute, idle trimming).
- Video checks on by default once remote-video attribution lands for v3 shared MP4 files.

## Development

```bash
pip install -e .[dev]
python -m pytest                      # unit tests (synthetic defects + false-positive regressions)
ROLLOUTQA_E2E=1 python -m pytest      # + end-to-end against lerobot/pusht
```

Apache-2.0. Named for *kinema*, "motion" - the thing your dataset is supposed to contain.

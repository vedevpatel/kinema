# Severity proposal for `timestamp-grid` (not implemented)

Status: **proposal only.** This PR changes no detector, threshold, severity,
score, README, `causes.json`, board or history. Implementing any item below
would need a separate PR with tests, and the nightly workflow would then
regenerate the published output.

The evidence is from three cases, each one episode on one loader release with
the pyav decoder only: phase 2 ([report](../../loader-verification/2026-11-22/report.md))
and this experiment ([report](report.md)). No universal threshold follows from
that, so the proposals below change what a finding *claims* and where it sits
pending verification. They do not introduce new numeric cutoffs.

## Current behavior (main at `33c66f41`, `src/kinema/checks/timestamps.py`)

| Condition | Current severity | Current consequence wording |
| --- | --- | --- |
| First timestamp `> 0.5/fps` | **FAIL** if any video feature exists, otherwise INFO. The size of the offset is not considered | "video lookup uses timestamps; a nonzero start alone does not prove decoding will miss (compare the video timeline)" (narrowed by PR #19) |
| Spacing deviation `> grid_warn_frac` (10%) of a period | **WARN**, one finding per dataset | "LeRobot's default tolerance_s (1e-4 s) rejects these - needs tolerance_s >= …" |
| Spacing deviation `> 1e-4 s` but within 10% | INFO | "load with tolerance_s >= …" |

`causes.json` (`timestamp-grid-broken`) maps FAIL to high confidence and WARN to medium.

## Evidence available

| Case | Pattern | Release / path | Raw lookup (phase 2) | Shifted loader-derived lookup (this work) | Full loader |
| --- | --- | --- | --- | --- | --- |
| `r71:c1:f1` droid ep 0 | small start (0.733 s), on-grid spacing, video covers 0–14.13 s | 0.3.3 / pyav direct decoder | 9/9 ok | 60/60 batches ok (max distance 0) | blocked earlier by `KeyError('stats')` |
| `r94:c1:f1` openarm ep 0 | epoch-scale start (~1.784e9 s), float64 storage, video 0–14.53 s | 0.3.3 / pyav direct decoder | 12/12 tolerance errors | 80/80 batches tolerance errors; float32 collapses every query to `1784331520.0` | `check_timestamps_sync` raises at construction |
| `r1:c1:f0` FineART ep 40455 | declared 30 fps, rows and video actually 90 Hz ("67% deviation") | 0.4.0 / pyav direct decoder | 16/16 ok | 80/80 batches ok (max distance 1.53e-5 s); delta windows span about 1/3 of the intended time | blocked earlier by `language` dtype |

Source facts: 0.4.0 has no `check_timestamps_sync` call, while 0.3.3 calls it
at construction. Loader-derived shifted queries are stored row timestamps
(after float32 conversion), not `raw + k/fps`. Padded queries reuse the
boundary row.

## Proposals

### 1. Small nonzero start: stop emitting FAIL on the offset alone

*Evidence:* one episode, `r71:c1:f1`. A 0.733 s start decoded at raw
timestamps and under every preregistered shifted query set. The video
timeline begins at 0 and covers the row timestamps.

*Proposal:* when the check has nothing to compare the start against, report a
nonzero start below FAIL, as unverified WARN or INFO, and say what was not
compared. Keep FAIL only where the check can show the data timestamps fall
outside the video timeline, for example from v3
`videos/<key>/from_timestamp`/`to_timestamp` or video duration metadata. That
is a check the detector does not perform today.

*Not supported:* that small offsets are harmless in general. Two things were
not tested: image-content alignment (the droid video has 213 frames for 192
rows, so the rows may sit inside a longer recording, or may not) and the other
22 listed droid episodes. No numeric "small" threshold is proposed.

### 2. Epoch-scale timestamps: keep FAIL and name the mechanism

*Evidence:* one episode, `r94:c1:f1`, under 0.3.3. Construction rejects the
spacing because float32 conversion destroys it. The direct decoder misses at
raw and at every shifted query, and all delta windows collapse to a single
float32 value.

*Proposal:* keep FAIL for this pattern when video features exist. The message
should state the mechanisms observed: construction failure in 0.3.3, and lookup
misses against a video timeline that starts near 0. It should not reuse the
generic "decoding will miss" wording that also fires for small offsets. The
detector would need a magnitude split between proposals 1 and 2. **Where that
split should sit is open.** It could be tied to float32 resolution or to video
duration. Either can be derived from the source arithmetic, but neither was
tested here, and one dataset cannot calibrate it.

*Not supported:* the behavior of epoch-scale timestamps under 0.4.0/v3.0
(no spacing check exists there; untested), or for the other 39 listed
episodes.

### 3. Grid deviation: change the claim; keep WARN for now

*Evidence:* one episode, `r1:c1:f0`, under 0.4.0. The current wording
("LeRobot's default tolerance_s rejects these") was contradicted on the tested
path: 0.4.0 has no spacing check, and every raw and loader-derived shifted
lookup decoded within tolerance. The observed problem is different. The
declared fps (30) disagrees with the actual rate (90 Hz), so
`delta_timestamps = i/30` becomes row offset `i`, which spans `i/90` s.

*Proposal:*

- Make the tolerance-rejection wording version-qualified. It describes the
  0.3.3 `check_timestamps_sync` path. Release 0.3.3 declares
  `CODEBASE_VERSION = "v2.1"`, its version check only warns on a v3.0 dataset,
  and it reads the v2 metadata layout. Whether it would open FineART at all
  was not tested.
- Report uniform spacing at a rate different from the declared fps as a
  declared-fps mismatch, separate from irregular jitter. The consequence to
  state is that temporal windows cover a different time span than the policy
  config intends.
- Keep WARN. The effect on a trained policy is not measured, so FAIL is not
  supported.

*Not supported:* any statement about irregular, non-uniform jitter. No such
case was tested here. The 10% `grid_warn_frac` line and the INFO band between
1e-4 s and 10% are untouched and unevaluated.

### 4. Boundary and padding: no change proposed

The loader clamps out-of-episode rows to the boundary row and sets
`*_is_pad`. Every padded query decoded in the two cases where lookups work.
Queries outside the episode's time range failed only in the `nominal_grid`
comparator, which corresponds to no loader query. Nothing here argues for a
boundary-specific severity.

### 5. Backend scope

Every result above uses pyav. torchcodec is the default whenever it is
installed, and it maps timestamps to frame indices with `round(ts *
average_fps)` before its tolerance check. It could not load in either
environment. Until it is tested, any severity text should avoid implying
backend-independent behavior.

## Open items before implementation

- torchcodec, and 0.5.1 decoder paths, for all three patterns.
- A full-loader run on a dataset with each pattern that constructs without
  schema repair.
- Additional episodes per case, and datasets beyond these three, before
  choosing any magnitude cutoff.
- Image-content alignment for small offsets.
- The float32 distance margin on long v3 video timelines: about 1.5e-5 to
  3e-5 s was observed near 300 s, and the margin shrinks as timestamps grow.

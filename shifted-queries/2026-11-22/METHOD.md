# Shifted-timestamp queries: method (2026-11-22)

This directory is separate from both frozen directories:
[`audits/2026-11-22/`](../../audits/2026-11-22/) (phase 1) and
[`loader-verification/2026-11-22/`](../../loader-verification/2026-11-22/) (phase 2).
It reads them and never writes to them. It tests one question that phase 2 left
open: **what happens to video lookup when a training query is shifted away from
the current frame?** The cases are `r1:c1:f0` (FineART, grid deviation) and
`r71:c1:f1` (droid, small nonzero start). `r94:c1:f1` (epoch-scale start) is a
comparator.

## What the loader actually queries

The tested releases are LeRobot 0.3.3 for the v2.1 droid case and the v2.0
epoch-scale case, and 0.4.0 for v3.0 FineART. Phase 2 used the same releases,
and their environment records and source hashes are in
`loader-verification/2026-11-22/evidence/environments.json`. `plan.py`
re-hashes the installed files against the values in `expectations.json`. Paths
below are relative to `site-packages/lerobot/` in each environment.

| Step | 0.3.3 | 0.4.0 |
| --- | --- | --- |
| Offsets | `datasets/factory.py::resolve_delta_timestamps` builds `delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]` for every `observation.*` key | same |
| Offset check at construction | `utils.py::check_delta_timestamps`: each delta must be a multiple of `1/fps` within `tolerance_s` | same |
| Rows | `utils.py::get_delta_indices`: `round(d * fps)`, then `_get_query_indices`: `max(ep_start, min(ep_end - 1, idx + delta))`, with `<key>_is_pad` true outside the episode | same, with `ep_start`/`ep_end` from `meta.episodes[ep]["dataset_from_index"/"dataset_to_index"]` |
| Query timestamps | `_get_query_timestamps`: `torch.stack(hf_dataset.select(rows)["timestamp"]).tolist()`, the **stored timestamps of the clamped rows** | same (`hf_dataset[rows]`) |
| dtype | `hf_transform_to_torch` applies `torch.tensor(x)` to Python floats, which gives float32 | same |
| Video timeline | `_query_videos` passes the timestamps as-is | adds `videos/<key>/from_timestamp` |
| Data-spacing check | `__init__` calls `check_timestamps_sync(diffs vs 1/fps, tolerance_s)` and raises on violation | **absent**: `check_timestamps_sync` is not called anywhere in 0.4.0 |
| Decoder (pyav) | `video_utils.py::decode_video_frames_torchvision` seeks to the keyframe before `min(q)`, reads until `pts >= max(q)`, and asserts `min_q |float32(q) - float32(pts)| < tolerance_s` | identical function body |
| Default backend | `get_safe_default_codec()` returns `"torchcodec"` if the module spec exists | same |

Two consequences shape the experiment:

1. With the default `delta_timestamps=None`, `__getitem__` queries only the
   current row's timestamp. Phase 2's raw-timestamp probes already covered
   that. Shifted queries exist only when a policy config supplies
   `observation_delta_indices`.
2. The shifted queries are **stored row timestamps**, not `raw + k/fps`.
   Phase 2's remaining-gap wording (`raw timestamp + k/30`) is therefore not
   a LeRobot code path. It is kept as a separate, labeled `nominal_grid`
   comparator.

## Preregistered query set

No dataset default fixes the offsets. The rule, recorded in `expectations.json`,
was fixed before any probe: include every policy config shipped in the tested
release whose default `observation_delta_indices` is not `None`, evaluated at
its dataclass defaults, plus one `no_delta` control. That gives `smolvla` `[0]`,
`diffusion` `[-1, 0]`, `vqbet` `[-4..0]` and `tdmpc` `[0..5]`. `plan.py`
rescans `policies/**/configuration_*.py` from source text and refuses to plan if
the set differs.

The anchors are first, middle, worst-grid and last, using phase 2's worst-grid
rule. For each case, camera, anchor and config there is one batch: the exact
list `__getitem__(anchor)` would hand the decoder for that key. Padded batches,
which contain at least one clamped row, are kept apart from in-range batches.
`nominal_grid` queries (`current + k/fps`, k in the nonzero offsets above) are
single calls, and they are flagged when they fall outside the episode's stored
time range.

The whole query set, with rows, `is_pad`, stored and float32 timestamps,
`from_timestamp` and intended versus effective offsets, is written to
`plan/<case>.json` before any decoder runs. `preregistration.json` hashes
those plans together with `expectations.json` and the harness scripts.
`probe.py` refuses to run if any of them has changed.

## Execution

`probe.py` runs in the phase-2 isolated environment for each case
(`/tmp/kinema-phase2-v2` for 0.3.3 and `/tmp/kinema-phase2-v3` for 0.4.0).
Before any probe it:

- re-hashes every input against phase 2's ledger, checks the Hub snapshot
  revision in the cache path, and checks the frozen audit's LFS and info.json
  hashes where present (`plan.py` performs the first verification);
- compares the installed package set with the phase-2 freeze;
- checks the delta rows against upstream `check_delta_timestamps` and
  `get_delta_indices`, and checks the float32 query timestamps against
  upstream `hf_transform_to_torch`. For 0.3.3 it also checks them against a
  `load_dataset("parquet") + set_transform + select` replay of
  `load_hf_dataset`;
- makes one attempt on the default backend (torchcodec), whose outcome is
  recorded and never substituted.

Each batch then goes to upstream `decode_video_frames(path, queries, 1e-4,
"pyav")`. The returned frame pts come from an independent replay of the same
VideoReader window. The replay must reproduce the upstream outcome, and its
frames must be tensor-equal to the upstream output. If they are not, the batch
is a `harness_discrepancy` and counts as unresolved.

This is a **direct-decoder experiment**. `LeRobotDataset` is not constructed,
patched or subclassed. FineART's `language` dtype failure and droid's
`KeyError('stats')` are not bypassed, so a decoder success never establishes
full-loader or training success.

## Reproduce

```sh
D=shifted-queries/2026-11-22
# Inputs: phase-2 files under /tmp/kinema-phase2-inputs (recreate with
# loader-verification/2026-11-22/prepare.py at the pinned revisions; see that METHOD.md).
/tmp/kinema-phase2-v3/bin/python -I $D/plan.py r1:c1:f0
/tmp/kinema-phase2-v2/bin/python -I $D/plan.py r71:c1:f1
/tmp/kinema-phase2-v2/bin/python -I $D/plan.py r94:c1:f1
/tmp/kinema-phase2-v3/bin/python -I $D/probe.py r1:c1:f0
/tmp/kinema-phase2-v2/bin/python -I $D/probe.py r71:c1:f1
/tmp/kinema-phase2-v2/bin/python -I $D/probe.py r94:c1:f1
# Post-hoc, descriptive only (not preregistered): stream rate, frame count and pts range
/tmp/kinema-phase2-v3/bin/python -I $D/video_facts.py r1:c1:f0
/tmp/kinema-phase2-v2/bin/python -I $D/video_facts.py r71:c1:f1
/tmp/kinema-phase2-v2/bin/python -I $D/video_facts.py r94:c1:f1
python $D/make_report.py   # needs numpy only
python -m pytest tests/test_shifted_queries.py -q
/tmp/kinema-phase2-v2/bin/python -I -m unittest discover -s $D -p 'test_probe_env.py'
```

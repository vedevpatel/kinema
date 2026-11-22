# Phase 1 verification rubric (fixed before adjudication)

The unit is one published WARN or FAIL finding, not a check verdict, dataset,
score, or product. The frozen `board.json.gz` is the assertion being audited; it
is never evidence that the assertion is true. The board's row revision and
engine stamp must be retained. A later upstream revision cannot overturn the
earlier row. No finding is adjudicated from an unpinned live dataset.

## Labels

- **confirmed**: independent observation at the recorded revision supports
  the finding's specific factual assertion. For a compatibility claim, test
  the relevant LeRobot loader or document the precise contract and failure.
  An observed anomaly alone does not confirm causal language such as
  "broken motor", "training harm", or "copy-paste".
- **false alarm**: independent observation at the recorded revision
  contradicts the specific assertion, including a verified representation or
  sampling artifact. Explain the contradiction; a changed current dataset
  is insufficient.
- **unresolved**: original inputs cannot be recovered, the verification
  path is unsupported or incomplete, or the signal can be reproduced but
  its interpretation remains uncertain. Missing video, incomplete listings,
  unexamined shards, or untested loader claims require this label.

Record **signal exists** separately for heuristic warnings. Exact duplicate
action arrays, static joints, idle periods, extremes, outliers, brightness and
blur may be real signals without establishing a data defect or its cause.
Signal reproduction alone cannot promote such a case to confirmed. A
structure violation (a missing referenced file, absent rows at a declared
pointer, non-finite values, a missing required field) can be confirmed from
raw files and a complete pinned listing where needed. Dataset semantic claims
need external context or inspection.

For each case, record the pinned source, commands, files or API calls, observed
values, reason, and any limitation. Do not import kinema detection helpers into
the verifier. The sample selection was frozen before these labels were assigned.

## Sampling rule

`scripts/assurance_sample.py` enumerates every WARN/FAIL finding in the
frozen `docs/data.json`; an absent `docs/dataset/` page never removes a row.
The key is the board row/check/finding index. For each check ID in lexical
order, select two findings by SHA-256 of the fixed seed and key, covering one
of each severity where both exist. Prefer a previously unused dataset within
each draw. Fill remaining slots with FAIL findings, then all findings, using
the same ranking. This intentionally covers rare checks and FAILs more than
their natural board frequency; any precision for this sample is not board
prevalence. The complete frame counts, seed, keys and selected findings are
in `sample.json`.

## Known limits to seek during review

Check missing dataset pages, duplicated or correlated findings, nested dataset
roots, RLDS versus Hub sources, unverifiable video and loader claims, and
revision drift. Current main's detector implementation may differ from the
stored engine, so re-running its check is not independent evidence. Existing
regression fixtures influenced development and are a separate miss exercise,
not an independent benchmark. Do not infer a real-world miss rate.

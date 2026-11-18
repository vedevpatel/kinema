# Harm studies

`python scripts/harm_study.py --json /tmp/harm.json` runs the CPU ridge-regression
baseline. It uses pinned pusht episodes, holds out action-prediction targets,
and measures injected recipes against clean data. It is a limited model check,
not evidence of rollout success or a general estimate across robots.

The GPU harness is `scripts/harm_table.py`. Each manifest records completed,
interrupted and spent runs; completed runs resume rather than retrain. Use
`--help` to inspect the current training budget, rollout count, seed and ceiling
options before starting work. GPU runs spend money and are not part of the unit
suite. `--dry-run` exercises recipe checks on CPU, not GPU-policy outcomes.

Published evidence is `harm-table.json`, including exact dataset revision,
train/held-out episode IDs, unchanged raw runs, derived rows and accelerator
pairs. Derived percentages use the clean run at the same seed. Cross-GPU pairs
are confounded and are disclosed. The corrected newer rows cannot support an
isolated causal harm claim. Seed deltas straddling zero do not prove a defect is
benign. The historical score model and latest measurements are separate.

Before publishing new evidence, regenerate from original run records, check
that the validator reproduces every percentage from those records, inspect
accelerator comparability, report total listed-run compute separately from a
partial ledger, and render the table for inspection. Do not substitute a
success-looking subset when a pin or run failed.

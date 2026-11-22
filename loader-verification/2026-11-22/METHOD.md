# Loader consequence verification, 2026-11-22

This directory is separate from the frozen phase 1 audit. `expectations.json` was
written before running any loader experiment. It preserves the eleven original
unresolved assertions, pins, and the observations required to confirm or refute
their *loader consequences*. A metadata discrepancy alone cannot confirm a loader
consequence. Results are scoped to an explicitly named loader release, input
revision, episode and operation.

The four named checks contain twelve findings in the frozen sample. One
(`r413:c2:f2`) was already confirmed as a metadata mismatch and did not make an
unresolved loader-consequence claim; it is outside this phase's eleven cases.

## Environment and replay

The experiments used local CPU only. LeRobot 0.3.3 has `CODEBASE_VERSION =
"v2.1"` and explicitly accepts v2.0 global stats; it was selected for v2.0
and v2.1 cases. [LeRobot's v3 documentation](https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx)
says v3.0 support begins in 0.4.0; that release was selected for v3 cases.
LeRobot 0.5.1 (also declaring v3.0) was a secondary FineART schema check after
0.4.0 failed on its custom `language` dtype. Full package freezes and installed
upstream source hashes are in `evidence/environments.json` and the package
lists it references.

To recreate the three isolated environments without touching the project
`.venv`, use Python 3.11 for the first two and 3.12 for the optional third:

```sh
uv venv --python 3.11 /tmp/kinema-phase2-v2
uv venv --python 3.11 /tmp/kinema-phase2-v3
uv venv --python 3.12 /tmp/kinema-phase2-v3-051
for kind in v2 v3 v3_secondary; do
  case "$kind" in
    v2) env=/tmp/kinema-phase2-v2 ;;
    v3) env=/tmp/kinema-phase2-v3 ;;
    v3_secondary) env=/tmp/kinema-phase2-v3-051 ;;
  esac
  uv pip install --python "$env/bin/python" 'torch==2.7.1+cpu' 'torchvision==0.22.1+cpu' --index-url https://download.pytorch.org/whl/cpu
  rg -v '^(torch|torchvision)==' "loader-verification/2026-11-22/evidence/${kind}_packages.txt" > "/tmp/kinema-phase2-${kind}-other.txt"
  uv pip install --python "$env/bin/python" --no-deps -r "/tmp/kinema-phase2-${kind}-other.txt"
done
```

`report.json` lists the exact recovery and probe commands for each case. Run
`prepare.py` with the relevant environment's Python to fetch only `meta/` and
the named episode parquet/video paths at the frozen revision. Input files live
under `/tmp/kinema-phase2-inputs/` and are never committed; `evidence/` stores
their paths and SHA-256 hashes. The frozen audit's LFS hashes, where present,
are checked during acquisition and report generation.

## FineART verdict correction

The original `fineart_decoder_expectations.json` treated successful lookups at
raw frame timestamps as a refutation of the tolerance claim. That criterion
was too weak: shifted training queries (`raw timestamp + k/30`) were not
probed, and the full loader stopped earlier on the source's custom `language`
dtype. The preregistered file and its 16 successful raw-timestamp probes are
preserved as historical evidence. The consequence is **unresolved** in the
current report. No schema repair or substitute dataset was used.

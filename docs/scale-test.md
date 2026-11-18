# Loader scale checks

The harness runs growing episode budgets in separate processes and records
wall time, peak RSS and exit status. It does not train a policy or include video.

```bash
python scripts/scale_test.py --repo lerobot/droid_1.0.1 --episodes 50 100 250 500 --out /tmp/scaletest
```

Inspect `--help` before using it. Network downloads can be large; ensure storage
and memory headroom rather than treating the episode cap as a byte cap. The
frame-budget option controls when arrays spill to disk, not total process RSS.
Hub cache population changes cold-fetch time, so report cache state with timing.

Each run writes a report JSON if it finishes, plus a `.meta.json` with exit code,
wall time, peak RSS and CPU times. A killed or failed run is a result, not a
passing scale claim. A partial load is not a full-dataset correctness audit.
Compare in-memory and spilled arrays and check verdicts on the same input before
claiming parity. This document defines the procedure; it claims no new measured
throughput or maximum supported dataset size.

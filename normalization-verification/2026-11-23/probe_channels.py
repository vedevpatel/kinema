"""Raw-channel inspection for the near-constant cases (r390:c7:f8, r393:c7:f14).

Run with the phase-2 v2 environment (pyarrow + numpy only):
    /tmp/kinema-phase2-v2/bin/python -I probe_channels.py

Reads observation.state from every data file prepare.py selected under the
frozen sampling rule, and reports per file and pooled: the flagged dim, the
frozen control dim and the stored meta/stats.json entries for both. It also
checks the sin/cos structure that the GR00T modality layout implies (dims
28:56 = sin(0:28), 56:84 = cos(0:28)). It saves the flagged and control
columns for every sampled row, plus every 25th full 256-dim row, for the
normalizer probes. No kinema code is imported and no detector rule is applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

EXP = common.load_json(common.EXPECTATIONS)
FULL_ROW_STRIDE = 25


def column(path: str, name: str) -> np.ndarray:
    t = pq.read_table(path, columns=[name])
    return np.asarray(t.column(name).combine_chunks().flatten().to_numpy(zero_copy_only=False),
                      dtype=np.float64).reshape(t.num_rows, -1)


def ints(path: str, name: str) -> np.ndarray | None:
    schema = pq.read_schema(path)
    if name not in schema.names:
        return None
    return np.asarray(pq.read_table(path, columns=[name]).column(name).to_numpy(), dtype=np.int64)


def channel_summary(x: np.ndarray) -> dict:
    s = common.finite_summary(x)
    fin = x[np.isfinite(x)]
    s.update(
        n_exact_zero=int((fin == 0).sum()),
        n_unique=int(np.unique(fin).size),
        n_abs_gt_1e_11=int((np.abs(fin) > 1e-11).sum()),
        n_abs_gt_1e_9=int((np.abs(fin) > 1e-9).sum()),
        q01=float(np.quantile(fin, 0.01)) if fin.size else None,
        q50=float(np.quantile(fin, 0.50)) if fin.size else None,
        q99=float(np.quantile(fin, 0.99)) if fin.size else None,
    )
    return s


def stored(stats: dict, feature: str, dim: int) -> dict:
    entry = stats.get(feature, {})
    out = {}
    for k, v in entry.items():
        a = np.asarray(v, dtype=np.float64).reshape(-1)
        if a.size > dim:
            out[k] = float(a[dim])
        elif a.size == 1:
            out[k] = float(a[0])
    return out


def main() -> None:
    common.check_preregistration(EXP["frozen_files"])
    inputs = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "inputs.json")["cases"]}
    common.SAMPLES.mkdir(parents=True, exist_ok=True)
    result = {"schema": "kinema.normverify.channels/1",
              "environment": common.environment_record(["numpy", "pyarrow"]), "cases": []}
    for case in EXP["cases"]:
        if case["kind"] != "near-constant":
            continue
        cid, feat = case["case_id"], case["finding"]["details"]["feature"]
        d = case["finding"]["details"]["dim"]
        inp = inputs[cid]
        stats = common.load_json(Path(inp["root"]) / "meta/stats.json")
        ctrl = common.control_dim({j: stored(stats, feat, j)["std"] for j in range(*case["slice"])}, d, case["slice"])
        per_file, flagged_all, control_all, full_rows, episodes_all = [], [], [], [], []
        sincos_err = []
        for f in inp["selected_data_files"]:
            x = column(f["cache_path"], feat)
            ep = ints(f["cache_path"], "episode_index")
            flagged_all.append(x[:, d])
            control_all.append(x[:, ctrl])
            full_rows.append(x[::FULL_ROW_STRIDE])
            if ep is not None:
                episodes_all.append(ep)
            q = x[:, 0:28]
            sincos_err.append(float(max(np.abs(np.sin(q) - x[:, 28:56]).max(), np.abs(np.cos(q) - x[:, 56:84]).max())))
            per_file.append({
                "path": f["path"], "sha256": f["sha256"], "rows": int(x.shape[0]), "width": int(x.shape[1]),
                "episodes": sorted({int(e) for e in ep}) if ep is not None else None,
                "flagged": channel_summary(x[:, d]), "control": channel_summary(x[:, ctrl]),
                "feature_nonfinite": int((~np.isfinite(x)).sum()),
                "sincos_max_abs_err": sincos_err[-1],
            })
        fa, ca = np.concatenate(flagged_all), np.concatenate(control_all)
        sample_path = common.SAMPLES / f"{cid.replace(':', '_')}.npz"
        np.savez(sample_path, flagged=fa, control=ca, full_rows=np.concatenate(full_rows),
                 episode_index=np.concatenate(episodes_all) if episodes_all else np.array([]))
        slice_ = case["slice"]
        sib = [j for j in range(slice_[0], slice_[1]) if j != d]
        sib_std = [stored(stats, feat, j).get("std") for j in sib]
        result["cases"].append({
            "case_id": cid, "feature": feat, "dim": d, "control_dim": ctrl,
            "files": per_file,
            "pooled": {"files": len(per_file), "rows": int(fa.size),
                       "episodes": int(np.unique(np.concatenate(episodes_all)).size) if episodes_all else None,
                       "flagged": channel_summary(fa), "control": channel_summary(ca)},
            "stored_stats": {"flagged": stored(stats, feat, d), "control": stored(stats, feat, ctrl),
                             "slice": slice_, "sibling_std_median": float(np.median(sib_std)),
                             "sibling_std_min": float(np.min(sib_std)), "sibling_std_max": float(np.max(sib_std)),
                             "stat_names": sorted(stats.get(feat, {}).keys())},
            "sincos_max_abs_err": float(max(sincos_err)),
            "sample_file": {"path": str(sample_path), "sha256": common.sha256_file(sample_path),
                            "full_row_stride": FULL_ROW_STRIDE},
        })
    common.write_json(common.EVIDENCE / "channels.json", result)
    print("ok")


if __name__ == "__main__":
    main()

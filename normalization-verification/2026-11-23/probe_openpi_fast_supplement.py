"""Deviation D6 (not preregistered): S8 for PI0-FAST-mapped VLABench checkpoints.

    env-openpi-vlabench/bin/python -I probe_openpi_fast_supplement.py

probe_openpi.py fork builds S4 arrays with the stats_config (PI0, action_dim 32)
transforms, so 7-dim FAST checkpoint stats could not broadcast against them
(a harness shape artifact recorded in evidence/openpi_fork.json). In the fork,
VLABenchInputs pads state/actions to the model action_dim; for PI0-FAST
(action_dim 7) pad_to_dim is a no-op on VLABench's 7-dim arrays, and
DeltaActions acts on the first dims only, so the FAST pipeline's arrays equal the
first 7 dims of the PI0 pipeline's (its action horizon is 10, a subset of the
50-step chunk values). This script rebuilds S3/S4 exactly as probe_openpi.py
does, slices to 7 dims, and runs Normalize with each FAST checkpoint's stats and
its mapped config's use_quantile_norm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.argv = [sys.argv[0], "fork"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import probe_openpi as po  # noqa: E402  (module setup: fork stack, offline, private scratch)

import numpy as np  # noqa: E402


def items_for(case: dict, inp: dict, dc, cfg) -> list[dict]:
    root = Path(inp["root"])
    meta = po.lerobot_dataset.LeRobotDatasetMetadata(case["source"]["repo"], root=root, local_files_only=True)
    delta = {k: [t / meta.fps for t in range(cfg.model.action_horizon)] for k in dc.action_sequence_keys}
    ds = po.lerobot_dataset.LeRobotDataset(case["source"]["repo"], root=root, episodes=[0], delta_timestamps=delta, local_files_only=True)
    base = po._data_loader.TransformedDataset(ds, [po._transforms.PromptFromLeRobotTask(meta.tasks)]) if dc.prompt_from_task else ds
    out = []
    for i in range(len(ds)):
        x = base[i]
        for t in [*dc.repack_transforms.inputs, *dc.data_transforms.inputs]:
            x = t(x)
        out.append({"state": np.asarray(x["state"]), "actions": np.asarray(x["actions"])})
    return out


def main() -> None:
    common.check_preregistration(["common.py", "probe_openpi.py", "expectations.json"])
    inputs = common.load_json(common.EVIDENCE / "inputs.json")
    by_id = {c["case_id"]: c for c in inputs["cases"]}
    out = {"schema": "kinema.normverify.openpi_fast_supplement/1", "deviation": "D6",
           "environment": common.environment_record(["openpi", "lerobot", "jax", "numpy"]), "cases": []}
    for case in po.EXP["cases"]:
        if "openpi_fork" not in case.get("configurations", {}):
            continue
        conf = case["configurations"]["openpi_fork"]
        _, cfg, dc = po.config_record(conf["stats_config"], po.SCRATCH / "empty_assets")
        items = items_for(case, by_id[case["case_id"]], dc, cfg)
        fast_flags = {}
        for name in conf["config_names"]:
            r, _, fdc = po.config_record(name, po.SCRATCH / "empty_assets")
            if r["model_type"] == "ModelType.PI0_FAST":
                fast_flags[name] = bool(fdc.use_quantile_norm)
        rec = {"case_id": case["case_id"], "stats_config": conf["stats_config"], "frames": len(items), "checkpoints": []}
        for r in po.released_for(case, inputs):
            if r["mapped_config"] not in fast_flags:
                continue
            ns = po._normalize.load(Path(r["local"]).parent)
            dims = {k: int(np.asarray(v.mean).size) for k, v in ns.items()}
            sliced = [{"state": it["state"][..., :dims["state"]], "actions": it["actions"][..., :dims["actions"]]} for it in items]
            padded_tail_zero = all(not np.any(it["state"][..., dims["state"]:]) for it in items)
            rec["checkpoints"].append({
                "source": r["source"], "sha256": r["sha256"], "mapped_config": r["mapped_config"],
                "added_by_deviation": r.get("added_by_deviation"), "stats_dims": dims,
                "pi0_pipeline_state_tail_is_zero_padding": bool(padded_tail_zero),
                "normalize": po.normalize_record(ns, fast_flags[r["mapped_config"]], sliced)})
        out["cases"].append(rec)
    common.write_json(common.EVIDENCE / "openpi_fork_fast_supplement.json", out)
    print("ok")


if __name__ == "__main__":
    main()

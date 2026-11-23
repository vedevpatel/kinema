"""openpi stats-selection and normalization probes for the missing-stats cases.

    upstream openpi (r67):  env-openpi/bin/python -I probe_openpi.py upstream
    VLABench fork (r15, r51): env-openpi-vlabench/bin/python -I probe_openpi.py fork

Stages, each recorded separately so a failure in one is never read as a result
of a later one:
  S1 config selection  - pinned TrainConfig -> DataConfig (repack keys, asset id,
                         use_quantile_norm, norm stats found in an empty assets dir)
  S2 training guard    - data_loader.transform_dataset with no assets norm stats
  S3 construction      - the stack's own pinned LeRobotDataset on episode 0
  S4 batch selection   - repack + data transforms (as compute_norm_stats applies
                         them): which dataset keys become `state` / `actions`
  S5 training stats    - openpi RunningStats over the episode-0 frames
  S6 training normalize- openpi Normalize with S5 stats, per config
  S7 inference load    - checkpoints.load_norm_stats on S5 stats saved with
                         normalize.save (the inference loader), then Normalize
  S8 released stats    - normalize.load on each selected released checkpoint's
                         norm_stats.json, Normalize on the S4 arrays
  C1 control           - Normalize(None) on the S4 arrays (mechanism control)
Episode 0 only: the stats in S5-S7 are episode-0 stats, not dataset stats.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

STACK = sys.argv[1]
SCRATCH = common.SCRATCH / f"openpi-{STACK}"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=str(SCRATCH / "hf"),
                  OPENPI_DATA_HOME=str(SCRATCH / "openpi_data"), JAX_PLATFORMS="cpu")
# the upstream-pinned lerobot rejects the deprecated LEROBOT_HOME; the fork's older lerobot only reads it
os.environ["LEROBOT_HOME" if STACK == "fork" else "HF_LEROBOT_HOME"] = str(SCRATCH / "lerobot_home")

import numpy as np  # noqa: E402
import openpi.shared.normalize as _normalize  # noqa: E402
import openpi.training.checkpoints as _checkpoints  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.data_loader as _data_loader  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402
from lerobot.common.datasets import lerobot_dataset  # noqa: E402

EXP = common.load_json(common.EXPECTATIONS)


def repack_structure(dc) -> dict | None:
    for t in dc.repack_transforms.inputs:
        if isinstance(t, _transforms.RepackTransform):
            return dict(t.structure)
    return None


def config_record(name: str, assets: Path) -> tuple[dict, object, object]:
    cfg = _config.get_config(name)
    scope = "DataConfigFactory.create (full)"
    create_error = None
    try:
        dc = cfg.data.create(assets, cfg.model)
    except OSError as e:
        # FAST configs build the FAST tokenizer from the Hub inside create(); offline isolation
        # blocks it. create_base_config is the function that sets use_quantile_norm (source pins).
        create_error = common.error_record(e)
        if STACK == "fork" and type(cfg.data).__name__ == "SimpleDataConfig":
            raise  # the fork's SimpleDataConfig sets use_quantile_norm outside create_base_config
        dc = cfg.data.create_base_config(assets, cfg.model) if STACK == "upstream" else cfg.data.create_base_config(assets)
        scope = "DataConfigFactory.create_base_config only (model and data transforms not built)"
    return {"selection_scope": scope, "create_error": create_error,
        "config": name, "model_class": type(cfg.model).__name__, "model_type": str(cfg.model.model_type),
        "action_horizon": int(cfg.model.action_horizon), "repo_id": dc.repo_id, "asset_id": dc.asset_id,
        "use_quantile_norm": bool(dc.use_quantile_norm), "norm_stats_found_in_empty_assets": dc.norm_stats is not None,
        "repack": repack_structure(dc) if dc.repack_transforms.inputs else None, "action_sequence_keys": list(dc.action_sequence_keys),
        "prompt_from_task": bool(dc.prompt_from_task),
    }, cfg, dc


def stats_record(ns: dict) -> dict:
    out = {}
    for k, s in ns.items():
        out[k] = {"dims": int(np.asarray(s.mean).size), "has_q01_q99": s.q01 is not None and s.q99 is not None,
                  "std_min": float(np.min(s.std)), "std_max": float(np.max(s.std)),
                  "q_range_min": float(np.min(np.asarray(s.q99) - np.asarray(s.q01))) if s.q01 is not None else None}
    return out


def normalize_record(ns, use_q: bool, items: list[dict]) -> dict:
    try:
        norm = _transforms.Normalize(ns, use_quantiles=use_q)
        outs = [norm({"state": it["state"], "actions": it["actions"]}) for it in items]
    except Exception as e:  # noqa: BLE001 - every outcome is evidence
        return common.error_record(e)
    return {"outcome": "ok", "use_quantiles": use_q,
            "state": common.finite_summary(np.stack([o["state"] for o in outs])),
            "actions": common.finite_summary(np.stack([o["actions"] for o in outs]))}


def run_case(case: dict, inp: dict, released: list[dict]) -> dict:
    cid, conf = case["case_id"], case["configurations"][f"openpi_{STACK}"]
    root = Path(inp["root"])
    rec = {"case_id": cid, "stack": STACK, "configs": [], "stages": {}}
    empty_assets = SCRATCH / "empty_assets"
    empty_assets.mkdir(exist_ok=True)
    dcs = {}
    for name in conf["config_names"]:
        try:
            r, cfg, dc = config_record(name, empty_assets)
            dcs[name] = (cfg, dc)
        except Exception as e:  # noqa: BLE001
            r = {"config": name, **common.error_record(e)}
        rec["configs"].append(r)
    primary = conf["stats_config"]
    cfg, dc = dcs[primary]
    # S2: the real training guard with no assets norm stats
    try:
        _data_loader.transform_dataset([], dc)
        rec["stages"]["S2_training_guard"] = {"outcome": "no_error"}
    except Exception as e:  # noqa: BLE001
        rec["stages"]["S2_training_guard"] = common.error_record(e)
    # S3: construction with the stack's pinned lerobot, episode 0, action chunks
    try:
        meta_kwargs = {"local_files_only": True} if STACK == "fork" else {}
        meta = lerobot_dataset.LeRobotDatasetMetadata(case["source"]["repo"], root=root, **meta_kwargs)
        delta = {k: [t / meta.fps for t in range(cfg.model.action_horizon)] for k in dc.action_sequence_keys}
        ds_kwargs = {"local_files_only": True} if STACK == "fork" else {"revision": case["source"]["revision"], "download_videos": False}
        ds = lerobot_dataset.LeRobotDataset(case["source"]["repo"], root=root, episodes=[0], delta_timestamps=delta, **ds_kwargs)
        rec["stages"]["S3_construction"] = {
            "outcome": "ok", "len": len(ds), "fps": meta.fps, "codebase_version": meta.info.get("codebase_version"),
            "meta_stats_is_none": meta.stats is None,
            "meta_stats_keys": sorted(meta.stats) if meta.stats else [], "delta_timestamps_keys": sorted(delta),
            "dataset_feature_keys": sorted(meta.features)}
    except Exception as e:  # noqa: BLE001
        rec["stages"]["S3_construction"] = common.error_record(e)
        return rec
    # S4: the transforms compute_norm_stats applies (repack + data transforms + RemoveStrings)
    try:
        base = _data_loader.TransformedDataset(ds, [_transforms.PromptFromLeRobotTask(meta.tasks)]) if dc.prompt_from_task else ds
        pipeline = [*dc.repack_transforms.inputs, *dc.data_transforms.inputs]
        items = []
        for i in range(len(ds)):
            x = base[i]
            for t in pipeline:
                x = t(x)
            items.append({"state": np.asarray(x["state"]), "actions": np.asarray(x["actions"])})
        raw_keys = sorted(base[0].keys())
        rec["stages"]["S4_batch_selection"] = {
            "outcome": "ok", "frames": len(items), "raw_item_keys": raw_keys,
            "state_shape": list(items[0]["state"].shape), "actions_shape": list(items[0]["actions"].shape),
            "repack": repack_structure(dc),
            "canonical_label_in_item": {k: (k in raw_keys) for k in ("observation.state", "action")}}
    except Exception as e:  # noqa: BLE001
        rec["stages"]["S4_batch_selection"] = common.error_record(e)
        return rec
    # S5: RunningStats as compute_norm_stats does, over episode 0 only
    try:
        rs = {k: _normalize.RunningStats() for k in ("state", "actions")}
        for k in rs:
            values = np.stack([it[k] for it in items])
            # both compute_norm_stats scripts feed 2-D batches (upstream reshapes inside update)
            rs[k].update(values.reshape(-1, values.shape[-1]))
        ns = {k: v.get_statistics() for k, v in rs.items()}
        rec["stages"]["S5_training_stats"] = {"outcome": "ok", "scope": "episode 0 only", "stats": stats_record(ns)}
    except Exception as e:  # noqa: BLE001
        rec["stages"]["S5_training_stats"] = common.error_record(e)
        return rec
    # S6: training-stage Normalize with S5 stats, per pinned config's selection
    rec["stages"]["S6_training_normalize"] = {
        name: normalize_record(ns, dcs[name][1].use_quantile_norm, items) for name in dcs}
    rec["stages"]["S6_note"] = (f"every config's use_quantile_norm is applied to the arrays produced by {primary}'s "
                                "transforms and S5 stats; configs with another action_dim pad differently in their own pipeline")
    # S7: save as compute_norm_stats does; load with the inference loader
    try:
        asset_dir = SCRATCH / "assets" / cid.replace(":", "_")
        _normalize.save(asset_dir / dc.asset_id, ns)
        loaded = _checkpoints.load_norm_stats(asset_dir, dc.asset_id)
        rec["stages"]["S7_inference_load"] = {
            "outcome": "ok", "loader": "openpi.training.checkpoints.load_norm_stats", "keys": sorted(loaded),
            "normalize": {name: normalize_record(loaded, dcs[name][1].use_quantile_norm, items) for name in dcs}}
    except Exception as e:  # noqa: BLE001
        rec["stages"]["S7_inference_load"] = common.error_record(e)
    # S8: released checkpoint stats selected by the frozen rule
    s8 = []
    for r in released:
        entry = {"source": r["source"], "asset_id": r["asset_id"], "mapped_config": r["mapped_config"],
                 "candidates": r.get("candidates"), "added_by_deviation": r.get("added_by_deviation"), "sha256": r["sha256"]}
        try:
            rel = _normalize.load(Path(r["local"]).parent)
            entry["stats"] = stats_record(rel)
            if r["mapped_config"] and r["mapped_config"] in dcs:
                use_q = dcs[r["mapped_config"]][1].use_quantile_norm
                entry["normalize"] = normalize_record(rel, use_q, items)
            else:
                entry["normalize"] = {"outcome": "not_run", "reason": "no pinned executable config mapped to this checkpoint"}
        except Exception as e:  # noqa: BLE001
            entry.update(common.error_record(e))
        s8.append(entry)
    rec["stages"]["S8_released_stats"] = s8
    # C1: mechanism control - Normalize with no stats is the identity in openpi
    out = _transforms.Normalize(None)({"state": items[0]["state"]})
    rec["stages"]["C1_none_control"] = {"label": "mechanism control, not a dataset result",
                                        "identity": bool(np.array_equal(out["state"], items[0]["state"]))}
    return rec


def released_for(case: dict, inputs: dict) -> list[dict]:
    """Released norm_stats files the frozen rule maps to this case and stack."""
    mapping = list(case["configurations"][f"openpi_{STACK}"]["released"])
    if STACK == "fork":
        # D5: the frozen org-scan rule selected checkpoints the frozen list omitted; map them by the
        # frozen mechanical rule (model type, asset id, aligned flag; unanimous use_quantile_norm)
        asset = case["configurations"]["openpi_fork"]["fork_repo_id"]
        listed = {m["source"] for m in mapping}
        inv = EXP["fork_config_inventory"]["rows"]
        quant = {r[0]: r[3] for r in inv}
        for ck in inputs["checkpoints"]:
            if ck.get("rule") == "vlabench_org_scan" and ck["selected"] and ck["repo"] not in listed \
                    and any(a == f"assets/{asset}/norm_stats.json" for a in ck["norm_stats_assets"]):
                cands = common.vlabench_candidates(ck["repo"], asset, inv)
                cfg = cands[0] if cands and len({quant[c] for c in cands}) == 1 else None
                mapping.append({"source": ck["repo"], "asset_id": asset, "config": cfg, "candidates": cands,
                                "added_by_deviation": "D5"})
    found = []
    for ck in inputs["checkpoints"]:
        for f in ck.get("files", []):
            name = f.get("path") or f.get("name")
            if not name or not name.endswith("norm_stats.json") or "local" not in f:
                continue
            source = (ck.get("repo") or ck.get("gcs_prefix"))
            for m in mapping:
                if m["source"] == source and name.endswith(m["asset_id"] + "/norm_stats.json"):
                    found.append({"source": source, "asset_id": m["asset_id"], "mapped_config": m["config"],
                                  "candidates": m.get("candidates"), "added_by_deviation": m.get("added_by_deviation"),
                                  "local": f["local"], "sha256": f["sha256"]})
    return found


def main() -> None:
    common.check_preregistration(EXP["frozen_files"])
    inputs = common.load_json(common.EVIDENCE / "inputs.json")
    by_id = {c["case_id"]: c for c in inputs["cases"]}
    out = {"schema": "kinema.normverify.openpi/1", "stack": STACK,
           "environment": common.environment_record(["openpi", "lerobot", "jax", "numpy", "torch", "datasets"]),
           "openpi_source": EXP["environments"][f"openpi_{STACK}"]["source"], "cases": []}
    for case in EXP["cases"]:
        if f"openpi_{STACK}" in case.get("configurations", {}):
            out["cases"].append(run_case(case, by_id[case["case_id"]], released_for(case, inputs)))
    common.write_json(common.EVIDENCE / f"openpi_{STACK}.json", out)
    print("ok")


if __name__ == "__main__":
    main()

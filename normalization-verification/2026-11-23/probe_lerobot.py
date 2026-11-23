"""LeRobot stats-selection and normalization probes.

    LeRobot 0.3.3 (v2.0 / v2.1 cases):  /tmp/kinema-phase2-v2/bin/python -I probe_lerobot.py 0.3.3
    LeRobot 0.4.0 (v3.0 case, and the v2.0 compatibility gate):
                                        /tmp/kinema-phase2-v3/bin/python -I probe_lerobot.py 0.4.0

0.3.3, missing-stats cases:
  L1 metadata      LeRobotDatasetMetadata on the pinned meta/ -> stats source
  L2 features      dataset_to_policy_features -> which dataset keys become STATE/ACTION
  L3 policy        make_policy(cfg, ds_meta) for the frozen policy list (no weights)
  L4 forward       one training forward on two real episode-0 frames
  L5 named target  the policy's own target normalizer on the named feature, when selected
0.3.3, near-constant r390: L1 only, then Normalize on the sampled rows if L1 yields stats.
0.4.0: the v2.0 compatibility gate on the missing-stats cases (construction only), and for
r393: M1 metadata stats, M2 features, M3 NormalizerProcessorStep per STATE mode used by
the shipped policy configs, on sampled rows and on the labelled synthetic shift inputs.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RELEASE = sys.argv[1]
SCRATCH = common.SCRATCH / f"lerobot-{RELEASE}"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=str(SCRATCH / "hf"),
                  HF_LEROBOT_HOME=str(SCRATCH / "lerobot_home"), TORCH_HOME=str(SCRATCH / "torch"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import lerobot  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402

EXP = common.load_json(common.EXPECTATIONS)
PKG = Path(lerobot.__file__).parent


def metadata(case: dict, root: Path) -> tuple[dict, object]:
    try:
        meta = LeRobotDatasetMetadata(case["source"]["repo"], root=root, revision=case["source"]["revision"])
    except Exception as e:  # noqa: BLE001
        return common.error_record(e), None
    stats = meta.stats
    return {"outcome": "ok", "codebase_version": meta.info.get("codebase_version"),
            "stats_is_none": stats is None, "stats_keys": sorted(stats) if stats else [],
            "feature_keys": sorted(meta.features)}, meta


def features(meta) -> dict:
    pf = dataset_to_policy_features(meta.features)
    return {k: {"type": v.type.name, "shape": list(v.shape)} for k, v in pf.items()}


def lerobot033_missing_stats(case: dict, inp: dict) -> dict:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.policies.factory import make_policy, make_policy_config

    root = Path(inp["root"])
    rec = {"case_id": case["case_id"], "release": RELEASE}
    rec["L1_metadata"], meta = metadata(case, root)
    if meta is None:
        return rec
    rec["L2_features"] = features(meta)
    named = case["finding"]["details"]["feature"]
    dataset_key = case["dataset_key"]
    rec["L2_named_feature_selected"] = named in rec["L2_features"]
    rec["L2_dataset_key_selected"] = rec["L2_features"].get(dataset_key)
    rec["policies"] = []
    for pol in EXP["lerobot_033_policies"]:
        prec = {"policy": pol["type"], "overrides": pol["overrides"]}
        try:
            cfg = make_policy_config(pol["type"], **pol["overrides"])
            policy = make_policy(cfg, ds_meta=meta)
            prec["L3_make_policy"] = {
                "outcome": "ok",
                "input_features": {k: v.type.name for k, v in cfg.input_features.items()},
                "output_features": {k: v.type.name for k, v in cfg.output_features.items()}}
        except Exception as e:  # noqa: BLE001
            prec["L3_make_policy"] = common.error_record(e)
            rec["policies"].append(prec)
            continue
        buffers = {}
        for mod_name in ("normalize_inputs", "normalize_targets", "unnormalize_outputs"):
            mod = getattr(policy, mod_name, None)
            if mod is None:
                continue
            for bname, buf in mod.named_buffers():
                buffers[f"{mod_name}.{bname}"] = {"all_inf": bool(torch.isinf(buf).all()), "shape": list(buf.shape)}
        prec["normalizer_buffers"] = buffers
        try:
            delta = resolve_delta_timestamps(cfg, meta)
            ds = LeRobotDataset(case["source"]["repo"], root=root, episodes=[0], delta_timestamps=delta,
                                revision=case["source"]["revision"], download_videos=False, video_backend="pyav")
            batch = torch.utils.data.default_collate([ds[0], ds[1]])
            policy.train()
            policy.forward(batch)
            prec["L4_forward"] = {"outcome": "ok"}
        except Exception as e:  # noqa: BLE001
            prec["L4_forward"] = common.error_record(e)
            batch = None
        if dataset_key in cfg.output_features and batch is not None:
            try:
                policy.normalize_targets({dataset_key: batch[dataset_key]})
                prec["L5_named_target"] = {"outcome": "ok", "key": dataset_key}
            except Exception as e:  # noqa: BLE001
                prec["L5_named_target"] = {"key": dataset_key, **common.error_record(e)}
        else:
            prec["L5_named_target"] = {"outcome": "not_selected", "key": dataset_key,
                                       "selected_as": rec["L2_dataset_key_selected"]}
        rec["policies"].append(prec)
    return rec


def sampled_rows(cid: str) -> tuple[np.ndarray, dict]:
    ch = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "channels.json")["cases"]}[cid]
    z = np.load(ch["sample_file"]["path"])
    if common.sha256_file(ch["sample_file"]["path"]) != ch["sample_file"]["sha256"]:
        raise SystemExit(f"{cid}: sample file hash changed")
    return z, ch


def shift_inputs(z, ch: dict, rule: dict) -> tuple[np.ndarray, list[dict]]:
    """Labelled synthetic deployment shift: copies of the first full sampled
    row with the flagged dim set to stored mean + delta. Not dataset values."""
    base = z["full_rows"][0].copy()
    mean = ch["stored_stats"]["flagged"]["mean"]
    deltas = list(rule["absolute_deltas"]) + [rule["sibling_std_multiplier"] * ch["stored_stats"]["sibling_std_median"]]
    rows, labels = [], []
    for sign in (1.0, -1.0):
        for dlt in deltas:
            r = base.copy()
            r[ch["dim"]] = mean + sign * dlt
            rows.append(r)
            labels.append({"delta": sign * dlt})
    return np.stack(rows), labels


def channel_outputs(col: np.ndarray, raw: np.ndarray) -> dict:
    rec = common.finite_summary(col)
    fin = np.isfinite(col).all()
    rec["spread_q01_q99"] = float(np.quantile(col, 0.99) - np.quantile(col, 0.01)) if fin else None
    rec["std_out"] = float(col.std()) if fin else None
    dr = raw - np.median(raw)
    do = col - np.median(col)
    nz = np.abs(dr) > 0
    rec["gain_median"] = float(np.median(np.abs(do[nz]) / np.abs(dr[nz]))) if nz.any() and fin else None
    return rec


CHUNK = 20000


def column_through(run, base: np.ndarray, d: int, values: np.ndarray) -> np.ndarray:
    """Pass `values` through the real normalizer as dim d of copies of a real
    sampled row (the normalizers are elementwise per dim), in chunks."""
    out = []
    for i in range(0, len(values), CHUNK):
        v = values[i:i + CHUNK]
        rows = np.repeat(base[None, :], len(v), axis=0)
        rows[:, d] = v
        out.append(run(rows)[:, d])
    return np.concatenate(out)


def evaluate(run, z, ch: dict) -> dict:
    d, ctrl = ch["dim"], ch["control_dim"]
    base = z["full_rows"][0]
    out_flag = column_through(run, base, d, z["flagged"])
    out_ctrl = column_through(run, base, ctrl, z["control"])
    out_full = np.concatenate([run(z["full_rows"][i:i + CHUNK]) for i in range(0, len(z["full_rows"]), CHUNK)])
    shift_rows, shift_labels = shift_inputs(z, ch, EXP["synthetic_shift"])
    out_shift = run(shift_rows)
    return {
        "outcome": "ok",
        "flagged_in_sample": channel_outputs(out_flag, z["flagged"]),
        "control_in_sample": {"dim": ctrl, **channel_outputs(out_ctrl, z["control"])},
        "full_rows": {"rows": int(len(z["full_rows"])), "nonfinite": int((~np.isfinite(out_full)).sum()),
                      "abs_max_all_dims": float(np.nanmax(np.abs(out_full))),
                      "flagged_dim_abs_max": float(np.nanmax(np.abs(out_full[:, d])))},
        "synthetic_shift": {"label": "synthetic deployment shift, not dataset values",
                            "rows": [{**lab, "flagged_out": float(out_shift[i, d])} for i, lab in enumerate(shift_labels)]},
    }


def to_loader_tensor(rows: np.ndarray) -> torch.Tensor:
    # equals torch.tensor(rows.tolist()) (what hf_transform_to_torch does); checked in loader_dtype_check
    return torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32))


def loader_dtype_check(rows: np.ndarray) -> dict:
    a = torch.tensor(rows[:4].tolist())
    b = to_loader_tensor(rows[:4])
    return {"torch_tensor_of_python_floats_dtype": str(a.dtype), "equal_to_float32_cast": bool(torch.equal(a, b))}


def lerobot040_near_constant(case: dict, inp: dict) -> dict:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.converters import create_transition
    from lerobot.processor.core import TransitionKey
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    cid, feat, d = case["case_id"], case["finding"]["details"]["feature"], case["finding"]["details"]["dim"]
    rec = {"case_id": cid, "release": RELEASE}
    rec["M1_metadata"], meta = metadata(case, Path(inp["root"]))
    if meta is None:
        return rec
    st = meta.stats.get(feat, {})
    rec["M1_stats_for_dim"] = {k: float(np.asarray(v).reshape(-1)[d]) for k, v in st.items() if np.asarray(v).size > d}
    rec["M2_features"] = features(meta)
    modes = {}
    for f in sorted((PKG / "policies").glob("*/configuration_*.py")):
        m = re.search(r'"STATE":\s*NormalizationMode\.(\w+)', f.read_text())
        if m:
            modes.setdefault(m.group(1), []).append(f.parent.name)
    rec["M3_state_modes_in_shipped_configs"] = modes
    if sorted(modes) != sorted(EXP["lerobot_040_state_modes"]):
        rec["M3_refused"] = f"mode set {sorted(modes)} differs from frozen {EXP['lerobot_040_state_modes']}"
        return rec
    z, ch = sampled_rows(cid)
    width = int(z["full_rows"].shape[1])
    rec["loader_dtype"] = loader_dtype_check(z["full_rows"])
    rec["M3_normalize"] = {}
    for mode in sorted(modes):
        step = NormalizerProcessorStep(features={feat: PolicyFeature(type=FeatureType.STATE, shape=(width,))},
                                       norm_map={FeatureType.STATE: NormalizationMode[mode]}, stats=meta.stats)

        def run(rows: np.ndarray, step=step) -> np.ndarray:
            obs = step(create_transition(observation={feat: to_loader_tensor(rows)}))[TransitionKey.OBSERVATION][feat]
            return obs.detach().numpy().astype(np.float64)

        try:
            rec["M3_normalize"][mode] = {"policies": modes[mode], "eps": step.eps, **evaluate(run, z, ch)}
        except Exception as e:  # noqa: BLE001
            rec["M3_normalize"][mode] = common.error_record(e)
    return rec


def lerobot033_near_constant(case: dict, inp: dict) -> dict:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.normalize import Normalize

    cid, feat = case["case_id"], case["finding"]["details"]["feature"]
    rec = {"case_id": cid, "release": RELEASE}
    rec["L1_metadata"], meta = metadata(case, Path(inp["root"]))
    if meta is None or not meta.stats or feat not in meta.stats:
        rec["normalize"] = {"outcome": "not_reached", "reason": "metadata construction gave no stats for the feature"}
        return rec
    z, ch = sampled_rows(cid)
    width = int(z["full_rows"].shape[1])
    rec["normalize"] = {}
    for mode in ("MEAN_STD", "MIN_MAX"):
        norm = Normalize({feat: PolicyFeature(type=FeatureType.STATE, shape=(width,))},
                         {"STATE": NormalizationMode[mode]}, meta.stats)

        def run(rows: np.ndarray, norm=norm) -> np.ndarray:
            return norm({feat: to_loader_tensor(rows)})[feat].detach().numpy().astype(np.float64)

        try:
            rec["normalize"][mode] = evaluate(run, z, ch)
        except Exception as e:  # noqa: BLE001
            rec["normalize"][mode] = common.error_record(e)
    return rec


def lerobot_checkpoint_provenance(inputs: dict) -> list[dict]:
    """Released LeRobot checkpoints carry normalizer state in processor files.
    Read their names, norm_map and stats tensors (no weights)."""
    from safetensors.numpy import load_file

    out = []
    for ck in inputs["checkpoints"]:
        if ck.get("role") != "lerobot_comparator":
            continue
        files = {Path(f["path"]).name: f for f in ck["files"] if f.get("present")}
        rec = {"repo": ck["repo"], "revision": ck["revision"], "license": ck.get("license")}
        pre = common.load_json(files["policy_preprocessor.json"]["local"])
        norm = [st for st in pre["steps"] if st.get("registry_name") == "normalizer_processor"]
        rec["preprocessor_steps"] = [st.get("registry_name") for st in pre["steps"]]
        rec["norm_map"] = norm[0]["config"]["norm_map"] if norm else None
        rec["normalizer_features"] = sorted(norm[0]["config"]["features"]) if norm else None
        tc = common.load_json(files["train_config.json"]["local"])
        rec["train_dataset"] = tc.get("dataset", {}).get("repo_id")
        state_file = norm[0].get("state_file") if norm else None
        if state_file and state_file in files:
            tensors = load_file(files[state_file]["local"])
            rec["normalizer_tensors"] = {k: list(v.shape) for k, v in sorted(tensors.items())}
            rec["has_state_quantiles"] = any(k.startswith("observation.state.q01") for k in tensors)
        out.append(rec)
    return out


def main() -> None:
    common.check_preregistration(EXP["frozen_files"])
    inputs = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "inputs.json")["cases"]}
    out = {"schema": "kinema.normverify.lerobot/1", "release": RELEASE,
           "environment": common.environment_record(["lerobot", "torch", "datasets", "numpy", "av"]), "cases": []}
    for case in EXP["cases"]:
        inp = inputs[case["case_id"]]
        if RELEASE == "0.3.3":
            if case["kind"] == "missing-stats":
                out["cases"].append(lerobot033_missing_stats(case, inp))
            elif case["source"]["format"] != "v3.0":
                out["cases"].append(lerobot033_near_constant(case, inp))
        else:
            if case["kind"] == "missing-stats":
                rec = {"case_id": case["case_id"], "release": RELEASE}
                rec["L1_metadata"], _ = metadata(case, Path(inp["root"]))
                out["cases"].append(rec)
            elif case["source"]["format"] == "v3.0":
                out["cases"].append(lerobot040_near_constant(case, inp))
    if RELEASE == "0.4.0":
        out["checkpoint_provenance"] = lerobot_checkpoint_provenance(common.load_json(common.EVIDENCE / "inputs.json"))
    common.write_json(common.EVIDENCE / f"lerobot_{RELEASE}.json", out)
    print("ok")


if __name__ == "__main__":
    main()

"""Isaac-GR00T n1.6.1 probes (r79:c17:f0 exumi; r390:c7:f8 BEHAVIOR R1Pro).

    env-gr00t/bin/python -I probe_gr00t.py

The GR00T source is the pinned n1.6.1-release checkout, put on sys.path; only
gr00t.data modules (numpy/pandas/scipy) are imported, nothing is patched.

r79 (missing stats, GR00T-flavoured v2.0 per its card):
  G1 stats generation   gr00t.data.stats.generate_stats on a private copy of the
                        pinned meta/ + episode-0 parquet (training calls it first)
  G2 loader             LeRobotEpisodeLoader construction (asserts meta/stats.json)
  G3 schema facts       modality.json keys, presence of a state modality, and
                        whether any shipped embodiment config names this dataset
r390 (near-constant dim):
  G4 selection          released GR00T-N1.6-BEHAVIOR1k processor state keys ->
                        pinned modality.json slices -> is the flagged dim consumed?
                        Also: do the checkpoint statistics carry its slice?
  G5 loader + processor LeRobotEpisodeLoader (state modality only) on the first
                        selected episode, then StateActionProcessor.apply_state with
                        the checkpoint statistics and the released processor flags
  G6 hypothetical       normalize_values_minmax on the flagged dim with the dataset
                        stats.json min/max (NOT the released config: that config does
                        not select this dim) - labelled control
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

EXP = common.load_json(common.EXPECTATIONS)
GR00T_SRC = common.UPSTREAM / "Isaac-GR00T-n1.6.1"
sys.path.insert(0, str(GR00T_SRC))
SCRATCH = common.SCRATCH / "gr00t"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
os.environ.update(HF_HUB_OFFLINE="1")

import numpy as np  # noqa: E402
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.data.state_action.state_action_processor import StateActionProcessor  # noqa: E402
from gr00t.data.stats import generate_stats  # noqa: E402
from gr00t.data.utils import normalize_values_minmax  # noqa: E402


def private_copy(src_root: Path, name: str) -> Path:
    dst = SCRATCH / name
    shutil.copytree(src_root, dst, symlinks=False)
    return dst


def r79(case: dict, inp: dict) -> dict:
    rec = {"case_id": case["case_id"]}
    root = private_copy(Path(inp["root"]), "r79_generate_stats")
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    try:
        generate_stats(root)
        rec["G1_generate_stats"] = {"outcome": "ok"}
    except Exception as e:  # noqa: BLE001
        rec["G1_generate_stats"] = common.error_record(e)
    after = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    rec["G1_generate_stats"]["files_written"] = sorted(set(after) - set(before))
    root2 = private_copy(Path(inp["root"]), "r79_loader")
    try:
        LeRobotEpisodeLoader(root2, modality_configs={})
        rec["G2_loader"] = {"outcome": "ok"}
    except Exception as e:  # noqa: BLE001
        rec["G2_loader"] = common.error_record(e)
    modality = common.load_json(Path(inp["root"]) / "meta/modality.json")
    info = common.load_json(Path(inp["root"]) / "meta/info.json")
    rec["G3_schema"] = {"modality_keys": sorted(modality), "has_state_modality": "state" in modality,
                        "info_has_features": "features" in info,
                        "shipped_embodiment_configs": sorted(MODALITY_CONFIGS)}
    return rec


def selected_indices(modality: dict, keys: list[str]) -> dict:
    idx, missing = [], []
    for k in keys:
        if k not in modality["state"]:
            missing.append(k)
            continue
        idx.extend(range(modality["state"][k]["start"], modality["state"][k]["end"]))
    return {"indices": idx, "missing_keys": missing}


def slice_of(modality: dict, dim: int) -> dict | None:
    for k, v in modality["state"].items():
        if v["start"] <= dim < v["end"]:
            return {"key": k, "start": v["start"], "end": v["end"], "offset": dim - v["start"]}
    return None


def r390(case: dict, inp: dict, ckpt: dict) -> dict:
    d = case["finding"]["details"]["dim"]
    rec = {"case_id": case["case_id"]}
    root = Path(inp["root"])
    modality = common.load_json(root / "meta/modality.json")
    proc = common.load_json(ckpt["processor_config.json"])["processor_kwargs"]
    stats = common.load_json(ckpt["statistics.json"])
    tag = EXP["gr00t_behavior"]["embodiment_tag"]
    state_cfg = proc["modality_configs"][tag]["state"]
    sel = selected_indices(modality, state_cfg["modality_keys"])
    flagged_slice = slice_of(modality, d)
    rec["G4_selection"] = {
        "released_state_keys": state_cfg["modality_keys"],
        "selected_dim_count": len(sel["indices"]), "keys_missing_from_modality_json": sel["missing_keys"],
        "flagged_dim": d, "flagged_slice": flagged_slice, "flagged_dim_selected": d in sel["indices"],
        "checkpoint_state_stat_keys": sorted(stats[tag]["state"]),
        "checkpoint_has_flagged_slice": flagged_slice is not None and flagged_slice["key"] in stats[tag]["state"],
        "shipped_config_state_keys_equal_released": MODALITY_CONFIGS[tag]["state"].modality_keys == state_cfg["modality_keys"],
        "processor_flags": {k: proc[k] for k in ("use_percentiles", "clip_outliers", "apply_sincos_state_encoding")},
    }
    # G5: real loader (state only) on the first selected episode + released processor
    work = SCRATCH / "r390"
    shutil.copytree(root, work)
    first = inp["selected_data_files"][0]
    dst = work / first["path"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(first["cache_path"], dst)
    try:
        loader = LeRobotEpisodeLoader(work, modality_configs={"state": MODALITY_CONFIGS[tag]["state"]})
        ep_index = int(Path(first["path"]).stem.split("_")[-1])
        pos = [int(e["episode_index"]) for e in loader.episodes_metadata].index(ep_index)
        df = loader[pos]
        state_cols = [c for c in df.columns if c.startswith("state.")]
        state = {c[len("state."):]: np.stack(df[c].to_numpy()) for c in state_cols}
        rec["G5_loader"] = {"outcome": "ok", "episode_index": ep_index, "rows": int(len(df)),
                            "state_columns": state_cols,
                            "joint_qeffort_loaded": any("qeffort" in c for c in state_cols)}
        sap = StateActionProcessor(modality_configs={tag: {"state": MODALITY_CONFIGS[tag]["state"]}},
                                   statistics={tag: {"state": stats[tag]["state"]}},
                                   use_percentiles=proc["use_percentiles"], clip_outliers=proc["clip_outliers"],
                                   apply_sincos_state_encoding=proc["apply_sincos_state_encoding"])
        out = sap.apply_state(state, tag)
        cat = np.concatenate([np.asarray(out[k]).reshape(len(df), -1) for k in state_cfg["modality_keys"]], axis=1)
        rec["G5_processor"] = {"outcome": "ok", "output_width": int(cat.shape[1]), **common.finite_summary(cat)}
    except Exception as e:  # noqa: BLE001
        rec.setdefault("G5_loader", common.error_record(e))
        rec["G5_processor"] = {"outcome": "not_reached_or_error", **common.error_record(e)}
    # G6: hypothetical control with the dataset's own stats for the flagged dim
    ds_stats = common.load_json(root / "meta/stats.json")[case["finding"]["details"]["feature"]]
    ch = {c["case_id"]: c for c in common.load_json(common.EVIDENCE / "channels.json")["cases"]}[case["case_id"]]
    z = np.load(ch["sample_file"]["path"])
    params = {"min": np.array([ds_stats["min"][d]]), "max": np.array([ds_stats["max"][d]])}
    in_sample = normalize_values_minmax(z["flagged"].reshape(-1, 1).copy(), params)
    deltas = list(EXP["synthetic_shift"]["absolute_deltas"]) + [EXP["synthetic_shift"]["sibling_std_multiplier"] * ch["stored_stats"]["sibling_std_median"]]
    shifted = np.array([[ds_stats["mean"][d] + s * x] for s in (1.0, -1.0) for x in deltas])
    rec["G6_hypothetical_minmax"] = {
        "label": "hypothetical: the released config does not select this dim; dataset stats.json min/max used",
        "min": ds_stats["min"][d], "max": ds_stats["max"][d],
        "isclose_masked": bool(np.isclose(ds_stats["max"][d], ds_stats["min"][d])),
        "in_sample": common.finite_summary(in_sample),
        "synthetic_shift": {"label": "synthetic deployment shift, not dataset values",
                            "inputs": shifted.ravel().tolist(),
                            "outputs": normalize_values_minmax(shifted.copy(), params).ravel().tolist()},
    }
    return rec


def main() -> None:
    common.check_preregistration(EXP["frozen_files"])
    inputs = common.load_json(common.EVIDENCE / "inputs.json")
    by_id = {c["case_id"]: c for c in inputs["cases"]}
    ck = next(c for c in inputs["checkpoints"] if c.get("repo") == EXP["gr00t_behavior"]["repo"])
    ckpt = {Path(f["path"]).name: f["local"] for f in ck["files"] if f.get("present")}
    out = {"schema": "kinema.normverify.gr00t/1", "gr00t_source": EXP["environments"]["gr00t"]["source"],
           "environment": common.environment_record(["numpy", "pandas", "scipy", "pyarrow", "torch"]), "cases": []}
    for case in EXP["cases"]:
        if case["case_id"] == "r79:c17:f0":
            out["cases"].append(r79(case, by_id[case["case_id"]]))
        elif case["case_id"] == "r390:c7:f8":
            out["cases"].append(r390(case, by_id[case["case_id"]], ckpt))
    common.write_json(common.EVIDENCE / "gr00t.json", out)
    print("ok")


if __name__ == "__main__":
    main()

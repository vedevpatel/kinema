"""Derive outcomes and labels from the evidence with the frozen rules.

    python3 -I make_report.py

Writes report.json (machine-readable), adjudications.json (new labels with
pointers to the original cases) and report.md. Thresholds and the overall rule
come from expectations.json; nothing here reads a probe's free text to decide.
Case-level label = the label of the case's consequence assertion (A4 / B2);
the factual assertions (A1, A2, B1) are reported beside it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

E = common.EVIDENCE
EXP = common.load_json(common.EXPECTATIONS)
INPUTS = common.load_json(E / "inputs.json")
CH = {c["case_id"]: c for c in common.load_json(E / "channels.json")["cases"]}
OP_UP = {c["case_id"]: c for c in common.load_json(E / "openpi_upstream.json")["cases"]}
OP_FORK = {c["case_id"]: c for c in common.load_json(E / "openpi_fork.json")["cases"]}
OP_FAST = {c["case_id"]: c for c in common.load_json(E / "openpi_fork_fast_supplement.json")["cases"]}
LR033 = {c["case_id"]: c for c in common.load_json(E / "lerobot_0.3.3.json")["cases"]}
LR040_ALL = common.load_json(E / "lerobot_0.4.0.json")
LR040 = {c["case_id"]: c for c in LR040_ALL["cases"]}
GR = {c["case_id"]: c for c in common.load_json(E / "gr00t.json")["cases"]}
DEV = common.load_json(common.DEVIATIONS)
AMP_HI, AMP_LO = 0.1, 0.01  # frozen in expectations B2 outcomes
DECISIVE = {"normalizes_finite", "fails_missing_stats", "amplified", "not_amplified", "nonfinite"}


def err(rec: dict) -> str:
    return f"{rec.get('exception')}: {rec.get('message')}"


# --- missing-stats cases ------------------------------------------------------------

def a1(case: dict) -> dict:
    inp = {c["case_id"]: c for c in INPUTS["cases"]}[case["case_id"]]
    paths = [r["path"] for r in inp["meta_listing_at_revision"]]
    hits = [p for p in paths if p.endswith("meta/stats.json") or p.endswith("meta/episodes_stats.jsonl")]
    return {"label": "refuted" if hits else "confirmed", "kind": "provenance (Hub listing at the pinned revision)",
            "observation": f"{len(paths)} files under meta/: {paths}; stats files present: {hits}"}


def a2(case: dict) -> dict:
    key, label = case["dataset_key"], case["finding"]["details"]["feature"]
    via = "literal column name" if key == label else "kinema alias (source pin kinema.alias.*)"
    return {"label": "confirmed", "kind": "source inspection + pinned schema",
            "observation": f"the label `{label}` is bound to the dataset column `{key}` ({via})",
            "dataset_column": key}


def openpi_a4(case: dict, ev: dict, stack: str) -> dict:
    key = "actions" if case["dataset_key"] == "actions" else "state"
    st = ev["stages"]
    out = {"configuration": f"openpi_{stack}", "role": "primary", "dataset_key": case["dataset_key"], "openpi_key": key}
    if st.get("S3_construction", {}).get("outcome") != "ok" or st.get("S4_batch_selection", {}).get("outcome") != "ok":
        out.update(outcome="not_reached", reason="construction or batch selection failed")
        return out
    s7 = st["S7_inference_load"]
    per = {n: (r.get("outcome") == "ok" and r[key]["n_nonfinite"] == 0) for n, r in s7["normalize"].items()}
    released = []
    for r in st["S8_released_stats"]:
        n = r.get("normalize", {})
        released.append({"source": r["source"], "mapped_config": r["mapped_config"], "added_by": r.get("added_by_deviation"),
                         "outcome": n.get("outcome"), "finite": n.get("outcome") == "ok" and n[key]["n_nonfinite"] == 0,
                         "note": n.get("reason") or (err(n) if n.get("outcome") == "error" else None)})
    for r in OP_FAST.get(case["case_id"], {}).get("checkpoints", []) if stack == "fork" else []:
        n = r["normalize"]
        released.append({"source": r["source"], "mapped_config": r["mapped_config"], "added_by": "D6" + ("+" + r["added_by_deviation"] if r.get("added_by_deviation") else ""),
                         "outcome": n.get("outcome"), "finite": n.get("outcome") == "ok" and n[key]["n_nonfinite"] == 0, "note": "7-dim FAST slice"})
    ok = all(per.values()) and per
    out.update(
        outcome="normalizes_finite" if ok else "fails_missing_stats",
        kind="implementation execution (episode 0 only)",
        observation=(f"pinned metadata stats are None (S3); the training guard demands assets norm stats (S2: {err(st['S2_training_guard'])}); "
                     f"openpi RunningStats over {st['S4_batch_selection']['frames']} episode-0 frames gave stats (S5); "
                     f"Normalize after the inference loader (S7) was finite for `{key}` under {sum(per.values())}/{len(per)} configs"),
        per_config_finite=per, released=released)
    return out


def lerobot033_a4(case: dict) -> dict:
    ev = LR033[case["case_id"]]
    out = {"configuration": "lerobot_0.3.3", "role": "secondary"}
    if ev["L1_metadata"].get("outcome") != "ok":
        out.update(outcome="not_reached", stage="metadata construction", observation=err(ev["L1_metadata"]))
        return out
    sel = ev["L2_dataset_key_selected"]
    pols = {p["policy"]: p["L3_make_policy"] for p in ev["policies"]}
    constructed = {k: v.get("outcome") == "ok" for k, v in pols.items()}
    if sel is None:
        out.update(outcome="not_consumed", stage="feature selection (dataset_to_policy_features)",
                   observation=f"`{case['dataset_key']}` is not typed STATE/ACTION (only observation* / action* keys are); policies: "
                               + "; ".join(f"{k}: {err(v)}" for k, v in pols.items() if v.get('outcome') != 'ok'))
    elif not any(constructed.values()):
        out.update(outcome="not_reached", stage="policy construction (feature selection by exact name)",
                   observation=f"`{case['dataset_key']}` is typed {sel['type']} but every policy failed construction: "
                               + "; ".join(f"{k}: {err(v)}" for k, v in pols.items()))
    else:
        out.update(outcome="see_policies", observation=str(ev["policies"]))
    out["meta_stats_is_none"] = ev["L1_metadata"].get("stats_is_none")
    return out


def lerobot040_gate(case: dict) -> dict:
    ev = LR040[case["case_id"]]["L1_metadata"]
    return {"configuration": "lerobot_0.4.0", "role": "secondary (compatibility gate)",
            "outcome": "not_reached" if ev.get("outcome") != "ok" else "constructed",
            "stage": "metadata construction", "observation": err(ev) if ev.get("outcome") != "ok" else "ok"}


def gr00t_r79() -> dict:
    ev = GR["r79:c17:f0"]
    return {"configuration": "gr00t", "role": "primary", "outcome": "not_reached", "stage": "stats generation / loader construction",
            "observation": (f"generate_stats: {err(ev['G1_generate_stats'])} (files written: {ev['G1_generate_stats']['files_written']}); "
                            f"LeRobotEpisodeLoader: {err(ev['G2_loader'])}; modality.json keys {ev['G3_schema']['modality_keys']} "
                            f"(state modality: {ev['G3_schema']['has_state_modality']}); info.json has features: {ev['G3_schema']['info_has_features']}")}


def a3(case: dict, configs: list[dict]) -> dict:
    per = {}
    for c in configs:
        name = c["configuration"]
        if name.startswith("openpi"):
            per[name] = {"label": "refuted", "kind": "source inspection + execution",
                         "observation": "stats come from compute_norm_stats over transformed data into assets/<asset_id> (S5-S7) and from checkpoint assets at inference (S8); meta/stats.json is never read"}
        elif name == "lerobot_0.3.3":
            per[name] = {"label": "confirmed" if c.get("meta_stats_is_none") is not None else "not_reached",
                         "kind": "source inspection + execution",
                         "observation": "make_policy passes ds_meta.stats (meta/stats.json for v2.0); it is None here" if c.get("meta_stats_is_none") else c.get("observation")}
        elif name == "gr00t":
            per[name] = {"label": "refuted", "kind": "source inspection (execution not reached)",
                         "observation": "GR00T training generates meta/stats.json from the parquet data when absent (factory -> generate_stats); here generation failed on the missing info.json features"}
        elif name == "lerobot_0.4.0":
            per[name] = {"label": "not_reached", "kind": "execution", "observation": c["observation"]}
    return per


def overall(primary: dict, others: list[dict]) -> tuple[str, str]:
    po = primary["outcome"]
    diff = [o for o in others if o["outcome"] in DECISIVE and o["outcome"] != po]
    if po in ("not_reached",):
        if any(o["outcome"] in DECISIVE for o in others):
            return "partially_resolved", "primary not reached; a secondary configuration was decisive"
        return "unresolved", "no tested configuration reached normalization of the named feature"
    base = {"normalizes_finite": "refuted", "not_amplified": "refuted", "not_consumed": "refuted",
            "fails_missing_stats": "confirmed", "amplified": "confirmed", "nonfinite": "confirmed"}.get(po, "partially_resolved")
    if diff:
        return "partially_resolved", f"secondary configuration(s) reached a different decisive outcome: {[o['configuration'] for o in diff]}"
    return base, f"primary configuration outcome: {po}"


def missing_case(case: dict) -> dict:
    cid = case["case_id"]
    if "openpi_fork" in case["configurations"]:
        prim = openpi_a4(case, OP_FORK[cid], "fork")
    elif "openpi_upstream" in case["configurations"]:
        prim = openpi_a4(case, OP_UP[cid], "upstream")
    else:
        prim = gr00t_r79()
    others = [lerobot033_a4(case), lerobot040_gate(case)]
    label, why = overall(prim, others)
    return {"A1_stats_absent": a1(case), "A2_feature_binding": a2(case),
            "A3_stats_source": a3(case, [prim] + others),
            "A4_inference_normalization": {"label": label, "rule": why, "configurations": [prim] + others},
            "case_label": label}


# --- near-constant cases --------------------------------------------------------------

def b1(case: dict) -> dict:
    ch = CH[case["case_id"]]
    stored = ch["stored_stats"]["flagged"]["std"]
    files = [f["flagged"]["std"] for f in ch["files"]]
    pooled = ch["pooled"]["flagged"]["std"]
    if stored < 1e-9 and all(s < 1e-9 for s in files):
        label = "confirmed"
    elif stored >= 1e-6 or pooled >= 1e-6:
        label = "refuted"
    else:
        label = "partially_resolved"
    p = ch["pooled"]["flagged"]
    return {"label": label, "kind": "pinned stats + raw parquet sample",
            "board_std": case["finding"]["details"]["std"], "stored_std": stored, "sampled_std_per_file": files, "sampled_std_pooled": pooled,
            "observation": (f"{ch['pooled']['files']} files, {ch['pooled']['rows']} rows, {ch['pooled']['episodes']} episodes: "
                            f"{p['n_exact_zero']} exact zeros, {p['n_unique']} distinct values, range [{p['min']:.3g}, {p['max']:.3g}], "
                            f"{p['n_abs_gt_1e_11']} rows with |x| > 1e-11; stored range [{ch['stored_stats']['flagged']['min']:.3g}, {ch['stored_stats']['flagged']['max']:.3g}]"),
            "note": "near-zero, not constant; the board std comes from the detector's own loaded rows (source pin kinema.nan.std_from_loaded_rows)"}


def amp_outcome(rec: dict) -> str:
    f = rec["flagged_in_sample"]
    if f["n_nonfinite"]:
        return "nonfinite"
    if f["std_out"] >= AMP_HI:
        return "amplified"
    if f["std_out"] <= AMP_LO:
        return "not_amplified"
    return "intermediate"


def b2_r390() -> dict:
    g = GR["r390:c7:f8"]
    prim = {"configuration": "gr00t_released (nvidia/GR00T-N1.6-BEHAVIOR1k)", "role": "primary",
            "outcome": "not_consumed" if not g["G4_selection"]["flagged_dim_selected"] and not g["G5_loader"].get("joint_qeffort_loaded") else "see_evidence",
            "kind": "provenance + implementation execution",
            "observation": (f"released state keys select {g['G4_selection']['selected_dim_count']} dims; dim 117 is "
                            f"{g['G4_selection']['flagged_slice']['key']}[{g['G4_selection']['flagged_slice']['offset']}], not selected; "
                            f"checkpoint statistics carry it: {g['G4_selection']['checkpoint_has_flagged_slice']}; the GR00T loader on episode "
                            f"{g['G5_loader'].get('episode_index')} returned {len(g['G5_loader'].get('state_columns', []))} state groups without joint_qeffort; "
                            f"processor output width {g['G5_processor'].get('output_width')}, finite {g['G5_processor'].get('n_nonfinite') == 0}, "
                            f"range [{g['G5_processor'].get('min')}, {g['G5_processor'].get('max')}]")}
    lr = LR033["r390:c7:f8"]
    sec = {"configuration": "lerobot_0.3.3", "role": "secondary", "outcome": "not_reached",
           "stage": "metadata construction", "observation": err(lr["L1_metadata"])}
    hyp = g["G6_hypothetical_minmax"]
    control = {"label": hyp["label"], "isclose_masked": hyp["isclose_masked"], "in_sample_abs_max": hyp["in_sample"]["abs_max"],
               "synthetic_shift_outputs": hyp["synthetic_shift"]["outputs"]}
    label, why = overall(prim, [sec])
    return {"label": label, "rule": why, "configurations": [prim, sec], "hypothetical_control": control}


def b2_r393() -> dict:
    ev = LR040["r393:c7:f14"]
    modes = []
    for mode, rec in sorted(ev["M3_normalize"].items()):
        if rec.get("outcome") != "ok":
            modes.append({"mode": mode, "outcome": "not_reached", "observation": err(rec)})
            continue
        f, c = rec["flagged_in_sample"], rec["control_in_sample"]
        oc = "not_consumed" if mode == "IDENTITY" else amp_outcome(rec)
        modes.append({"mode": mode, "policies": rec["policies"], "outcome": oc, "flagged_std_out": f["std_out"],
                      "flagged_abs_max_out": f["abs_max"], "gain_median": f["gain_median"], "nonfinite": f["n_nonfinite"],
                      "control_dim": c["dim"], "control_std_out": c["std_out"], "control_abs_max_out": c["abs_max"],
                      "full_rows_nonfinite": rec["full_rows"]["nonfinite"], "full_rows_abs_max_all_dims": rec["full_rows"]["abs_max_all_dims"],
                      "synthetic_shift": rec["synthetic_shift"]["rows"]})
    decisive = {m["outcome"] for m in modes if m["outcome"] in DECISIVE}
    if decisive == {"amplified"} or decisive == {"nonfinite"}:
        label = "confirmed"
    elif decisive <= {"not_amplified"} and decisive:
        label = "refuted"
    elif decisive:
        label = "partially_resolved"
    else:
        label = "unresolved"
    return {"label": label, "rule": "primary configuration evaluated per shipped STATE mode; mixed decisive outcomes give partially_resolved",
            "configurations": [{"configuration": "lerobot_0.4.0", "role": "primary", "modes": modes,
                                "stats_selected": "dataset.meta.stats (lerobot-train; source pin lr040.train_stats)",
                                "stats_for_dim": ev["M1_stats_for_dim"], "loader_dtype": ev["loader_dtype"]}]}


def near_case(case: dict) -> dict:
    b2 = b2_r390() if case["case_id"] == "r390:c7:f8" else b2_r393()
    ch = CH[case["case_id"]]
    return {"B1_near_zero_std": b1(case), "B2_amplification": b2,
            "layout": {"slice": case["slice"], "sincos_max_abs_err": ch["sincos_max_abs_err"],
                       "basis": ("pinned modality.json (joint_qeffort = state[112:140])" if case["case_id"] == "r390:c7:f8"
                                 else "inferred: GR00T modality.json of the sibling R1Pro conversion + sin/cos structure of dims 0:84")},
            "case_label": b2["label"]}


# --- assembly -----------------------------------------------------------------------

def scope(cid: str, c: dict) -> str:
    """One generated sentence per case: which assertion, stage, configuration, revision, operation."""
    rev = c["original"]["revision"][:8]
    if "A4_inference_normalization" in c:
        cfgs = "; ".join(f"{x['configuration']}: {x['outcome']}" for x in c["A4_inference_normalization"]["configurations"])
        return (f"@{rev}: stats absence {c['A1_stats_absent']['label']} (meta listing); label binds to column "
                f"`{c['A2_feature_binding']['dataset_column']}`; normalization consequence {c['A4_inference_normalization']['label']} "
                f"[{cfgs}]." + (" openpi results use episode-0 stats computed by openpi and, where mapped, released checkpoint stats; no trained policy was run."
                                if "openpi" in cfgs else " No configuration reached normalization; no trained policy was run."))
    b2 = c["B2_amplification"]
    if cid == "r390:c7:f8":
        detail = "; ".join(f"{x['configuration']}: {x['outcome']}" for x in b2["configurations"])
    else:
        detail = "; ".join(f"{m['mode']}: {m['outcome']}" + (f" (normalized std {m['flagged_std_out']:.3g})" if m.get("flagged_std_out") is not None else "")
                           for m in b2["configurations"][0]["modes"])
    return (f"@{rev}: near-zero std {c['B1_near_zero_std']['label']} (stored stats + {len(c['B1_near_zero_std']['sampled_std_per_file'])} sampled files); "
            f"amplification {b2['label']} [{detail}]. In-sample values only decide; synthetic shifts are descriptive.")


def build() -> tuple[dict, dict]:
    cases = {}
    for case in EXP["cases"]:
        res = missing_case(case) if case["kind"] == "missing-stats" else near_case(case)
        res.update(original={"audit": f"audits/2026-11-22 {case['case_id']}", "phase1_label": case["prior"]["phase1"]["label"],
                             "phase2_label": case["prior"].get("phase2", {}).get("classification"),
                             "dataset": case["finding"]["dataset"], "revision": case["finding"]["revision"],
                             "engine": case["finding"]["engine"], "message": case["finding"]["message"]})
        cases[case["case_id"]] = res
    report = {"schema": "kinema.normverify.report/1", "date_utc": "2026-11-23",
              "preregistration": common.load_json(common.PREREGISTRATION)["sha256"],
              "deviations": [{"id": d["id"], "file": d["file"]} for d in DEV["deviations"]],
              "evidence_files": sorted(p.name for p in E.iterdir()),
              "lerobot_checkpoint_provenance": LR040_ALL.get("checkpoint_provenance"),
              "counts": {lab: sum(1 for c in cases.values() if c["case_label"] == lab)
                         for lab in ("confirmed", "refuted", "partially_resolved", "unresolved")},
              "cases": cases}
    adj = {cid: {"original_case": c["original"], "label": c["case_label"], "scope": scope(cid, c),
                 "assertions": {k: v["label"] for k, v in c.items() if isinstance(v, dict) and "label" in v and k[:2] in ("A1", "A2", "A4", "B1", "B2")},
                 "evidence": "report.json#cases." + cid}
           for cid, c in cases.items()}
    return report, adj


def main() -> None:
    report, adj = build()
    common.write_json(common.HERE / "report.json", report)
    common.write_json(common.HERE / "adjudications.json", adj)
    print(report["counts"], {k: v["label"] for k, v in adj.items()})


if __name__ == "__main__":
    main()

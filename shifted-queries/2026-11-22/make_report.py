"""Build report.json and report.md from the frozen plan and probe evidence.

    python shifted-queries/2026-11-22/make_report.py

Runs with numpy only (no LeRobot). Every verdict is computed from evidence
with the preregistered rules in queryplan.py. The phase-1 and phase-2
results are copied verbatim from their frozen files and are not reinterpreted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import queryplan as qp  # noqa: E402

REPO = HERE.parents[1]
BUCKETS = ("loader_derived/in_range", "loader_derived/padded", "nominal_grid/in_range", "nominal_grid/outside_episode")
VERDICT_LABEL = {"supports_claim": "confirmed", "contradicts_claim": "refuted", "unresolved": "unresolved", "not_applicable": "not applicable"}
# Bucket-level wording: a bucket is evidence about queries, not a verdict on the claim.
BUCKET_LABEL = {"supports_claim": "tolerance failure observed", "contradicts_claim": "all within tolerance", "unresolved": "unresolved", "not_applicable": "not applicable"}

ENV_NAME = {"v2": "LeRobot 0.3.3", "v3": "LeRobot 0.4.0"}

# Fixed scope statements. Each is checked in tests against the computed verdicts.
SCOPE = {
    "r1:c1:f0": {
        "full_loader": "LeRobotDataset construction fails in 0.4.0 and 0.5.1 on source dtype 'language' (phase 2). Not bypassed, so __getitem__ was never exercised.",
        "source_note": "LeRobot 0.4.0 has no check_timestamps_sync call: spacing is never validated against tolerance_s, so the decoder is the only place tolerance_s can reject a timestamp (datasets/lerobot_dataset.py, sha256 2fc7bbd5...).",
        "offset_note": "info.json declares 30 fps while rows and all four videos are 90 Hz. resolve_delta_timestamps(i/30) maps to row offset i, which spans i/90 s: about one third of the window each config intends.",
        "gaps": [
            "The full-loader query path is still blocked by the language dtype; this result does not establish that training loads.",
            "Default torchcodec backend: libtorchcodec failed to load (dependency), so it is untested. torchcodec maps timestamps to frame indices using average_fps, a different rule.",
            "Only episode 40455, four anchors, default configs. Not tested in 0.5.1.",
            "The 1/3 window compression's effect on a trained policy is not measured.",
        ],
    },
    "r71:c1:f1": {
        "full_loader": "LeRobotDataset construction raises KeyError('stats') on episode metadata (phase 2). Not bypassed.",
        "source_note": "0.3.3 runs check_timestamps_sync at construction, but this check is on spacing, and droid spacing is on the 1/15 s grid. Phase 2's construction failure happened earlier, on episode stats.",
        "offset_note": "Rows span 0.7333 to 13.4667 s; the videos span 0 to 14.1333 s (213 frames for 192 rows). Lookups succeed because the video timeline covers the row timestamps. Whether the frame at 0.7333 s shows the same instant as row 0 cannot be determined from timestamps.",
        "gaps": [
            "The other 22 listed episodes were not probed.",
            "Image-content alignment between rows and video frames is untested.",
            "Full loader blocked by KeyError('stats'); default torchcodec backend untested (dependency).",
        ],
    },
    "r94:c1:f1": {
        "full_loader": "In 0.3.3, LeRobotDataset construction raises check_timestamps_sync ValueError (phase 2), so __getitem__ is never reached.",
        "source_note": "hf_transform_to_torch converts the float64 epoch timestamps to float32. Every value in the episode rounds to 1784331520.0 (the float32 spacing at this magnitude is 128 s), so all delta windows collapse to one query value.",
        "offset_note": "Video pts run 0 to 14.53 s; queries are about 1.784e9 s. Because the episode's float32 time range also collapses to one value, every nominal query is classed as outside the episode, which is an artifact of the preregistered rule.",
        "gaps": [
            "Comparator only: 0.3.3 and episode 0 only; the other 39 listed episodes were not probed.",
            "Epoch-scale timestamps under 0.4.0/v3.0 are not tested.",
        ],
    },
}


def load(path: Path):
    return json.loads(path.read_text())


def bucket_stats(batches: list[dict]) -> dict:
    out = {}
    for bucket in BUCKETS:
        chosen = [b for b in batches if qp.batch_bucket(b) == bucket]
        if not chosen:
            continue
        per_query = [q for b in chosen for q in (b["replay"].get("per_query") or []) if q]
        ok_d = [q["decoder_distance_s"] for b in chosen if b["outcome"] == "ok" for q in b["replay"]["per_query"]]
        bad_d = [q["decoder_distance_s"] for q in per_query if not q["within_tolerance"]]
        out[bucket] = {
            "batches": len(chosen),
            "queries": sum(len(b["queries"]) for b in chosen),
            "counts": qp.summarize(chosen)[bucket],
            "verdict": qp.bucket_verdict(qp.summarize(chosen)[bucket]),
            "max_distance_ok_s": max(ok_d) if ok_d else None,
            "min_distance_failing_query_s": min(bad_d) if bad_d else None,
            "harness_discrepancies": sum(b["outcome"] == "harness_discrepancy" for b in chosen),
        }
    return out


def offset_fidelity(batches: list[dict]) -> dict:
    """Intended (i/fps) versus effective (row timestamp difference) at the middle anchor."""
    out = {}
    for b in batches:
        if b["query_class"] == "loader_derived" and b["anchor"] == "middle" and b["config"] in {"diffusion", "vqbet", "tdmpc"} and b["camera"] == batches[0]["camera"]:
            out[b["config"]] = [{"delta_rows": q["delta_rows"], "intended_s": q["intended_offset_s"], "effective_s": q["effective_offset_s"]} for q in b["queries"]]
    return out


def failures(batches: list[dict]) -> list[dict]:
    return [
        {
            "camera": b["camera"], "anchor": b["anchor"], "config": b["config"], "bucket": qp.batch_bucket(b),
            "queries_s": [q["video_query_s"] for q in b["queries"]],
            "returned_pts": [q["returned_pts"] if q else None for q in (b["replay"].get("per_query") or [])],
            "decoder_distance_s": [q["decoder_distance_s"] if q else None for q in (b["replay"].get("per_query") or [])],
            "outcome": b["outcome"],
        }
        for b in batches if b["outcome"] != "ok"
    ]


def case_report(case: dict, phase2: dict, audit_case: dict) -> dict:
    key = case["case_id"].replace(":", "_")
    plan = load(HERE / "plan" / f"{key}.json")
    probe = load(HERE / "evidence" / f"{key}_probe.json")
    video = load(HERE / "evidence" / f"{key}_video_facts.json")["videos"]
    batches = probe.get("batches", [])
    stats = bucket_stats(batches)
    loader_counts = qp.summarize([b for b in batches if b["query_class"] == "loader_derived"])
    merged = {k: sum(loader_counts.get(b, {}).get(k, 0) for b in ("loader_derived/in_range", "loader_derived/padded")) for k in ("ok", "tolerance_error", "other_error", "not_run")}
    shifted = qp.bucket_verdict(merged) if probe["status"] == "executed" else "unresolved"
    padded_rows_on_boundary = all(
        q["row"] in (0, plan["plan"]["rows"] - 1)
        for b in batches if b["query_class"] == "loader_derived" for q in b["queries"] if q["is_pad"]
    )
    env = probe["environment"]
    return {
        "case_id": case["case_id"],
        "role": case["role"],
        "dataset": case["dataset"],
        "revision": case["revision"],
        "episode": case["episode"],
        "original": {
            "phase1_label": audit_case["label"],
            "phase1_severity": audit_case.get("severity"),
            "assertion": audit_case["message"],
            "phase2_classification": phase2["classification"],
            "phase2_tested_scope": phase2.get("tested_scope"),
            "phase2_reason": phase2.get("reason"),
        },
        "loader_release": env["lerobot"],
        "backend": "pyav",
        "experiment": probe["experiment"],
        "identity": {
            "plan_identity_ok": plan["identity_ok"],
            "inputs_rehashed_ok": probe["inputs_rehashed_ok"],
            "inputs": [{k: e.get(k) for k in ("path", "sha256", "phase2_sha256", "frozen_lfs_sha256", "snapshot_revision", "ok")} for e in plan["inputs"]["entries"]],
            "upstream_sources_ok": plan["sources"]["ok"],
            "packages_differing_from_phase2_freeze": env["packages_differing_from_phase2_freeze"],
            "packages_not_in_phase2_freeze": env["packages_not_in_phase2_freeze"],
        },
        "data_facts": plan["data_facts"],
        "video_facts": video,
        "anchors": plan["plan"]["anchors"],
        "conformance": {"delta": probe["delta_conformance"], "loader_timestamps": probe["loader_timestamp_conformance"]},
        "default_backend": probe["default_backend_attempt"],
        "buckets": stats,
        "padded_rows_on_episode_boundary": padded_rows_on_boundary,
        "offset_fidelity_middle_anchor": offset_fidelity(batches),
        "failing_batches": failures(batches),
        "conclusions": {
            "shifted_loader_derived_lookup": {
                "result": VERDICT_LABEL[shifted],
                "scope": f"{ENV_NAME[case['env']]} decode_video_frames(backend='pyav', tolerance_s=1e-4), direct decoder, loader-derived query sets for default diffusion/vqbet/tdmpc/smolvla configs plus no-delta control, episode {case['episode']}, anchors first/middle/worst_grid/last, {len(case['cameras'])} camera(s)",
            },
            "nominal_grid_comparator": {b: BUCKET_LABEL[s["verdict"]] for b, s in stats.items() if b.startswith("nominal_grid")},
            "full_loader": {"result": "unresolved", "reason": SCOPE[case["case_id"]]["full_loader"]},
            "default_torchcodec_backend": {"result": "unresolved" if probe["default_backend_attempt"]["outcome"] != "ok" else "ok", "reason": probe["default_backend_attempt"].get("message", "")[:200]},
        },
        "source_note": SCOPE[case["case_id"]]["source_note"],
        "offset_note": SCOPE[case["case_id"]]["offset_note"],
        "remaining_gaps": SCOPE[case["case_id"]]["gaps"],
        "evidence": [f"plan/{key}.json", f"evidence/{key}_probe.json", f"evidence/{key}_video_facts.json"],
    }


def build_report() -> dict:
    expectations = load(HERE / "expectations.json")
    prereg = load(HERE / "preregistration.json")
    changed = [p for p, digest in prereg["sha256"].items() if qp.sha256_file(HERE / p) != digest]
    if changed:
        raise SystemExit(f"preregistered files changed: {changed}")
    phase2 = {c["case_id"]: c for c in load(REPO / "loader-verification/2026-11-22/report.json")["cases"]}
    audit = {c["key"]: c for c in load(REPO / "audits/2026-11-22/audit.json")["cases"]}
    cases = [case_report(c, phase2[c["case_id"]], audit[c["case_id"]]) for c in expectations["cases"]]
    return {
        "schema": "kinema.shifted_queries.report/1",
        "date_utc": expectations["date_utc"],
        "frozen_audit_sha256": expectations["frozen_sources"]["audit_json"]["sha256"],
        "phase2_expectations_sha256": expectations["frozen_sources"]["phase2_expectations"]["sha256"],
        "preregistration_sha256": qp.sha256_file(HERE / "preregistration.json"),
        "preregistered_files": prereg["sha256"],
        "decision_rules": expectations["decision_rules"],
        "offset_selection": expectations["offset_selection"],
        "cases": cases,
        "not_claimed": [
            "No product accuracy, aggregate accuracy or training-benefit claim.",
            "No full-loader or training success for any case.",
            "No conclusion about unprobed episodes, releases, backends or non-default policy configs.",
        ],
    }


def fmt(x, digits=3):
    if x is None:
        return "–"
    return f"{x:.{digits}g}"


def render(report: dict) -> str:
    lines = [
        "# Shifted-timestamp video lookup (2026-11-22)",
        "",
        "This report tests the shifted-query consequence that [phase 2](../../loader-verification/2026-11-22/report.md) left open for FineART (`r1:c1:f0`). It runs the same query class on droid (`r71:c1:f1`) and uses `r94:c1:f1` as an epoch-scale comparator. It does not revise the frozen [phase-1 audit](../../audits/2026-11-22/audit.md) or the phase-2 report; the new results are listed separately below. Method: [METHOD.md](METHOD.md). Severity proposal (not implemented): [severity-proposal.md](severity-proposal.md).",
        "",
        f"Frozen audit SHA-256 `{report['frozen_audit_sha256']}`. Phase-2 expectations SHA-256 `{report['phase2_expectations_sha256']}`. Preregistration SHA-256 `{report['preregistration_sha256']}`. The preregistration was committed before any probe ran on pinned inputs.",
        "",
        "**All results are direct-decoder results.** `LeRobotDataset` was not constructed or patched for any case, and the full-loader consequence stays unresolved for all three.",
        "",
        "## Results, kept separate from the original reports",
        "",
        "| Case | Phase 1 | Phase 2 (unchanged) | Shifted loader-derived lookup (new) | Loader / backend | Full loader |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["cases"]:
        o = c["original"]
        lines.append(f"| `{c['case_id']}` ({c['role']}) | {o['phase1_label']} ({o['phase1_severity']}) | {o['phase2_classification']} | **{c['conclusions']['shifted_loader_derived_lookup']['result']}** | {c['loader_release']} / pyav | {c['conclusions']['full_loader']['result']} |")
    lines += ["", "\"Refuted\" and \"confirmed\" apply only to the executed query class, release, backend, episode and anchors named in each case. They are not verdicts on the original findings as a whole.", ""]
    lines += ["## What the loader queries", "", "These come from the installed source, with hashes in `expectations.json` (paths under `site-packages/lerobot/`):", "",
              "- `datasets/factory.py::resolve_delta_timestamps` sets `delta_timestamps = [i / fps for i in cfg.observation_delta_indices]` for every `observation.*` key, including video keys. `LeRobotDataset` defaults to `delta_timestamps=None`, which queries only the current timestamp.",
              "- `get_delta_indices` gives `round(d * fps)` row offsets, and `_get_query_indices` clamps them to the episode and sets `<key>_is_pad`. The video queries are then the **stored timestamps of the clamped rows** (`_get_query_timestamps`), converted to float32 by `hf_transform_to_torch`. Release 0.4.0 also adds `from_timestamp`.",
              "- Phase 2's `raw + k/30` is therefore not a LeRobot query. It is kept below as a labeled `nominal_grid` comparator.",
              "- Release 0.3.3 calls `check_timestamps_sync` at construction; 0.4.0 does not call it anywhere. In both releases the pyav decoder requires `|float32(q) - float32(pts)| < 1e-4` for every query in a batch.",
              "- The default backend is torchcodec whenever its module spec exists. libtorchcodec failed to load in both environments, so only pyav was exercised.", ""]
    lines += ["## Per-case results", ""]
    for c in report["cases"]:
        d = c["data_facts"]
        cam0 = next(iter(c["video_facts"].values()))
        lines += [f"### `{c['case_id']}` — {c['dataset']}@`{c['revision']}`, episode {c['episode']}", "",
                  f"Original assertion (phase 1, verbatim): {c['original']['assertion']}", "",
                  f"Identity: inputs re-hashed against the phase-2 ledger and Hub snapshot revision: {'ok' if c['identity']['inputs_rehashed_ok'] and c['identity']['plan_identity_ok'] else 'FAILED'}. Upstream sources match the preregistered hashes: {'ok' if c['identity']['upstream_sources_ok'] else 'FAILED'}. Packages differing from the phase-2 freeze: {len(c['identity']['packages_differing_from_phase2_freeze'])}. The frozen audit holds LFS hashes for {sum(1 for e in c['identity']['inputs'] if e['frozen_lfs_sha256'])}/{len(c['identity']['inputs'])} inputs; the rest are checked against the phase-2 ledger only.", "",
                  f"Data: {d['episode_rows']} rows, declared fps {d['declared_fps']:g}, stored `{d['stored_dtype']}` (info `{d['info_timestamp_dtype']}`), first/last {d['first_timestamp']:.6g}/{d['last_timestamp']:.6g} s. Video (first camera): {cam0['frames']} frames at {fmt(cam0['average_rate'])} fps, pts {fmt(cam0['first_pts'])}–{fmt(cam0['last_pts'], 6)} s. Anchors: {c['anchors']}.", "",
                  f"Conformance: delta rows equal upstream `get_delta_indices` and pass `check_delta_timestamps` for every config. The float32 query timestamps match upstream `hf_transform_to_torch` ({c['conformance']['loader_timestamps']['mismatches']} mismatches in {c['conformance']['loader_timestamps']['batches_checked']} batches). " + (f"The 0.3.3 `load_dataset` replay gave {c['conformance']['loader_timestamps']['component_replay']['mismatches']} mismatches." if 'mismatches' in c['conformance']['loader_timestamps']['component_replay'] else "No 0.4.0 `load_hf_dataset` replay was run (it would hit the language dtype)."), "",
                  "| Bucket | Batches | Queries | ok | tolerance error | other | Max distance, ok (s) | Min distance, failing query (s) | Bucket rule |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for bucket, s in c["buckets"].items():
            k = s["counts"]
            lines.append(f"| {bucket} | {s['batches']} | {s['queries']} | {k['ok']} | {k['tolerance_error']} | {k['other_error'] + k['not_run']} | {fmt(s['max_distance_ok_s'])} | {fmt(s['min_distance_failing_query_s'])} | {BUCKET_LABEL[s['verdict']]} |")
        fid = c["offset_fidelity_middle_anchor"].get("tdmpc", [])
        lines += ["", f"Padding: every padded query reads the boundary row (first or last) and decodes that boundary frame; `is_pad` marks it: {'yes' if c['padded_rows_on_episode_boundary'] else 'NO'}. Effective versus intended offsets at the middle anchor (tdmpc, s): " + ", ".join(f"{q['effective_s']:.4g}/{q['intended_s']:.4g}" for q in fid) + ".", "",
                  f"Source note: {c['source_note']}", "", f"Timeline note: {c['offset_note']}", ""]
        if c["failing_batches"]:
            groups: dict[str, dict] = {}
            for f in c["failing_batches"]:
                g = groups.setdefault(f["bucket"], {"n": 0, "anchors": set(), "configs": set()})
                g["n"] += 1
                g["anchors"].add(f["anchor"])
                g["configs"].add(f["config"])
            lines += [f"Failing batches ({len(c['failing_batches'])}; full query, returned-pts and distance lists are in `report.json`): " + "; ".join(f"{b}: {g['n']} (anchors {', '.join(sorted(g['anchors']))}; configs {', '.join(sorted(g['configs']))})" for b, g in sorted(groups.items())) + ".", ""]
        lines += ["Conclusion (scoped): shifted loader-derived lookup is **" + c["conclusions"]["shifted_loader_derived_lookup"]["result"] + "** for " + c["conclusions"]["shifted_loader_derived_lookup"]["scope"] + ". Full loader: unresolved (" + c["conclusions"]["full_loader"]["reason"] + ") Default torchcodec backend: " + c["conclusions"]["default_torchcodec_backend"]["result"] + " (dependency failure).", "",
                  "Remaining gaps:", ""] + [f"- {g}" for g in c["remaining_gaps"]] + ["", "Evidence: " + ", ".join(f"`{e}`" for e in c["evidence"]), ""]
    lines += ["## Adversarial review", "",
              "- **Cherry-picked queries.** The offsets come from a rule: every default config in the release whose `observation_delta_indices` is not None, plus a no-delta control. `plan.py` rescans all policy configs and refuses a mismatch. The anchors are fixed, every camera is probed, and the complete query set was committed in the preregistration commit before execution. The 4 anchors are not every frame, and for droid and r94 the worst-grid anchor deviates only slightly.",
              "- **Padding behavior.** Clamping and `is_pad` follow `_get_query_indices` in both releases. Padded queries re-decode the boundary frame; the loader never queries outside the episode. The failing FineART `nominal_grid/outside_episode` queries (before 0 s or past the last pts) correspond to no loader query, and they are not counted as evidence for the claim. `is_pad` itself was not observed through `LeRobotDataset`. With `episodes=[...]`, the 0.4.0 index semantics (global `dataset_from_index` against the loaded table) were not exercised.",
              "- **Revision drift.** Each input was re-hashed against the phase-2 ledger, its cache snapshot path names the pinned 40-character revision, and the frozen LFS hash was checked where one exists. FineART data and videos have no frozen-audit LFS hash, so their identity rests on the phase-2 ledger and the snapshot revision. The upstream source files and the package set match the phase-2 records.",
              "- **Harness fidelity.** For each batch, an independent VideoReader replay reproduced the upstream pass or fail, and its frames were tensor-equal to the upstream output on every success (0 discrepancies). The float32 query timestamps match upstream `hf_transform_to_torch`.",
              "- **Float32 margin.** FineART successes reach distances of 1.53e-5 s (loader-derived) and 3.05e-5 s (nominal), on the order of float32 resolution near 300 s (about 3.05e-5 s) and within a third of `tolerance_s`. Longer timelines, such as large v3 `from_timestamp` offsets, would shrink that margin. That was not tested.",
              "- **Broader-than-evidence conclusions.** Every conclusion names its release, backend, query class, episode and anchors. Decoder success is not reported as loader or training success. Nothing here is aggregated into an accuracy figure.",
              "- **Prediction check.** The preregistered predictions said loader-derived queries would decode for FineART and droid and fail for r94, and they did. They did not anticipate the FineART `nominal_grid` failures outside the episode range, or r94's empty `nominal_grid/in_range` bucket, which comes from the float32 range collapse. Both are reported as observed.", "",
              "## Reproduce", "", "See [METHOD.md](METHOD.md#reproduce). Inputs are not committed. The evidence files record their SHA-256 hashes and the outcome of every batch.", ""]
    return "\n".join(lines)


def main() -> None:
    report = build_report()
    (HERE / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (HERE / "report.md").write_text(render(report))
    print(json.dumps({c["case_id"]: c["conclusions"]["shifted_loader_derived_lookup"]["result"] for c in report["cases"]}))


if __name__ == "__main__":
    main()

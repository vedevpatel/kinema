"""Build report.json and report.md from this phase's evidence and the frozen phase-3 evidence.

    python torchcodec-backend/2026-11-22/make_report.py   # needs numpy only

Reads evidence/, predictions/ and selection records here, plus the frozen phase-3
plan and probe evidence. Writes only report.json and report.md in this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import PHASE3, case_key, frozen_source_checks, load_expectations, load_json, sha256_file  # noqa: E402

sys.path.insert(1, str(PHASE3))
import queryplan as qp  # noqa: E402
import tcmodel as tm  # noqa: E402

BUCKETS = ("loader_derived/in_range", "loader_derived/padded", "nominal_grid/in_range", "nominal_grid/outside_episode")
BACKENDS = ("torchcodec", "pyav_control", "phase3_pyav")
SELECTION_FILES = [("tc-v2", "no_ffmpeg"), ("tc-v2", "ffmpeg_aliases"), ("tc-v3", "no_ffmpeg"), ("tc-v3", "ffmpeg_aliases")]


def outcome(batch: dict, backend: str, phase3: dict) -> str:
    return phase3["outcome"] if backend == "phase3_pyav" else batch[backend]["outcome"]


def distances(batch: dict, backend: str, phase3: dict) -> list[float]:
    if backend == "phase3_pyav":
        return [q["decoder_distance_s"] for q in phase3["replay"].get("per_query") or [] if q]
    rows = batch[backend]["replay"].get("per_query") or []
    return [q["decoder_distance_s"] for q in rows if q and q.get("decoder_distance_s") is not None]


def bucket_table(batches: list[dict], phase3_batches: list[dict]) -> dict:
    table = {}
    for backend in BACKENDS:
        table[backend] = {}
        for bucket in BUCKETS:
            pairs = [(b, p) for b, p in zip(batches, phase3_batches, strict=True) if qp.batch_bucket(b) == bucket]
            counts = dict.fromkeys(tm.OUTCOMES, 0)
            ok_dist, bad_dist = [], []
            for b, p in pairs:
                o = outcome(b, backend, p)
                counts[o if o in counts else "other_error"] += 1
                (ok_dist if o == "ok" else bad_dist).extend(distances(b, backend, p))
            entry = {"batches": len(pairs), "queries": sum(len(b["queries"]) for b, _ in pairs), "counts": {k: v for k, v in counts.items() if v}, **tm.bucket_verdict(counts)}
            if ok_dist:
                entry["max_distance_ok_s"] = max(ok_dist)
            if bad_dist:
                failing = [d for d in bad_dist if d >= tm.TOLERANCE_S]
                if failing:
                    entry["min_distance_failing_query_s"] = min(failing)
            table[backend][bucket] = entry
    return table


def merged_loader_verdict(batches: list[dict], phase3_batches: list[dict], backend: str) -> dict:
    counts = dict.fromkeys(tm.OUTCOMES, 0)
    for b, p in zip(batches, phase3_batches, strict=True):
        if b["query_class"] == "loader_derived":
            o = outcome(b, backend, p)
            counts[o if o in counts else "other_error"] += 1
    verdict = tm.bucket_verdict(counts)
    return {"counts": {k: v for k, v in counts.items() if v}, **verdict, "result": tm.VERDICT_LABEL[verdict["verdict"]]}


def agreement(batches: list[dict], phase3_batches: list[dict]) -> dict:
    out = {}
    for bucket in BUCKETS:
        pairs = [(b, p) for b, p in zip(batches, phase3_batches, strict=True) if qp.batch_bucket(b) == bucket]
        if not pairs:
            continue
        pts = {"both_returned": {"same": 0, "different": 0}, "selected_then_rejected": {"same": 0, "different": 0}, "replay_only_own_index": {"same": 0, "different": 0}}
        px_equal = px_differ = 0
        px_stats, offsets, differ_examples = [], {}, []
        transitions: dict[str, int] = {}
        for b, _ in pairs:
            key = f"torchcodec={b['torchcodec']['outcome']} / pyav={b['pyav_control']['outcome']}"
            transitions[key] = transitions.get(key, 0) + 1
            both_ok = b["torchcodec"]["outcome"] == "ok" and b["pyav_control"]["outcome"] == "ok"
            for k, q in enumerate(b["comparison"]["per_query"]):
                if "same_pts" in q:
                    slot = pts["both_returned" if both_ok else "selected_then_rejected"]
                    slot["same" if q["same_pts"] else "different"] += 1
                    if not q["same_pts"] and len(differ_examples) < 12:
                        differ_examples.append({"camera": b["camera"], "anchor": b["anchor"], "config": b["config"], "outcomes": key, "query_s": b["queries"][k]["video_query_s"], "torchcodec_ticks": q["tc_ticks"], "pyav_ticks": q["pyav_ticks"], "pts_difference_s": q["pts_difference_s"]})
                elif "same_pts_own_frame" in q:
                    pts["replay_only_own_index"]["same" if q["same_pts_own_frame"] else "different"] += 1
                if "pixels" in q:
                    if q["pixels"]["equal"]:
                        px_equal += 1
                    else:
                        px_differ += 1
                        px_stats.append(q["pixels"])
                        off = q["pixels"].get("closest_pyav_offset")
                        offsets[str(off)] = offsets.get(str(off), 0) + 1
        entry = {
            "batch_outcomes": dict(sorted(transitions.items())),
            "same_outcome_batches": sum(b["comparison"]["same_outcome"] for b, _ in pairs),
            "batches": len(pairs),
            "returned_pts": pts,
            "pixels_both_succeeded": {"equal": px_equal, "different": px_differ},
        }
        if differ_examples:
            entry["pts_difference_examples"] = differ_examples
        if px_stats:
            entry["pixel_difference"] = {
                "max_abs_max": max(s["max_abs"] for s in px_stats),
                "mean_abs_max": max(s["mean_abs"] for s in px_stats),
                "mean_abs_min": min(s["mean_abs"] for s in px_stats),
                "fraction_values_differing_max": max(s["fraction_values_differing"] for s in px_stats),
                "closest_pyav_window_offset": offsets,
            }
        out[bucket] = entry
    return out


def control_drift(batches: list[dict], phase3_batches: list[dict]) -> dict:
    outcome_drift, pts_drift, compared = [], 0, 0
    for b, p in zip(batches, phase3_batches, strict=True):
        if (b["camera"], b["anchor"], b["config"]) != (p["camera"], p["anchor"], p["config"]):
            raise SystemExit("batch order differs from phase-3 evidence")
        if b["pyav_control"]["outcome"] != p["outcome"]:
            outcome_drift.append({"camera": b["camera"], "anchor": b["anchor"], "config": b["config"], "control": b["pyav_control"]["outcome"], "phase3": p["outcome"]})
        for q, q3 in zip(b["pyav_control"]["replay"].get("per_query") or [], p["replay"].get("per_query") or [], strict=True):
            if q and q3:
                compared += 1
                pts_drift += q["returned_pts_s"] != q3["returned_pts"]
    return {"batches_compared": len(batches), "outcome_drift": outcome_drift, "queries_compared": compared, "returned_pts_drift": pts_drift}


def model_agreement(batches: list[dict], predictions: dict) -> dict:
    batch_mismatch, query_mismatch, compared = [], 0, 0
    for b, pr in zip(batches, predictions["batches"], strict=True):
        if b["torchcodec"]["outcome"] != pr["predicted_outcome"]:
            batch_mismatch.append({"camera": b["camera"], "anchor": b["anchor"], "config": b["config"], "observed": b["torchcodec"]["outcome"], "predicted": pr["predicted_outcome"]})
        for q, p in zip(b["torchcodec"]["replay"].get("per_query") or [], pr["per_query"], strict=True):
            compared += 1
            same_index = q["raw_index"] == p["raw_index"]
            same_pts = q.get("returned_pts_s") == p.get("returned_pts_s")
            query_mismatch += not (same_index and same_pts)
    return {"batches_compared": len(batches), "batch_outcome_mismatches": batch_mismatch, "queries_compared": compared, "query_index_or_pts_mismatches": query_mismatch}


def failure_mechanisms(batches: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for b in batches:
        o = b["torchcodec"]["outcome"]
        if o == "ok":
            continue
        bucket = qp.batch_bucket(b)
        entry = out.setdefault(f"{bucket}: {o}", {"batches": 0, "wrapped_negative_index": 0, "index_at_or_above_nb_frames": 0, "examples": []})
        entry["batches"] += 1
        nb = b["torchcodec"]["replay"]["metadata"]["num_frames_from_header"]
        rows = b["torchcodec"]["replay"]["per_query"]
        entry["wrapped_negative_index"] += any(r["raw_index"] < 0 for r in rows)
        entry["index_at_or_above_nb_frames"] += any(r["index"] >= nb for r in rows)
        if len(entry["examples"]) < 3:
            entry["examples"].append({"camera": b["camera"], "anchor": b["anchor"], "config": b["config"], "queries_s": [q["video_query_s"] for q in b["queries"]], "raw_index": [r["raw_index"] for r in rows], "index": [r["index"] for r in rows], "returned_pts_s": [r.get("returned_pts_s") for r in rows], "nb_frames": nb, "message": b["torchcodec"]["upstream"].get("message", "")[:240]})
    return out


def video_metadata(batches: list[dict]) -> dict:
    out = {}
    for b in batches:
        md = b["torchcodec"]["replay"].get("metadata")
        if md and b["camera"] not in out:
            out[b["camera"]] = md
    return out


def case_report(case: dict, expectations: dict) -> dict:
    key = case_key(case["case_id"])
    evidence = load_json(HERE / "evidence" / f"{key}_torchcodec.json")
    phase3 = load_json(PHASE3 / "evidence" / f"{key}_probe.json")
    predictions = load_json(HERE / "predictions" / f"{key}.json")
    base = {
        "case_id": case["case_id"],
        "role": case["role"],
        "dataset": case["dataset"],
        "revision": case["revision"],
        "episode": case["episode"],
        "environment": case["environment"],
        "lerobot": case["lerobot"],
        "original_assertion": case["original_assertion"],
        "phase3_pyav_result": case["phase3_pyav_result"],
        "status": evidence["status"],
        "identity": evidence["identity"]["checks"],
        "ffmpeg": evidence["identity"]["torchcodec_environment"].get("ffmpeg_library_versions", {}).get("ffmpeg_version"),
    }
    if evidence["status"] != "executed":
        base["conclusion"] = {"torchcodec_shifted_loader_derived_lookup": "unresolved", "reason": "not executed: identity or environment check failed"}
        return base
    batches = evidence["batches"]
    phase3_batches = phase3["batches"]
    merged = {backend: merged_loader_verdict(batches, phase3_batches, backend) for backend in BACKENDS}
    scope = f"LeRobot {case['lerobot']} decode_video_frames(backend='torchcodec', tolerance_s=1e-4) with torchcodec 0.5 and FFmpeg {base['ffmpeg']} (PyAV-bundled), direct decoder, frozen phase-3 loader-derived query sets, episode {case['episode']}, anchors first/middle/worst_grid/last, {len(case['cameras'])} camera(s)"
    base.update(
        video_metadata=video_metadata(batches),
        buckets=bucket_table(batches, phase3_batches),
        loader_derived_merged=merged,
        backend_agreement=agreement(batches, phase3_batches),
        pyav_control_vs_phase3=control_drift(batches, phase3_batches),
        source_model_agreement=model_agreement(batches, predictions),
        torchcodec_rejections=failure_mechanisms(batches),
        replay_consistency={b: sum(x[b]["replay_consistent"] for x in batches) for b in ("torchcodec", "pyav_control")},
        batches_total=len(batches),
        default_backend_attempt=evidence["default_backend_attempt"],
        selection_before_batches=evidence["selection_before_batches"],
        conclusion={
            "torchcodec_shifted_loader_derived_lookup": merged["torchcodec"]["result"],
            "via": merged["torchcodec"]["via"],
            "scope": scope,
            "pyav_control": merged["pyav_control"]["result"],
            "phase3_pyav": merged["phase3_pyav"]["result"],
            "full_loader": case["full_loader"],
        },
    )
    return base


def selection_records() -> list[dict]:
    out = []
    for env, variant in SELECTION_FILES:
        path = HERE / "evidence" / f"selection_{env}_{variant}.json"
        if not path.exists():
            out.append({"env": env, "variant": variant, "status": "not_run"})
            continue
        r = load_json(path)
        out.append({
            "env": env,
            "variant": variant,
            "lerobot": r["lerobot"],
            "find_spec_torchcodec": r["find_spec_torchcodec"],
            "get_safe_default_codec": r["get_safe_default_codec"],
            "import_torchcodec_decoders": r["import_torchcodec_decoders"]["outcome"],
            "decode_default_backend": r["decode_default_backend"]["outcome"],
            "decode_default_backend_exception": r["decode_default_backend"].get("exception"),
            "decode_explicit_pyav": r["decode_explicit_pyav"]["outcome"],
            "default_equals_explicit_torchcodec": (r.get("default_equals_explicit_torchcodec") or {}).get("value"),
            "source_lines_present": all(all(v.values()) for v in r["source_lines_present"].values()),
        })
    return out


def build_report() -> dict:
    expectations = load_expectations()
    prereg_sha = sha256_file(HERE / "preregistration.json")
    return {
        "schema": "kinema.torchcodec_backend.report/1",
        "date_utc": "2026-11-22",
        "preregistration_sha256": prereg_sha,
        "expectations_sha256": sha256_file(HERE / "expectations.json"),
        "frozen_sources_unchanged": all(frozen_source_checks(expectations).values()),
        "experiment": expectations["experiment"],
        "selection": selection_records(),
        "cases": [case_report(c, expectations) for c in expectations["cases"]],
        "limits": expectations["limits"],
    }


def fmt(x, digits=3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def counts_text(counts: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in counts.items()) or "–"


def render(report: dict) -> str:
    lines = [
        "# torchcodec backend comparison (2026-11-22)",
        "",
        "This report replays the frozen [phase-3](../../shifted-queries/2026-11-22/report.md) shifted-query plans through LeRobot's **torchcodec** decoder path and compares each batch with **pyav**. It does not revise the phase-1 audit, phase 2 or phase 3. Their files are unchanged and their hashes are re-checked: " + ("ok" if report["frozen_sources_unchanged"] else "**changed**") + ". Method: [METHOD.md](METHOD.md). Environment: [environment_spec.json](environment_spec.json).",
        "",
        f"Preregistration SHA-256 `{report['preregistration_sha256']}` (committed before any batch ran). Expectations SHA-256 `{report['expectations_sha256']}`.",
        "",
        "**All results are direct-decoder results.** `LeRobotDataset` was not constructed or patched, and the full-loader consequence stays unresolved for every case. The environment is distinct from phases 2 and 3: the same venvs and packages, with FFmpeg 7.1.1 shared libraries added on the library path.",
        "",
        "## Results, kept separate by phase, backend and release",
        "",
        "| Case | Release | Phase 3, pyav (frozen) | pyav control (this run) | torchcodec (this run) | torchcodec rejection mechanism | Full loader |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["cases"]:
        con = c["conclusion"]
        lines.append(f"| `{c['case_id']}` ({c['role']}) | {c['lerobot']} | {con.get('phase3_pyav', '–')} | {con.get('pyav_control', '–')} | **{con['torchcodec_shifted_loader_derived_lookup']}** | {', '.join(con.get('via') or []) or '–'} | unresolved |")
    lines += [
        "",
        "Each verdict covers loader-derived queries only (in-range and padded together, as in phase 3) for the named release, backend, episode and anchors. Verdicts are not pooled and do not judge the original findings as a whole.",
        "",
        "## Default-backend selection and import",
        "",
        "| Env | LeRobot | FFmpeg on path | `get_safe_default_codec()` | import torchcodec | `decode_video_frames(..., backend=None)` | explicit pyav |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in report["selection"]:
        if s.get("status") == "not_run":
            lines.append(f"| {s['env']} | – | {s['variant']} | not run | | | |")
            continue
        default = s["decode_default_backend"] + (f" ({s['decode_default_backend_exception']})" if s["decode_default_backend_exception"] else "")
        lines.append(f"| {s['env']} | {s['lerobot']} | {'yes' if s['variant'] == 'ffmpeg_aliases' else 'no'} | `{s['get_safe_default_codec']}` | {s['import_torchcodec_decoders']} | {default} | {s['decode_explicit_pyav']} |")
    lines += ["", "These rows use a synthetic video. `get_safe_default_codec()` checks only `importlib.util.find_spec(\"torchcodec\")`, so in both releases the default is torchcodec whether or not its libraries load. `LeRobotDataset.__init__` and `DatasetConfig.video_backend` use that default (source lines checked in each release). No pyav fallback happens when the import fails; the error propagates.", ""]
    lines += ["## Per-case results", ""]
    for c in report["cases"]:
        lines += [f"### `{c['case_id']}` — {c['dataset']}@`{c['revision']}`, episode {c['episode']}", "", f"Original assertion (phase 1, verbatim): {c['original_assertion']}", ""]
        lines.append("Identity: " + ", ".join(f"{k} {'ok' if v else 'FAILED'}" for k, v in c["identity"].items()) + f". FFmpeg reported by torchcodec: {c['ffmpeg']}.")
        lines.append("")
        if c["status"] != "executed":
            lines += [f"Not executed: {c['conclusion']['reason']}", ""]
            continue
        md = next(iter(c["video_metadata"].values()))
        lines.append(f"torchcodec metadata (first camera, approximate mode): average_fps {fmt(md['average_fps'])} (= average_fps_from_header), num_frames {md['num_frames']} (from header; num_frames_from_content {md['num_frames_from_content']}), {md['width']}x{md['height']} {md['codec']}.")
        lines += ["", "| Bucket | Backend | Batches | Queries | Outcomes | Verdict | Max distance, ok (s) | Min distance, failing query (s) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for bucket in BUCKETS:
            for backend in BACKENDS:
                e = c["buckets"][backend][bucket]
                if not e["batches"]:
                    continue
                verdict = e["verdict"] + (f" ({', '.join(e['via'])})" if e["via"] else "")
                lines.append(f"| {bucket} | {backend} | {e['batches']} | {e['queries']} | {counts_text(e['counts'])} | {verdict} | {fmt(e.get('max_distance_ok_s'))} | {fmt(e.get('min_distance_failing_query_s'))} |")
        lines += ["", "`nominal_grid` rows are the labeled comparator (`raw + k/fps`, not a LeRobot query); their verdict strings are bucket-rule outputs and are not evidence for or against the claim.", "", "Backend agreement (torchcodec vs the same-process pyav control):", ""]
        for bucket, a in c["backend_agreement"].items():
            px = a["pixels_both_succeeded"]
            rp = a["returned_pts"]
            text = f"- `{bucket}`: same outcome in {a['same_outcome_batches']}/{a['batches']} batches ({'; '.join(f'{k}: {v}' for k, v in a['batch_outcomes'].items())}). Returned pts where both returned frames: {rp['both_returned']['same']} same, {rp['both_returned']['different']} different. Pixels there: {px['equal']} equal, {px['different']} different."
            if sum(rp["selected_then_rejected"].values()):
                text += f" Frames each backend selected in batches then rejected by tolerance: {rp['selected_then_rejected']['same']} same pts, {rp['selected_then_rejected']['different']} different."
            if sum(rp["replay_only_own_index"].values()):
                text += f" Replay-only own-index frames: {rp['replay_only_own_index']['same']} same pts, {rp['replay_only_own_index']['different']} different."
            if "pixel_difference" in a:
                pd = a["pixel_difference"]
                text += f" Pixel differences: max abs {pd['max_abs_max']}, mean abs {fmt(pd['mean_abs_min'])}–{fmt(pd['mean_abs_max'])}, up to {fmt(pd['fraction_values_differing_max'])} of values; closest pyav window offset {pd['closest_pyav_window_offset']}."
            lines.append(text)
        drift = c["pyav_control_vs_phase3"]
        model = c["source_model_agreement"]
        lines += [
            "",
            f"pyav control vs frozen phase 3: {len(drift['outcome_drift'])} outcome changes in {drift['batches_compared']} batches, {drift['returned_pts_drift']} returned-pts changes in {drift['queries_compared']} queries. Replays reproduced upstream in {c['replay_consistency']['torchcodec']}/{c['batches_total']} torchcodec batches and {c['replay_consistency']['pyav_control']}/{c['batches_total']} pyav batches.",
            "",
            f"Source model vs binary: {len(model['batch_outcome_mismatches'])} batch-outcome mismatches in {model['batches_compared']}, {model['query_index_or_pts_mismatches']} index or returned-pts mismatches in {model['queries_compared']} queries.",
            "",
        ]
        if c["torchcodec_rejections"]:
            lines.append("torchcodec rejections by mechanism:")
            lines.append("")
            for k, v in c["torchcodec_rejections"].items():
                ex = v["examples"][0]
                lines.append(f"- `{k}`: {v['batches']} batches; {v['wrapped_negative_index']} contain a negative index wrapped to the end of the video, {v['index_at_or_above_nb_frames']} an index at or above nb_frames ({ex['nb_frames']}). Example ({ex['camera']}, {ex['anchor']}, {ex['config']}): queries {[fmt(q, 7) for q in ex['queries_s']]} → raw index {ex['raw_index']} → index {ex['index']} → returned pts {[fmt(p, 7) for p in ex['returned_pts_s']]}.")
            lines.append("")
        d = c["default_backend_attempt"]
        lines.append(f"Default backend after all batches: `get_safe_default_codec()` = `{d['get_safe_default_codec']}`, outcome {d['outcome']}" + (f", equal to explicit torchcodec: {d.get('equals_explicit_torchcodec')}" if d["outcome"] == "ok" else f" ({d.get('exception')})") + ".")
        lines += ["", f"Conclusion (scoped): torchcodec shifted loader-derived lookup is **{c['conclusion']['torchcodec_shifted_loader_derived_lookup']}**" + (f" via {', '.join(c['conclusion']['via'])}" if c["conclusion"]["via"] else "") + f" for {c['conclusion']['scope']}. Full loader: {c['conclusion']['full_loader']}.", ""]
        lines += CASE_NOTES.get(c["case_id"], [])
        lines += ["Remaining gaps (unresolved):", ""] + [f"- {g}" for g in GAPS[c["case_id"]]] + [""]
        lines += [f"Evidence: `evidence/{case_key(c['case_id'])}_torchcodec.json`, `predictions/{case_key(c['case_id'])}.json`; plan `shifted-queries/2026-11-22/plan/{case_key(c['case_id'])}.json` (unchanged).", ""]
    lines += ADVERSARIAL + ["", "## Limits", ""] + [f"- {x}" for x in report["limits"]] + ["", "## Reproduce", "", "See [METHOD.md](METHOD.md#reproduce). Inputs are not committed. Evidence files record input hashes and the outcome of every batch for both backends."]
    return "\n".join(lines) + "\n"


CASE_NOTES: dict[str, list[str]] = {
    "r1:c1:f0": [
        "Selection note: every stream has r_frame_rate 90, matching the 90 Hz rows. At every probed loader-derived query, `round(ts * 90)` named the frame whose pts equals the stored row timestamp, which is the same frame pyav's nearest-pts rule chose. That includes the worst-grid anchor (row 1458). The tolerance claim is therefore refuted on this path for both backends. This says nothing about phase 3's separate timeline finding (info.json declares 30 fps, so delta index 1 spans 1/90 s).",
        "",
        "Comparator note (`nominal_grid/outside_episode`, not loader queries): the backends reject these queries by different mechanisms. Queries before 0 s give a negative index, which `VideoDecoder.get_frames_at` wraps to the end of the video, and tolerance then rejects them at about 299.4 s distance. Queries past the last frame raise `Invalid frame index`. pyav returns the nearest window frame and rejects at 0.0111 s. LeRobot's clamped loader queries never produce these values.",
        "",
    ],
    "r71:c1:f1": [
        "Selection note: r_frame_rate 15 and nb_frames 213. Every loader-derived and nominal query mapped to an in-range index, and both backends returned the same frame with identical pixels. Whether the frame at 0.7333 s shows the same instant as row 0 (the video starts at 0 s, the rows at 0.7333 s) is a semantic-alignment question that neither backend tests.",
        "",
    ],
    "r94:c1:f1": [
        "Mechanism note: the lookup is rejected under both backends, but differently. torchcodec raises `RuntimeError: Invalid frame index=53529945600 ... must be less than 437` from `validateFrameIndex` before LeRobot's tolerance assertion runs. pyav raises the tolerance `AssertionError`. The comparator's role is unchanged, and the exception class a caller would see differs by backend. No frames are returned, so no pixel comparison exists.",
        "",
    ],
}

GAPS: dict[str, list[str]] = {
    "r1:c1:f0": [
        "Full loader: LeRobotDataset construction fails on source dtype 'language' (phase 2), so neither backend was exercised through __getitem__.",
        "LeRobot 0.5.1's torchcodec path was not run.",
        "Only episode 40455, four anchors and the default policy configs were probed.",
        "Phase 3's timeline finding (declared 30 fps vs 90 Hz rows and video, so delta windows span a third of the intended time) is untouched, and its effect on training is not measured.",
        "torchcodec's filtergraph colour path (width not a multiple of 32) and other FFmpeg 7 builds were not exercised.",
    ],
    "r71:c1:f1": [
        "Full loader: LeRobotDataset construction raises KeyError('stats') (phase 2).",
        "The other 22 listed episodes were not probed.",
        "Whether the frame decoded at 0.7333 s shows the instant recorded in row 0 is not testable from timestamps or pixels alone.",
        "Only LeRobot 0.3.3 was run for this v2.1 dataset.",
    ],
    "r94:c1:f1": [
        "Comparator only: episode 0 and LeRobot 0.3.3; the other 39 listed episodes were not probed.",
        "Full loader: construction raises check_timestamps_sync ValueError in 0.3.3 (phase 2), before any lookup.",
        "How callers react to the torchcodec RuntimeError instead of the pyav AssertionError is not tested.",
        "Epoch-scale timestamps under 0.4.0/v3.0 are not tested.",
    ],
}

ADVERSARIAL = [
    "## Adversarial review",
    "",
    "- **Changed inputs.** Every input was re-hashed against the phase-3 plan record before any batch ran. Phase 3 had checked those records against the phase-2 ledger and the Hub snapshot revision. Every frozen phase-1/2/3 artifact used here was re-hashed too (`frozen_sources_ok`). FineART data and videos still have no frozen-audit LFS hash, so their identity rests on the phase-2 ledger and the snapshot revision, as in phase 3.",
    "- **Changed dependencies.** The Python package set equals the phase-2 freeze in both venvs, and the LeRobot source hashes equal phase 3's. The only addition is FFmpeg 7.1.1 shared libraries on `LD_LIBRARY_PATH`, taken from PyAV's own wheel by soname aliases, which makes this a distinct environment. The pyav control in that environment reproduced every phase-3 outcome and every returned pts (0 drift in 1056 queries), so at these queries the added library path did not change the pyav results. The pre-freeze smoke test wrote 8 bytecode caches into the v2 venv (`evidence/venv_writes.json`); no source or package file changed.",
    "- **Unverifiable binary.** The installed torchcodec `.so` files cannot be matched to the v0.5.0 C++ source. Their behaviour matched the source model's preregistered prediction on all 616 batches and 1056 queries (index, returned pts and outcome). That supports the source reading on these streams. A post-hoc demux (`evidence/stream_facts_posthoc.json`) shows every probed stream is H.264 High profile with B-frames (packet pts out of decode order), so presentation-order reordering was exercised. Untested: variable frame rate, a missing `nb_frames` header, seek-mode `exact`, and a header `r_frame_rate` that differs from the true frame spacing (the synthetic tests show torchcodec then indexes with the guessed rate).",
    "- **Query drift.** The plans are phase 3's preregistered files, byte-identical (hash-checked). The tests assert that every evidence batch equals its plan batch, field by field, including every query value. No query was regenerated, added, dropped or chosen after seeing an outcome, and the predictions were committed before execution.",
    "- **Harness fidelity.** Independent replays reproduced the upstream outcome, and on success the output tensor, in every batch for both backends. The torchcodec replay uses a fresh decoder, whereas 0.4.0 upstream uses its per-path cache. Their agreement shows no cache-order effect at these queries.",
    "- **Pixels.** Every query where both backends returned frames had the same pts and bit-identical uint8 pixels, so the neighbour-frame investigation never triggered. Three conditions produced this: the same FFmpeg binaries, H.264 yuv420p input, and widths that are multiples of 32 (640 and 256), so torchcodec used swscale and not its filtergraph fallback. A different FFmpeg build, pixel format or width could produce pixel differences. Those are not tested here, and frame equality says nothing about semantic alignment.",
    "- **Mechanism versus claim.** The r94 confirmation under torchcodec rests on index validation, not on the tolerance check. It is reported as `confirmed via index_error`, not as a tolerance confirmation. The FineART and droid refutations rest on all loader-derived batches decoding within tolerance under both backends.",
    "- **Float32 margin.** The loader-derived success distances are unchanged across backends (FineART up to 1.53e-5 s, droid 0), because both backends returned the same frames. The phase-3 caveat stands: longer timelines or large v3 `from_timestamp` offsets would shrink the margin, and that is untested.",
    "- **Broader-than-evidence conclusions.** Each verdict names its release, backend, query class, episode and anchors. Untested: LeRobot 0.5.1, other episodes, non-default configs, GPU decoding, DataLoader workers, the full loader and training. Default-backend selection was verified only in 0.3.3 and 0.4.0. Nothing here is aggregated into an accuracy figure or used to set a severity cutoff.",
]


def main() -> None:
    report = build_report()
    (HERE / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (HERE / "report.md").write_text(render(report))
    print(json.dumps([{k: c[k] for k in ("case_id", "status")} | {"conclusion": c["conclusion"]} for c in report["cases"]], indent=1))


if __name__ == "__main__":
    main()

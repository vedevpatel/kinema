"""Build a scoped loader-consequence report from the frozen case frame."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare import HERE, source_parts, verify_frozen


CHECKS = {"episode-file-pointers", "dropped-frames", "timestamp-grid", "deployment-readiness"}
LABELS = {"confirmed", "refuted", "unresolved"}
EXPECTED_EXPECTATIONS_SHA256 = "ee6d4f27e12e184d310fee484bcb7c162247e36a9f83424645f7ce9ce51a1b91"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_command(case_id: str, name: str, loader_version: str) -> str:
    if "_loader" in name:
        suffix = name.removesuffix(".json").split("_loader", 1)[1]
        extra = f" --suffix {suffix}" if suffix else ""
        program = f"run_loader.py {case_id}{extra}"
    elif name == "r1_c1_f0_decoder.json":
        program = "decode_fineart.py"
    elif name.endswith("_decoder.json"):
        program = f"decode_probe.py {case_id}"
    elif name.endswith("_hand_right.json"):
        program = "probe_traly.py"
    else:
        raise ValueError(f"Unknown evidence command: {name}")
    environment = "v2" if loader_version == "0.3.3" else "v3_secondary" if loader_version == "0.5.1" else "v3"
    python = {"v2": "/tmp/kinema-phase2-v2/bin/python", "v3": "/tmp/kinema-phase2-v3/bin/python", "v3_secondary": "/tmp/kinema-phase2-v3-051/bin/python"}[environment]
    return f"{python} loader-verification/2026-11-22/{program}"


def build_report(base: Path = HERE) -> dict:
    source = base.parents[1] / "audits/2026-11-22"
    audit_path = source / "audit.json"
    audit = json.loads(audit_path.read_text())
    expectations_path = base / "expectations.json"
    expectations = json.loads(expectations_path.read_text())
    if digest(audit_path) != expectations["source_audit_sha256"]:
        raise ValueError("Frozen audit changed")
    if digest(expectations_path) != EXPECTED_EXPECTATIONS_SHA256:
        raise ValueError("Pre-run expectations changed")
    expected_cases = {case["key"]: case for case in audit["cases"] if case["check"] in CHECKS and case["label"] == "unresolved"}
    planned = {case["case_id"]: case for case in expectations["cases"]}
    decisions = json.loads((base / "adjudications.json").read_text())
    if len(expected_cases) != 11 or set(planned) != set(expected_cases) or set(decisions) != set(expected_cases):
        raise ValueError("The eleven-case frame is incomplete or changed")
    frozen_hashes = json.loads((source / "artifact_hashes.json").read_text())
    records = []
    for case_id, plan in planned.items():
        original = expected_cases[case_id]
        decision = decisions[case_id]
        if decision["classification"] not in LABELS or plan["assertion"] != original["message"] or plan["revision"] != original["revision"]:
            raise ValueError(f"Invalid verdict or drift for {case_id}")
        case_stem = case_id.replace(":", "_")
        ledger_path = base / "evidence" / f"{case_stem}.json"
        ledger = json.loads(ledger_path.read_text())
        verify_frozen(plan, ledger, frozen_hashes)
        files = [f"evidence/{case_stem}.json", *["evidence/" + name for name in decision["evidence"]]]
        calls = []
        for name in decision["evidence"]:
            evidence = json.loads((base / "evidence" / name).read_text())
            if evidence["case_id"] != case_id:
                raise ValueError(f"Wrong evidence case: {name}")
            loader_version = (evidence.get("environment") or {}).get("lerobot") or ("0.5.1" if "0.5.1" in name else "0.3.3" if name.endswith("_decoder.json") and case_id != "r1:c1:f0" else "0.4.0")
            calls.append({"evidence_file": "evidence/" + name, "command": evidence_command(case_id, name, loader_version), "operation": evidence.get("operation"), "environment": evidence.get("environment") or {"lerobot": loader_version}, "steps": [{k: value for k, value in step.items() if k not in ("traceback", "message", "keys", "path")} for step in evidence.get("steps", evidence.get("probes", []))]})
        for name in files:
            if not (base / name).is_file():
                raise FileNotFoundError(name)
        repo, prefix = source_parts(plan["dataset"])
        recovery_commands = [f"python loader-verification/2026-11-22/prepare.py {case_id} --metadata"]
        recovery_commands.extend(f"python loader-verification/2026-11-22/prepare.py {case_id} --path {item['path']}" for item in ledger if not item["path"].startswith("meta/"))
        records.append({"case_id": case_id, "check": plan["check"], "dataset": plan["dataset"], "repo": repo, "subdir": prefix.removesuffix("/"), "revision": plan["revision"], "original_assertion": plan["assertion"], "original_audit_classification": original["label"], "metadata_anomaly_at_freeze": plan["metadata_anomaly_at_freeze"], "preregistered_operation": plan["operation"], "preregistered_confirm_if": plan["confirm_if"], "preregistered_refute_if": plan["refute_if"], "classification": decision["classification"], "tested_scope": decision["tested_scope"], "observed_values": decision["observation"], "reason": decision["reason"], "remaining_gap": decision["remaining_gap"], "artifact_hashes": ledger, "recovery_commands": recovery_commands, "calls": calls, "evidence_files": files})
    per_check = defaultdict(Counter)
    for record in records:
        per_check[record["check"]][record["classification"]] += 1
    counts = Counter(record["classification"] for record in records)
    return {"schema": 1, "date_utc": "2026-11-22", "scope": "Loader consequences in the eleven unresolved frozen phase-1 cases; no product certification or aggregate accuracy", "frozen_audit_sha256": digest(audit_path), "frozen_expectations_sha256": digest(expectations_path), "frozen_board_commit": audit["manifest"]["board_commit"], "frozen_board_engine": audit["manifest"]["board_engine"], "loader_versions": {"v2.0/v2.1": "LeRobot 0.3.3", "v3.0": "LeRobot 0.4.0", "secondary_v3_schema_probe": "LeRobot 0.5.1 for FineART only"}, "version_sources": {"v2_loader_source": "https://github.com/huggingface/lerobot/blob/v0.3.3/src/lerobot/datasets/lerobot_dataset.py", "v3_format_documentation": "https://github.com/huggingface/lerobot/blob/main/docs/source/lerobot-dataset-v3.mdx", "v3_loader_source": "https://github.com/huggingface/lerobot/blob/v0.4.0/src/lerobot/datasets/lerobot_dataset.py", "v3_secondary_source": "https://github.com/huggingface/lerobot/blob/v0.5.1/src/lerobot/datasets/dataset_metadata.py"}, "version_selection": "LeRobot 0.3.3 declares CODEBASE_VERSION v2.1 and its source explicitly supports v2.0 global stats. Official LeRobot v3 documentation dates v3.0 support to 0.4.0. Version 0.5.1 also declares v3.0 and was used only to check FineART's custom language dtype, not to seek a preferred verdict.", "environment_file": "evidence/environments.json", "result_counts": {label: counts[label] for label in sorted(LABELS)}, "per_check": {check: {label: counts[label] for label in sorted(LABELS)} for check, counts in sorted(per_check.items())}, "cases": records, "adversarial_review": {"selection_bias": "All eleven unresolved findings in the four named checks were included; no case removed because of unavailable per-dataset pages or difficult loaders. The previously confirmed r413:c2:f2 is excluded by the frozen unresolved filter.", "shared_assumptions": "The verifier imports no kinema detector code. It uses upstream LeRobot releases, raw parquet/MP4 and the frozen audit only to identify assertions and pins. A direct decoder probe cannot replace full dataset iteration; raw timestamp probes do not settle shifted training queries.", "revision_drift": "Every recovered path was fetched at a 40-character commit and SHA-256 hashed. Frozen info/stats and available LFS hashes were checked. No latest-branch data were substituted.", "incomplete_access": "Frame probes use first/middle/last and claim-specific positions, not every frame. BitRobot and VLABench pointer cases inspect the selected loader table but do not run training. FineART and droid cannot complete dataset construction for independent schema reasons. FineART shifted queries were not tested.", "weak_labels": "No policy checkpoint is specified for four inference-normalization claims. Those remain unresolved despite successful raw frame loading in three cases. No training or real-world miss-rate claim is made."}}


def markdown(report: dict) -> str:
    counts = report["result_counts"]
    lines = ["# Phase 2: pinned loader consequence verification (2026-11-22)", "", "This report measures eleven unresolved assertions from the [frozen audit](../../audits/2026-11-22/audit.md). It does not certify kinema or revise the phase-1 verdicts.", "", f"Frozen audit SHA-256: `{report['frozen_audit_sha256']}`. Preregistered expectations SHA-256: `{report['frozen_expectations_sha256']}`.", "", f"Scoped case results: **{counts['confirmed']} confirmed, {counts['refuted']} refuted, {counts['unresolved']} unresolved**. These are neither product accuracy nor a training outcome measure.", "", "LeRobot 0.3.3 tested v2.0/v2.1; 0.4.0 tested v3.0. FineART also received a 0.5.1 schema-compatibility probe. These versions follow their declared dataset-format support; full package versions and source hashes are in `evidence/environments.json`.", "", "| Case | Check | Consequence tested | Result |", "| --- | --- | --- | --- |"]
    for case in report["cases"]:
        lines.append(f"| `{case['case_id']}` | {case['check']} | {case['tested_scope']} | **{case['classification']}** |")
    lines += ["", "## Per-case observations", ""]
    for case in report["cases"]:
        lines += [f"### `{case['case_id']}` — {case['classification']}", "", f"Source: `{case['dataset']}@{case['revision']}`. Original assertion: {case['original_assertion']}", "", f"Observed: {case['observed_values']}", "", f"Reason: {case['reason']}", "", f"Remaining gap: {case['remaining_gap']}", "", "Evidence: " + ", ".join(f"`{name}`" for name in case["evidence_files"]), ""]
    lines += ["## Interpretation limits", "", "The four deployment-readiness claims remain unresolved at the asserted inference-policy consequence. Three sources load raw episode items but use different feature names; one source fails earlier because `features` is absent. A loader opening data cannot establish whether a particular pretrained policy can normalize at inference. No policy checkpoint or inference configuration was specified.", "", "The two file-pointer probes confirm that the affected episode has zero rows in the selected loader table. They do not observe a trainer, so the words ‘silently drops out of training’ remain untested. Cybernetic-Labs video lookup fails at the raw epoch-scale timestamps, but ordinary dataset construction rejects timestamp spacing first. The droid episode-0 video lookup claim is contradicted by direct decoder probes, though full dataset construction has a separate episode-stats blocker. FineART raw-timestamp probes succeeded, but shifted training queries were not tested and its full loader is blocked by a custom language dtype; its claimed tolerance consequence remains unresolved.", "", "## Adversarial review", ""]
    for key, value in report["adversarial_review"].items():
        lines.append(f"- **{key.replace('_', ' ').capitalize()}:** {value}")
    lines += ["", "## Reproduce", "", "```sh", "# Install the declared LeRobot releases in separate temporary Python environments (see evidence/environments.json).", "# Recover only the listed metadata and affected episode files at their immutable revisions:", "python loader-verification/2026-11-22/prepare.py CASE_ID --metadata", "python loader-verification/2026-11-22/prepare.py CASE_ID --path RELATIVE_PATH", "python loader-verification/2026-11-22/run_loader.py CASE_ID", "python loader-verification/2026-11-22/make_report.py", "```", "", "The exact prepared paths, content hashes, operations, observed outputs and errors are retained in `report.json` and `evidence/`. Local episode and video bytes are not committed.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    (HERE / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (HERE / "report.md").write_text(markdown(report))
    print(f"Wrote {len(report['cases'])} cases to {HERE / 'report.json'}")

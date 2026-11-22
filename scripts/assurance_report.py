"""Validate and render the frozen phase 1 assurance audit.

Run: python3 scripts/assurance_report.py audits/2026-11-22
No kinema imports, detector calls, or network requests.
"""

from __future__ import annotations

import hashlib
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE_COMMIT = "f701ff4bdd7be514644fa2769aa1bb22e979e4d6"
BOARD_ENGINE = "f9a1edd38b66"
MAIN_ENGINE_AT_FREEZE = "01fb98a42eb7"
LABELS = ("confirmed", "false alarm", "unresolved")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tally(cases: list[dict], key: str) -> dict[str, dict[str, int]]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for c in cases:
        groups[c[key]][c["label"]] += 1
    return {name: {**{label: counts[label] for label in LABELS}, "total": sum(counts.values())}
            for name, counts in sorted(groups.items())}


def assemble(root: Path) -> dict:
    sample = json.loads((root / "sample.json").read_text())
    board_raw = gzip.decompress((root / "board.json.gz").read_bytes())
    if hashlib.sha256(board_raw).hexdigest() != sample["board_sha256"]:
        raise ValueError("frozen board hash mismatch")
    board = json.loads(board_raw)
    if len(board["datasets"]) != sample["frame"]["board_rows"]:
        raise ValueError("frame does not match board rows")
    sources = {p["dataset"]: p for p in json.loads((root / "provenance.json").read_text())}
    obs = json.loads((root / "observations.json").read_text())
    judgments = json.loads((root / "adjudications.json").read_text())
    misses = json.loads((root / "misses.json").read_text())
    artifacts = json.loads((root / "artifact_hashes.json").read_text())
    keys = {c["key"] for c in sample["selected"]}
    if keys != set(judgments) or keys != set(obs) or len(keys) != len(sample["selected"]):
        raise ValueError("every selected key needs one observation and adjudication")
    if any(c["engine"] != BOARD_ENGINE for c in sample["selected"]):
        raise ValueError("mixed board engines need a separate manifest")
    cases = []
    for c in sample["selected"]:
        j = judgments[c["key"]]
        if j["label"] not in LABELS or j["signal_exists"] not in (True, False, None):
            raise ValueError(f"invalid adjudication: {c['key']}")
        if not j["reason"].strip():
            raise ValueError(f"missing reason: {c['key']}")
        p = sources[c["dataset"]]
        if p["revision"] != c["revision"]:
            raise ValueError(f"revision drift: {c['key']}")
        if j["label"] != "unresolved" and p["status"] != "revision_resolved":
            raise ValueError(f"resolved label without pinned source: {c['key']}")
        o = obs[c["key"]]
        if j["label"] != "unresolved" and not o.get("observed"):
            raise ValueError(f"resolved label without raw observation: {c['key']}")
        metadata = p.get("metadata") or {}
        cases.append({
            "key": c["key"], "dataset": c["dataset"], "revision": c["revision"],
            "engine": c["engine"], "check": c["check"], "severity": c["severity"],
            "message": c["message"], "finding_details": c["details"],
            "label": j["label"], "signal_exists": j["signal_exists"], "reason": j["reason"],
            "pinned_inputs": {"source_status": p["status"], "source": o.get("source", c["dataset"]),
                              "info_sha256": metadata.get("meta/info.json", {}).get("sha256"),
                              "stats_sha256": metadata.get("meta/stats.json", {}).get("sha256")},
            "commands": [f".venv/bin/python scripts/assurance_probe.py {root}",
                         f".venv/bin/python scripts/assurance_observe.py {root} {c['key']}"],
            "observed": o.get("observed", {}),
            "verification_limit": o.get("unsupported") or o.get("error"),
        })
    total = len(cases)
    counts = Counter(c["label"] for c in cases)
    return {
        "schema": 1, "date_utc": "2026-11-22", "scope": "frozen published finding instances; product not certified",
        "manifest": {"board_commit": BASE_COMMIT, "board_engine": BOARD_ENGINE,
                     "main_engine_at_freeze": MAIN_ENGINE_AT_FREEZE,
                     "board_generated": sample["board_generated"], "sampling_seed": sample["seed"],
                     "score_model": sample["score_model"],
                     "board_json_sha256": sample["board_sha256"],
                     "sha256": {name: sha(root / name) for name in
                                ("board.json.gz", "sample.json", "provenance.json", "observations.json",
                                 "adjudications.json", "misses.json", "artifact_hashes.json")}},
        "sampling_frame": sample["frame"],
        "results": {"selected": total, **{label: counts[label] for label in LABELS},
                    "unresolved_over_total": f"{counts['unresolved']}/{total}",
                    "datasets_selected": len({c["dataset"] for c in cases})},
        "per_check": tally(cases, "check"), "per_dataset": tally(cases, "dataset"),
        "cases": cases, "miss_exercise": misses,
        "source_artifact_count": len(artifacts),
        "adversarial_review": {
            "selection_bias": "Two-per-check and FAIL fill overrepresent rare checks and FAIL findings; distinct-dataset preference changes prevalence. The two bridge task-string rows appear to be mirrors, so 39 dataset IDs are not 39 independent sources. No population precision estimate is claimed.",
            "shared_assumptions": "Raw probes use source info.json path templates and published episode IDs to locate targets, but never kinema detection helpers. Metadata can be stale: the OpenGraph case was cross-checked against raw parquet. Heuristic arithmetic and source stats alone are not treated as proof of a defect.",
            "revision_drift": "All 35 selected Hub dataset IDs resolved to the exact row commit; source metadata and available object IDs are frozen. Four RLDS cases have only non-immutable GCS stamps and stay unresolved. Main's engine 01fb98a42eb7 differs from the board's f9a1edd38b66; no main rerun adjudicates a stored row.",
            "weak_labels": "The structural regression pack and corpus influenced development and cannot estimate misses. Most heuristic warnings lack video or task context. The one false alarm rejects the motor interpretation of a static quaternion component while preserving the static signal.",
            "page_eligibility": "Sampling reads all WARN/FAIL findings in frozen data.json, not per-dataset pages. Missing pages cannot silently remove rows; the test runs the sampler with no page directory.",
        },
    }


def markdown(doc: dict) -> str:
    m, r, frame = doc["manifest"], doc["results"], doc["sampling_frame"]
    lines = [
        "# Kinema assurance phase 1: frozen finding audit (2026-11-22)", "",
        "This independently audits a sample of published assertions. It does **not** certify kinema, its scores, or the board.", "",
        f"Frozen board commit: `{m['board_commit']}`; uncompressed board SHA-256: `{m['board_json_sha256']}`; "
        f"board engine: `{m['board_engine']}`; main engine at freeze: `{m['main_engine_at_freeze']}`. "
        "The stored board engine differs from main, and no current detector output is used as truth.", "",
        f"Board generated `{m['board_generated']}`; seed `{m['sampling_seed']}`. "
        f"The frame contains {frame['eligible']} WARN/FAIL finding instances on {frame['eligible_datasets']} datasets "
        f"out of {frame['board_rows']} board rows. Of {frame['all_findings']} total findings, "
        f"{frame['excluded_findings']} were excluded ({frame['excluded_by_severity']}). {frame['exclusions']}.", "",
        "Selection: two per check by fixed hash, covering both severities where available and preferring distinct datasets; "
        "remaining slots draw from FAIL findings. This deliberately oversamples rare checks and FAILs. "
        "The selection used only the frozen board JSON. No evidence or page availability influenced it.", "",
        f"Results: **{r['confirmed']} confirmed, {r['false alarm']} false alarm, "
        f"{r['unresolved']} unresolved** of {r['selected']} selected; unresolved/total = "
        f"**{r['unresolved_over_total']}**. Cases span {r['datasets_selected']} datasets. "
        "No aggregate accuracy or real-world miss rate is inferred. Precision is not reported because most cases remain unresolved.", "",
        "`audit.json` has every finding, pinned revision, commands, raw observations and reason. "
        "`board.json.gz` is the frozen selection source; `provenance.json` records resolved source commits and metadata hashes. "
        f"`artifact_hashes.json` records {doc['source_artifact_count']} pinned paths with Hub LFS SHA-256 or Git blob IDs where exposed. "
        "`observations.json` preserves raw-source probes. See `RUBRIC.md` for the preregistered labels.", "",
        "## By check", "", "| Check | Confirmed | False alarm | Unresolved | Total |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, count in doc["per_check"].items():
        lines.append(f"| {name} | {count['confirmed']} | {count['false alarm']} | {count['unresolved']} | {count['total']} |")
    lines += ["", "## By dataset", "", "| Dataset | Confirmed | False alarm | Unresolved | Total |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, count in doc["per_dataset"].items():
        lines.append(f"| {name.replace('|', '&#124;')} | {count['confirmed']} | {count['false alarm']} | {count['unresolved']} | {count['total']} |")
    miss = doc["miss_exercise"]
    lines += ["", "## Separate miss exercise", "",
              f"The structural regression pack has {sum(c['role'] == 'known_broken' for c in miss['fixtures'])} known-broken fixtures "
              f"and {sum('clean_control' in c['role'] for c in miss['fixtures'])} clean controls. "
              f"Its test run: {miss['test_results']['structural_pack']}; "
              f"the adjudicated corpus: {miss['test_results']['adjudicated_corpus']}. "
              "The FastUMI missing-video case and its v2.1 clean sibling are matched controls. "
              "`misses.json` lists each fixture's pinned revision and expected defects.", "",
              miss["development_influence"], "", miss["held_out_search"], "",
              "No real-world miss rate is inferred from these fixtures."]
    lines += ["", "## Adversarial review", ""]
    for heading, finding in doc["adversarial_review"].items():
        lines.append(f"- **{heading.replace('_', ' ').capitalize()}:** {finding}")
    lines += ["", "## Proposed next step", "",
              "Open a separate detector-fix PR for the quaternion-as-actuator false alarm after preserving this evidence. "
              "For the assurance layer, pin complete GCS object generations, run the relevant LeRobot loader on pointer and timestamp cases, "
              "inspect selected videos and trajectories, and acquire genuinely held-out broken and matched clean examples with independent labels."]
    for label, title in (("false alarm", "False alarms"), ("unresolved", "Unresolved cases"),
                         ("confirmed", "Confirmed cases")):
        lines += ["", f"## {title}", ""]
        for c in doc["cases"]:
            if c["label"] == label:
                lines.append(f"- `{c['key']}` {c['check']} ({c['dataset']}@{c['revision'][:12]}): {c['reason']}")
    lines += ["", "## Reproduction", "", "```sh",
              "python3 scripts/assurance_sample.py docs/data.json audits/2026-11-22",
              ".venv/bin/python scripts/assurance_probe.py audits/2026-11-22",
              ".venv/bin/python scripts/assurance_observe.py audits/2026-11-22",
              ".venv/bin/python scripts/assurance_artifacts.py audits/2026-11-22",
              "python3 scripts/assurance_report.py audits/2026-11-22", "```", "",
              "The Hub and GCS can change availability; pinned Hub commits and the frozen observations preserve what this audit saw. "
              "The GCS rows are unresolved because the board stamps are not immutable object generations.", ""]
    return "\n".join(lines)


def main() -> None:
    root = Path(sys.argv[1])
    doc = assemble(root)
    (root / "audit.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    (root / "audit.md").write_text(markdown(doc))
    print(doc["results"])


if __name__ == "__main__":
    main()

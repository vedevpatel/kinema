"""Verify pinned inputs and upstream sources, then write the complete query plan.

Run with the case's isolated environment Python, for example:

    /tmp/kinema-phase2-v3/bin/python -I shifted-queries/2026-11-22/plan.py r1:c1:f0

No kinema module and no LeRobot module is imported. The installed LeRobot
package is located and read as source text only. Inputs are the files phase 2
recovered under /tmp/kinema-phase2-inputs; nothing is downloaded.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from queryplan import build_case_plan, is_finite_plan, sha256_file  # noqa: E402

REPO = HERE.parents[1]
PHASE2 = REPO / "loader-verification" / "2026-11-22"
AUDIT = REPO / "audits" / "2026-11-22"
INPUT_ROOT = Path("/tmp/kinema-phase2-inputs")


def load_expectations() -> dict:
    return json.loads((HERE / "expectations.json").read_text())


def case_root(case_id: str, root: Path = INPUT_ROOT) -> Path:
    return root / case_id.replace(":", "_")


def lerobot_root() -> Path:
    spec = importlib.util.find_spec("lerobot")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("lerobot is not installed in this interpreter")
    return Path(list(spec.submodule_search_locations)[0])


def policy_observation_deltas(path: Path) -> list[int] | None | str:
    """Evaluate a config's default observation_delta_indices from source text.

    Returns the list, None, or "unparsed" when the property is absent. Only
    class-level integer defaults are bound to `self`; builtins are limited to
    range and list.
    """
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        defaults = {}
        prop = None
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and isinstance(item.value, ast.Constant) and isinstance(item.value.value, int):
                defaults[item.target.id] = item.value.value
            if isinstance(item, ast.FunctionDef) and item.name == "observation_delta_indices":
                prop = item
        if prop is None:
            continue
        returns = [n for n in ast.walk(prop) if isinstance(n, ast.Return)]
        if len(returns) != 1 or returns[0].value is None:
            return "unparsed"
        expression = ast.unparse(returns[0].value)
        value = eval(expression, {"__builtins__": {}, "range": range, "list": list}, {"self": SimpleNamespace(**defaults)})  # noqa: S307
        return value
    return "unparsed"


def verify_sources(env: str, expectations: dict, root: Path) -> dict:
    expected = expectations["upstream_sources"][env]
    phase2_env = json.loads((PHASE2 / "evidence" / "environments.json").read_text())[env]
    files = {}
    for relative, digest in expected.items():
        actual = sha256_file(root / relative)
        files[relative] = {"sha256": actual, "matches_preregistered": actual == digest}
        phase2_digest = phase2_env["installed_source_sha256"].get(relative)
        if phase2_digest is not None:
            files[relative]["matches_phase2_record"] = actual == phase2_digest
    scanned = {}
    for config in sorted(root.glob("policies/**/configuration_*.py")):
        value = policy_observation_deltas(config)
        if value != "unparsed":
            scanned[str(config.relative_to(root))] = value
    non_none = sorted({Path(p).parent.name for p, v in scanned.items() if v is not None})
    configs = expectations["offset_selection"]["configs"]
    derived_match = all(scanned.get(f"policies/{name}/configuration_{name}.py") == configs[name]["observation_delta_indices"] for name in non_none)
    ok = (
        all(f["matches_preregistered"] and f.get("matches_phase2_record", True) for f in files.values())
        and non_none == expectations["offset_selection"]["non_none_policy_files"][env]
        and derived_match
    )
    return {
        "lerobot_root": str(root),
        "files": files,
        "policy_observation_delta_indices": scanned,
        "non_none_policies": non_none,
        "derived_defaults_match_preregistered": derived_match,
        "ok": ok,
    }


def snapshot_revision(path: Path) -> str | None:
    """Revision segment of the Hub cache snapshot the local symlink points at."""
    parts = Path(path.readlink() if path.is_symlink() else path).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts else None


def verify_inputs(case: dict, relative_paths: list[str], root: Path) -> dict:
    ledger = {e["path"]: e for e in json.loads((PHASE2 / "evidence" / f"{case['case_id'].replace(':', '_')}.json").read_text())}
    repo = case["dataset"]
    frozen = {(e["repo"], e["revision"], e["path"]): e for e in json.loads((AUDIT / "artifact_hashes.json").read_text())}
    phase2_case = {c["case_id"]: c for c in json.loads((PHASE2 / "expectations.json").read_text())["cases"]}[case["case_id"]]
    entries = []
    for relative in relative_paths:
        local = root / relative
        entry = {"path": relative, "present": local.exists()}
        if not local.exists():
            entry["ok"] = False
            entries.append(entry)
            continue
        digest = sha256_file(local)
        frozen_entry = frozen.get((repo, case["revision"], relative))
        entry.update(
            sha256=digest,
            bytes=local.stat().st_size,
            snapshot_revision=snapshot_revision(local),
            phase2_sha256=ledger.get(relative, {}).get("sha256"),
            frozen_lfs_sha256=(frozen_entry or {}).get("lfs_sha256"),
        )
        checks = [entry["phase2_sha256"] == digest, entry["snapshot_revision"] == case["revision"]]
        if entry["frozen_lfs_sha256"]:
            checks.append(entry["frozen_lfs_sha256"] == digest)
        if relative == "meta/info.json":
            checks.append(digest == phase2_case["pinned_inputs"]["info_sha256"])
        entry["ok"] = all(checks)
        entries.append(entry)
    return {"entries": entries, "ok": all(e["ok"] for e in entries)}


def episode_timestamps(case: dict, root: Path) -> tuple[np.ndarray, dict]:
    table = pq.read_table(root / case["data_path"], columns=["episode_index", "timestamp", "index"])
    episodes = np.asarray(table["episode_index"])
    rows = np.nonzero(episodes == case["episode"])[0]
    if len(rows) == 0 or not np.all(np.diff(rows) == 1):
        raise ValueError("episode rows missing or not contiguous")
    column = table["timestamp"]
    stored = np.asarray(column.combine_chunks() if hasattr(column, "combine_chunks") else column)[rows]
    global_index = np.asarray(table["index"])[rows]
    facts = {
        "stored_dtype": str(stored.dtype),
        "parquet_type": str(table.schema.field("timestamp").type),
        "episode_rows": int(len(rows)),
        "global_index_first": int(global_index[0]),
        "global_index_last": int(global_index[-1]),
        "first_timestamp": float(stored[0]),
        "last_timestamp": float(stored[-1]),
    }
    return stored, facts


def episode_meta(case: dict, root: Path) -> dict | None:
    if "episodes_meta_path" not in case:
        return None
    table = pq.read_table(root / case["episodes_meta_path"])
    index = np.asarray(table["episode_index"])
    row = table.slice(int(np.nonzero(index == case["episode"])[0][0]), 1).to_pylist()[0]
    keep = ["episode_index", "length", "dataset_from_index", "dataset_to_index"]
    for camera in case["cameras"]:
        keep += [f"videos/{camera}/from_timestamp", f"videos/{camera}/to_timestamp", f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"]
    return {k: row[k] for k in keep}


def plan_case(case_id: str, root: Path = INPUT_ROOT) -> dict:
    expectations = load_expectations()
    case = {c["case_id"]: c for c in expectations["cases"]}[case_id]
    audit_sha = sha256_file(REPO / expectations["frozen_sources"]["audit_json"]["path"])
    phase2_sha = sha256_file(REPO / expectations["frozen_sources"]["phase2_expectations"]["path"])
    audit_case = {c["key"]: c for c in json.loads((AUDIT / "audit.json").read_text())["cases"]}[case_id]
    local = case_root(case_id, root)
    info = json.loads((local / "meta/info.json").read_text())
    videos = [case["video_path_template"].format(camera=camera) for camera in case["cameras"]]
    paths = ["meta/info.json", case["data_path"], *videos] + ([case["episodes_meta_path"]] if "episodes_meta_path" in case else [])
    record = {
        "case_id": case_id,
        "env": case["env"],
        "python": platform.python_version(),
        "frozen_audit_sha256": audit_sha,
        "frozen_audit_sha256_ok": audit_sha == expectations["frozen_sources"]["audit_json"]["sha256"],
        "phase2_expectations_sha256_ok": phase2_sha == expectations["frozen_sources"]["phase2_expectations"]["sha256"],
        "original_assertion": audit_case["message"],
        "original_severity": audit_case.get("severity"),
        "original_label": audit_case["label"],
        "revision": case["revision"],
        "revision_matches_audit": audit_case["revision"] == case["revision"],
        "sources": verify_sources(case["env"], expectations, lerobot_root()),
        "inputs": verify_inputs(case, paths, local),
    }
    identity_ok = all([record["frozen_audit_sha256_ok"], record["phase2_expectations_sha256_ok"], record["revision_matches_audit"], record["sources"]["ok"], record["inputs"]["ok"]])
    record["identity_ok"] = identity_ok
    if not identity_ok:
        record["plan"] = None
        return record
    stored, facts = episode_timestamps(case, local)
    meta = episode_meta(case, local)
    if meta is not None:
        facts["episode_meta"] = meta
        facts["meta_range_matches_rows"] = meta["dataset_to_index"] - meta["dataset_from_index"] == facts["episode_rows"] and meta["dataset_from_index"] == facts["global_index_first"]
        if not facts["meta_range_matches_rows"]:
            raise ValueError("episode metadata range does not match the contiguous episode rows")
    fps = float(info["fps"])
    facts["declared_fps"] = fps
    facts["info_timestamp_dtype"] = info["features"]["timestamp"]["dtype"]
    facts["video_info_fps"] = {k: v.get("info", {}).get("video.fps") for k, v in info["features"].items() if v.get("dtype") == "video"}
    configs = {name: value["observation_delta_indices"] for name, value in expectations["offset_selection"]["configs"].items()}
    from_ts = None if meta is None else {camera: float(meta[f"videos/{camera}/from_timestamp"]) for camera in case["cameras"]}
    plan = build_case_plan(timestamps=stored, fps=fps, cameras=case["cameras"], from_timestamps=from_ts, configs=configs, nominal_offsets=expectations["offset_selection"]["nominal_offsets"])
    if not is_finite_plan(plan):
        raise ValueError("non-finite query")
    for batch, path in zip(plan["batches"], (case["video_path_template"].format(camera=b["camera"]) for b in plan["batches"]), strict=True):
        batch["video_path"] = path
    record["data_facts"] = facts
    record["plan"] = plan
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    args = parser.parse_args()
    record = plan_case(args.case_id)
    out = HERE / "plan" / f"{args.case_id.replace(':', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    plan = record["plan"] or {"batches": []}
    print(json.dumps({"case_id": args.case_id, "identity_ok": record["identity_ok"], "batches": len(plan["batches"]), "queries": sum(len(b["queries"]) for b in plan["batches"]), "anchors": plan.get("anchors"), "plan": str(out)}))


if __name__ == "__main__":
    main()

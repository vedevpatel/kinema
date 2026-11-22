"""Shared paths and frozen-artifact loaders (standard library only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASE3 = REPO / "shifted-queries" / "2026-11-22"
PHASE2 = REPO / "loader-verification" / "2026-11-22"
AUDIT = REPO / "audits" / "2026-11-22"
INPUT_ROOT = Path("/tmp/kinema-phase2-inputs")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_key(case_id: str) -> str:
    return case_id.replace(":", "_")


def case_root(case_id: str, root: Path = INPUT_ROOT) -> Path:
    return root / case_key(case_id)


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def load_expectations() -> dict:
    return load_json(HERE / "expectations.json")


def load_environment_spec() -> dict:
    return load_json(HERE / "environment_spec.json")


def load_phase3_plan(case_id: str) -> tuple[dict, str]:
    """The phase-3 plan record, refused unless it still matches the phase-3 preregistration."""
    relative = f"plan/{case_key(case_id)}.json"
    digest = sha256_file(PHASE3 / relative)
    frozen = load_json(PHASE3 / "preregistration.json")["sha256"][relative]
    if digest != frozen:
        raise SystemExit(f"phase-3 {relative} changed after its preregistration: {digest} != {frozen}")
    record = load_json(PHASE3 / relative)
    if not record.get("identity_ok") or record.get("plan") is None:
        raise SystemExit(f"phase-3 {relative} has no executable plan")
    return record, digest


def load_phase3_probe(case_id: str) -> dict:
    return load_json(PHASE3 / "evidence" / f"{case_key(case_id)}_probe.json")


def check_preregistration() -> dict:
    """Refuse to execute if any file frozen in this directory's preregistration changed."""
    prereg = load_json(HERE / "preregistration.json")
    changed = [p for p, digest in prereg["sha256"].items() if sha256_file(HERE / p) != digest]
    if changed:
        raise SystemExit(f"preregistered files changed after freeze: {changed}")
    return prereg


def frozen_source_checks(expectations: dict) -> dict[str, bool]:
    """sha256 of every frozen phase-1/2/3 artifact named in expectations.json."""
    sources = expectations["frozen_sources"]
    entries = [sources[k] for k in ("audit_json", "phase2_expectations", "phase3_preregistration", "phase3_expectations")]
    entries += list(sources["phase3_plans"].values()) + list(sources["phase3_pyav_evidence"].values())
    return {e["path"]: sha256_file(REPO / e["path"]) == e["sha256"] for e in entries}

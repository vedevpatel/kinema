"""Shared helpers for the normalization-verification probes (no kinema imports).

Every probe runs in its own isolated environment with `python -I`, so this file
is loaded by path, never by an import from the repository's package.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVIDENCE = HERE / "evidence"
EXPECTATIONS = HERE / "expectations.json"
PREREGISTRATION = HERE / "preregistration.json"

# Large inputs never enter the repository. Paths are recorded with hashes.
WORK = Path("/tmp/kinema-normverify")
INPUTS = WORK / "inputs"
UPSTREAM = WORK / "upstream"
CKPT = WORK / "ckpt-files"
SAMPLES = Path.home() / ".cache/kinema-normverify/samples"  # on disk: /tmp is a small tmpfs
SCRATCH = WORK / "scratch"
PHASE2_INPUTS = Path("/tmp/kinema-phase2-inputs")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path | str):
    with open(path) as f:
        return json.load(f)


def _clean(obj):
    """JSON-safe: numpy scalars/arrays to Python, non-finite floats to strings."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - every probe env has numpy
        np = None
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [_clean(x) for x in obj.tolist()]
        if isinstance(obj, np.generic):
            obj = obj.item()
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_json(path: Path | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, sort_keys=True) + "\n")


def error_record(exc: BaseException) -> dict:
    return {
        "outcome": "error",
        "exception": type(exc).__name__,
        "message": str(exc)[:2000],
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-6000:],
    }


def environment_record(packages: list[str]) -> dict:
    from importlib import metadata

    versions = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": versions,
    }


def check_preregistration(files: list[str]) -> dict:
    """Refuse to run when a frozen file changed after preregistration."""
    prereg = load_json(PREREGISTRATION)["sha256"]
    observed = {}
    for name in files:
        actual = sha256_file(HERE / name)
        observed[name] = actual
        if prereg.get(name) != actual:
            raise SystemExit(f"{name} changed after preregistration ({actual} != {prereg.get(name)})")
    return observed


def finite_summary(values) -> dict:
    import numpy as np

    a = np.asarray(values, dtype=np.float64)
    fin = np.isfinite(a)
    out = {"n": int(a.size), "n_nonfinite": int((~fin).sum())}
    if fin.any():
        f = a[fin]
        out.update(min=float(f.min()), max=float(f.max()), abs_max=float(np.abs(f).max()),
                   mean=float(f.mean()), std=float(f.std()))
    return out


# --- frozen selection rules (used by prepare.py / probe_channels.py, tested offline) ---

def rank(seed: str, case: str, path: str) -> str:
    return hashlib.sha256(f"{seed}|{case}|{path}".encode()).hexdigest()


def select_files(rule: dict, case: str, listing: list[dict]) -> list[dict]:
    """The lexically first data file, then files in SHA-256 rank order, until
    `count` distinct files; the byte cap drops the last-added files first."""
    files = sorted((r for r in listing if r["path"].endswith(".parquet")), key=lambda r: r["path"])
    chosen = [files[0]]
    for r in sorted(files[1:], key=lambda r: rank(rule["seed"], case, r["path"])):
        if len(chosen) >= rule["count"]:
            break
        chosen.append(r)
    while sum(r["bytes"] for r in chosen) > rule["byte_cap"] and len(chosen) > 1:
        chosen.pop()
    return chosen


def control_dim(stds: dict[int, float], dim: int, slice_: list[int]) -> int:
    """Within the flagged dim's slice, the other dim whose stored std is the
    lower median (ties broken by the lower index)."""
    sib = [j for j in range(slice_[0], slice_[1]) if j != dim]
    ranked = sorted(sib, key=lambda j: (stds[j], j))
    return ranked[(len(ranked) - 1) // 2]


def checkpoint_model_type(repo: str) -> str | None:
    name = repo.split("/")[-1]
    for prefix, mt in (("pi0-fast-", "PI0_FAST"), ("pifast-", "PI0_FAST"), ("pi05-", "PI05"), ("pi0-", "PI0")):
        if name.startswith(prefix):
            return mt
    return None


def vlabench_candidates(repo: str, asset_id: str, inventory: list[list]) -> list[str]:
    """Fork configs with the checkpoint's model type, asset id and '-aligned' flag."""
    mt = checkpoint_model_type(repo)
    aligned = "-aligned" in repo.split("/")[-1]
    return [r[0] for r in inventory if r[1] == mt and r[2] == asset_id and ("_aligned" in r[0]) == aligned]

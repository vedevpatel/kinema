"""Create the FFmpeg soname alias directory for one environment (reversible).

    python3 torchcodec-backend/2026-11-22/setup_ffmpeg_aliases.py tc-v2

Reads environment_spec.json, verifies the sha256 of every PyAV-bundled target
library inside the venv, and creates `<alias_dir>/<soname> -> <av.libs>/<target>`
symlinks. Nothing inside the venv is written. Undo with `rm -rf <alias_dir>`.
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def listing_digest(directory: Path) -> str:
    """sha256 of `sha256sum *` text output inside the directory (sorted names)."""
    lines = "".join(f"{sha256_file(p)}  {p.name}\n" for p in sorted(directory.iterdir(), key=lambda p: p.name))
    return hashlib.sha256(lines.encode()).hexdigest()


def build_aliases(spec: dict, env: str) -> dict:
    environment = spec["environments"][env]
    av_libs = Path(environment["av_libs"])
    alias_dir = Path(environment["alias_dir"])
    targets = spec["ffmpeg_provisioning"]["target_sha256"]
    checks = {}
    for soname, entry in targets.items():
        target = av_libs / entry["target"]
        actual = sha256_file(target) if target.exists() else None
        checks[soname] = {"target": str(target), "sha256": actual, "ok": actual == entry["sha256"]}
    listing = listing_digest(av_libs)
    ok = all(c["ok"] for c in checks.values()) and listing == spec["ffmpeg_provisioning"]["av_libs_listing_sha256"]
    record = {"env": env, "alias_dir": str(alias_dir), "av_libs": str(av_libs), "targets": checks, "av_libs_listing_sha256": listing, "ok": ok}
    if not ok:
        return record
    alias_dir.mkdir(parents=True, exist_ok=True)
    for soname, check in checks.items():
        link = alias_dir / soname
        if link.is_symlink() and os.readlink(link) == check["target"]:
            continue
        if link.exists() or link.is_symlink():
            raise SystemExit(f"{link} exists and does not point at {check['target']}")
        link.symlink_to(check["target"])
    unexpected = sorted(p.name for p in alias_dir.iterdir() if p.name not in targets)
    if unexpected:
        raise SystemExit(f"unexpected entries in {alias_dir}: {unexpected}")
    record["ld_library_path"] = f"{alias_dir}:{av_libs}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("env", choices=["tc-v2", "tc-v3"])
    args = parser.parse_args()
    spec = json.loads((HERE / "environment_spec.json").read_text())
    record = build_aliases(spec, args.env)
    print(json.dumps(record, indent=1, sort_keys=True))
    if not record["ok"]:
        raise SystemExit("target verification failed; no aliases created")


if __name__ == "__main__":
    main()

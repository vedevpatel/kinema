"""Unit test for scripts/scale_test.py platform-correct RSS units."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "scale_test.py"


def _load():
    spec = importlib.util.spec_from_file_location("scale_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_peak_rss_units_linux_vs_macos():
    """ru_maxrss is KiB on Linux and BYTES on macOS - same process must
    report the same MiB on both."""
    mod = _load()
    linux = mod.peak_rss_mb(2_450 * 1024, platform="linux")  # 2450 MiB in KiB
    macos = mod.peak_rss_mb(2_450 * 1024 * 1024, platform="darwin")  # in bytes
    assert abs(linux - 2450.0) < 0.1
    assert abs(macos - 2450.0) < 0.1
    assert abs(linux - macos) < 0.1

"""Engine fingerprint: a hash of the code that decides verdicts.

Stored alongside every leaderboard result so a results cache can tell
when it was produced by different check logic and must be re-audited -
bumping a version string by hand is exactly the step that gets forgotten.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ENGINE_FILES = ("dataset.py", "stats.py", "checks/*.py")


@lru_cache(maxsize=1)
def engine_version() -> str:
    h = hashlib.sha1()
    for pattern in _ENGINE_FILES:
        for p in sorted(_ROOT.glob(pattern)):
            h.update(p.relative_to(_ROOT).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]

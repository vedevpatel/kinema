"""Hub reachability for the network suites.

The KINEMA_E2E suites hit the real Hugging Face Hub on every assertion.
When the Hub itself is down, those failures say nothing about kinema - so
the suites skip with a visible reason instead of failing CI. Reachability
is probed once per pytest process (module-level cache), cheap on the
runner and free locally where KINEMA_E2E is unset anyway.
"""

from __future__ import annotations

import os
import urllib.request

import pytest

_probe: bool | None = None


def e2e_enabled() -> bool:
    return os.environ.get("KINEMA_E2E", os.environ.get("ROLLOUTQA_E2E")) == "1"


def hub_reachable() -> bool:
    global _probe
    if _probe is None:
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/datasets/lerobot/pusht",
                headers={"User-Agent": "kinema-ci"},
            )
            with urllib.request.urlopen(req, timeout=15) as r:
                _probe = r.status == 200
        except Exception:  # noqa: BLE001 - any failure means "cannot reach the Hub"
            _probe = False
    return _probe


def e2e_mark():
    """The pytestmark for network suites: skip on opt-out or on a Hub outage."""
    if not e2e_enabled():
        return pytest.mark.skipif(True, reason="set KINEMA_E2E=1 to run")
    if not hub_reachable():
        return pytest.mark.skipif(
            True,
            reason="huggingface.co unreachable - skipping the network suites "
            "rather than failing CI on a Hub outage",
        )
    return pytest.mark.skipif(False, reason="")

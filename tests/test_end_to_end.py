"""End-to-end against a real public dataset (lerobot/pusht, v3, ~25k frames).

Network-dependent. Skipped unless ROLLOUTQA_E2E=1; CI sets it on the
nightly leg, and it is the command used to generate the README demo.
"""

from __future__ import annotations

import os

import pytest

from kinema.checks import ALL_CHECKS, CheckConfig, Severity
from kinema.dataset import load_dataset
from kinema.report import DatasetReport, render_markdown

pytestmark = pytest.mark.skipif(
    os.environ.get("ROLLOUTQA_E2E") != "1", reason="set ROLLOUTQA_E2E=1 to run"
)


def test_pusht_end_to_end():
    ctx = load_dataset("lerobot/pusht")
    assert ctx.total_episodes == 206
    assert ctx.fps == 10.0
    assert ctx.state is not None and ctx.action is not None

    rep = DatasetReport("lerobot/pusht", ctx.format_version, ctx.total_episodes, ctx.row_count, ctx.fps)
    for check in ALL_CHECKS:
        rep.results.append(check.run(ctx, CheckConfig()))
    assert 0 <= rep.score <= 100
    md = render_markdown(rep)
    assert "lerobot/pusht" in md
    # the point of the tool: every check runs without crashing on real data
    crashed = [r.check_id for r in rep.results if "crashed" in r.skip_reason]
    assert not crashed, f"checks crashed on pusht: {crashed}"

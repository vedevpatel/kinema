"""The Hub-outage skip: network suites must skip visibly, never fail CI."""

from __future__ import annotations

import netcheck


def _mark(m):
    return getattr(m, "mark", m)  # MarkDecorator -> Mark


def test_skips_without_opt_in(monkeypatch):
    monkeypatch.delenv("KINEMA_E2E", raising=False)
    monkeypatch.delenv("ROLLOUTQA_E2E", raising=False)
    mark = _mark(netcheck.e2e_mark())
    assert mark.name == "skipif" and mark.args[0] is True
    assert "KINEMA_E2E" in mark.kwargs["reason"]


def test_skips_with_reason_when_hub_is_down(monkeypatch):
    monkeypatch.setenv("KINEMA_E2E", "1")
    monkeypatch.setattr(netcheck, "hub_reachable", lambda: False)
    mark = _mark(netcheck.e2e_mark())
    assert mark.args[0] is True and "unreachable" in mark.kwargs["reason"]


def test_runs_when_enabled_and_reachable(monkeypatch):
    monkeypatch.setenv("KINEMA_E2E", "1")
    monkeypatch.setattr(netcheck, "hub_reachable", lambda: True)
    mark = _mark(netcheck.e2e_mark())
    assert mark.args[0] is False

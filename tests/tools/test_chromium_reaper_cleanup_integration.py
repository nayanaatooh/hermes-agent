"""Integration tests for the chromium orphan reaper hook inside the cleanup thread.

These validate the wiring between ``tools.terminal_tool._maybe_reap_chromium_orphans``
and ``tools.chromium_profile_reaper``: the flag gate, the safe-import path,
and the exception-swallowing contract that keeps the cleanup thread alive
when the janitor trips.
"""

from __future__ import annotations

import pytest

from tools import terminal_tool


def test_reaper_hook_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_CHROMIUM_ORPHAN_REAPER_ENABLED", raising=False)
    calls = []
    monkeypatch.setattr(
        "tools.chromium_profile_reaper.run_chromium_profile_reaper",
        lambda **kwargs: calls.append(kwargs),
    )

    terminal_tool._maybe_reap_chromium_orphans()

    assert calls == [], "reaper must be off by default until operator flips the flag"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_reaper_hook_activates_on_common_truthy_flag_values(monkeypatch, value):
    monkeypatch.setenv("HERMES_CHROMIUM_ORPHAN_REAPER_ENABLED", value)
    calls = []
    monkeypatch.setattr(
        "tools.chromium_profile_reaper.run_chromium_profile_reaper",
        lambda **kwargs: calls.append(kwargs) or None,
    )

    terminal_tool._maybe_reap_chromium_orphans()

    assert len(calls) == 1
    assert calls[0]["dry_run"] is True, "integration must only invoke dry-run mode"
    assert "process_snapshot_provider" in calls[0]
    assert "socket_roots" in calls[0]
    assert "log_event" in calls[0]


def test_reaper_hook_swallows_exceptions_from_reaper(monkeypatch):
    monkeypatch.setenv("HERMES_CHROMIUM_ORPHAN_REAPER_ENABLED", "1")

    def boom(**kwargs):
        raise RuntimeError("simulated scan failure")

    monkeypatch.setattr(
        "tools.chromium_profile_reaper.run_chromium_profile_reaper", boom,
    )

    # Must not propagate — cleanup thread would die otherwise.
    terminal_tool._maybe_reap_chromium_orphans()


def test_reaper_hook_passes_active_sessions_from_browser_tool(monkeypatch):
    monkeypatch.setenv("HERMES_CHROMIUM_ORPHAN_REAPER_ENABLED", "1")

    from tools import browser_tool  # imported lazily inside the hook; import eagerly here for the patch
    fake_sessions = {"task-1": {"session_name": "h_test123"}}
    monkeypatch.setattr(browser_tool, "_active_sessions", fake_sessions, raising=False)

    seen = {}
    monkeypatch.setattr(
        "tools.chromium_profile_reaper.run_chromium_profile_reaper",
        lambda **kwargs: seen.update(kwargs) or None,
    )

    terminal_tool._maybe_reap_chromium_orphans()

    assert seen["active_sessions"] == fake_sessions

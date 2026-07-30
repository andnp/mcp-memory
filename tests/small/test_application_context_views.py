from __future__ import annotations

from pathlib import Path

import pytest

from mcp_memory.context import ApplicationContext


pytestmark = pytest.mark.small


def test_management_view_exposes_only_management_capabilities() -> None:
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        workspace_root=Path("/tmp/workspace"),
        memory_path=Path("/tmp/memory"),
        db_manager=object(),
        repository=object(),
        internal_tool_call_tracker=object(),
    )

    view = ctx.management_view()

    assert view.workspace_id == "workspace-a"
    assert view.repository is ctx.repository

    view.provider_usage = "usage"
    assert ctx.provider_usage == "usage"

    with pytest.raises(AttributeError):
        getattr(view, "internal_tool_call_tracker")


def test_task_runtime_view_exposes_only_runtime_capabilities() -> None:
    tracker = object()
    ctx = ApplicationContext(
        task_queue="queue",
        ai_json_provider="json",
        ai_agent_provider="agent",
        provider_policy_events="events",
        journal="journal",
        internal_tool_call_tracker=tracker,
    )

    view = ctx.task_runtime_view()

    assert view.task_queue == "queue"
    assert view.ai_json_provider == "json"
    assert view.internal_tool_call_tracker is tracker

    with pytest.raises(AttributeError):
        getattr(view, "read_cache")
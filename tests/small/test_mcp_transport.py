from __future__ import annotations

import json

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.mcp import transport


def _payload(response):
    return json.loads(response[0].text)


@pytest.mark.asyncio
async def test_dispatch_memory_tool_rejects_uninitialized_context() -> None:
    payload = _payload(await transport.dispatch_memory_tool(object(), "record_thought", {}))

    assert payload == {
        "error": "runtime_not_initialized",
        "status": "error",
        "tool": "record_thought",
    }


@pytest.mark.asyncio
async def test_dispatch_memory_tool_rejects_unknown_tool_name() -> None:
    payload = _payload(await transport.dispatch_memory_tool(ApplicationContext(), "typo_tool", {"query": "auth"}))

    assert payload == {
        "arguments": {"query": "auth"},
        "error": "unknown_tool",
        "status": "error",
        "tool": "typo_tool",
    }


@pytest.mark.parametrize(
    ("error", "exception"),
    [
        ("invalid_arguments", ValueError("bad arguments")),
        ("file_not_found", FileNotFoundError("missing memory.md")),
    ],
)
def test_call_service_sync_maps_common_service_errors(error: str, exception: Exception) -> None:
    def _failing_service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        raise exception

    payload = _payload(transport._call_service_sync(_failing_service, ApplicationContext(), {"path": "demo"}))

    assert payload == {
        "detail": str(exception),
        "error": error,
        "status": "error",
    }


@pytest.mark.asyncio
async def test_internal_dispatch_records_tracker_counts_via_session_bound_task_resolution() -> None:
    tracker = InternalToolCallTracker()
    ctx = ApplicationContext(
        session_id="session-123",
        internal_tool_call_tracker=tracker,
    )

    def _ok_service(ctx: ApplicationContext, arguments: dict) -> dict:
        return {"status": "ok"}

    services = {
        "internal_get_next_curator_batch": _ok_service,
        "internal_update_memory_record": _ok_service,
    }

    first = _payload(
        await transport._dispatch_tool(
            ctx,
            "internal_get_next_curator_batch",
            {"task_id": "task-1"},
            service_resolver=lambda: services,
            on_success=transport._record_internal_tool_call,
        )
    )
    second = _payload(
        await transport._dispatch_tool(
            ctx,
            "internal_update_memory_record",
            {},
            service_resolver=lambda: services,
            on_success=transport._record_internal_tool_call,
        )
    )
    snapshot = tracker.finalize_task("task-1")

    assert first == {"status": "ok"}
    assert second == {"status": "ok"}
    assert snapshot is not None
    assert snapshot.total_calls == 2
    assert snapshot.mutating_calls == 1
    assert snapshot.by_name == {
        "internal_get_next_curator_batch": 1,
        "internal_update_memory_record": 1,
    }


@pytest.mark.asyncio
async def test_internal_dispatch_unknown_tool_does_not_record_tracker_counts() -> None:
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-123")
    ctx = ApplicationContext(
        session_id="session-123",
        internal_tool_call_tracker=tracker,
    )

    payload = _payload(await transport.dispatch_internal_memory_tool(ctx, "missing_internal_tool", {}))
    snapshot = tracker.snapshot_task("task-1")

    assert payload == {
        "arguments": {},
        "error": "unknown_tool",
        "status": "error",
        "tool": "missing_internal_tool",
    }
    assert snapshot is not None
    assert snapshot.total_calls == 0
    assert snapshot.mutating_calls == 0
    assert snapshot.by_name == {}
from __future__ import annotations

import json

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.direct_mutation_evidence import DirectMutationEvidence, reconcile_direct_mutation_evidence
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
        "error": "unknown_tool",
        "status": "error",
        "tool": "typo_tool",
    }


@pytest.mark.asyncio
async def test_dispatch_memory_tool_omits_success_status() -> None:
    def _ok_service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        return {"status": "ok", "results": []}

    payload = _payload(
        await transport._dispatch_tool(
            ApplicationContext(),
            "search_memory_records",
            {},
            service_resolver=lambda: {"search_memory_records": _ok_service},
            compact_success=True,
        )
    )

    assert payload == {"results": []}


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
async def test_internal_dispatch_assigns_scoped_call_identity() -> None:
    observed = []
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-123", execution_epoch=3)
    ctx = ApplicationContext(session_id="session-123", internal_tool_call_tracker=tracker)

    def _service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        from mcp_memory.core.curator_evidence import current_curator_execution

        observed.append(current_curator_execution())
        return {"status": "ok"}

    services = {"internal_peek_record": _service}
    for _ in range(2):
        await transport._dispatch_tool(
            ctx,
            "internal_peek_record",
            {},
            service_resolver=lambda: services,
            on_success=transport._record_internal_tool_call,
        )

    first, second = observed
    assert first is not None
    assert second is not None
    assert [first.sequence, second.sequence] == [1, 2]
    assert first.call_id != second.call_id
    assert first.task_id == second.task_id == "task-1"
    assert first.execution_epoch == second.execution_epoch == 3


def test_direct_mutation_evidence_rejects_late_execution_epoch() -> None:
    evidence = DirectMutationEvidence.start(
        task_id="task-1",
        execution_epoch=3,
        session_id="session-123",
        call_id="call-1",
        sequence=1,
        tool_name="internal_update_memory_record",
        arguments={"task_id": "task-1", "memory_id": "memory-1"},
    )

    reconciled = reconcile_direct_mutation_evidence(
        evidence,
        ledger_entry={
            "status": "success",
            "tool_name": "internal_update_memory_record",
            "task_id": "task-1",
            "execution_epoch": 2,
            "call_id": "call-1",
        },
        payload={"status": "ok"},
    )

    assert reconciled.outcome == "ledger_invalid"
    assert reconciled.error_code == "ledger_missing_or_mismatched"


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
        "error": "unknown_tool",
        "status": "error",
        "tool": "missing_internal_tool",
    }
    assert snapshot is not None
    assert snapshot.total_calls == 0
    assert snapshot.mutating_calls == 0
    assert snapshot.by_name == {}


@pytest.mark.asyncio
async def test_internal_dispatch_records_service_error_without_counting_mutation() -> None:
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-123")
    ctx = ApplicationContext(
        session_id="session-123",
        internal_tool_call_tracker=tracker,
    )

    def _error_service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        return {"status": "error", "error": "stale_memory"}

    payload = _payload(
        await transport._dispatch_tool(
            ctx,
            "internal_update_memory_record",
            {"memory_id": "memory-1", "task_id": "task-1"},
            service_resolver=lambda: {"internal_update_memory_record": _error_service},
            on_success=transport._record_internal_tool_call,
        )
    )
    snapshot = tracker.finalize_task("task-1")

    assert payload == {"status": "error", "error": "stale_memory"}
    assert snapshot is not None
    assert snapshot.total_calls == 1
    assert snapshot.mutating_calls == 0
    assert snapshot.tool_call_ledger[0]["status"] == "error"


@pytest.mark.asyncio
async def test_internal_mutation_dispatch_persists_independent_evidence() -> None:
    class EvidenceStore:
        def __init__(self) -> None:
            self.items = []

        def save(self, evidence) -> None:
            self.items.append(evidence)

    store = EvidenceStore()
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-123", execution_epoch=3)
    ctx = ApplicationContext(
        session_id="session-123", internal_tool_call_tracker=tracker,
        direct_mutation_evidence=store, repository=object(),
    )

    def _service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        return {"status": "ok", "record": {"id": "memory-1", "updated_at": "after"}}

    payload = _payload(await transport._dispatch_tool(
        ctx, "internal_update_memory_record", {"task_id": "task-1", "memory_id": "memory-1"},
        service_resolver=lambda: {"internal_update_memory_record": _service},
        on_success=transport._record_internal_tool_call,
    ))

    assert payload["record"]["id"] == "memory-1"
    assert len(store.items) == 1
    assert store.items[0].outcome == "applied_verified"
    assert store.items[0].execution_epoch == 3
    snapshot = tracker.snapshot_task("task-1")
    assert snapshot is not None and snapshot.mutating_calls == 1
    assert snapshot.tool_call_ledger[0]["execution_epoch"] == 3


@pytest.mark.asyncio
async def test_internal_mutation_dispatch_records_evidence_persistence_failure() -> None:
    class FailingEvidenceStore:
        def save(self, evidence) -> None:
            raise RuntimeError("database write failed")

    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-123")
    ctx = ApplicationContext(
        session_id="session-123", internal_tool_call_tracker=tracker,
        direct_mutation_evidence=FailingEvidenceStore(), repository=object(),
    )

    def _service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        return {"status": "ok", "record": {"id": "memory-1", "updated_at": "after"}}

    with pytest.raises(RuntimeError, match="database write failed"):
        await transport._dispatch_tool(
            ctx, "internal_update_memory_record", {"task_id": "task-1", "memory_id": "memory-1"},
            service_resolver=lambda: {"internal_update_memory_record": _service},
            on_success=transport._record_internal_tool_call,
        )

    snapshot = tracker.snapshot_task("task-1")
    assert snapshot is not None
    assert snapshot.runtime_errors == ["direct_mutation_evidence_persistence_failed"]

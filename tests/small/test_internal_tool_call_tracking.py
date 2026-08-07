from __future__ import annotations

from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker


def test_internal_tool_call_tracker_binds_session_and_counts_mutations() -> None:
    """Verify session-bound internal tool accounting for one task.

    This keeps curator metadata deterministic even when provider-reported tool stats are missing.
    """

    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1", session_id="session-1")

    tracker.record_call(
        "internal_get_next_curator_batch",
        task_id="task-1",
        session_id="session-1",
        arguments={"task_id": "task-1", "memory_id": "memory-1"},
    )
    tracker.record_call("internal_update_memory_record", session_id="session-1")
    tracker.record_call("task_complete", session_id="session-1")

    snapshot = tracker.finalize_task("task-1")

    assert snapshot is not None
    assert snapshot.task_id == "task-1"
    assert snapshot.total_calls == 3
    assert snapshot.mutating_calls == 1
    assert snapshot.by_name == {
        "internal_get_next_curator_batch": 1,
        "internal_update_memory_record": 1,
        "task_complete": 1,
    }
    assert snapshot.tool_names_used == [
        "internal_get_next_curator_batch",
        "internal_update_memory_record",
        "task_complete",
    ]
    assert snapshot.tool_call_ledger[0] == {
        "sequence": 1,
        "tool_name": "internal_get_next_curator_batch",
        "kind": "read",
        "status": "success",
        "argument_keys": ["memory_id", "task_id"],
        "memory_ids": ["memory-1"],
    }


def test_internal_tool_call_tracker_does_not_count_failed_mutation() -> None:
    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1")

    tracker.record_call(
        "internal_update_memory_record",
        task_id="task-1",
        success=False,
        arguments={"memory_id": "memory-1", "content": "secret"},
    )

    snapshot = tracker.finalize_task("task-1")

    assert snapshot is not None
    assert snapshot.total_calls == 1
    assert snapshot.mutating_calls == 0
    assert snapshot.tool_call_ledger == [
        {
            "sequence": 1,
            "tool_name": "internal_update_memory_record",
            "kind": "mutation",
            "status": "error",
            "argument_keys": ["content", "memory_id"],
            "memory_ids": ["memory-1"],
        }
    ]
    assert "secret" not in str(snapshot.tool_call_ledger)


def test_internal_tool_call_tracker_falls_back_to_single_active_task() -> None:
    """Verify calls without session or task ids resolve to the sole tracked run.

    This preserves deterministic accounting for direct runtime tests and single-run execution paths.
    """

    tracker = InternalToolCallTracker()
    tracker.reset_task("task-1")

    resolved_task_id = tracker.record_call("internal_merge_memory_into_canonical")
    snapshot = tracker.finalize_task("task-1")

    assert resolved_task_id == "task-1"
    assert snapshot is not None
    assert snapshot.total_calls == 1
    assert snapshot.mutating_calls == 1
    assert snapshot.by_name == {"internal_merge_memory_into_canonical": 1}

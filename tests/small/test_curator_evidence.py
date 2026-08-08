from __future__ import annotations

from mcp_memory.core.curator_evidence import (
    CuratorExecutionIdentity,
    begin_tool_call,
    current_curator_execution,
    finalize_curator_execution,
    finish_tool_call,
    reset_curator_execution,
)


def test_execution_identity_resets_sequence() -> None:
    reset_curator_execution("task-1", execution_epoch=4, session_id="session-1")
    first_token = begin_tool_call()
    first = current_curator_execution()
    finish_tool_call(first_token)
    second_token = begin_tool_call()
    second = current_curator_execution()
    finish_tool_call(second_token)

    assert first is not None
    assert first.call_id
    assert first.sequence == 1
    assert first.task_id == "task-1"
    assert first.execution_epoch == 4
    assert first.session_id == "session-1"
    assert second is not None
    assert second.call_id
    assert second.call_id != first.call_id
    assert second.sequence == 2
    finalize_curator_execution()


def test_tool_call_can_start_a_scoped_context() -> None:
    token = begin_tool_call(task_id="task-2", execution_epoch=8, session_id="session-2")
    context = current_curator_execution()
    finish_tool_call(token)

    assert context is not None
    assert context.identity == CuratorExecutionIdentity("task-2", 8, "session-2")
    assert context.sequence == 1
    finalize_curator_execution()
    assert current_curator_execution() is None

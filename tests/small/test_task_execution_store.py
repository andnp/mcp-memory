from __future__ import annotations

import pytest

from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from tests.small.task_execution_attempt_contract import (
    assert_preserves_first_terminal_reason_on_late_finish,
    assert_records_attempt_lifecycle,
    assert_reopens_terminal_attempt_for_new_provider_call,
)


@pytest.mark.small
def test_task_execution_attempt_repository_records_attempt_lifecycle(db_manager) -> None:
    assert_records_attempt_lifecycle(
        lambda: TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    )


@pytest.mark.small
def test_task_execution_attempt_repository_preserves_first_terminal_reason_on_late_finish(db_manager) -> None:
    assert_preserves_first_terminal_reason_on_late_finish(
        lambda: TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    )


@pytest.mark.small
def test_task_execution_attempt_repository_lists_workspace_scoped_attempts(db_manager) -> None:
    workspace_a = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    workspace_b = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-b")

    workspace_a.start_attempt(
        task_id="task-a",
        execution_epoch=1,
        task_name="memory-curator",
        request_id="req-a",
        subprocess_pid=1001,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    workspace_a.finish_attempt(
        task_id="task-a",
        execution_epoch=1,
        status="finished",
        completed_at=11.0,
    )

    workspace_b.start_attempt(
        task_id="task-b",
        execution_epoch=1,
        task_name="memory-deduper",
        request_id="req-b",
        subprocess_pid=2002,
        provider_key="claude-code",
        provider_name="Claude Code",
        model_name="sonnet",
        started_at=12.0,
    )

    workspace_a_attempts = workspace_a.list_attempts()
    assert [attempt.task_id for attempt in workspace_a_attempts] == ["task-a"]

    running_attempts = workspace_b.list_attempts(status="running")
    assert [attempt.task_id for attempt in running_attempts] == ["task-b"]

    task_a_attempts = workspace_a.list_attempts(task_id="task-a")
    assert len(task_a_attempts) == 1
    assert task_a_attempts[0].workspace_id == "workspace-a"


@pytest.mark.small
def test_task_execution_attempt_repository_requires_existing_attempt_for_updates(db_manager) -> None:
    repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")

    with pytest.raises(ValueError, match="missing-task@3"):
        repository.heartbeat_attempt(task_id="missing-task", execution_epoch=3, heartbeat_at=10.0)

    with pytest.raises(ValueError, match="missing-task@3"):
        repository.finish_attempt(task_id="missing-task", execution_epoch=3, status="error", completed_at=11.0)


@pytest.mark.small
def test_task_execution_attempt_repository_reopens_terminal_attempt_for_new_provider_call(db_manager) -> None:
    assert_reopens_terminal_attempt_for_new_provider_call(
        lambda: TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    )


@pytest.mark.small
def test_task_execution_attempt_repository_ignores_duplicate_late_start_for_same_provider_call(db_manager) -> None:
    repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")

    repository.start_attempt(
        task_id="task-3",
        execution_epoch=8,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    finished = repository.finish_attempt(
        task_id="task-3",
        execution_epoch=8,
        status="error",
        completed_at=12.0,
        request_id="req-1",
        subprocess_pid=1111,
        error_text="Provider subprocess 1111 exited unexpectedly",
        termination_reason="provider_exited",
    )

    duplicate_start = repository.start_attempt(
        task_id="task-3",
        execution_epoch=8,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=20.0,
    )

    assert duplicate_start.id == finished.id
    assert duplicate_start.request_id == "req-1"
    assert duplicate_start.subprocess_pid == 1111
    assert duplicate_start.status == "error"
    assert duplicate_start.started_at == 10.0
    assert duplicate_start.last_heartbeat_at == 12.0
    assert duplicate_start.completed_at == 12.0
    assert duplicate_start.error_text == "Provider subprocess 1111 exited unexpectedly"
    assert duplicate_start.termination_reason == "provider_exited"

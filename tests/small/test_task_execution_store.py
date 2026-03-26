from __future__ import annotations

import pytest

from mcp_memory.task_execution_store import TaskExecutionAttemptRepository


@pytest.mark.small
def test_task_execution_attempt_repository_records_attempt_lifecycle(db_manager) -> None:
    repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")

    started = repository.start_attempt(
        task_id="task-1",
        execution_epoch=1,
        task_name="memory-curator",
        request_id="req-1",
        subprocess_pid=1234,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )

    assert started.task_id == "task-1"
    assert started.execution_epoch == 1
    assert started.status == "running"
    assert started.started_at == 10.0
    assert started.last_heartbeat_at == 10.0
    assert started.completed_at is None

    duplicate_start = repository.start_attempt(
        task_id="task-1",
        execution_epoch=1,
        task_name="memory-curator",
        request_id="req-1",
        subprocess_pid=1234,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=12.0,
    )
    assert duplicate_start.id == started.id
    assert duplicate_start.started_at == 10.0
    assert duplicate_start.last_heartbeat_at == 12.0

    heartbeated = repository.heartbeat_attempt(
        task_id="task-1",
        execution_epoch=1,
        heartbeat_at=14.0,
        request_id="req-1",
        subprocess_pid=5678,
    )
    assert heartbeated.last_heartbeat_at == 14.0
    assert heartbeated.subprocess_pid == 5678

    finished = repository.finish_attempt(
        task_id="task-1",
        execution_epoch=1,
        status="finished",
        completed_at=16.0,
        request_id="req-1",
        subprocess_pid=5678,
        termination_reason="provider_finished",
    )
    assert finished.status == "finished"
    assert finished.completed_at == 16.0
    assert finished.last_heartbeat_at == 16.0
    assert finished.termination_reason == "provider_finished"

    late_heartbeat = repository.heartbeat_attempt(
        task_id="task-1",
        execution_epoch=1,
        heartbeat_at=99.0,
        subprocess_pid=9999,
    )
    assert late_heartbeat.status == "finished"
    assert late_heartbeat.last_heartbeat_at == 16.0
    assert late_heartbeat.subprocess_pid == 5678

    duplicate_finish = repository.finish_attempt(
        task_id="task-1",
        execution_epoch=1,
        status="error",
        completed_at=100.0,
        error_text="late duplicate finish",
    )
    assert duplicate_finish.status == "finished"
    assert duplicate_finish.completed_at == 16.0
    assert duplicate_finish.error_text == "late duplicate finish"


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

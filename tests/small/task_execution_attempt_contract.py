from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mcp_memory.task_execution_store import TaskExecutionAttemptRecord


class TaskExecutionAttemptRepositoryLike(Protocol):
    def start_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        task_name: str | None,
        request_id: str | None,
        subprocess_pid: int | None,
        provider_key: str | None,
        provider_name: str | None,
        model_name: str | None,
        started_at: float | None = None,
        status: str = "running",
    ) -> TaskExecutionAttemptRecord: ...

    def heartbeat_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        heartbeat_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
    ) -> TaskExecutionAttemptRecord: ...

    def finish_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        status: str,
        completed_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
        error_text: str | None = None,
        termination_reason: str | None = None,
    ) -> TaskExecutionAttemptRecord: ...


RepositoryFactory = Callable[[], TaskExecutionAttemptRepositoryLike]


def assert_records_attempt_lifecycle(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

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
        termination_reason="late_duplicate_finish",
    )
    assert duplicate_finish.status == "finished"
    assert duplicate_finish.completed_at == 16.0
    assert duplicate_finish.error_text is None
    assert duplicate_finish.termination_reason == "provider_finished"


def assert_preserves_first_terminal_reason_on_late_finish(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    repository.start_attempt(
        task_id="task-late-finish",
        execution_epoch=9,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=31337,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    finished = repository.finish_attempt(
        task_id="task-late-finish",
        execution_epoch=9,
        status="error",
        completed_at=12.0,
        request_id="req-1",
        subprocess_pid=31337,
        error_text="Provider subprocess 31337 exited unexpectedly",
        termination_reason="provider_subprocess_exited_retry",
    )

    duplicate_finish = repository.finish_attempt(
        task_id="task-late-finish",
        execution_epoch=9,
        status="cancelled",
        completed_at=20.0,
        request_id="req-1",
        subprocess_pid=31337,
        error_text="Command cancelled",
        termination_reason="provider_cancelled",
    )

    assert finished.status == "error"
    assert finished.error_text == "Provider subprocess 31337 exited unexpectedly"
    assert finished.termination_reason == "provider_subprocess_exited_retry"
    assert duplicate_finish.status == "error"
    assert duplicate_finish.completed_at == 12.0
    assert duplicate_finish.error_text == "Provider subprocess 31337 exited unexpectedly"
    assert duplicate_finish.termination_reason == "provider_subprocess_exited_retry"


def assert_reopens_terminal_attempt_for_new_provider_call(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    repository.start_attempt(
        task_id="task-2",
        execution_epoch=7,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    repository.finish_attempt(
        task_id="task-2",
        execution_epoch=7,
        status="error",
        completed_at=12.0,
        request_id="req-1",
        subprocess_pid=1111,
        error_text="Provider subprocess 1111 exited unexpectedly",
        termination_reason="provider_exited",
    )

    reopened = repository.start_attempt(
        task_id="task-2",
        execution_epoch=7,
        task_name="ingest-system1",
        request_id="req-2",
        subprocess_pid=2222,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=20.0,
    )

    assert reopened.request_id == "req-2"
    assert reopened.subprocess_pid == 2222
    assert reopened.status == "running"
    assert reopened.started_at == 20.0
    assert reopened.last_heartbeat_at == 20.0
    assert reopened.completed_at is None
    assert reopened.error_text is None
    assert reopened.termination_reason is None

    completed = repository.finish_attempt(
        task_id="task-2",
        execution_epoch=7,
        status="success",
        completed_at=24.0,
        request_id="req-2",
        subprocess_pid=2222,
    )

    assert completed.request_id == "req-2"
    assert completed.subprocess_pid == 2222
    assert completed.status == "success"
    assert completed.started_at == 20.0
    assert completed.last_heartbeat_at == 24.0
    assert completed.completed_at == 24.0
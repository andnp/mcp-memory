"""Provider-neutral task records and queue application contracts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from mcp_memory.core.task_results import TaskRunResult, TaskRunResultSource


def is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class TaskRecord:
    id: str
    task_name: str
    data: dict[str, Any]
    workspace_id: str | None
    status: str
    priority: int
    retries_count: int
    max_retries: int
    created_at: float
    updated_at: float
    available_at: float
    claimed_at: float | None
    started_at: float | None
    completed_at: float | None
    last_error: str | None
    execution_epoch: int = 0
    subprocess_pid: int | None = None
    active_request_id: str | None = None
    cancellation_requested_at: float | None = None
    cancelled_at: float | None = None
    cancellation_reason: str | None = None
    cancelled_by: str | None = None


@dataclass
class TaskRunRecord:
    id: str
    task_id: str
    task_name: str
    workspace_id: str | None
    status: str
    started_at: float
    completed_at: float
    duration_seconds: float
    result: TaskRunResult = field(default_factory=TaskRunResult)
    error_text: str | None = None


@dataclass
class TaskRunSummary:
    task_name: str
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    retry_runs: int = 0
    last_status: str | None = None
    last_started_at: float | None = None
    last_completed_at: float | None = None
    last_error: str | None = None
    last_result: TaskRunResult = field(default_factory=TaskRunResult)
    avg_duration_seconds: float = 0.0
    total_lines_compressed: int = 0


class TaskQueue(Protocol):
    def enqueue(self, task_name: str, data: dict[str, Any] | None = None, workspace_id: str | None = None, priority: int = 100, max_retries: int = 3, available_at: float | None = None, task_id: str | None = None) -> TaskRecord: ...
    def enqueue_unique(self, task_name: str, data: dict[str, Any] | None = None, workspace_id: str | None = None, priority: int = 100, max_retries: int = 3, available_at: float | None = None) -> tuple[TaskRecord, bool]: ...
    def claim_next(self, now: float | None = None, workspace_id: str | None | object = ...) -> TaskRecord | None: ...
    def complete(self, task_id: str, completed_at: float | None = None, run_result: TaskRunResultSource = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def fail(self, task_id: str, error: str, retry_delay_seconds: float = 0.0, failed_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def fail_permanently(self, task_id: str, error: str, failed_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def retry_running_task(self, task_id: str, error: str, available_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def set_running_process(self, task_id: str, *, subprocess_pid: int | None, request_id: str | None, updated_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def clear_running_process(self, task_id: str, *, updated_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def touch_running_task(self, task_id: str, *, updated_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def recover_abandoned_running_tasks(self, *, workspace_id: str | None = None, stale_after_seconds: float = 300.0, now: float | None = None) -> list[TaskRecord]: ...
    def get_task(self, task_id: str) -> TaskRecord: ...
    def request_cancel(self, task_id: str, *, cancelled_by: str, reason: str, requested_at: float | None = None) -> TaskRecord: ...
    def finalize_cancellation(self, task_id: str, *, cancelled_at: float | None = None, execution_epoch: int | None = None) -> TaskRecord: ...
    def is_cancellation_requested(self, task_id: str) -> bool: ...
    def clear_running_task_data_keys(self, task_id: str, *, field_names: list[str]) -> TaskRecord: ...
    def extend_running_task_data_int_list(self, task_id: str, *, field_name: str, values: list[int]) -> TaskRecord: ...
    def extend_running_task_data_object_list(self, task_id: str, *, field_name: str, values: list[dict[str, Any]]) -> TaskRecord: ...
    def update_pending_task(self, task_id: str, *, data: dict[str, Any] | None = None, priority: int | None = None, available_at: float | None = None) -> TaskRecord: ...
    def find_open_task(self, task_name: str, workspace_id: str | None = None) -> TaskRecord | None: ...
    def find_open_task_any_workspace(self, task_name: str) -> TaskRecord | None: ...
    def list_open_tasks_any_workspace(self, task_name: str) -> list[TaskRecord]: ...
    def find_open_task_with_data(self, task_name: str, *, workspace_id: str | None = None, data_fields: dict[str, Any]) -> TaskRecord | None: ...
    def find_open_task_with_data_any_workspace(self, task_name: str, *, data_fields: dict[str, Any]) -> TaskRecord | None: ...
    def list_tasks(self, status: str | None = None, workspace_id: str | None = None, limit: int = 100) -> list[TaskRecord]: ...
    def get_latest_successful_task_completion(self, task_name: str) -> float | None: ...
    def list_task_runs(self, task_id: str | None = None, task_name: str | None = None, workspace_id: str | None = None, limit: int = 50) -> list[TaskRunRecord]: ...
    def summarize_task_runs(self, task_names: list[str], *, workspace_id: str | None = None) -> list[TaskRunSummary]: ...
    def count_by_status(self) -> dict[str, int]: ...


__all__ = ["TaskQueue", "TaskRecord", "TaskRunRecord", "TaskRunSummary", "is_process_alive"]

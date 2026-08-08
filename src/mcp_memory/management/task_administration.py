from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    TRIGGERABLE_BACKGROUND_TASK_NAMES,
    task_priority,
)
from mcp_memory.management.agent_run_reporting import build_agent_run_history_payload
from mcp_memory.management.models import TaskDetailPayload, TaskListPayload


@dataclass(frozen=True)
class TaskAdministrationServiceDependencies:
    task_queue: Any
    terminate_process: Callable[[int], bool]
    task_payload: Callable[[Any], dict]


_CURATOR_CAMPAIGN_ALIAS_TASK_NAMES = frozenset({
    CONFLICT_DETECTOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
})


def _resolve_manual_maintenance_task_name(task_name: str) -> tuple[str, str | None]:
    if task_name in _CURATOR_CAMPAIGN_ALIAS_TASK_NAMES:
        return CURATOR_TASK_NAME, task_name
    return task_name, None


class TaskAdministrationService:
    def __init__(self, dependencies: TaskAdministrationServiceDependencies) -> None:
        self._dependencies = dependencies

    def enqueue_background_task(self, task_name: str, *, force: bool = False) -> dict:
        dependencies = self._dependencies
        canonical_task_name, redirected_from_task_name = _resolve_manual_maintenance_task_name(task_name)
        if canonical_task_name not in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            raise ValueError(f"unknown_background_task:{task_name}")

        payload = {"workspace_id": None}
        if force:
            task = dependencies.task_queue.enqueue(
                task_name=canonical_task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(canonical_task_name),
            )
            result = {"status": "enqueued", "created": True, "task": dependencies.task_payload(task)}
        else:
            task, created = dependencies.task_queue.enqueue_unique(
                task_name=canonical_task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(canonical_task_name),
            )
            result = {
                "status": "enqueued" if created else "already_pending",
                "created": created,
                "task": dependencies.task_payload(task),
            }

        if redirected_from_task_name is not None:
            result["redirected_from_task_name"] = redirected_from_task_name
        return result

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        grouped_task_names: dict[str, list[str]] = {}
        for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            canonical_task_name, redirected_from_task_name = _resolve_manual_maintenance_task_name(task_name)
            grouped_task_names.setdefault(canonical_task_name, [])
            if redirected_from_task_name is not None:
                grouped_task_names[canonical_task_name].append(redirected_from_task_name)

        results: list[dict] = []
        for canonical_task_name, redirected_from_task_names in grouped_task_names.items():
            result = self.enqueue_background_task(canonical_task_name, force=force)
            if redirected_from_task_names:
                result["redirected_from_task_names"] = redirected_from_task_names
            results.append(result)
        return results

    def cancel_task(self, task_id: str, *, cancelled_by: str = "cli", reason: str = "cancelled_by_user") -> dict:
        dependencies = self._dependencies
        task = dependencies.task_queue.request_cancel(task_id, cancelled_by=cancelled_by, reason=reason)
        signal_sent = False
        if task.status == "running" and task.subprocess_pid is not None:
            signal_sent = dependencies.terminate_process(task.subprocess_pid)
        return {
            "status": "cancelled" if task.status == "cancelled" else "cancellation_requested",
            "signal_sent": signal_sent,
            "task": dependencies.task_payload(dependencies.task_queue.get_task(task_id)),
        }

    def list_tasks(self, status: str | None = None, workspace_id: str | None = None, limit: int = 20) -> TaskListPayload:
        tasks = self._dependencies.task_queue.list_tasks(status=status, workspace_id=workspace_id, limit=limit)
        return TaskListPayload(tasks=[self._dependencies.task_payload(task) for task in tasks])

    def get_task_detail(self, task_id: str) -> TaskDetailPayload:
        dependencies = self._dependencies
        task = dependencies.task_queue.get_task(task_id)
        runs = dependencies.task_queue.list_task_runs(task_id=task_id, limit=50)
        return TaskDetailPayload(
            task=dependencies.task_payload(task),
            runs=[
                build_agent_run_history_payload(
                    task_id=run.task_id,
                    task_name=run.task_name,
                    status=run.status,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    duration_seconds=run.duration_seconds,
                    error_text=run.error_text,
                    result=run.result,
                    detail_level="full",
                )
                for run in runs
            ],
        )

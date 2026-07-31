from __future__ import annotations

import time
from typing import Any
from typing import cast

from mcp_memory.core.journal import System1Journal
from mcp_memory.core.maintenance_schedule import (
    CONFLICT_DETECTOR_TASK_NAME,
    AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES,
    AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
    TAXONOMIST_TASK_NAME,
)
from mcp_memory.core.task_handlers import task_priority
from mcp_memory.core.recurring_jitter import compute_recurring_jitter_seconds
from mcp_memory.core.task_results import TaskRunResult
from mcp_memory.core.ports.tasks import TaskQueue, TaskRecord, TaskRunSummary


AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS = 3600.0
AUTONOMOUS_MAINTENANCE_TRIGGERS = {"recurring_schedule", "recurring_follow_up"}
LEGACY_CLEANUP_TASK_NAMES = (
    PROJECT_MANAGER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    CONFLICT_DETECTOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    TAXONOMIST_TASK_NAME,
)
LEGACY_CLEANUP_MIGRATION_TRIGGER = "legacy_cleanup_migration"
LEGACY_CLEANUP_MIGRATION_REASON = "legacy_cleanup_task_migrated_to_memory_curator"


def get_global_latest_thought_timestamp(journal: System1Journal) -> float | None:
    return journal.get_latest_thought_timestamp()


def build_global_idle_state(
    journal: System1Journal,
    *,
    now: float | None = None,
) -> dict[str, float | None]:
    current_time = time.time() if now is None else now
    last_thought_at = get_global_latest_thought_timestamp(journal)
    idle_seconds = None if last_thought_at is None else max(current_time - last_thought_at, 0.0)
    return {
        "last_thought_at": last_thought_at,
        "idle_seconds": idle_seconds,
        "idle_threshold_seconds": AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS,
    }


def should_pause_autonomous_recurring_maintenance(
    task: TaskRecord,
    journal: System1Journal,
    *,
    now: float | None = None,
) -> dict[str, float | None] | None:
    if task.task_name not in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES:
        return None
    if not isinstance(task.data, dict):
        return None
    if task.data.get("trigger") not in AUTONOMOUS_MAINTENANCE_TRIGGERS:
        return None

    idle_state = build_global_idle_state(journal, now=now)
    idle_seconds = idle_state["idle_seconds"]
    if idle_state["last_thought_at"] is None:
        return idle_state
    if idle_seconds is not None and idle_seconds >= AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS:
        return idle_state
    return None


def build_idle_pause_result(
    task: TaskRecord,
    idle_state: dict[str, float | None],
) -> dict[str, float | bool | int | None]:
    interval_seconds = None
    if isinstance(task.data, dict):
        raw_interval_seconds = task.data.get("interval_seconds")
        if isinstance(raw_interval_seconds, (int, float)) and not isinstance(raw_interval_seconds, bool):
            interval_seconds = float(raw_interval_seconds)
    return {
        "paused_for_idle": True,
        "idle_seconds": idle_state.get("idle_seconds"),
        "last_thought_at": idle_state.get("last_thought_at"),
        "idle_threshold_seconds": AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS,
        "task_priority": task.priority,
        "interval_seconds": interval_seconds,
    }


def should_preserve_idle_pause(
    summary: TaskRunSummary,
    journal: System1Journal,
    *,
    now: float | None = None,
) -> bool:
    if summary.last_status != "completed":
        return False
    if not summary.last_result.get("paused_for_idle"):
        return False

    idle_state = build_global_idle_state(journal, now=now)
    if not _timestamps_match(idle_state.get("last_thought_at"), summary.last_result.get("last_thought_at")):
        return False

    if idle_state["last_thought_at"] is None:
        return True

    idle_seconds = idle_state["idle_seconds"]
    return bool(
        idle_seconds is not None and idle_seconds >= AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS
    )


def resume_paused_recurring_maintenance(
    task_queue: TaskQueue,
    *,
    now: float | None = None,
) -> list[TaskRecord]:
    resumed_tasks: list[TaskRecord] = []
    current_time = time.time() if now is None else now

    for task_name in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES:
        summary = task_queue.summarize_task_runs([task_name], workspace_id=None)[0]
        if not summary.last_result.get("paused_for_idle"):
            continue

        existing = task_queue.find_open_task(task_name, None)
        if existing is not None:
            if existing.status == "pending" and _is_autonomous_recurring_data(existing.data):
                interval_seconds = _interval_seconds_from_result(task_name, summary.last_result)
                jitter_seconds = compute_recurring_jitter_seconds(interval_seconds)
                resumed_tasks.append(
                    task_queue.update_pending_task(
                        existing.id,
                        data=_build_resumed_task_data(task_name, summary.last_result, jitter_seconds=jitter_seconds),
                        available_at=current_time + jitter_seconds,
                        priority=_task_priority_from_result(summary.last_result, existing.priority),
                    )
                )
            continue

        interval_seconds = _interval_seconds_from_result(task_name, summary.last_result)
        jitter_seconds = compute_recurring_jitter_seconds(interval_seconds)
        resumed_task, _ = task_queue.enqueue_unique(
            task_name,
            _build_resumed_task_data(task_name, summary.last_result, jitter_seconds=jitter_seconds),
            None,
            _task_priority_from_result(summary.last_result),
            3,
            current_time + jitter_seconds,
        )
        resumed_tasks.append(resumed_task)

    return resumed_tasks


def drain_legacy_cleanup_tasks(
    task_queue: TaskQueue,
    *,
    now: float | None = None,
) -> list[TaskRecord]:
    drained_at = time.time() if now is None else now
    drained_tasks: list[TaskRecord] = []
    migration_sources: list[dict[str, Any]] = []

    for task_name in LEGACY_CLEANUP_TASK_NAMES:
        open_tasks = _list_open_legacy_cleanup_tasks(task_queue, task_name)
        for source_task in open_tasks:
            cancelled_task = _cancel_legacy_cleanup_task(task_queue, source_task, drained_at=drained_at)
            drained_tasks.append(cancelled_task)
            migration_sources.append(_legacy_cleanup_migration_source(cancelled_task, drained_at=drained_at))

    if migration_sources:
        _ensure_memory_curator_migration_task(
            task_queue,
            migration_sources,
            drained_at=drained_at,
        )

    return drained_tasks


def _build_resumed_task_data(task_name: str, last_result: TaskRunResult, *, jitter_seconds: float) -> dict[str, Any]:
    data: dict[str, Any] = {
        "workspace_id": None,
        "trigger": "recurring_resume",
        "jitter_seconds": jitter_seconds,
    }
    raw_interval_seconds = last_result.get("interval_seconds")
    if isinstance(raw_interval_seconds, (int, float)) and not isinstance(raw_interval_seconds, bool):
        data["interval_seconds"] = float(raw_interval_seconds)
    elif task_name in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES:
        data["interval_seconds"] = _interval_seconds_from_result(task_name, last_result)
    return data


def _legacy_cleanup_migration_source(task: TaskRecord, *, drained_at: float) -> dict[str, Any]:
    return {
        "task_id": task.id,
        "task_name": task.task_name,
        "status": task.status,
        "drained_at": drained_at,
        "migration_reason": LEGACY_CLEANUP_MIGRATION_REASON,
    }


def _ensure_memory_curator_migration_task(
    task_queue: TaskQueue,
    migration_sources: list[dict[str, Any]],
    *,
    drained_at: float,
) -> TaskRecord:
    existing_task = task_queue.find_open_task(CURATOR_TASK_NAME, None)
    if existing_task is None:
        task, created = task_queue.enqueue_unique(
            CURATOR_TASK_NAME,
            data={
                "workspace_id": None,
                "trigger": LEGACY_CLEANUP_MIGRATION_TRIGGER,
                "migration_reason": LEGACY_CLEANUP_MIGRATION_REASON,
                "migration_sources": migration_sources,
            },
            workspace_id=None,
            priority=task_priority(CURATOR_TASK_NAME),
            available_at=drained_at,
        )
        if created:
            return task
        return _merge_memory_curator_migration_sources(task_queue, task, migration_sources)

    return _merge_memory_curator_migration_sources(task_queue, existing_task, migration_sources)


def _merge_memory_curator_migration_sources(
    task_queue: TaskQueue,
    task: TaskRecord,
    migration_sources: list[dict[str, Any]],
) -> TaskRecord:
    if task.status == "pending":
        next_data = dict(task.data)
        existing_sources = list(next_data.get("migration_sources", []))
        existing_sources.extend(migration_sources)
        next_data["migration_sources"] = existing_sources
        next_data.setdefault("workspace_id", None)
        next_data.setdefault("migration_reason", LEGACY_CLEANUP_MIGRATION_REASON)
        next_data.setdefault("trigger", LEGACY_CLEANUP_MIGRATION_TRIGGER)
        return task_queue.update_pending_task(task.id, data=next_data)
    if task.status != "running":
        return task
    return task_queue.extend_running_task_data_object_list(
        task.id,
        field_name="migration_sources",
        values=migration_sources,
    )


def _list_open_legacy_cleanup_tasks(
    task_queue: TaskQueue,
    task_name: str,
) -> list[TaskRecord]:
    list_helper = getattr(task_queue, "list_open_tasks_any_workspace", None)
    if callable(list_helper):
        return cast(list[TaskRecord], list_helper(task_name))
    existing = task_queue.find_open_task_any_workspace(task_name)
    return [] if existing is None else [existing]


def _cancel_legacy_cleanup_task(
    task_queue: TaskQueue,
    source_task: TaskRecord,
    *,
    drained_at: float,
) -> TaskRecord:
    cancelled_task = task_queue.request_cancel(
        source_task.id,
        cancelled_by="system",
        reason=LEGACY_CLEANUP_MIGRATION_REASON,
        requested_at=drained_at,
    )
    if cancelled_task.status == "running":
        cancelled_task = task_queue.finalize_cancellation(cancelled_task.id, cancelled_at=drained_at)
    return cancelled_task


def _is_autonomous_recurring_data(data: dict[str, Any] | None) -> bool:
    if not isinstance(data, dict):
        return False
    return data.get("trigger") in AUTONOMOUS_MAINTENANCE_TRIGGERS


def _timestamps_match(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return abs(float(left) - float(right)) <= 1e-6


def _task_priority_from_result(last_result: TaskRunResult, default: int = 100) -> int:
    raw_priority = last_result.get("task_priority")
    if isinstance(raw_priority, int) and not isinstance(raw_priority, bool):
        return raw_priority
    if isinstance(raw_priority, float):
        return int(raw_priority)
    return default


def _interval_seconds_from_result(task_name: str, last_result: TaskRunResult) -> float:
    raw_interval_seconds = last_result.get("interval_seconds")
    if isinstance(raw_interval_seconds, (int, float)) and not isinstance(raw_interval_seconds, bool):
        return float(raw_interval_seconds)
    interval_seconds = AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS.get(task_name)
    if interval_seconds is not None:
        return interval_seconds
    return RECURRING_TASK_INTERVAL_SECONDS[task_name]

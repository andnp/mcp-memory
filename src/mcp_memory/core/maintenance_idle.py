from __future__ import annotations

import time
from typing import Any

from mcp_memory.core.journal import System1Journal
from mcp_memory.core.maintenance_schedule import MAINTENANCE_TASK_NAMES, RECURRING_TASK_INTERVAL_SECONDS
from mcp_memory.core.recurring_jitter import compute_recurring_jitter_seconds
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord, TaskRunSummary


AUTONOMOUS_MAINTENANCE_IDLE_THRESHOLD_SECONDS = 3600.0
AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES = MAINTENANCE_TASK_NAMES
AUTONOMOUS_MAINTENANCE_TRIGGERS = {"recurring_schedule", "recurring_follow_up"}


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
    if not isinstance(summary.last_result, dict) or not summary.last_result.get("paused_for_idle"):
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
    task_queue: SQLiteTaskQueue,
    *,
    now: float | None = None,
) -> list[TaskRecord]:
    resumed_tasks: list[TaskRecord] = []
    current_time = time.time() if now is None else now

    for task_name in AUTONOMOUS_RECURRING_MAINTENANCE_TASK_NAMES:
        summary = task_queue.summarize_task_runs([task_name], workspace_id=None)[0]
        if not isinstance(summary.last_result, dict) or not summary.last_result.get("paused_for_idle"):
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


def _build_resumed_task_data(task_name: str, last_result: dict[str, Any], *, jitter_seconds: float) -> dict[str, Any]:
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


def _task_priority_from_result(last_result: dict[str, Any], default: int = 100) -> int:
    raw_priority = last_result.get("task_priority")
    if isinstance(raw_priority, int) and not isinstance(raw_priority, bool):
        return raw_priority
    if isinstance(raw_priority, float):
        return int(raw_priority)
    return default


def _interval_seconds_from_result(task_name: str, last_result: dict[str, Any]) -> float:
    raw_interval_seconds = last_result.get("interval_seconds")
    if isinstance(raw_interval_seconds, (int, float)) and not isinstance(raw_interval_seconds, bool):
        return float(raw_interval_seconds)
    return RECURRING_TASK_INTERVAL_SECONDS[task_name]

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from mcp_memory.core.journal import System1Journal, _ALL_WORKSPACES
from mcp_memory.core.ports.tasks import TaskQueue, TaskRecord


_ALL_WORKSPACES_WIRE = "*"


@dataclass(slots=True)
class System1IngestScheduleResult:
    task: TaskRecord
    created: bool
    pending_count: int
    trigger: str


def schedule_system1_ingest(
    task_queue: TaskQueue,
    journal: System1Journal,
    workspace_id: str | None,
    *,
    now: float | None = None,
    suppression_config=None,
):
    from mcp_memory.core.task_handlers.constants import (
        SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS,
        SYSTEM1_INGEST_DEBOUNCE_SECONDS,
        SYSTEM1_INGEST_TASK_NAME,
        SYSTEM1_INGEST_THRESHOLD,
    )

    scheduled_at = time.time() if now is None else now
    scheduling_context = _build_ingest_scheduling_context(
        journal,
        scheduled_at=scheduled_at,
        workspace_id=workspace_id,
    )
    if scheduling_context is None:
        return None

    pending_count = scheduling_context.pending_count
    oldest_pending_timestamp = scheduling_context.oldest_pending_timestamp

    trigger = "system1_threshold"
    available_at = scheduled_at
    if pending_count < SYSTEM1_INGEST_THRESHOLD:
        trigger = "system1_debounce"
        available_at = scheduled_at + SYSTEM1_INGEST_DEBOUNCE_SECONDS

    rate_limited_until = _resolve_auto_ingest_rate_limit_until(
        task_queue,
        task_name=SYSTEM1_INGEST_TASK_NAME,
        cooldown_seconds=SYSTEM1_AUTO_INGEST_RATE_LIMIT_SECONDS,
    )
    if rate_limited_until is not None and available_at < rate_limited_until:
        available_at = rate_limited_until
        trigger = f"{trigger}_rate_limited"

    suppressed_until = _resolve_ingest_suppression_until(scheduled_at, suppression_config)
    if suppressed_until is not None and available_at < suppressed_until:
        available_at = max(available_at, suppressed_until)
        trigger = f"{trigger}_suppressed"

    return _enqueue_system1_ingest_task(
        task_queue,
        workspace_id=workspace_id,
        journal_workspace_id=scheduling_context.journal_workspace_id,
        pending_count=pending_count,
        oldest_pending_timestamp=oldest_pending_timestamp,
        trigger=trigger,
        available_at=available_at,
        rate_limited_until=rate_limited_until,
        suppressed_until=suppressed_until,
    )


def schedule_system1_ingest_continuation(
    task_queue: TaskQueue,
    journal: System1Journal,
    workspace_id: str | None,
    *,
    now: float | None = None,
    suppression_config=None,
):
    scheduled_at = time.time() if now is None else now
    scheduling_context = _build_ingest_scheduling_context(
        journal,
        scheduled_at=scheduled_at,
        workspace_id=workspace_id,
    )
    if scheduling_context is None:
        return None

    trigger = "system1_backlog_continuation"
    available_at = scheduled_at
    suppressed_until = _resolve_ingest_suppression_until(scheduled_at, suppression_config)
    if suppressed_until is not None and available_at < suppressed_until:
        available_at = suppressed_until
        trigger = f"{trigger}_suppressed"

    return _enqueue_system1_ingest_task(
        task_queue,
        workspace_id=workspace_id,
        journal_workspace_id=scheduling_context.journal_workspace_id,
        pending_count=scheduling_context.pending_count,
        oldest_pending_timestamp=scheduling_context.oldest_pending_timestamp,
        trigger=trigger,
        available_at=available_at,
        suppressed_until=suppressed_until,
    )


@dataclass(slots=True)
class _System1IngestSchedulingContext:
    journal_workspace_id: object
    pending_count: int
    oldest_pending_timestamp: float


def _build_ingest_scheduling_context(
    journal: System1Journal,
    *,
    scheduled_at: float,
    workspace_id: str | None,
) -> _System1IngestSchedulingContext | None:
    del scheduled_at
    journal_workspace_id = resolve_pending_workspace_id(journal, None)
    pending_count = journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    if pending_count <= 0:
        return None

    oldest_pending_timestamp = journal.get_oldest_pending_timestamp(workspace_id=journal_workspace_id)
    if oldest_pending_timestamp is None:
        return None

    del workspace_id
    return _System1IngestSchedulingContext(
        journal_workspace_id=journal_workspace_id,
        pending_count=pending_count,
        oldest_pending_timestamp=oldest_pending_timestamp,
    )


def _enqueue_system1_ingest_task(
    task_queue: TaskQueue,
    *,
    workspace_id: str | None,
    journal_workspace_id: object,
    pending_count: int,
    oldest_pending_timestamp: float,
    trigger: str,
    available_at: float,
    rate_limited_until: float | None = None,
    suppressed_until: float | None = None,
):
    from mcp_memory.core.task_handlers.constants import (
        SYSTEM1_INGEST_PRIORITY,
        SYSTEM1_INGEST_TASK_NAME,
    )

    task_data = {
        "workspace_id": workspace_id,
        "journal_workspace_id": _serialize_pending_workspace_id(journal_workspace_id),
        "trigger": trigger,
        "pending_count": pending_count,
        "oldest_pending_timestamp": oldest_pending_timestamp,
    }
    if rate_limited_until is not None:
        task_data["rate_limited_until"] = rate_limited_until
    if suppressed_until is not None:
        task_data["suppressed_until"] = suppressed_until
    existing = task_queue.find_open_task_any_workspace(SYSTEM1_INGEST_TASK_NAME)
    if existing is not None:
        task = existing
        created = False
        if (
            task.status == "pending"
            and _is_auto_scheduled_ingest_task(task)
            and (
                task.workspace_id is not None
                or abs(available_at - task.available_at) > 1e-6
                or task.priority != SYSTEM1_INGEST_PRIORITY
                or task.data != task_data
            )
        ):
            task = task_queue.update_pending_task(
                task.id,
                available_at=available_at,
                data=task_data,
                priority=SYSTEM1_INGEST_PRIORITY,
            )
    else:
        task, created = task_queue.enqueue_unique(
            task_name=SYSTEM1_INGEST_TASK_NAME,
            workspace_id=None,
            data=task_data,
            priority=SYSTEM1_INGEST_PRIORITY,
            available_at=available_at,
        )
        if (
            not created
            and task.status == "pending"
            and _is_auto_scheduled_ingest_task(task)
            and (
                task.workspace_id is not None
                or abs(available_at - task.available_at) > 1e-6
                or task.priority != SYSTEM1_INGEST_PRIORITY
                or task.data != task_data
            )
        ):
            task = task_queue.update_pending_task(
                task.id,
                available_at=available_at,
                data=task_data,
                priority=SYSTEM1_INGEST_PRIORITY,
            )

    return System1IngestScheduleResult(
        task=task,
        created=created,
        pending_count=pending_count,
        trigger=trigger,
    )


def resolve_pending_workspace_id(journal: System1Journal, workspace_id: str | None):
    if workspace_id == _ALL_WORKSPACES_WIRE:
        return _ALL_WORKSPACES
    if workspace_id is None:
        return _ALL_WORKSPACES
    pending_count = journal.count_by_status(workspace_id=workspace_id).get("pending", 0)
    if pending_count > 0 or workspace_id is None:
        return workspace_id
    if journal.count_by_status(workspace_id=_ALL_WORKSPACES).get("pending", 0) > 0:
        return _ALL_WORKSPACES
    return workspace_id


def _serialize_pending_workspace_id(workspace_id: object) -> str | None:
    if workspace_id is _ALL_WORKSPACES:
        return _ALL_WORKSPACES_WIRE
    return workspace_id if isinstance(workspace_id, str) else None


def _resolve_ingest_suppression_until(now: float, suppression_config) -> float | None:
    if suppression_config is None or not getattr(suppression_config, "enabled", False):
        return None
    windows = getattr(suppression_config, "windows", [])
    if not windows:
        return None

    current = datetime.fromtimestamp(now).astimezone()
    current_hour = current.hour + (current.minute / 60.0) + (current.second / 3600.0)
    candidate_end_times: list[float] = []

    for window in windows:
        days = set(getattr(window, "days_of_week", []))
        start_hour = int(getattr(window, "start_hour", 0))
        end_hour = int(getattr(window, "end_hour", 0))
        if start_hour == end_hour:
            if not days or current.weekday() in days:
                end_dt = (current + timedelta(days=1)).replace(hour=end_hour, minute=0, second=0, microsecond=0)
                candidate_end_times.append(end_dt.timestamp())
            continue
        if start_hour < end_hour:
            if (not days or current.weekday() in days) and start_hour <= current_hour < end_hour:
                end_dt = current.replace(hour=end_hour, minute=0, second=0, microsecond=0)
                candidate_end_times.append(end_dt.timestamp())
            continue
        if current_hour >= start_hour and (not days or current.weekday() in days):
            end_dt = (current + timedelta(days=1)).replace(hour=end_hour, minute=0, second=0, microsecond=0)
            candidate_end_times.append(end_dt.timestamp())
            continue
        previous_day = (current.weekday() - 1) % 7
        if current_hour < end_hour and (not days or previous_day in days):
            end_dt = current.replace(hour=end_hour, minute=0, second=0, microsecond=0)
            candidate_end_times.append(end_dt.timestamp())

    return min(candidate_end_times) if candidate_end_times else None


def _resolve_auto_ingest_rate_limit_until(
    task_queue: TaskQueue,
    *,
    task_name: str,
    cooldown_seconds: float,
) -> float | None:
    latest_completed_at = task_queue.get_latest_successful_task_completion(task_name)
    if latest_completed_at is None:
        return None
    return latest_completed_at + cooldown_seconds


def _is_auto_scheduled_ingest_task(task: TaskRecord) -> bool:
    trigger = task.data.get("trigger")
    return isinstance(trigger, str) and trigger.startswith("system1_")

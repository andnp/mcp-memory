from __future__ import annotations

import time
from dataclasses import dataclass

from mcp_memory.core.journal import System1Journal, _ALL_WORKSPACES
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord


_ALL_WORKSPACES_WIRE = "*"


@dataclass(slots=True)
class System1IngestScheduleResult:
    task: TaskRecord
    created: bool
    pending_count: int
    trigger: str


def schedule_system1_ingest(
    task_queue: SQLiteTaskQueue,
    journal: System1Journal,
    workspace_id: str | None,
    *,
    now: float | None = None,
):
    from mcp_memory.core.task_handlers.constants import (
        SYSTEM1_INGEST_DEBOUNCE_SECONDS,
        SYSTEM1_INGEST_TASK_NAME,
        SYSTEM1_INGEST_THRESHOLD,
    )

    scheduled_at = time.time() if now is None else now
    journal_workspace_id = resolve_pending_workspace_id(journal, workspace_id)
    pending_count = journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    if pending_count <= 0:
        return None

    oldest_pending_timestamp = journal.get_oldest_pending_timestamp(workspace_id=journal_workspace_id)
    if oldest_pending_timestamp is None:
        return None

    trigger = "system1_threshold"
    available_at = scheduled_at
    if pending_count < SYSTEM1_INGEST_THRESHOLD:
        trigger = "system1_debounce"
        available_at = max(oldest_pending_timestamp + SYSTEM1_INGEST_DEBOUNCE_SECONDS, scheduled_at)

    task_data = {
        "workspace_id": workspace_id,
        "journal_workspace_id": _serialize_pending_workspace_id(journal_workspace_id),
        "trigger": trigger,
        "pending_count": pending_count,
        "oldest_pending_timestamp": oldest_pending_timestamp,
    }
    task, created = task_queue.enqueue_unique(
        task_name=SYSTEM1_INGEST_TASK_NAME,
        workspace_id=workspace_id,
        data=task_data,
        available_at=available_at,
    )
    if not created and task.status == "pending" and available_at < task.available_at:
        task = task_queue.update_pending_task(
            task.id,
            available_at=available_at,
            data=task_data,
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
    if journal.count_by_status(workspace_id=None).get("pending", 0) > 0:
        return None
    return workspace_id


def _serialize_pending_workspace_id(workspace_id: object) -> str | None:
    if workspace_id is _ALL_WORKSPACES:
        return _ALL_WORKSPACES_WIRE
    return workspace_id if isinstance(workspace_id, str) else None

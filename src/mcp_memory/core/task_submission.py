from __future__ import annotations

from mcp_memory.context import TaskQueueContext
from mcp_memory.core.task_handlers.constants import SUMMARIZE_MEMORY_PRIORITY, SUMMARIZE_MEMORY_TASK_NAME
from mcp_memory.core.tasks import TaskRecord


def enqueue_summary_refresh_task(
    ctx: TaskQueueContext,
    *,
    memory_id: str,
    workspace_ids: list[str] | None = None,
) -> TaskRecord | None:
    if ctx.task_queue is None:
        return None

    del workspace_ids

    existing = ctx.task_queue.find_open_task_with_data_any_workspace(
        SUMMARIZE_MEMORY_TASK_NAME,
        data_fields={"memory_id": memory_id},
    )
    if existing is not None:
        return existing

    return ctx.task_queue.enqueue(
        task_name=SUMMARIZE_MEMORY_TASK_NAME,
        workspace_id=None,
        data={"memory_id": memory_id},
        priority=SUMMARIZE_MEMORY_PRIORITY,
    )

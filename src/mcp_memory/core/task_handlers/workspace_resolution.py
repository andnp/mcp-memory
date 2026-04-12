from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import TaskRecord


def resolve_task_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    del ctx
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    if isinstance(task.workspace_id, str) and task.workspace_id.strip():
        return task.workspace_id.strip()
    return None


def resolve_task_or_context_workspace_id(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    fallback: str | None = None,
) -> str | None:
    workspace_id = resolve_task_workspace_id(ctx, task)
    if workspace_id is not None:
        return workspace_id
    if isinstance(ctx.workspace_id, str) and ctx.workspace_id.strip():
        return ctx.workspace_id.strip()
    return fallback
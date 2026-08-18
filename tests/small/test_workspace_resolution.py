from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.workspace_resolution import (
    resolve_task_or_context_workspace_id,
    resolve_task_workspace_id,
)
from mcp_memory.core.tasks import TaskRecord


def test_resolve_task_workspace_id_prefers_task_payload_then_task_field() -> None:
    ctx = ApplicationContext(workspace_id="ctx-workspace")

    payload_task = cast(Any, SimpleNamespace(data={"workspace_id": "payload-workspace"}, workspace_id="task-workspace"))
    field_task = cast(Any, SimpleNamespace(data={}, workspace_id="task-workspace"))

    assert resolve_task_workspace_id(ctx, cast(TaskRecord, payload_task)) == "payload-workspace"
    assert resolve_task_workspace_id(ctx, cast(TaskRecord, field_task)) == "task-workspace"


def test_resolve_task_or_context_workspace_id_falls_back_to_context_then_default() -> None:
    ctx = ApplicationContext(workspace_id="ctx-workspace")
    task = cast(TaskRecord, cast(Any, SimpleNamespace(data={}, workspace_id=None)))

    assert resolve_task_or_context_workspace_id(ctx, task) == "ctx-workspace"
    assert resolve_task_or_context_workspace_id(ApplicationContext(), task, fallback="workspace-unknown") == "workspace-unknown"
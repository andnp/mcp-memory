from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.validation import optional_string


def internal_task_complete_service(ctx: ApplicationContext, arguments: dict) -> dict:
    del ctx
    summary = optional_string(arguments, "summary")
    task_id = optional_string(arguments, "task_id")
    task_name = optional_string(arguments, "task_name")
    return {
        "status": "ok",
        "task_id": task_id,
        "task_name": task_name,
        "summary": summary,
        "completion_recorded": True,
    }
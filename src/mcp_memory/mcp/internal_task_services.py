from __future__ import annotations

from mcp_memory.context import ApplicationContext
def internal_task_complete_service(ctx: ApplicationContext, arguments: dict) -> dict:
    del ctx
    return {
        "status": "ok",
        "completion_recorded": True,
    }

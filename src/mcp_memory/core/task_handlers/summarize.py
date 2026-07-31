from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.summaries import build_deterministic_summary
from mcp_memory.core.ports.tasks import TaskRecord


async def handle_summarize_memory_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"memory_id": None, "summary": None}

    memory_id = str(task.data.get("memory_id", "")).strip()
    if not memory_id:
        return {"memory_id": None, "summary": None}

    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"memory_id": memory_id, "summary": None}

    del provider
    summary = build_deterministic_summary(
        title=record.title,
        content=record.content,
        memory_type=record.type,
    )

    updated = ctx.repository.update_memory(memory_id, summary=summary)
    return {
        "memory_id": memory_id,
        "summary": updated.summary if updated is not None else summary,
    }

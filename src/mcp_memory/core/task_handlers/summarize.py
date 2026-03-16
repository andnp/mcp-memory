from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import TaskRecord


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

    summary = None
    if provider is not None:
        try:
            summary_response = await provider.ask(
                "Write a concise 1-2 sentence summary as JSON: "
                '{"summary": "..."}\n\n'
                f"Title: {record.title}\nContent: {record.content}"
            )
            maybe_summary = summary_response.get("summary")
            if isinstance(maybe_summary, str) and maybe_summary.strip():
                summary = maybe_summary.strip()
        except Exception:
            summary = None

    if summary is None:
        summary = _build_summary(record.content)

    updated = ctx.repository.update_memory(memory_id, summary=summary)
    return {
        "memory_id": memory_id,
        "summary": updated.summary if updated is not None else summary,
    }


def _build_summary(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ""
    sentences = [segment.strip() for segment in stripped.split(".") if segment.strip()]
    if len(sentences) >= 2:
        return ". ".join(sentences[:2]) + "."
    if len(stripped) <= 220:
        return stripped
    return stripped[:217].rstrip() + "..."
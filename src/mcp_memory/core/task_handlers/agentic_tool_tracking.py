from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext


def reset_agentic_tool_tracking(ctx: ApplicationContext, task_id: str) -> None:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    reset_task = getattr(tracker, "reset_task", None)
    if callable(reset_task):
        reset_task(task_id, session_id=ctx.session_id)


def snapshot_agentic_tool_tracking(ctx: ApplicationContext, task_id: str) -> Any:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    snapshot_task = getattr(tracker, "snapshot_task", None)
    if callable(snapshot_task):
        return snapshot_task(task_id)
    return None


def finalize_agentic_tool_tracking(ctx: ApplicationContext, task_id: str) -> Any:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    finalize_task = getattr(tracker, "finalize_task", None)
    if callable(finalize_task):
        return finalize_task(task_id)
    return None


def prefer_deterministic_agentic_counts(
    result: dict[str, Any],
    *,
    deterministic_counts: Any,
) -> dict[str, Any]:
    if deterministic_counts is None:
        return result
    normalized = dict(result)
    normalized["tool_calls_executed"] = int(getattr(deterministic_counts, "total_calls", 0))
    normalized["mutations"] = int(getattr(deterministic_counts, "mutating_calls", 0))
    normalized["tool_names_used"] = list(getattr(deterministic_counts, "tool_names_used", []))
    return normalized
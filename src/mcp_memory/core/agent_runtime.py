from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    handle_defragmenter_task,
    handle_conflict_detector_task,
    handle_deduplicator_task,
    handle_fact_checker_task,
    handle_graph_linker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
    handle_taxonomist_task,
)
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import TaskRecord


def build_runtime_task_worker(
    ctx: ApplicationContext,
    provider: Any = None,
) -> RuntimeTaskWorker:
    active_provider = provider if provider is not None else getattr(ctx, "ai_provider", None)
    return RuntimeTaskWorker(
        ctx,
        handlers=build_default_task_handlers(active_provider),
        poll_interval_seconds=0.05,
    )


def build_default_task_handlers(
    provider: Any = None,
) -> dict[str, Callable[[ApplicationContext, TaskRecord], Any]]:
    return {
        SYSTEM1_INGEST_TASK_NAME: lambda ctx, task: handle_ingest_system1_task(ctx, task, provider),
        SUMMARIZE_MEMORY_TASK_NAME: lambda ctx, task: handle_summarize_memory_task(ctx, task, provider),
        GRAPH_LINKER_TASK_NAME: lambda ctx, task: handle_graph_linker_task(ctx, task, provider),
        CONFLICT_DETECTOR_TASK_NAME: lambda ctx, task: handle_conflict_detector_task(ctx, task, provider),
        DEFRAGMENTER_TASK_NAME: lambda ctx, task: handle_defragmenter_task(ctx, task, provider),
        DEDUPLICATOR_TASK_NAME: lambda ctx, task: handle_deduplicator_task(ctx, task, provider),
        TAXONOMIST_TASK_NAME: lambda ctx, task: handle_taxonomist_task(ctx, task, provider),
        PROJECT_MANAGER_TASK_NAME: handle_project_manager_task,
        FACT_CHECKER_TASK_NAME: handle_fact_checker_task,
        SWEEPER_TASK_NAME: handle_sweeper_task,
    }


def bootstrap_background_tasks(ctx: ApplicationContext) -> None:
    task_queue = getattr(ctx, "task_queue", None)
    if task_queue is None:
        return

    workspace_id = getattr(ctx, "workspace_id", None)
    journal = getattr(ctx, "journal", None)

    if journal is not None:
        schedule_system1_ingest(
            task_queue,
            journal,
            workspace_id,
        )

    for task_name, interval_seconds in RECURRING_TASK_INTERVAL_SECONDS.items():
        _ensure_recurring_task_scheduled(
            task_queue,
            task_name=task_name,
            workspace_id=workspace_id,
            interval_seconds=interval_seconds,
        )


def _ensure_recurring_task_scheduled(
    task_queue,
    *,
    task_name: str,
    workspace_id: str | None,
    interval_seconds: float,
) -> None:
    existing = task_queue.find_open_task(task_name, workspace_id)
    if existing is not None:
        return

    summary = task_queue.summarize_task_runs([task_name], workspace_id=workspace_id)[0]
    available_at = None
    if summary.last_completed_at is not None:
        available_at = max(summary.last_completed_at + interval_seconds, time.time())

    task_queue.enqueue_unique(
        task_name=task_name,
        workspace_id=workspace_id,
        available_at=available_at,
        data={
            "workspace_id": workspace_id,
            "trigger": "recurring_schedule",
            "interval_seconds": interval_seconds,
        },
    )
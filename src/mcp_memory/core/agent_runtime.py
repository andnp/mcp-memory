from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
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
    handle_memory_curator_task,
    handle_deduplicator_task,
    handle_fact_checker_task,
    handle_graph_linker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
    handle_taxonomist_task,
    task_priority,
)
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import TaskRecord


logger = logging.getLogger(__name__)


AGENTIC_TASK_NAMES = {
    SYSTEM1_INGEST_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    CURATOR_TASK_NAME,
}
DEFAULT_RUNTIME_TASK_RETRY_DELAY_SECONDS = 300.0


def build_runtime_task_worker(
    ctx: ApplicationContext,
    provider: Any = None,
) -> RuntimeTaskWorker:
    active_json_provider = (
        provider
        if provider is not None
        else getattr(ctx, "ai_json_provider", None) or getattr(ctx, "ai_provider", None)
    )
    active_agentic_provider = getattr(ctx, "ai_agent_provider", None)
    handlers = build_default_task_handlers(active_json_provider, active_agentic_provider)
    expected_handlers = {
        SYSTEM1_INGEST_TASK_NAME,
        SUMMARIZE_MEMORY_TASK_NAME,
        GRAPH_LINKER_TASK_NAME,
        CONFLICT_DETECTOR_TASK_NAME,
        DEFRAGMENTER_TASK_NAME,
        DEDUPLICATOR_TASK_NAME,
        TAXONOMIST_TASK_NAME,
        CURATOR_TASK_NAME,
        PROJECT_MANAGER_TASK_NAME,
        FACT_CHECKER_TASK_NAME,
        SWEEPER_TASK_NAME,
    }
    missing_handlers = sorted(expected_handlers - set(handlers))
    if missing_handlers:
        logger.error(
            "Runtime task worker initialized with incomplete handler registry",
            extra={"missing_handlers": missing_handlers, "handler_names": sorted(handlers)},
        )
    else:
        logger.info(
            "Runtime task worker handler registry ready",
            extra={"handler_names": sorted(handlers), "handler_count": len(handlers)},
        )
    return RuntimeTaskWorker(
        ctx,
        handlers=handlers,
        handler_factory=lambda: build_default_task_handlers(active_json_provider, active_agentic_provider),
        poll_interval_seconds=0.05,
        retry_delay_seconds=DEFAULT_RUNTIME_TASK_RETRY_DELAY_SECONDS,
    )


def build_default_task_handlers(
    provider: Any = None,
    agentic_provider: Any = None,
) -> dict[str, Callable[[ApplicationContext, TaskRecord], Any]]:
    def scoped(ctx: ApplicationContext, task_name: str, task: TaskRecord):
        return _provider_for_task(ctx, provider, agentic_provider, task_name, task)

    return {
        SYSTEM1_INGEST_TASK_NAME: lambda ctx, task: handle_ingest_system1_task(ctx, task, scoped(ctx, SYSTEM1_INGEST_TASK_NAME, task)),
        SUMMARIZE_MEMORY_TASK_NAME: lambda ctx, task: handle_summarize_memory_task(ctx, task, scoped(ctx, SUMMARIZE_MEMORY_TASK_NAME, task)),
        GRAPH_LINKER_TASK_NAME: lambda ctx, task: handle_graph_linker_task(ctx, task, scoped(ctx, GRAPH_LINKER_TASK_NAME, task)),
        CONFLICT_DETECTOR_TASK_NAME: lambda ctx, task: handle_conflict_detector_task(ctx, task, scoped(ctx, CONFLICT_DETECTOR_TASK_NAME, task)),
        DEFRAGMENTER_TASK_NAME: lambda ctx, task: handle_defragmenter_task(ctx, task, scoped(ctx, DEFRAGMENTER_TASK_NAME, task)),
        DEDUPLICATOR_TASK_NAME: lambda ctx, task: handle_deduplicator_task(ctx, task, scoped(ctx, DEDUPLICATOR_TASK_NAME, task)),
        TAXONOMIST_TASK_NAME: lambda ctx, task: handle_taxonomist_task(ctx, task, scoped(ctx, TAXONOMIST_TASK_NAME, task)),
        CURATOR_TASK_NAME: lambda ctx, task: handle_memory_curator_task(ctx, task, scoped(ctx, CURATOR_TASK_NAME, task)),
        PROJECT_MANAGER_TASK_NAME: handle_project_manager_task,
        FACT_CHECKER_TASK_NAME: handle_fact_checker_task,
        SWEEPER_TASK_NAME: handle_sweeper_task,
    }


def _provider_for_task(ctx: ApplicationContext, provider: Any, agentic_provider: Any, task_name: str, task: TaskRecord):
    registry = getattr(ctx, "ai_provider_registry", None) or {}
    routing = None if ctx.config is None else ctx.config.provider_routing
    prefer_agentic = task_name in AGENTIC_TASK_NAMES

    candidate_route_keys: list[str] = []
    if routing is not None:
        if task_name in routing.task_routes:
            candidate_route_keys = routing.task_routes[task_name]
        elif prefer_agentic and routing.default_agentic_route:
            candidate_route_keys = routing.default_agentic_route
        elif not prefer_agentic and routing.default_json_route:
            candidate_route_keys = routing.default_json_route

    if candidate_route_keys:
        found_routed_provider = False
        for route_key in candidate_route_keys:
            bundle = registry.get(route_key)
            if not isinstance(bundle, dict):
                continue
            selected_provider = _select_provider_from_bundle(bundle, prefer_agentic=prefer_agentic)
            if selected_provider is None:
                continue
            found_routed_provider = True
            budget_available = getattr(selected_provider, "budget_available", None)
            if callable(budget_available) and not budget_available():
                logger.warning("Skipping over-budget provider route", extra={"task_name": task_name, "route_key": route_key})
                continue
            return _bind_provider(selected_provider, task_name=task_name, task=task)
        if found_routed_provider:
            return None

    selected_provider = provider
    if prefer_agentic and agentic_provider is not None:
        selected_provider = agentic_provider
    if selected_provider is None:
        return None
    budget_available = getattr(selected_provider, "budget_available", None)
    if callable(budget_available) and not budget_available():
        return None
    return _bind_provider(selected_provider, task_name=task_name, task=task)


def _select_provider_from_bundle(bundle: dict[str, Any], *, prefer_agentic: bool):
    if prefer_agentic and bundle.get("agentic") is not None:
        return bundle.get("agentic")
    if bundle.get("json") is not None:
        return bundle.get("json")
    return bundle.get("agentic")


def _bind_provider(selected_provider: Any, *, task_name: str, task: TaskRecord):
    binder = getattr(selected_provider, "with_usage_context", None)
    if not callable(binder):
        return selected_provider
    return binder(task_name=task_name, task_id=task.id, workspace_id=task.workspace_id)


def bootstrap_background_tasks(ctx: ApplicationContext) -> None:
    task_queue = getattr(ctx, "task_queue", None)
    if task_queue is None:
        return

    journal = getattr(ctx, "journal", None)

    if journal is not None:
        schedule_system1_ingest(
            task_queue,
            journal,
            None,
            suppression_config=None if ctx.config is None else ctx.config.ingest_suppression,
        )

    for task_name, interval_seconds in RECURRING_TASK_INTERVAL_SECONDS.items():
        _ensure_recurring_task_scheduled(
            task_queue,
            task_name=task_name,
            workspace_id=None,
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
        priority = task_priority(task_name)
        if existing.status == "pending" and existing.priority != priority:
            task_queue.update_pending_task(existing.id, priority=priority)
        return

    summary = task_queue.summarize_task_runs([task_name], workspace_id=workspace_id)[0]
    available_at = None
    if summary.last_completed_at is not None:
        available_at = max(summary.last_completed_at + interval_seconds, time.time())

    task_queue.enqueue_unique(
        task_name=task_name,
        workspace_id=workspace_id,
        priority=task_priority(task_name),
        available_at=available_at,
        data={
            "workspace_id": workspace_id,
            "trigger": "recurring_schedule",
            "interval_seconds": interval_seconds,
        },
    )

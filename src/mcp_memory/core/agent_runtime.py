from __future__ import annotations

from collections.abc import Callable
import logging
import time
from typing import Any

from mcp_memory.context import BackgroundTaskBootstrapContext, ProviderSelectionContext, TaskRuntimeContext
from mcp_memory.core.maintenance_idle import should_preserve_idle_pause
from mcp_memory.core.provider_policy import ProviderSelectionInputs, select_provider_for_inputs
from mcp_memory.core.task_policy import DEFAULT_AGENTIC_TASK_NAMES
from mcp_memory.core.system1_scheduling import schedule_system1_ingest
from mcp_memory.core.recurring_jitter import compute_recurring_jitter_seconds
from mcp_memory.core.recurring_jitter import read_recurring_jitter_seconds
from mcp_memory.core.task_handlers import (
    AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS,
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    EMBEDDING_REPAIR_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINK_DISCOVERY_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    handle_defragmenter_task,
    handle_embedding_repair_task,
    handle_conflict_detector_task,
    handle_memory_curator_task,
    handle_deduplicator_task,
    handle_fact_checker_task,
    handle_graph_link_discovery_task,
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

AGENTIC_TASK_NAMES = set(DEFAULT_AGENTIC_TASK_NAMES)
DEFAULT_RUNTIME_TASK_RETRY_DELAY_SECONDS = 300.0


def build_runtime_task_worker(
    ctx: TaskRuntimeContext,
    provider: Any = None,
) -> RuntimeTaskWorker:
    provider_selection_inputs = _provider_selection_inputs_from_context(ctx)
    active_json_provider = (
        provider
        if provider is not None
        else getattr(ctx, "ai_json_provider", None) or getattr(ctx, "ai_provider", None)
    )
    active_agentic_provider = getattr(ctx, "ai_agent_provider", None)
    handlers = build_default_task_handlers(active_json_provider, active_agentic_provider, provider_selection_inputs=provider_selection_inputs)
    expected_handlers = {
        SYSTEM1_INGEST_TASK_NAME,
        SUMMARIZE_MEMORY_TASK_NAME,
        EMBEDDING_REPAIR_TASK_NAME,
        GRAPH_LINK_DISCOVERY_TASK_NAME,
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
        handler_factory=lambda: build_default_task_handlers(active_json_provider, active_agentic_provider, provider_selection_inputs=provider_selection_inputs),
        poll_interval_seconds=0.05,
        retry_delay_seconds=DEFAULT_RUNTIME_TASK_RETRY_DELAY_SECONDS,
    )


def build_default_task_handlers(
    provider: Any = None,
    agentic_provider: Any = None,
    *,
    provider_selection_inputs: ProviderSelectionInputs | None = None,
) -> dict[str, Callable[[Any, TaskRecord], Any]]:
    def scoped(ctx: ProviderSelectionContext, task_name: str, task: TaskRecord):
        inputs = provider_selection_inputs or _provider_selection_inputs_from_context(ctx)
        return _provider_for_task_inputs(inputs, provider, agentic_provider, task_name, task)

    return {
        SYSTEM1_INGEST_TASK_NAME: lambda ctx, task: handle_ingest_system1_task(ctx, task, scoped(ctx, SYSTEM1_INGEST_TASK_NAME, task)),
        SUMMARIZE_MEMORY_TASK_NAME: lambda ctx, task: handle_summarize_memory_task(ctx, task, scoped(ctx, SUMMARIZE_MEMORY_TASK_NAME, task)),
        EMBEDDING_REPAIR_TASK_NAME: handle_embedding_repair_task,
        GRAPH_LINK_DISCOVERY_TASK_NAME: lambda ctx, task: handle_graph_link_discovery_task(ctx, task),
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


def _provider_selection_inputs_from_context(ctx: ProviderSelectionContext) -> ProviderSelectionInputs:
    return ProviderSelectionInputs(
        config=ctx.config,
        ai_provider_registry=getattr(ctx, "ai_provider_registry", None),
        provider_policy_events=getattr(ctx, "provider_policy_events", None),
    )


def _provider_for_task_inputs(inputs: ProviderSelectionInputs, provider: Any, agentic_provider: Any, task_name: str, task: TaskRecord):
    selected = select_provider_for_inputs(
        inputs,
        provider,
        agentic_provider,
        task_name,
        task,
        agentic_task_names=AGENTIC_TASK_NAMES,
    )
    if selected is None:
        return None
    selected_provider_key = getattr(selected, "_provider_key", None)
    if selected_provider_key is not None:
        logger.debug("Selected provider for task", extra={"task_name": task_name, "provider_key": selected_provider_key})
    return selected


def _provider_for_task(ctx: ProviderSelectionContext, provider: Any, agentic_provider: Any, task_name: str, task: TaskRecord):
    return _provider_for_task_inputs(_provider_selection_inputs_from_context(ctx), provider, agentic_provider, task_name, task)


def bootstrap_background_tasks(ctx: BackgroundTaskBootstrapContext) -> None:
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

    for task_name, interval_seconds in AUTONOMOUS_RECURRING_TASK_INTERVAL_SECONDS.items():
        _ensure_recurring_task_scheduled(
            task_queue,
            journal=journal,
            task_name=task_name,
            workspace_id=None,
            interval_seconds=interval_seconds,
        )


def _ensure_recurring_task_scheduled(
    task_queue,
    *,
    journal,
    task_name: str,
    workspace_id: str | None,
    interval_seconds: float,
) -> None:
    current_time = time.time()
    summary = task_queue.summarize_task_runs([task_name], workspace_id=workspace_id)[0]
    base_available_at = current_time
    if summary.last_completed_at is not None:
        base_available_at = max(summary.last_completed_at + interval_seconds, current_time)

    existing = task_queue.find_open_task(task_name, workspace_id)
    if existing is not None:
        priority = task_priority(task_name)
        if existing.status == "pending":
            next_data = dict(existing.data)
            next_available_at = None
            needs_update = False
            jitter_seconds = read_recurring_jitter_seconds(existing.data)
            if jitter_seconds is None:
                jitter_seconds = compute_recurring_jitter_seconds(interval_seconds)
                next_data["jitter_seconds"] = jitter_seconds
                needs_update = True
            expected_available_at = base_available_at + jitter_seconds
            if existing.priority != priority:
                needs_update = True
            if _is_recurring_task(existing.data):
                if next_data.get("workspace_id") != workspace_id:
                    next_data["workspace_id"] = workspace_id
                    needs_update = True
                if next_data.get("interval_seconds") != interval_seconds:
                    next_data["interval_seconds"] = interval_seconds
                    needs_update = True
                if abs(existing.available_at - expected_available_at) > 1e-6:
                    next_available_at = expected_available_at
                    needs_update = True
            if needs_update:
                task_queue.update_pending_task(
                    existing.id,
                    data=next_data,
                    available_at=next_available_at,
                    priority=priority,
                )
        return

    if journal is not None and should_preserve_idle_pause(summary, journal):
        return

    jitter_seconds = compute_recurring_jitter_seconds(interval_seconds)
    expected_available_at = base_available_at + jitter_seconds

    task_queue.enqueue_unique(
        task_name=task_name,
        workspace_id=workspace_id,
        priority=task_priority(task_name),
        available_at=expected_available_at,
        data={
            "workspace_id": workspace_id,
            "trigger": "recurring_schedule",
            "interval_seconds": interval_seconds,
            "jitter_seconds": jitter_seconds,
        },
    )


def _is_recurring_task(data: dict[str, object] | None) -> bool:
    if not isinstance(data, dict):
        return False
    trigger = data.get("trigger")
    return trigger in {"recurring_schedule", "recurring_follow_up"}

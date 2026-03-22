from __future__ import annotations

from mcp_memory.core.provider_policy import ProviderSelectionInputs, ProviderSelectionRequest, select_provider_for_request
from mcp_memory.core.task_handlers import SUMMARIZE_MEMORY_TASK_NAME, TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.core.task_policy import DEFAULT_AGENTIC_TASK_NAMES, DEFAULT_LOW_PRIORITY_TASK_NAMES, task_class_for_task
from mcp_memory.management.models import TaskRouteAuditPayload


def build_task_route_audit(
    *,
    config,
    workspace_id: str | None,
    provider_usage_repo,
    ai_json_provider,
    ai_agent_provider,
    ai_provider_registry,
) -> list[TaskRouteAuditPayload]:
    task_names = [*TRIGGERABLE_BACKGROUND_TASK_NAMES, SUMMARIZE_MEMORY_TASK_NAME]
    usage_by_task: dict[str, tuple[int, int]] = {}
    top_provider_by_task: dict[str, tuple[str, str]] = {}
    top_provider_rank_by_task: dict[str, tuple[int, int]] = {}
    for summary in provider_usage_repo.summarize_usage(workspace_id=workspace_id):
        task_name = summary.task_name
        if task_name is None:
            continue
        success_count, failure_count = usage_by_task.get(task_name, (0, 0))
        usage_by_task[task_name] = (
            success_count + summary.calls_last_day - summary.failures_last_day,
            failure_count + summary.failures_last_day,
        )
        rank = (summary.calls_last_day, -summary.failures_last_day)
        previous_rank = top_provider_rank_by_task.get(task_name)
        if previous_rank is None or rank > previous_rank:
            top_provider_by_task[task_name] = (summary.provider_key, summary.model_name)
            top_provider_rank_by_task[task_name] = rank

    audits: list[TaskRouteAuditPayload] = []
    for task_name in task_names:
        task_class = task_class_for_task(config, task_name)
        execution_kind = (
            "agentic"
            if task_class in {"cheap_agentic", "premium_agentic"}
            else "deterministic" if task_class == "deterministic" else "json"
        )
        configured_routes = _configured_routes_for_task(config, task_name, execution_kind=execution_kind)
        selected = _select_provider_for_audit(
            config=config,
            workspace_id=workspace_id,
            ai_provider_registry=ai_provider_registry,
            ai_json_provider=ai_json_provider,
            ai_agent_provider=ai_agent_provider,
            task_name=task_name,
        )
        selected_provider_key = None if selected is None else getattr(selected, "_provider_key", None)
        selected_model_name = None if selected is None else getattr(selected, "_model_name", None)
        underlying_provider = None if selected is None else getattr(selected, "_provider", None)
        supports_agentic = None
        if selected is not None:
            supports_agentic_method = getattr(selected, "supports_agentic", None)
            if callable(supports_agentic_method):
                supports_agentic = bool(supports_agentic_method())
        recent = provider_usage_repo.list_conversations(task_name=task_name, limit=1)
        recent_record = recent[0] if recent else None
        recent_success_count, recent_failure_count = usage_by_task.get(task_name, (0, 0))
        usage_fallback = top_provider_by_task.get(task_name)

        audits.append(
            TaskRouteAuditPayload(
                task_name=task_name,
                task_class=task_class,
                execution_kind=execution_kind,
                low_priority=task_name in set((config.provider_routing.low_priority_task_names if config is not None else []) or DEFAULT_LOW_PRIORITY_TASK_NAMES),
                configured_primary_route=configured_routes[0] if configured_routes else None,
                configured_fallback_routes=configured_routes[1:] if len(configured_routes) > 1 else [],
                resolved_provider_key=selected_provider_key,
                resolved_model_name=selected_model_name,
                resolved_provider_type=None if underlying_provider is None else type(underlying_provider).__name__,
                resolved_supports_agentic=supports_agentic,
                recent_provider_key=(None if recent_record is None else recent_record.provider_key) or (None if usage_fallback is None else usage_fallback[0]),
                recent_model_name=(None if recent_record is None else recent_record.model_name) or (None if usage_fallback is None else usage_fallback[1]),
                recent_status=None if recent_record is None else recent_record.status,
                recent_success_count=recent_success_count,
                recent_failure_count=recent_failure_count,
                on_primary_route=None if selected_provider_key is None or not configured_routes else selected_provider_key == configured_routes[0] or str(selected_provider_key).startswith(configured_routes[0]),
            )
        )
    return audits


def _configured_routes_for_task(config, task_name: str, *, execution_kind: str) -> list[str]:
    if config is None:
        return []
    routing = config.provider_routing
    if execution_kind == "deterministic":
        return []
    if task_name in routing.task_routes:
        return list(routing.task_routes[task_name])
    if execution_kind == "agentic":
        return list(routing.default_agentic_route)
    return list(routing.default_json_route)


def _select_provider_for_audit(
    *,
    config,
    workspace_id: str | None,
    ai_provider_registry,
    ai_json_provider,
    ai_agent_provider,
    task_name: str,
):
    return select_provider_for_request(
        ProviderSelectionInputs(
            config=config,
            ai_provider_registry=ai_provider_registry,
        ),
        ai_json_provider,
        ai_agent_provider,
        ProviderSelectionRequest(
            task_name=task_name,
            task_id=f"audit:{task_name}",
            workspace_id=workspace_id,
        ),
        agentic_task_names=set(DEFAULT_AGENTIC_TASK_NAMES),
        record_admission_skips=False,
    )

from __future__ import annotations

import logging
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_policy import is_deterministic_task
from mcp_memory.core.tasks import TaskRecord


logger = logging.getLogger(__name__)


def select_provider_for_task(
    ctx: ApplicationContext,
    provider: Any,
    agentic_provider: Any,
    task_name: str,
    task: TaskRecord,
    *,
    agentic_task_names: set[str],
):
    if is_deterministic_task(ctx.config, task_name):
        return None

    registry = getattr(ctx, "ai_provider_registry", None) or {}
    routing = None if ctx.config is None else ctx.config.provider_routing
    prefer_agentic = task_name in agentic_task_names

    candidate_route_keys: list[str] = []
    if routing is not None:
        if task_name in routing.task_routes:
            candidate_route_keys = routing.task_routes[task_name]
        elif prefer_agentic and routing.default_agentic_route:
            candidate_route_keys = routing.default_agentic_route
        elif not prefer_agentic and routing.default_json_route:
            candidate_route_keys = routing.default_json_route

    if candidate_route_keys:
        logger.debug(
            "Provider routing attempting configured routes",
            extra={"task_name": task_name, "candidate_routes": candidate_route_keys},
        )
        found_routed_provider = False
        for route_key in candidate_route_keys:
            bundle = registry.get(route_key)
            if not isinstance(bundle, dict):
                continue
            selected_provider = select_provider_from_bundle(bundle, prefer_agentic=prefer_agentic)
            if selected_provider is None:
                continue
            found_routed_provider = True
            if not _budget_available(selected_provider):
                logger.debug(
                    "Provider route skipped because daily budget is exhausted",
                    extra={"task_name": task_name, "route_key": route_key},
                )
                continue
            return bind_provider_context(selected_provider, task_name=task_name, task=task)
        if found_routed_provider:
            logger.warning(
                "Provider routing exhausted all configured routes",
                extra={"task_name": task_name, "candidate_routes": candidate_route_keys},
            )
            return None

    selected_provider = provider
    if prefer_agentic and agentic_provider is not None:
        selected_provider = agentic_provider
    if selected_provider is None:
        logger.warning(
            "No provider available for task after routing and legacy fallback",
            extra={"task_name": task_name, "prefer_agentic": prefer_agentic},
        )
        return None
    if not _budget_available(selected_provider):
        logger.warning(
            "Legacy fallback provider is over budget",
            extra={"task_name": task_name},
        )
        return None
    return bind_provider_context(selected_provider, task_name=task_name, task=task)


def _budget_available(provider: Any) -> bool:
    budget_available = getattr(provider, "budget_available", None)
    if not callable(budget_available):
        return True
    return bool(budget_available())


def select_provider_from_bundle(bundle: dict[str, Any], *, prefer_agentic: bool):
    if prefer_agentic and bundle.get("agentic") is not None:
        return bundle.get("agentic")
    if bundle.get("json") is not None:
        return bundle.get("json")
    return bundle.get("agentic")


def bind_provider_context(selected_provider: Any, *, task_name: str, task: TaskRecord):
    binder = getattr(selected_provider, "with_usage_context", None)
    if not callable(binder):
        return selected_provider
    return binder(task_name=task_name, task_id=task.id, workspace_id=task.workspace_id)

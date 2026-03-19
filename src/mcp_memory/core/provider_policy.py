from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_policy import is_deterministic_task
from mcp_memory.core.tasks import TaskRecord


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSelectionInputs:
    config: Any = None
    ai_provider_registry: dict[str, Any] | None = None


class AgenticRouteFailoverProvider:
    def __init__(self, providers: list[Any], *, route_keys: list[str], task_name: str) -> None:
        self._providers = list(providers)
        self._route_keys = list(route_keys)
        self._task_name = task_name
        self._provider_key = getattr(self._providers[0], "_provider_key", None) if self._providers else None

    def supports_agentic(self) -> bool:
        return True

    async def run_agent(self, prompt: str):
        last_error: Exception | None = None
        for index, provider in enumerate(self._providers):
            run_agent = getattr(provider, "run_agent", None)
            if not callable(run_agent):
                continue
            try:
                return await cast(Callable[[str], Awaitable[Any]], run_agent)(prompt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                if not _is_same_run_failover_eligible_error(exc) or index >= len(self._providers) - 1:
                    raise
                logger.warning(
                    "Agentic provider route failed during run_agent; attempting next configured route",
                    extra={
                        "task_name": self._task_name,
                        "failed_route_key": self._route_keys[index],
                        "next_route_key": self._route_keys[index + 1],
                        "error": str(exc),
                    },
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("agentic_provider_not_configured")

    def __getattr__(self, name: str) -> Any:
        if not self._providers:
            raise AttributeError(name)
        return getattr(self._providers[0], name)




def select_provider_for_task(
    ctx: ApplicationContext,
    provider: Any,
    agentic_provider: Any,
    task_name: str,
    task: TaskRecord,
    *,
    agentic_task_names: set[str],
):
    return select_provider_for_inputs(
        ProviderSelectionInputs(
            config=ctx.config,
            ai_provider_registry=getattr(ctx, "ai_provider_registry", None),
        ),
        provider,
        agentic_provider,
        task_name,
        task,
        agentic_task_names=agentic_task_names,
    )


def select_provider_for_inputs(
    inputs: ProviderSelectionInputs,
    provider: Any,
    agentic_provider: Any,
    task_name: str,
    task: TaskRecord,
    *,
    agentic_task_names: set[str],
):
    if is_deterministic_task(inputs.config, task_name):
        return None

    registry = inputs.ai_provider_registry or {}
    routing = None if inputs.config is None else inputs.config.provider_routing
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
        first_routed_provider = None
        routed_agentic_providers: list[Any] = []
        routed_agentic_route_keys: list[str] = []
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
            bound_provider = bind_provider_context(selected_provider, task_name=task_name, task=task)
            if first_routed_provider is None:
                first_routed_provider = bound_provider
                if not prefer_agentic or not _supports_agentic_execution(bound_provider):
                    return first_routed_provider
                routed_agentic_providers.append(bound_provider)
                routed_agentic_route_keys.append(route_key)
                continue
            if prefer_agentic and _supports_agentic_execution(bound_provider):
                routed_agentic_providers.append(bound_provider)
                routed_agentic_route_keys.append(route_key)
        if first_routed_provider is not None:
            if len(routed_agentic_providers) > 1:
                return AgenticRouteFailoverProvider(
                    routed_agentic_providers,
                    route_keys=routed_agentic_route_keys,
                    task_name=task_name,
                )
            return first_routed_provider
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


def _supports_agentic_execution(provider: Any) -> bool:
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    return callable(run_agent) and (not callable(supports_agentic) or bool(supports_agentic()))


def _is_same_run_failover_eligible_error(exc: Exception) -> bool:
    retry_delay_seconds = getattr(exc, "retry_delay_seconds", None)
    return not isinstance(retry_delay_seconds, (int, float))


def bind_provider_context(selected_provider: Any, *, task_name: str, task: TaskRecord):
    binder = getattr(selected_provider, "with_usage_context", None)
    if not callable(binder):
        return selected_provider
    return binder(task_name=task_name, task_id=task.id, workspace_id=task.workspace_id)

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any, Awaitable, Callable, cast

from mcp_memory.core.provider_admission import ProviderAdmissionDecision
from mcp_memory.core.task_handlers.constants import TAXONOMIST_TASK_NAME
from mcp_memory.core.task_policy import is_deterministic_task
from mcp_memory.core.tasks import TaskRecord


logger = logging.getLogger(__name__)
_PROVIDER_WARNING_MIN_INTERVAL_SECONDS = 600.0
_provider_warning_state: dict[tuple[str, str | None, tuple[str, ...], str | None], float] = {}


@dataclass(frozen=True)
class ProviderSelectionInputs:
    config: Any = None
    ai_provider_registry: dict[str, Any] | None = None
    provider_policy_events: Any = None


@dataclass(frozen=True)
class ProviderSelectionRequest:
    task_name: str
    task_id: str | None = None
    execution_epoch: int | None = None
    workspace_id: str | None = None


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


def select_provider_for_request(
    inputs: ProviderSelectionInputs,
    provider: Any,
    agentic_provider: Any,
    request: ProviderSelectionRequest,
    *,
    agentic_task_names: set[str],
    record_admission_skips: bool = True,
):
    task_name = request.task_name
    if is_deterministic_task(inputs.config, task_name):
        return None

    registry = inputs.ai_provider_registry or {}
    routing = None if inputs.config is None else inputs.config.provider_routing
    prefer_agentic = task_name in agentic_task_names

    candidate_route_keys = _candidate_route_keys_for_task(
        routing=routing,
        registry=registry,
        task_name=task_name,
        prefer_agentic=prefer_agentic,
    )

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
            bound_provider = bind_provider_context(selected_provider, request=request)
            admission = _provider_admission_decision(selected_provider)
            if not admission.allowed:
                if record_admission_skips:
                    _record_admission_skip(bound_provider, admission)
                _record_provider_policy_event(
                    inputs,
                    request=request,
                    event_kind="route_skipped",
                    provider=selected_provider,
                    route_key=route_key,
                    candidate_routes=candidate_route_keys,
                    reason=admission,
                )
                logger.debug(
                    "Provider route skipped because admission control denied availability",
                    extra={
                        "task_name": task_name,
                        "route_key": route_key,
                        "reason": admission.reason,
                        "retry_delay_seconds": admission.retry_delay_seconds,
                    },
                )
                continue
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
            warning_suppressed = _log_provider_warning_once(
                "Provider routing exhausted all configured routes",
                warning_kind="routes_exhausted",
                task_name=task_name,
                route_keys=candidate_route_keys,
            )
            _record_provider_policy_event(
                inputs,
                request=request,
                event_kind="route_exhausted",
                warning_kind="routes_exhausted",
                candidate_routes=candidate_route_keys,
                warning_suppressed=warning_suppressed,
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
    admission = _provider_admission_decision(selected_provider)
    if not admission.allowed:
        if record_admission_skips:
            _record_admission_skip(bind_provider_context(selected_provider, request=request), admission)
        warning_suppressed = _log_provider_warning_once(
            "Legacy fallback provider is unavailable due to admission control",
            warning_kind="legacy_fallback_denied",
            task_name=task_name,
            route_keys=(),
            reason=admission.reason,
        )
        _record_provider_policy_event(
            inputs,
            request=request,
            event_kind="legacy_fallback_denied",
            warning_kind="legacy_fallback_denied",
            provider=selected_provider,
            reason=admission,
            warning_suppressed=warning_suppressed,
        )
        return None
    return bind_provider_context(selected_provider, request=request)


def select_provider_for_inputs(
    inputs: ProviderSelectionInputs,
    provider: Any,
    agentic_provider: Any,
    task_name: str,
    task: TaskRecord,
    *,
    agentic_task_names: set[str],
):
    return select_provider_for_request(
        inputs,
        provider,
        agentic_provider,
        ProviderSelectionRequest(
            task_name=task_name,
            task_id=task.id,
            execution_epoch=task.execution_epoch,
            workspace_id=task.workspace_id,
        ),
        agentic_task_names=agentic_task_names,
    )


def _budget_available(provider: Any) -> bool:
    budget_available = getattr(provider, "budget_available", None)
    if not callable(budget_available):
        return True
    return bool(budget_available())


def _provider_admission_decision(provider: Any) -> ProviderAdmissionDecision:
    admission_decision = getattr(provider, "admission_decision", None)
    if callable(admission_decision):
        decision = admission_decision()
        if isinstance(decision, ProviderAdmissionDecision):
            return decision
    return ProviderAdmissionDecision(allowed=_budget_available(provider))


def _record_admission_skip(provider: Any, admission: ProviderAdmissionDecision) -> None:
    recorder = getattr(provider, "record_admission_skip", None)
    if callable(recorder):
        recorder(admission)


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
    if bool(getattr(exc, "same_run_failover_eligible", False)):
        return True
    retry_delay_seconds = getattr(exc, "retry_delay_seconds", None)
    return not isinstance(retry_delay_seconds, (int, float))


def bind_provider_context(selected_provider: Any, *, request: ProviderSelectionRequest):
    binder = getattr(selected_provider, "with_usage_context", None)
    if not callable(binder):
        return selected_provider
    return binder(
        task_name=request.task_name,
        task_id=request.task_id,
        execution_epoch=request.execution_epoch,
        workspace_id=request.workspace_id,
    )


def _candidate_route_keys_for_task(*, routing, registry: dict[str, Any], task_name: str, prefer_agentic: bool) -> list[str]:
    if routing is None:
        return []
    if task_name in routing.task_routes:
        return routing.task_routes[task_name]
    if prefer_agentic and routing.default_agentic_route:
        return routing.default_agentic_route
    if not prefer_agentic and routing.default_json_route:
        return routing.default_json_route
    if not prefer_agentic and task_name == TAXONOMIST_TASK_NAME:
        return [route_key for route_key in ["copilot-mini"] if route_key in registry]
    return []


def _record_provider_policy_event(
    inputs: ProviderSelectionInputs,
    *,
    request: ProviderSelectionRequest,
    event_kind: str,
    warning_kind: str | None = None,
    provider: Any = None,
    route_key: str | None = None,
    candidate_routes: list[str] | tuple[str, ...] = (),
    reason: ProviderAdmissionDecision | None = None,
    warning_suppressed: bool = False,
) -> None:
    repository = inputs.provider_policy_events
    if repository is None:
        return
    repository.record_event(
        task_name=request.task_name,
        task_id=request.task_id,
        event_kind=event_kind,
        warning_kind=warning_kind,
        provider_key=getattr(provider, "_provider_key", None),
        provider_name=getattr(provider, "_provider_name", None),
        model_name=getattr(provider, "_model_name", None),
        route_key=route_key,
        candidate_routes=list(candidate_routes),
        reason_category=None if reason is None else reason.reason_category,
        reason_code=None if reason is None else reason.reason,
        retry_delay_seconds=None if reason is None else reason.retry_delay_seconds,
        warning_suppressed=warning_suppressed,
    )


def _log_provider_warning_once(
    message: str,
    *,
    warning_kind: str,
    task_name: str,
    route_keys: list[str] | tuple[str, ...],
    reason: str | None = None,
    now: float | None = None,
 ) -> bool:
    current_time = time.time() if now is None else now
    key = (warning_kind, task_name, tuple(route_keys), reason)
    previous_logged_at = _provider_warning_state.get(key)
    if (
        previous_logged_at is not None
        and current_time - previous_logged_at < _PROVIDER_WARNING_MIN_INTERVAL_SECONDS
    ):
        return True
    _provider_warning_state[key] = current_time
    extra = {
        "task_name": task_name,
        "candidate_routes": list(route_keys),
    }
    if reason is not None:
        extra["reason"] = reason
    logger.warning(message, extra=extra)
    return False

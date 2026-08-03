from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from mcp_memory.core.provider_admission import build_provider_admission_exception
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.provider_admission import evaluate_provider_admission
from mcp_memory.core.provider_admission import should_persist_admission_backoff
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.providers.interfaces import ProviderAttemptFinishedEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptHeartbeatEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptStartedEvent
from mcp_memory.core.providers.interfaces import ProviderObserverEvent
from mcp_memory.core.providers.interfaces import ProviderJSONCall
from mcp_memory.core.ports.providers import ProviderUsagePort, TaskExecutionAttemptPort


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _ProviderObserverState:
    last_event: ProviderObserverEvent | None = None


class InstrumentedAIProvider:
    def __init__(
        self,
        provider,
        *,
        usage_repository: ProviderUsagePort,
        provider_key: str,
        provider_name: str,
        model_name: str,
        task_name: str | None = None,
        task_id: str | None = None,
        execution_epoch: int | None = None,
        workspace_id: str | None = None,
        task_queue = None,
        task_execution_attempts: TaskExecutionAttemptPort | None = None,
        budget_key: str | None = None,
        daily_call_limit: int | None = None,
        model_burst_call_limit: int | None = None,
        model_burst_window_seconds: float | None = None,
        block_test_execution: bool = False,
        provider_trust_class: str | None = None,
        provider_allowlisted: bool | None = None,
        provider_profile: str | None = None,
        route_available: bool | None = None,
    ) -> None:
        self._provider = provider
        self._usage_repository = usage_repository
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._model_name = model_name
        self._provider_profile = provider_profile or provider_key
        self._route_available = route_available
        self._task_name = task_name
        self._task_id = task_id
        self._execution_epoch = execution_epoch
        self._workspace_id = workspace_id
        self._task_queue = task_queue
        self._task_execution_attempts = task_execution_attempts
        self._budget_key = budget_key or provider_key
        self._daily_call_limit = daily_call_limit
        self._model_burst_call_limit = model_burst_call_limit
        self._model_burst_window_seconds = model_burst_window_seconds
        self._block_test_execution = block_test_execution
        self._provider_trust_class = provider_trust_class or str(
            getattr(provider, "provider_trust_class", "external")
        )
        configured_allowlist = (
            getattr(provider, "provider_allowlisted", None)
            if provider_allowlisted is None
            else provider_allowlisted
        )
        self._provider_allowlisted = (
            self._provider_trust_class == "local"
            if configured_allowlist is None
            else bool(configured_allowlist)
        )

    def with_usage_context(
        self,
        *,
        task_name: str | None,
        task_id: str | None = None,
        execution_epoch: int | None = None,
        workspace_id: str | None = None,
    ):
        return InstrumentedAIProvider(
            self._provider,
            usage_repository=self._usage_repository,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            task_name=task_name,
            task_id=task_id,
            execution_epoch=execution_epoch,
            workspace_id=workspace_id,
            task_queue=self._task_queue,
            task_execution_attempts=self._task_execution_attempts,
            budget_key=self._budget_key,
            daily_call_limit=self._daily_call_limit,
            model_burst_call_limit=self._model_burst_call_limit,
            model_burst_window_seconds=self._model_burst_window_seconds,
            block_test_execution=self._block_test_execution,
            provider_trust_class=self._provider_trust_class,
            provider_allowlisted=self._provider_allowlisted,
            provider_profile=self._provider_profile,
            route_available=self._route_available,
        )

    def with_route_context(
        self,
        *,
        provider_profile: str,
        route_available: bool,
    ):
        return InstrumentedAIProvider(
            self._provider,
            usage_repository=self._usage_repository,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            task_name=self._task_name,
            task_id=self._task_id,
            execution_epoch=self._execution_epoch,
            workspace_id=self._workspace_id,
            task_queue=self._task_queue,
            task_execution_attempts=self._task_execution_attempts,
            budget_key=self._budget_key,
            daily_call_limit=self._daily_call_limit,
            model_burst_call_limit=self._model_burst_call_limit,
            model_burst_window_seconds=self._model_burst_window_seconds,
            block_test_execution=self._block_test_execution,
            provider_trust_class=self._provider_trust_class,
            provider_allowlisted=self._provider_allowlisted,
            provider_profile=provider_profile,
            route_available=route_available,
        )

    def with_allowed_tool_names(self, allowed_tool_names: tuple[str, ...]):
        scoped_provider = getattr(self._provider, "with_allowed_tool_names", None)
        if not callable(scoped_provider):
            return self
        return InstrumentedAIProvider(
            scoped_provider(allowed_tool_names),
            usage_repository=self._usage_repository,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            task_name=self._task_name,
            task_id=self._task_id,
            execution_epoch=self._execution_epoch,
            workspace_id=self._workspace_id,
            task_queue=self._task_queue,
            task_execution_attempts=self._task_execution_attempts,
            budget_key=self._budget_key,
            daily_call_limit=self._daily_call_limit,
            model_burst_call_limit=self._model_burst_call_limit,
            model_burst_window_seconds=self._model_burst_window_seconds,
            block_test_execution=self._block_test_execution,
            provider_trust_class=self._provider_trust_class,
            provider_allowlisted=self._provider_allowlisted,
            provider_profile=self._provider_profile,
            route_available=self._route_available,
        )

    def _record_attempt_start(self, *, request_id: str, event: ProviderAttemptStartedEvent) -> None:
        if self._task_execution_attempts is None or self._task_id is None or self._execution_epoch is None:
            return
        self._task_execution_attempts.start_attempt(
            task_id=self._task_id,
            execution_epoch=self._execution_epoch,
            task_name=self._task_name,
            request_id=request_id,
            subprocess_pid=event.subprocess_pid,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            started_at=event.started_at,
        )

    def _record_attempt_heartbeat(self, *, request_id: str, event: ProviderAttemptHeartbeatEvent) -> None:
        if self._task_execution_attempts is None or self._task_id is None or self._execution_epoch is None:
            return
        self._task_execution_attempts.heartbeat_attempt(
            task_id=self._task_id,
            execution_epoch=self._execution_epoch,
            heartbeat_at=event.heartbeat_at,
            request_id=request_id,
            subprocess_pid=event.subprocess_pid,
        )

    def _finish_attempt(
        self,
        *,
        request_id: str,
        status: str,
        completed_at: float,
        subprocess_pid: int | None,
        error_text: str | None,
        termination_reason: str | None,
    ) -> None:
        if self._task_execution_attempts is None or self._task_id is None or self._execution_epoch is None:
            return
        self._task_execution_attempts.finish_attempt(
            task_id=self._task_id,
            execution_epoch=self._execution_epoch,
            status=status,
            completed_at=completed_at,
            request_id=request_id,
            subprocess_pid=subprocess_pid,
            error_text=error_text,
            termination_reason=termination_reason,
        )

    def _guard_test_execution(self) -> None:
        if not self._block_test_execution:
            return
        raise RuntimeError("provider_execution_blocked_in_tests")

    def _build_observer(
        self,
        *,
        prompt: str,
        request_id: str,
        state: _ProviderObserverState,
    ) -> Callable[[ProviderObserverEvent], None]:
        def _observer(event: ProviderObserverEvent) -> None:
            if not isinstance(
                event,
                (
                    ProviderAttemptStartedEvent,
                    ProviderAttemptHeartbeatEvent,
                    ProviderAttemptFinishedEvent,
                ),
            ):
                return
            state.last_event = event
            self._handle_observer_event(request_id=request_id, prompt=prompt, event=event)

        return _observer

    def _handle_observer_event(
        self,
        *,
        request_id: str,
        prompt: str,
        event: ProviderObserverEvent,
    ) -> None:
        if isinstance(event, ProviderAttemptStartedEvent):
            self._record_attempt_start(request_id=request_id, event=event)
            self._usage_repository.record_conversation(
                request_id=request_id,
                attempt=event.attempt,
                task_name=self._task_name,
                task_id=self._task_id,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                subprocess_pid=event.subprocess_pid,
                prompt_text=prompt,
                response_text="",
                parsed=None,
                status="running",
                error_text=None,
                reason_category=None,
                reason_code=None,
                retry_delay_seconds=None,
                started_at=event.started_at,
                completed_at=event.started_at,
            )
            return
        if isinstance(event, ProviderAttemptHeartbeatEvent):
            self._record_attempt_heartbeat(request_id=request_id, event=event)
            if self._task_queue is not None and self._task_id is not None:
                task_queue = self._task_queue
                task_id = self._task_id
                _safe_task_queue_update(
                    task_queue,
                    task_id=task_id,
                    operation="touch_running_task",
                    callback=lambda: task_queue.touch_running_task(
                        task_id,
                        updated_at=event.heartbeat_at,
                        execution_epoch=self._execution_epoch,
                    ),
                )
            self._usage_repository.touch_running_conversation(
                request_id=request_id,
                completed_at=event.heartbeat_at,
                subprocess_pid=event.subprocess_pid,
            )
            return

        assert isinstance(event, ProviderAttemptFinishedEvent)
        self._finish_attempt(
            request_id=request_id,
            status=event.status if event.status is not None else "finished",
            completed_at=event.completed_at,
            subprocess_pid=event.subprocess_pid,
            error_text=event.error_text,
            termination_reason=event.reason_code,
        )
        self._usage_repository.record_conversation(
            request_id=request_id,
            attempt=event.attempt,
            task_name=self._task_name,
            task_id=self._task_id,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            subprocess_pid=event.subprocess_pid,
            prompt_text=prompt,
            response_text=event.raw_text if event.raw_text is not None else "",
            parsed=event.parsed,
            status=event.status if event.status is not None else "error",
            error_text=event.error_text,
            reason_category=event.reason_category,
            reason_code=event.reason_code,
            retry_delay_seconds=event.retry_delay_seconds,
            started_at=event.started_at,
            completed_at=event.completed_at,
        )

    def admission_decision(self, *, now: float | None = None):
        return evaluate_provider_admission(
            self._usage_repository,
            provider_key=self._provider_key,
            budget_key=self._budget_key,
            model_name=self._model_name,
            daily_call_limit=self._daily_call_limit,
            model_burst_call_limit=self._model_burst_call_limit,
            model_burst_window_seconds=self._model_burst_window_seconds,
            now=now,
        )

    def budget_available(self, *, now: float | None = None) -> bool:
        return self.admission_decision(now=now).allowed

    def _enforce_rate_limits(self) -> None:
        decision = self.admission_decision()
        if decision.allowed:
            return
        raise build_provider_admission_exception(
            decision,
            provider_key=self._budget_key,
            model_name=self._model_name,
        )

    def record_admission_skip(self, decision, *, now: float | None = None) -> None:
        created_at = time.time() if now is None else now
        self._usage_repository.record_call(
            task_name=self._task_name,
            task_id=self._task_id,
            request_id=None,
            subprocess_pid=None,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status="skipped",
            duration_seconds=0.0,
            created_at=created_at,
            error_text=decision.error_text,
            reason_category=decision.reason_category,
            reason_code=decision.reason,
            retry_delay_seconds=decision.retry_delay_seconds,
        )

    def supports_agentic(self) -> bool:
        return callable(getattr(self._provider, "run_agent", None))

    async def open_agent_session(
        self,
        *,
        allowed_tool_names: tuple[str, ...] | None = None,
        tools: list[Any] | None = None,
    ):
        provider = self._provider
        binder = getattr(provider, "with_observer", None)
        if callable(binder):
            provider = binder(
                self._build_observer(
                    prompt="",
                    request_id=str(uuid4()),
                    state=_ProviderObserverState(),
                )
            )
        opener = getattr(provider, "open_agent_session", None)
        if not callable(opener):
            raise RuntimeError("agentic_session_not_supported")
        opener_kwargs: dict[str, Any] = {"allowed_tool_names": allowed_tool_names}
        if tools is not None:
            opener_kwargs["tools"] = tools
        session = await cast(Callable[..., Awaitable[Any]], opener)(**opener_kwargs)
        return _InstrumentedAgenticSession(self, session)

    async def ask_json(
        self,
        prompt: str,
        *,
        _on_call_complete: Callable[[ProviderJSONCall], None] | None = None,
    ) -> dict:
        started_at = time.time()
        request_id = str(uuid4())
        observer_state = _ProviderObserverState()
        self._guard_test_execution()

        try:
            self._enforce_rate_limits()
        except Exception as exc:
            classification = classify_provider_failure(exc)
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=None,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status="skipped",
                duration_seconds=0.0,
                created_at=started_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            _notify_json_call(
                _on_call_complete,
                ProviderJSONCall(
                    response=None,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                    model_name=self._model_name,
                    request_id=request_id,
                    attempt=0,
                    started_at=started_at,
                    completed_at=started_at,
                    status="skipped",
                    error_text=classification.error_text,
                    reason_code=classification.reason_code,
                    reason_category=classification.reason_category,
                    retry_delay_seconds=classification.retry_delay_seconds,
                    admission_status="skipped",
                    premium_request=False,
                ),
            )
            raise

        provider = self._provider
        binder = getattr(provider, "with_observer", None)
        if callable(binder):
            provider = binder(
                self._build_observer(
                    prompt=prompt,
                    request_id=request_id,
                    state=observer_state,
                )
            )
        try:
            ask_json = getattr(provider, "ask_json", None)
            if callable(ask_json):
                response = await cast(Callable[[str], Awaitable[dict[str, Any]]], ask_json)(prompt)
            else:
                response = await cast(Callable[[str], Awaitable[dict[str, Any]]], getattr(provider, "ask"))(prompt)
        except asyncio.CancelledError:
            completed_at = time.time()
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(observer_state.last_event, fallback="cancelled"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="cancelled"),
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
                completed_at=completed_at,
            )
            self._finish_attempt(
                request_id=request_id,
                status="cancelled",
                completed_at=completed_at,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                error_text="Command cancelled",
                termination_reason="provider_cancelled",
            )
            _notify_json_call(
                _on_call_complete,
                ProviderJSONCall(
                    response=None,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                    model_name=self._model_name,
                    request_id=request_id,
                    attempt=_extract_attempt(observer_state.last_event),
                    started_at=_extract_started_at(observer_state.last_event, fallback=started_at),
                    completed_at=_extract_completed_at(observer_state.last_event, fallback=completed_at),
                    status="cancelled",
                    error_text="Command cancelled",
                    reason_code="provider_cancelled",
                    raw_text=_extract_raw_text(observer_state.last_event),
                    parsed=None,
                    admission_status="admitted",
                    cancellation_requested=True,
                    premium_request=True,
                ),
            )
            raise
        except Exception as exc:
            completed_at = time.time()
            classification = classify_provider_failure(
                exc,
                event_status=_extract_status(observer_state.last_event, fallback="error"),
                raw_text=_extract_raw_text(observer_state.last_event),
            )
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(observer_state.last_event, fallback="error"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="error"),
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
                completed_at=completed_at,
            )
            if should_persist_admission_backoff(classification):
                self._usage_repository.upsert_admission_state(
                    provider_key=self._provider_key,
                    model_name=self._model_name,
                    reason_category=classification.reason_category,
                    reason_code=classification.reason_code,
                    error_text=classification.error_text,
                    retry_delay_seconds=classification.retry_delay_seconds,
                    active_until=completed_at + float(classification.retry_delay_seconds or 0.0),
                    updated_at=completed_at,
                )
            self._finish_attempt(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="error"),
                completed_at=completed_at,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                error_text=classification.error_text,
                termination_reason=classification.reason_code,
            )
            _notify_json_call(
                _on_call_complete,
                ProviderJSONCall(
                    response=None,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                    model_name=self._model_name,
                    request_id=request_id,
                    attempt=_extract_attempt(observer_state.last_event),
                    started_at=_extract_started_at(observer_state.last_event, fallback=started_at),
                    completed_at=_extract_completed_at(observer_state.last_event, fallback=completed_at),
                    status=_extract_status(observer_state.last_event, fallback="error"),
                    error_text=classification.error_text,
                    reason_code=classification.reason_code,
                    reason_category=classification.reason_category,
                    retry_delay_seconds=classification.retry_delay_seconds,
                    raw_text=_extract_raw_text(observer_state.last_event),
                    parsed=None,
                    admission_status="admitted",
                    premium_request=True,
                ),
            )
            raise
        self._usage_repository.clear_admission_state(provider_key=self._provider_key, model_name=self._model_name)
        completed_at = time.time()
        self._usage_repository.finalize_running_conversation(
            request_id=request_id,
            status=_extract_status(observer_state.last_event, fallback="success"),
            error_text=None,
            completed_at=completed_at,
            response_text=_extract_raw_text(observer_state.last_event) or json.dumps(response, sort_keys=True),
            parsed=response,
        )
        self._usage_repository.record_call(
            task_name=self._task_name,
            task_id=self._task_id,
            request_id=request_id,
            subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status=_extract_status(observer_state.last_event, fallback="success"),
            duration_seconds=max(completed_at - started_at, 0.0),
            created_at=completed_at,
            error_text=None,
            reason_category=None,
            reason_code=None,
            retry_delay_seconds=None,
        )
        self._finish_attempt(
            request_id=request_id,
            status=_extract_status(observer_state.last_event, fallback="success"),
            completed_at=completed_at,
            subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
            error_text=None,
            termination_reason=None,
        )
        _notify_json_call(
            _on_call_complete,
            ProviderJSONCall(
                response=response,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                request_id=request_id,
                attempt=_extract_attempt(observer_state.last_event),
                started_at=_extract_started_at(observer_state.last_event, fallback=started_at),
                completed_at=_extract_completed_at(observer_state.last_event, fallback=completed_at),
                status=_extract_status(observer_state.last_event, fallback="success"),
                raw_text=_extract_raw_text(observer_state.last_event) or json.dumps(response, sort_keys=True),
                parsed=response,
                admission_status="admitted",
                premium_request=True,
            ),
        )
        return response

    async def ask_json_with_telemetry(self, prompt: str) -> ProviderJSONCall:
        """Call JSON mode while returning the wrapper's authoritative metadata."""

        result: ProviderJSONCall | None = None

        def capture(value: ProviderJSONCall) -> None:
            nonlocal result
            result = value

        try:
            await self.ask_json(prompt, _on_call_complete=capture)
        except asyncio.CancelledError as exc:
            if result is not None:
                setattr(exc, "provider_json_call", result)
            raise
        except Exception:
            if result is not None:
                return result
            raise
        if result is None:
            raise RuntimeError("instrumented_provider_did_not_report_json_call")
        return result

    async def ask(self, prompt: str) -> dict:
        return await self.ask_json(prompt)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        started_at = time.time()
        request_id = str(uuid4())
        observer_state = _ProviderObserverState()
        self._guard_test_execution()

        try:
            self._enforce_rate_limits()
        except Exception as exc:
            classification = classify_provider_failure(exc)
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=None,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status="skipped",
                duration_seconds=0.0,
                created_at=started_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            raise

        provider = self._provider
        binder = getattr(provider, "with_observer", None)
        if callable(binder):
            provider = binder(
                self._build_observer(
                    prompt=prompt,
                    request_id=request_id,
                    state=observer_state,
                )
            )
        run_agent = getattr(provider, "run_agent", None)
        if not callable(run_agent):
            raise RuntimeError("agentic_provider_not_configured")
        try:
            result = await cast(Callable[[str], Awaitable[AgenticRunResult]], run_agent)(prompt)
        except asyncio.CancelledError:
            completed_at = time.time()
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(observer_state.last_event, fallback="cancelled"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="cancelled"),
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
                completed_at=completed_at,
            )
            self._finish_attempt(
                request_id=request_id,
                status="cancelled",
                completed_at=completed_at,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                error_text="Command cancelled",
                termination_reason="provider_cancelled",
            )
            raise
        except Exception as exc:
            completed_at = time.time()
            classification = classify_provider_failure(
                exc,
                event_status=_extract_status(observer_state.last_event, fallback="error"),
                raw_text=_extract_raw_text(observer_state.last_event),
            )
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(observer_state.last_event, fallback="error"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="error"),
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
                completed_at=completed_at,
            )
            if should_persist_admission_backoff(classification):
                self._usage_repository.upsert_admission_state(
                    provider_key=self._provider_key,
                    model_name=self._model_name,
                    reason_category=classification.reason_category,
                    reason_code=classification.reason_code,
                    error_text=classification.error_text,
                    retry_delay_seconds=classification.retry_delay_seconds,
                    active_until=completed_at + float(classification.retry_delay_seconds or 0.0),
                    updated_at=completed_at,
                )
            self._finish_attempt(
                request_id=request_id,
                status=_extract_status(observer_state.last_event, fallback="error"),
                completed_at=completed_at,
                subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
                error_text=classification.error_text,
                termination_reason=classification.reason_code,
            )
            raise
        self._usage_repository.clear_admission_state(provider_key=self._provider_key, model_name=self._model_name)
        completed_at = time.time()
        self._usage_repository.finalize_running_conversation(
            request_id=request_id,
            status=_extract_status(observer_state.last_event, fallback=result.status),
            error_text=None if result.status == "success" else result.summary,
            completed_at=completed_at,
            response_text=_extract_raw_text(observer_state.last_event) or result.raw_text or result.summary or "",
            parsed=result.parsed,
        )
        self._usage_repository.record_call(
            task_name=self._task_name,
            task_id=self._task_id,
            request_id=request_id,
            subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status=_extract_status(observer_state.last_event, fallback=result.status),
            duration_seconds=max(completed_at - started_at, 0.0),
            created_at=completed_at,
            error_text=None if result.status == "success" else result.summary,
            reason_category=None if result.status == "success" else "execution",
            reason_code=None if result.status == "success" else "agent_run_unsuccessful",
            retry_delay_seconds=None,
        )
        self._finish_attempt(
            request_id=request_id,
            status=_extract_status(observer_state.last_event, fallback=result.status),
            completed_at=completed_at,
            subprocess_pid=_extract_subprocess_pid(observer_state.last_event),
            error_text=None if result.status == "success" else result.summary,
            termination_reason=None if result.status == "success" else "agent_run_unsuccessful",
        )
        return result


class _InstrumentedAgenticSession:
    def __init__(self, provider: InstrumentedAIProvider, session) -> None:
        self._provider = provider
        self._session = session

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        started_at = time.time()
        request_id = str(uuid4())
        self._provider._guard_test_execution()
        self._provider._enforce_rate_limits()
        try:
            result = await self._session.run_agent(prompt)
        except asyncio.CancelledError:
            completed_at = time.time()
            self._provider._usage_repository.record_call(
                task_name=self._provider._task_name,
                task_id=self._provider._task_id,
                request_id=request_id,
                subprocess_pid=None,
                provider_key=self._provider._provider_key,
                provider_name=self._provider._provider_name,
                model_name=self._provider._model_name,
                status="cancelled",
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
            )
            raise
        except Exception as exc:
            completed_at = time.time()
            classification = classify_provider_failure(exc)
            self._provider._usage_repository.record_call(
                task_name=self._provider._task_name,
                task_id=self._provider._task_id,
                request_id=request_id,
                subprocess_pid=None,
                provider_key=self._provider._provider_key,
                provider_name=self._provider._provider_name,
                model_name=self._provider._model_name,
                status="error",
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            raise
        completed_at = time.time()
        self._provider._usage_repository.record_call(
            task_name=self._provider._task_name,
            task_id=self._provider._task_id,
            request_id=request_id,
            subprocess_pid=None,
            provider_key=self._provider._provider_key,
            provider_name=self._provider._provider_name,
            model_name=self._provider._model_name,
            status=result.status,
            duration_seconds=max(completed_at - started_at, 0.0),
            created_at=completed_at,
            error_text=None if result.status == "success" else result.summary,
            reason_category=None if result.status == "success" else "execution",
            reason_code=None if result.status == "success" else "agent_run_unsuccessful",
            retry_delay_seconds=None,
        )
        return result

    async def close(self) -> None:
        await self._session.close()


def _safe_task_queue_update(task_queue, *, task_id: str, operation: str, callback: Callable[[], object]) -> None:
    try:
        callback()
    except ValueError as exc:
        if "is not running" not in str(exc):
            raise
        logger.debug(
            "Ignoring provider observer task-state update after task terminalization",
            extra={"task_id": task_id, "operation": operation},
        )


def _extract_subprocess_pid(event: ProviderObserverEvent | None) -> int | None:
    if event is None:
        return None
    return event.subprocess_pid


def _extract_attempt(event: ProviderObserverEvent | None) -> int:
    if event is None:
        return 1
    return event.attempt


def _extract_started_at(event: ProviderObserverEvent | None, *, fallback: float) -> float:
    if isinstance(
        event,
        (ProviderAttemptStartedEvent, ProviderAttemptHeartbeatEvent, ProviderAttemptFinishedEvent),
    ):
        return event.started_at
    return fallback


def _extract_completed_at(event: ProviderObserverEvent | None, *, fallback: float) -> float:
    if isinstance(event, ProviderAttemptFinishedEvent):
        return event.completed_at
    return fallback


def _extract_status(event: ProviderObserverEvent | None, *, fallback: str) -> str:
    if not isinstance(event, ProviderAttemptFinishedEvent):
        return fallback
    return event.status if event.status else fallback


def _extract_raw_text(event: ProviderObserverEvent | None) -> str | None:
    if not isinstance(event, ProviderAttemptFinishedEvent):
        return None
    return event.raw_text


def _notify_json_call(
    callback: Callable[[ProviderJSONCall], None] | None,
    result: ProviderJSONCall,
) -> None:
    if callback is None:
        return
    try:
        callback(result)
    except Exception:
        logger.exception("Instrumented provider JSON telemetry callback failed")

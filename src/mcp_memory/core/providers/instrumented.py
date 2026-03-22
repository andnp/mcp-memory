from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

from mcp_memory.core.provider_admission import build_provider_admission_exception
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.provider_admission import evaluate_provider_admission
from mcp_memory.core.provider_admission import should_persist_admission_backoff
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.provider_usage_store import ProviderUsageRepository


class InstrumentedAIProvider:
    def __init__(
        self,
        provider,
        *,
        usage_repository: ProviderUsageRepository,
        provider_key: str,
        provider_name: str,
        model_name: str,
        task_name: str | None = None,
        task_id: str | None = None,
        workspace_id: str | None = None,
        task_queue = None,
        budget_key: str | None = None,
        daily_call_limit: int | None = None,
        model_burst_call_limit: int | None = None,
        model_burst_window_seconds: float | None = None,
    ) -> None:
        self._provider = provider
        self._usage_repository = usage_repository
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._model_name = model_name
        self._task_name = task_name
        self._task_id = task_id
        self._workspace_id = workspace_id
        self._task_queue = task_queue
        self._budget_key = budget_key or provider_key
        self._daily_call_limit = daily_call_limit
        self._model_burst_call_limit = model_burst_call_limit
        self._model_burst_window_seconds = model_burst_window_seconds

    def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
        return InstrumentedAIProvider(
            self._provider,
            usage_repository=self._usage_repository,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            task_name=task_name,
            task_id=task_id,
            workspace_id=workspace_id,
            task_queue=self._task_queue,
            budget_key=self._budget_key,
            daily_call_limit=self._daily_call_limit,
            model_burst_call_limit=self._model_burst_call_limit,
            model_burst_window_seconds=self._model_burst_window_seconds,
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

    async def ask_json(self, prompt: str) -> dict:
        started_at = time.time()
        request_id = str(uuid4())
        last_event: dict | None = None

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

        def _observer(payload: dict) -> None:
            nonlocal last_event
            last_event = payload
            if payload.get("event") == "started" and self._task_queue is not None and self._task_id is not None:
                self._task_queue.set_running_process(
                    self._task_id,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                    request_id=request_id,
                )
            if payload.get("event") == "started":
                started_value = float(payload.get("started_at", started_at))
                self._usage_repository.record_conversation(
                    request_id=request_id,
                    attempt=int(payload.get("attempt", 1)),
                    task_name=self._task_name,
                    task_id=self._task_id,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                    model_name=self._model_name,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                    prompt_text=prompt,
                    response_text="",
                    parsed=None,
                    status="running",
                    error_text=None,
                    reason_category=None,
                    reason_code=None,
                    retry_delay_seconds=None,
                    started_at=started_value,
                    completed_at=started_value,
                )
                return
            if payload.get("event") == "heartbeat":
                heartbeat_at = float(payload.get("heartbeat_at", time.time()))
                if self._task_queue is not None and self._task_id is not None:
                    self._task_queue.touch_running_task(self._task_id, updated_at=heartbeat_at)
                self._usage_repository.touch_running_conversation(
                    request_id=request_id,
                    completed_at=heartbeat_at,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                )
                return
            if payload.get("event") != "finished":
                return
            self._usage_repository.record_conversation(
                request_id=request_id,
                attempt=int(payload.get("attempt", 1)),
                task_name=self._task_name,
                task_id=self._task_id,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                prompt_text=prompt,
                response_text=str(payload.get("raw_text", "")),
                parsed=_coerce_parsed(payload.get("parsed")),
                status=str(payload.get("status", "error")),
                error_text=_coerce_text(payload.get("error")),
                reason_category=None,
                reason_code=None,
                retry_delay_seconds=None,
                started_at=float(payload.get("started_at", started_at)),
                completed_at=float(payload.get("completed_at", time.time())),
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)

        provider = self._provider
        binder = getattr(provider, "with_observer", None)
        if callable(binder):
            provider = binder(_observer)
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
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="cancelled"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(last_event, fallback="cancelled"),
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
                completed_at=completed_at,
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
        except Exception as exc:
            completed_at = time.time()
            classification = classify_provider_failure(
                exc,
                event_status=_extract_status(last_event, fallback="error"),
                raw_text=_extract_raw_text(last_event),
            )
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="error"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(last_event, fallback="error"),
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
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
        self._usage_repository.clear_admission_state(provider_key=self._provider_key, model_name=self._model_name)
        self._usage_repository.record_call(
            task_name=self._task_name,
            task_id=self._task_id,
            request_id=request_id,
            subprocess_pid=_extract_subprocess_pid(last_event),
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status=_extract_status(last_event, fallback="success"),
            duration_seconds=max(time.time() - started_at, 0.0),
            created_at=time.time(),
            error_text=None,
            reason_category=None,
            reason_code=None,
            retry_delay_seconds=None,
        )
        if self._task_queue is not None and self._task_id is not None:
            self._task_queue.clear_running_process(self._task_id)
        return response

    async def ask(self, prompt: str) -> dict:
        return await self.ask_json(prompt)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        started_at = time.time()
        request_id = str(uuid4())
        last_event: dict | None = None

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

        def _observer(payload: dict) -> None:
            nonlocal last_event
            last_event = payload
            if payload.get("event") == "started" and self._task_queue is not None and self._task_id is not None:
                self._task_queue.set_running_process(
                    self._task_id,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                    request_id=request_id,
                )
            if payload.get("event") == "started":
                started_value = float(payload.get("started_at", started_at))
                self._usage_repository.record_conversation(
                    request_id=request_id,
                    attempt=int(payload.get("attempt", 1)),
                    task_name=self._task_name,
                    task_id=self._task_id,
                    provider_key=self._provider_key,
                    provider_name=self._provider_name,
                    model_name=self._model_name,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                    prompt_text=prompt,
                    response_text="",
                    parsed=None,
                    status="running",
                    error_text=None,
                    reason_category=None,
                    reason_code=None,
                    retry_delay_seconds=None,
                    started_at=started_value,
                    completed_at=started_value,
                )
                return
            if payload.get("event") == "heartbeat":
                heartbeat_at = float(payload.get("heartbeat_at", time.time()))
                if self._task_queue is not None and self._task_id is not None:
                    self._task_queue.touch_running_task(self._task_id, updated_at=heartbeat_at)
                self._usage_repository.touch_running_conversation(
                    request_id=request_id,
                    completed_at=heartbeat_at,
                    subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                )
                return
            if payload.get("event") != "finished":
                return
            self._usage_repository.record_conversation(
                request_id=request_id,
                attempt=int(payload.get("attempt", 1)),
                task_name=self._task_name,
                task_id=self._task_id,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                subprocess_pid=_coerce_pid(payload.get("subprocess_pid")),
                prompt_text=prompt,
                response_text=str(payload.get("raw_text", "")),
                parsed=_coerce_parsed(payload.get("parsed")),
                status=str(payload.get("status", "error")),
                error_text=_coerce_text(payload.get("error")),
                reason_category=None,
                reason_code=None,
                retry_delay_seconds=None,
                started_at=float(payload.get("started_at", started_at)),
                completed_at=float(payload.get("completed_at", time.time())),
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)

        provider = self._provider
        binder = getattr(provider, "with_observer", None)
        if callable(binder):
            provider = binder(_observer)
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
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="cancelled"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(last_event, fallback="cancelled"),
                error_text="Command cancelled",
                reason_category="cancellation",
                reason_code="provider_cancelled",
                retry_delay_seconds=None,
                completed_at=completed_at,
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
        except Exception as exc:
            completed_at = time.time()
            classification = classify_provider_failure(
                exc,
                event_status=_extract_status(last_event, fallback="error"),
                raw_text=_extract_raw_text(last_event),
            )
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="error"),
                duration_seconds=max(completed_at - started_at, 0.0),
                created_at=completed_at,
                error_text=classification.error_text,
                reason_category=classification.reason_category,
                reason_code=classification.reason_code,
                retry_delay_seconds=classification.retry_delay_seconds,
            )
            self._usage_repository.finalize_running_conversation(
                request_id=request_id,
                status=_extract_status(last_event, fallback="error"),
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
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
        self._usage_repository.clear_admission_state(provider_key=self._provider_key, model_name=self._model_name)
        self._usage_repository.record_call(
            task_name=self._task_name,
            task_id=self._task_id,
            request_id=request_id,
            subprocess_pid=_extract_subprocess_pid(last_event),
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status=_extract_status(last_event, fallback=result.status),
            duration_seconds=max(time.time() - started_at, 0.0),
            created_at=time.time(),
            error_text=None if result.status == "success" else result.summary,
            reason_category=None if result.status == "success" else "execution",
            reason_code=None if result.status == "success" else "agent_run_unsuccessful",
            retry_delay_seconds=None,
        )
        if self._task_queue is not None and self._task_id is not None:
            self._task_queue.clear_running_process(self._task_id)
        return result


def _coerce_pid(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _coerce_parsed(value: object) -> dict | None:
    return value if isinstance(value, dict) else None


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _extract_subprocess_pid(event: dict | None) -> int | None:
    if event is None:
        return None
    return _coerce_pid(event.get("subprocess_pid"))


def _extract_status(event: dict | None, *, fallback: str) -> str:
    if event is None:
        return fallback
    status = event.get("status")
    return str(status) if isinstance(status, str) and status else fallback


def _extract_raw_text(event: dict | None) -> str | None:
    if event is None:
        return None
    raw_text = event.get("raw_text")
    return raw_text if isinstance(raw_text, str) else None

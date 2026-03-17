from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, cast
from uuid import uuid4

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
        )

    async def ask_json(self, prompt: str) -> dict:
        started_at = time.time()
        request_id = str(uuid4())
        last_event: dict | None = None

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
                    started_at=started_value,
                    completed_at=started_value,
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
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="cancelled"),
                duration_seconds=max(time.time() - started_at, 0.0),
                created_at=time.time(),
                error_text="Command cancelled",
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
        except Exception as exc:
            self._usage_repository.record_call(
                task_name=self._task_name,
                task_id=self._task_id,
                request_id=request_id,
                subprocess_pid=_extract_subprocess_pid(last_event),
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status=_extract_status(last_event, fallback="error"),
                duration_seconds=max(time.time() - started_at, 0.0),
                created_at=time.time(),
                error_text=str(exc),
            )
            if self._task_queue is not None and self._task_id is not None:
                self._task_queue.clear_running_process(self._task_id)
            raise
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
        )
        if self._task_queue is not None and self._task_id is not None:
            self._task_queue.clear_running_process(self._task_id)
        return response

    async def ask(self, prompt: str) -> dict:
        return await self.ask_json(prompt)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        provider = self._provider
        run_agent = getattr(provider, "run_agent", None)
        if not callable(run_agent):
            raise RuntimeError("agentic_provider_not_configured")
        return await cast(Callable[[str], Awaitable[AgenticRunResult]], run_agent)(prompt)


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

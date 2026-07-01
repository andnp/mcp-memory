from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from copilot import CopilotClient
from copilot.session import PermissionHandler
from copilot.session_events import AssistantMessageData
from copilot.session_events import SessionErrorData

from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers._json_cli import PROVIDER_SUBPROCESS_HEARTBEAT_SECONDS
from mcp_memory.core.providers._json_cli import build_cli_failure_exception
from mcp_memory.core.providers._json_cli import chain_observers
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.providers.interfaces import ProviderAttemptFinishedEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptHeartbeatEvent
from mcp_memory.core.providers.interfaces import ProviderAttemptStartedEvent
from mcp_memory.core.providers.interfaces import ProviderObserver
from mcp_memory.core.providers.interfaces import ProviderObserverEvent

import copy
import logging

logger = logging.getLogger(__name__)


class CopilotSDKProvider:
    provider_name = "Copilot SDK"

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        cwd: str | None = None,
        observer: ProviderObserver | None = None,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._cwd = cwd
        self._observer = observer

    def with_observer(self, observer: ProviderObserver):
        clone = copy.copy(self)
        existing = getattr(self, "_observer", None)
        if existing is None:
            clone._observer = observer
        else:
            clone._observer = chain_observers(existing, observer)
        return clone

    async def ask_json(self, prompt: str) -> dict[str, Any]:
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt, attempt=attempt + 1)
            if response.success:
                assert response.parsed is not None
                return response.parsed
            last_error = response.error
            if attempt < self._max_retries:
                logger.warning(
                    "%s attempt %d failed: %s. Retrying...",
                    self.provider_name,
                    attempt + 1,
                    last_error,
                )
        raise build_cli_failure_exception(self.provider_name, self._max_retries + 1, last_error)

    async def ask(self, prompt: str) -> dict[str, Any]:
        return await self.ask_json(prompt)

    def _mcp_servers(self) -> dict[str, Any] | None:
        return None

    async def _execute(self, prompt: str, *, attempt: int) -> AIResponse:
        started_at = time.time()
        self._notify(ProviderAttemptStartedEvent(attempt=attempt, prompt=prompt, subprocess_pid=None, started_at=started_at))
        try:
            event = await self._send_with_heartbeat(prompt, started_at=started_at, attempt=attempt)
        except TimeoutError:
            response = AIResponse(raw_text="", parsed=None, error="Command timed out")
            self._finish(attempt=attempt, prompt=prompt, started_at=started_at, status="timeout", response=response)
            return response
        except Exception as exc:  # noqa: BLE001 - SDK raises plain Exception for connection/session failures
            response = AIResponse(raw_text="", parsed=None, error=str(exc))
            self._finish(attempt=attempt, prompt=prompt, started_at=started_at, status="error", response=response)
            return response

        response = _to_ai_response(event)
        self._finish(
            attempt=attempt,
            prompt=prompt,
            started_at=started_at,
            status="success" if response.success else ("error" if response.error and "did not contain" not in response.error else "parse_error"),
            response=response,
        )
        return response

    async def _send_with_heartbeat(self, prompt: str, *, started_at: float, attempt: int):
        async with CopilotClient(working_directory=self._cwd) as client:
            session = await client.create_session(
                model=self._model,
                on_permission_request=PermissionHandler.approve_all,
                mcp_servers=self._mcp_servers(),
            )
            try:
                deadline = time.monotonic() + self._timeout_seconds
                send_task = asyncio.ensure_future(session.send_and_wait(prompt, timeout=self._timeout_seconds))
                while True:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise TimeoutError
                    try:
                        return await asyncio.wait_for(
                            asyncio.shield(send_task),
                            timeout=min(PROVIDER_SUBPROCESS_HEARTBEAT_SECONDS, remaining_seconds),
                        )
                    except asyncio.TimeoutError:
                        if send_task.done():
                            return send_task.result()
                        heartbeat_at = time.time()
                        self._notify(
                            ProviderAttemptHeartbeatEvent(
                                attempt=attempt,
                                prompt=prompt,
                                subprocess_pid=None,
                                started_at=started_at,
                                heartbeat_at=heartbeat_at,
                                elapsed_seconds=max(heartbeat_at - started_at, 0.0),
                            )
                        )
            finally:
                await session.disconnect()

    def _finish(self, *, attempt: int, prompt: str, started_at: float, status: str, response: AIResponse) -> None:
        completed_at = time.time()
        self._notify(
            ProviderAttemptFinishedEvent(
                attempt=attempt,
                prompt=prompt,
                subprocess_pid=None,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=max(completed_at - started_at, 0.0),
                status=status,
                returncode=None,
                raw_text=response.raw_text,
                parsed=response.parsed,
                error_text=response.error,
            )
        )

    def _notify(self, payload: ProviderObserverEvent) -> None:
        observer = getattr(self, "_observer", None)
        if observer is None:
            return
        try:
            observer(payload)
        except Exception:
            logger.exception("Provider observer failed")


class CopilotSDKAgenticProvider(CopilotSDKProvider):
    provider_name = "Copilot SDK Agentic"
    _DEFAULT_INTERNAL_TOOL_NAMES = (
        "task_complete",
        "internal_task_complete",
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_list_memory_records",
        "internal_get_next_dedup_batch",
        "internal_get_next_curator_batch",
        "internal_get_compatible_work_batch",
        "internal_get_next_ingest_batch",
        "internal_heartbeat_work_item",
        "internal_complete_work_item",
        "internal_defer_work_item",
        "internal_release_work_item",
        "internal_ingest_append_memory",
        "internal_ingest_create_memory",
        "internal_append_memory_content",
        "internal_archive_memory_record",
        "internal_merge_memory_into_canonical",
        "internal_split_memory_record",
        "internal_create_memory_record",
        "internal_update_memory_record",
        "internal_delete_memory_record",
        "internal_create_memory_link",
        "internal_delete_memory_link",
    )

    def __init__(
        self,
        *,
        model: str = "gpt-5.4-mini",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        cwd: str | None = None,
        observer: ProviderObserver | None = None,
        internal_server_name: str = "mcp-memory-internal",
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(model=model, timeout_seconds=timeout_seconds, max_retries=max_retries, cwd=cwd, observer=observer)
        self._internal_server_name = internal_server_name
        self._allowed_tool_names = tuple(allowed_tool_names or self._DEFAULT_INTERNAL_TOOL_NAMES)

    def _mcp_servers(self) -> dict[str, Any]:
        workspace_root = self._cwd or str(Path.cwd())
        return {
            self._internal_server_name: {
                "type": "stdio",
                "command": "uv",
                "args": ["run", "mcp-memory", "internal-run", "--workspace-root", workspace_root],
                "cwd": workspace_root,
                "tools": list(self._allowed_tool_names),
                "timeout": int(max(self._timeout_seconds, 1.0) * 1000),
            }
        }

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        last_response: AIResponse | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt, attempt=attempt + 1)
            last_response = response
            if response.success:
                return AgenticRunResult(
                    status="success",
                    summary=_extract_summary(response.parsed),
                    raw_text=response.raw_text,
                    parsed=response.parsed,
                    subprocess_pid=None,
                )
        assert last_response is not None
        raise build_cli_failure_exception(self.provider_name, self._max_retries + 1, last_response.error)


def _to_ai_response(event: Any) -> AIResponse:
    if event is None:
        return AIResponse(raw_text="", parsed=None, error="No assistant message received")
    if isinstance(event.data, SessionErrorData):
        return AIResponse(raw_text="", parsed=None, error=f"{event.data.error_type}: {event.data.message}")
    if not isinstance(event.data, AssistantMessageData):
        return AIResponse(raw_text="", parsed=None, error="No assistant message content found")
    content = event.data.content
    parsed = _extract_json_object(content)
    if parsed is None:
        return AIResponse(raw_text=content, parsed=None, error="Assistant message did not contain a JSON object")
    return AIResponse(raw_text=content, parsed=parsed)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_summary(parsed: dict[str, Any] | None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    for key in ("summary", "result", "response"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

from __future__ import annotations

import asyncio
from contextlib import suppress
import inspect
import logging
import os
import time as _time_module
from typing import Any, Callable, cast
from uuid import uuid4

from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import CallToolRequestParams, CallToolResult, ListToolsResult, PaginatedRequestParams, TextContent, Tool

from mcp_memory.client_clock import (
    remaining_suspend_aware_seconds,
    suspend_aware_deadline,
    suspend_aware_now,
)
from mcp_memory.config import resolve_workspace_root


def request_daemon_json(*args, **kwargs):
    from mcp_memory.daemon_transport import request_daemon_json as _request_daemon_json

    return _request_daemon_json(*args, **kwargs)


def ensure_daemon_started():
    from mcp_memory.daemon import ensure_daemon_started as _ensure_daemon_started

    return _ensure_daemon_started()


def stop_daemon():
    from mcp_memory.daemon import stop_daemon as _stop_daemon

    return _stop_daemon()


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

logger = logging.getLogger(__name__)
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"
_REQUEST_SESSION_ID_KEY = "__session_id"
_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS = 60.0
_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_POLL_SLICE_SECONDS = 0.1
_HOOK_TRANSPORT_HEALTH_PROBE_TIMEOUT_SECONDS = 0.2
_REQUEST_RECOVERY_RETRY_COUNT = 2
_SUSPEND_RESUME_CHECK_INTERVAL_SECONDS = 1.0
_SUSPEND_RESUME_DETECT_THRESHOLD_SECONDS = 2.0
_DAEMON_HEALTH_POLL_INTERVAL_SECONDS = 10.0
_DAEMON_HEALTH_PROBE_TIMEOUT_SECONDS = 2.0


class _DaemonRequestTimeout(Exception):
    """Internal sentinel for sync daemon timeouts surfaced from the worker thread."""


class _ClientRequestTimeout(Exception):
    """Internal sentinel for overall client timeout budget expiry."""


class MCPServer:
    def __init__(
        self,
        workspace_root: str | None = None,
        *,
        server_name: str = "mcp-memory",
        tool_path_prefix: str = "/internal/tools",
    ):
        self.workspace_root = workspace_root or str(resolve_workspace_root())
        self.server = Server(
            server_name,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )
        self._daemon: object | None = None
        self._daemon_recovery_task: asyncio.Task[None] | None = None
        self._tool_path_prefix = tool_path_prefix
        self._session_id: str | None = None
        self._session_started = False
        self._suspend_monitor_task: asyncio.Task[None] | None = None
        self._health_monitor_task: asyncio.Task[None] | None = None

    async def _list_tools(
        self,
        _context: ServerRequestContext[Any],
        _params: PaginatedRequestParams | None,
    ) -> ListToolsResult:
        payload = await self._request_daemon_json_with_client_timeout(self._tool_path_prefix, None)
        _raise_for_daemon_error_payload(payload)
        tools = payload.get("tools")
        if not isinstance(tools, list):
            raise ValueError("daemon_response_missing_tools")
        return ListToolsResult(
            tools=[
                Tool(
                    name=str(tool["name"]),
                    description=cast(str | None, tool.get("description")),
                    input_schema=cast(dict[str, object], tool["inputSchema"]),
                )
                for tool in tools
                if isinstance(tool, dict)
            ]
        )

    async def _call_tool(
        self,
        _context: ServerRequestContext[Any],
        params: CallToolRequestParams,
    ) -> CallToolResult:
        try:
            payload = await self._request_daemon_json_with_client_timeout(
                f"{self._tool_path_prefix}/{params.name}",
                params.arguments or {},
            )
            _raise_for_daemon_error_payload(payload)
            contents = payload.get("contents")
            if not isinstance(contents, list):
                raise ValueError("daemon_response_missing_contents")
            return CallToolResult(
                content=[
                    TextContent(type="text", text=str(item["text"]))
                    for item in contents
                    if isinstance(item, dict)
                ]
            )
        except (RuntimeError, TimeoutError) as exc:
            return CallToolResult(
                content=[TextContent(type="text", text=str(exc))],
                is_error=True,
            )

    async def _request_daemon_json_with_client_timeout(
        self,
        path: str,
        payload: dict | None,
    ) -> dict[str, object]:
        timeout_deadline = suspend_aware_deadline(_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS)

        async def request_with_wrapped_timeout() -> dict[str, object]:
            request_task = asyncio.create_task(
                asyncio.to_thread(
                    self._call_request_json_with_recovery,
                    path,
                    payload,
                    timeout_deadline=timeout_deadline,
                )
            )
            try:
                return await _await_suspend_aware_task_before_deadline(
                    request_task,
                    timeout_deadline=timeout_deadline,
                )
            except TimeoutError as exc:
                if not request_task.done():
                    request_task.cancel()
                if str(exc) == "mcp_client_request_timed_out":
                    raise _ClientRequestTimeout from exc
                raise _DaemonRequestTimeout from exc

        try:
            response = await request_with_wrapped_timeout()
            return response
        except _ClientRequestTimeout as exc:
            self._daemon = None
            self._start_background_daemon_recovery()
            raise TimeoutError("mcp_client_request_timed_out") from exc.__cause__
        except _DaemonRequestTimeout as exc:
            cause = exc.__cause__
            if isinstance(cause, TimeoutError):
                raise cause from None
            raise RuntimeError("daemon_request_timeout_wrapping_failed") from exc

    def _call_request_json_with_recovery(
        self,
        path: str,
        payload: dict | None,
        *,
        timeout_deadline: float,
    ) -> dict[str, object]:
        recovery_method = self._request_json_with_recovery
        if _callable_accepts_timeout_deadline(recovery_method):
            return recovery_method(path, payload, timeout_deadline=timeout_deadline)
        return recovery_method(path, payload)

    def _start_background_daemon_recovery(self) -> None:
        if self._daemon_recovery_task is not None and not self._daemon_recovery_task.done():
            return
        recovery_task = asyncio.create_task(self._refresh_daemon_metadata_in_background())
        self._daemon_recovery_task = recovery_task
        recovery_task.add_done_callback(self._clear_daemon_recovery_task)

    async def _refresh_daemon_metadata_in_background(self) -> None:
        try:
            self._daemon = await asyncio.to_thread(ensure_daemon_started)
        except Exception as exc:  # pragma: no cover - exercised via log assertion paths if needed
            logger.warning(
                "Background daemon metadata recovery failed after client timeout",
                extra={
                    "workspace_root": self.workspace_root,
                    "error": str(exc),
                },
            )

    def _clear_daemon_recovery_task(self, recovery_task: asyncio.Task[None]) -> None:
        if self._daemon_recovery_task is recovery_task:
            self._daemon_recovery_task = None

    async def _monitor_suspend_resume(self) -> None:
        last_boottime = suspend_aware_now()
        last_monotonic = _time_module.monotonic()
        while True:
            try:
                await asyncio.sleep(_SUSPEND_RESUME_CHECK_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            now_boottime = suspend_aware_now()
            now_monotonic = _time_module.monotonic()
            boottime_elapsed = now_boottime - last_boottime
            monotonic_elapsed = now_monotonic - last_monotonic
            last_boottime = now_boottime
            last_monotonic = now_monotonic
            if boottime_elapsed - monotonic_elapsed > _SUSPEND_RESUME_DETECT_THRESHOLD_SECONDS:
                logger.warning(
                    "Suspend/resume detected; invalidating daemon connection and scheduling recovery",
                    extra={
                        "boottime_elapsed_seconds": round(boottime_elapsed, 1),
                        "monotonic_elapsed_seconds": round(monotonic_elapsed, 3),
                    },
                )
                self._daemon = None
                self._start_background_daemon_recovery()

    async def _monitor_daemon_health(self) -> None:
        while True:
            try:
                await asyncio.sleep(_DAEMON_HEALTH_POLL_INTERVAL_SECONDS)
            except asyncio.CancelledError:
                return
            daemon = self._daemon
            if daemon is None:
                continue
            try:
                await asyncio.to_thread(
                    request_daemon_json,
                    daemon,
                    "/internal/health",
                    None,
                    timeout_seconds=_DAEMON_HEALTH_PROBE_TIMEOUT_SECONDS,
                )
            except (OSError, TimeoutError, ValueError):
                logger.warning(
                    "Daemon health probe failed; invalidating cached connection and scheduling recovery",
                )
                self._daemon = None
                self._start_background_daemon_recovery()

    async def run(self) -> None:
        logger.info("Initializing MCP Memory Server proxy...")
        self._session_id = str(uuid4())
        self._suspend_monitor_task = asyncio.create_task(self._monitor_suspend_resume())
        self._health_monitor_task = asyncio.create_task(self._monitor_daemon_health())
        # Start daemon in background so stdio_server starts immediately.
        # This prevents VSCode's MCP initialize from timing out when the daemon is slow
        # to start (e.g., when the database is unreachable after a network change or
        # sleep/wake cycle).
        self._start_background_daemon_recovery()
        session_start_task = asyncio.create_task(self._register_session_start())
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        finally:
            if self._suspend_monitor_task is not None:
                self._suspend_monitor_task.cancel()
                try:
                    await self._suspend_monitor_task
                except asyncio.CancelledError:
                    pass
                self._suspend_monitor_task = None
            if self._health_monitor_task is not None:
                self._health_monitor_task.cancel()
                try:
                    await self._health_monitor_task
                except asyncio.CancelledError:
                    pass
                self._health_monitor_task = None
            # Give asynchronous startup a short window to register this proxy
            # before sending session-end. This keeps normal MCP startup
            # responsive while ensuring quick stdio disconnects still clean up.
            if not session_start_task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(session_start_task), timeout=5.0)
                except Exception:
                    session_start_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await session_start_task
            if self._session_started:
                await asyncio.to_thread(self._send_session_hook, "session-end")

    async def _register_session_start(self) -> None:
        """Register this stdio proxy once its background daemon is available."""
        recovery_task = self._daemon_recovery_task
        if recovery_task is not None and not recovery_task.done():
            try:
                await asyncio.shield(recovery_task)
            except Exception:
                return
        if self._daemon is None:
            return
        self._session_started = await asyncio.to_thread(self._send_session_hook, "session-start")

    def _request_json(
        self,
        path: str,
        payload: dict | None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, object]:
        if self._daemon is None:
            raise RuntimeError("daemon_not_started")
        request_payload = self._request_payload(payload)
        response = request_daemon_json(self._daemon, path, request_payload, timeout_seconds=timeout_seconds)
        if not isinstance(response, dict):
            raise ValueError("daemon_response_must_be_object")
        return cast(dict[str, object], response)

    def _request_payload(self, payload: dict | None) -> dict | None:
        request_payload = {} if payload is None else dict(payload)
        if self.workspace_root is not None:
            request_payload.setdefault(_REQUEST_WORKSPACE_ROOT_KEY, self.workspace_root)
        if self._session_id is not None:
            request_payload.setdefault(_REQUEST_SESSION_ID_KEY, self._session_id)
        return request_payload or None

    def _request_json_with_recovery(
        self,
        path: str,
        payload: dict | None,
        *,
        timeout_deadline: float | None = None,
    ) -> dict[str, object]:
        max_attempts = _REQUEST_RECOVERY_RETRY_COUNT + 1
        for attempt_index in range(_REQUEST_RECOVERY_RETRY_COUNT + 1):
            remaining_timeout_seconds = _remaining_client_timeout_seconds(timeout_deadline)
            try:
                return self._call_request_json(
                    path,
                    payload,
                    timeout_seconds=remaining_timeout_seconds,
                )
            except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
                if not _is_recoverable_daemon_request_error(exc):
                    raise
                if timeout_deadline is not None and remaining_suspend_aware_seconds(timeout_deadline) <= 0.0:
                    raise TimeoutError("mcp_client_request_timed_out") from exc
                if attempt_index >= _REQUEST_RECOVERY_RETRY_COUNT:
                    raise
                logger.warning(
                    "Daemon request failed; refreshing daemon metadata before retry",
                    extra={
                        "path": path,
                        "attempt": attempt_index + 1,
                        "max_attempts": max_attempts,
                        "error": str(exc),
                    },
                )
                self._daemon = ensure_daemon_started()
        raise RuntimeError("daemon_request_retries_exhausted")

    def _call_request_json(
        self,
        path: str,
        payload: dict | None,
        *,
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        request_method = self._request_json
        if _callable_accepts_timeout_seconds(request_method):
            return request_method(path, payload, timeout_seconds=timeout_seconds)
        return request_method(path, payload)

    def _send_session_hook(self, event_name: str) -> bool:
        if self._session_id is None:
            return False
        try:
            self._request_json_with_recovery(
                f"/api/hooks/{event_name}",
                {
                    "session_id": self._session_id,
                    "source": "mcp-stdio",
                    "workspace_root": self.workspace_root,
                },
            )
            return True
        except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
            transport_health_snapshot = self._probe_transport_health_snapshot()
            logger.warning(
                "Failed to send %s hook for session %s: %s",
                event_name,
                self._session_id,
                exc,
                extra={
                    "event_name": event_name,
                    "session_id": self._session_id,
                    "transport_health_snapshot": transport_health_snapshot,
                },
            )
            return False

    def _probe_transport_health_snapshot(self) -> dict[str, Any] | None:
        if self._daemon is None:
            return None
        try:
            payload = request_daemon_json(
                self._daemon,
                "/internal/health",
                None,
                timeout_seconds=_HOOK_TRANSPORT_HEALTH_PROBE_TIMEOUT_SECONDS,
            )
        except (OSError, TimeoutError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload


def _is_recoverable_daemon_request_error(exc: Exception) -> bool:
    if isinstance(exc, (OSError, TimeoutError, ValueError)):
        return True
    return isinstance(exc, RuntimeError) and str(exc) == "daemon_not_started"


def _raise_for_daemon_error_payload(payload: dict[str, object]) -> None:
    status = payload.get("status")
    if status != "error":
        return
    error = payload.get("error")
    error_text = str(error) if error is not None else "daemon_request_failed"
    if error_text == "daemon_request_timed_out":
        raise TimeoutError(error_text)
    raise RuntimeError(error_text)


def _remaining_client_timeout_seconds(timeout_deadline: float | None) -> float | None:
    if timeout_deadline is None:
        return None
    remaining_seconds = remaining_suspend_aware_seconds(timeout_deadline)
    if remaining_seconds <= 0.0:
        raise TimeoutError("mcp_client_request_timed_out")
    return remaining_seconds


def _callable_accepts_timeout_deadline(func: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True

    timeout_parameter = signature.parameters.get("timeout_deadline")
    return timeout_parameter is not None


def _callable_accepts_timeout_seconds(func: Callable[..., object]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True

    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True

    timeout_parameter = signature.parameters.get("timeout_seconds")
    return timeout_parameter is not None


async def _await_suspend_aware_task_before_deadline(
    task: asyncio.Task[dict[str, object]],
    *,
    timeout_deadline: float,
) -> dict[str, object]:
    while True:
        remaining_seconds = remaining_suspend_aware_seconds(timeout_deadline)
        if remaining_seconds <= 0.0:
            raise TimeoutError("mcp_client_request_timed_out")
        done, _pending = await asyncio.wait(
            {task},
            timeout=min(remaining_seconds, _DAEMON_BACKED_MCP_CLIENT_TIMEOUT_POLL_SLICE_SECONDS),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if task in done:
            return await task

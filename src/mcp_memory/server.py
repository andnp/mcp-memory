from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast
from uuid import uuid4

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_memory.daemon import ensure_daemon_started, stop_daemon
from mcp_memory.daemon_transport import request_daemon_json


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

logger = logging.getLogger(__name__)
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"
_REQUEST_SESSION_ID_KEY = "__session_id"
_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS = 30.0
_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_ESCALATION_THRESHOLD = 2
_HOOK_TRANSPORT_HEALTH_PROBE_TIMEOUT_SECONDS = 0.2
_REQUEST_RECOVERY_RETRY_COUNT = 2


class _DaemonRequestTimeout(Exception):
    """Internal sentinel for sync daemon timeouts surfaced from the worker thread."""


class MCPServer:
    def __init__(
        self,
        workspace_root: str | None = None,
        *,
        server_name: str = "mcp-memory",
        tool_path_prefix: str = "/internal/tools",
    ):
        self.workspace_root = workspace_root
        self.server = Server(server_name)
        self._daemon: object | None = None
        self._daemon_recovery_task: asyncio.Task[None] | None = None
        self._daemon_recovery_force_restart_requested = False
        self._consecutive_client_timeout_failures = 0
        self._tool_path_prefix = tool_path_prefix
        self._session_id: str | None = None
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            payload = await self._request_daemon_json_with_client_timeout(self._tool_path_prefix, None)
            tools = payload.get("tools")
            if not isinstance(tools, list):
                raise ValueError("daemon_response_missing_tools")
            return [
                Tool(
                    name=str(tool["name"]),
                    description=cast(str | None, tool.get("description")),
                    inputSchema=cast(dict[str, object], tool["inputSchema"]),
                )
                for tool in tools
                if isinstance(tool, dict)
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            payload = await self._request_daemon_json_with_client_timeout(
                f"{self._tool_path_prefix}/{name}",
                arguments,
            )
            contents = payload.get("contents")
            if not isinstance(contents, list):
                raise ValueError("daemon_response_missing_contents")
            return [
                TextContent(type="text", text=str(item["text"]))
                for item in contents
                if isinstance(item, dict)
            ]

    async def _request_daemon_json_with_client_timeout(
        self,
        path: str,
        payload: dict | None,
    ) -> dict[str, object]:
        async def request_with_wrapped_timeout() -> dict[str, object]:
            try:
                return await asyncio.to_thread(self._request_json_with_recovery, path, payload)
            except TimeoutError as exc:
                raise _DaemonRequestTimeout from exc

        try:
            response = await asyncio.wait_for(
                request_with_wrapped_timeout(),
                timeout=_DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS,
            )
            self._consecutive_client_timeout_failures = 0
            return response
        except asyncio.TimeoutError as exc:
            self._consecutive_client_timeout_failures += 1
            self._daemon = None
            self._start_background_daemon_recovery(
                force_restart=(
                    self._consecutive_client_timeout_failures
                    >= _DAEMON_BACKED_MCP_CLIENT_TIMEOUT_ESCALATION_THRESHOLD
                )
            )
            raise TimeoutError("mcp_client_request_timed_out") from exc
        except _DaemonRequestTimeout as exc:
            cause = exc.__cause__
            if isinstance(cause, TimeoutError):
                raise cause from None
            raise RuntimeError("daemon_request_timeout_wrapping_failed") from exc

    def _start_background_daemon_recovery(self, *, force_restart: bool = False) -> None:
        self._daemon_recovery_force_restart_requested = (
            self._daemon_recovery_force_restart_requested or force_restart
        )
        if self._daemon_recovery_task is not None and not self._daemon_recovery_task.done():
            return
        recovery_task = asyncio.create_task(self._refresh_daemon_metadata_in_background())
        self._daemon_recovery_task = recovery_task
        recovery_task.add_done_callback(self._clear_daemon_recovery_task)

    async def _refresh_daemon_metadata_in_background(self) -> None:
        try:
            while True:
                force_restart = self._daemon_recovery_force_restart_requested
                self._daemon_recovery_force_restart_requested = False
                if force_restart:
                    await asyncio.to_thread(stop_daemon, self.workspace_root, None)
                self._daemon = await asyncio.to_thread(ensure_daemon_started, self.workspace_root, None)
                if not self._daemon_recovery_force_restart_requested:
                    break
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

    async def run(self) -> None:
        logger.info("Initializing MCP Memory Server proxy...")
        self._daemon = await asyncio.to_thread(ensure_daemon_started, self.workspace_root, None)
        self._session_id = str(uuid4())
        await asyncio.to_thread(self._send_session_hook, "session-start")
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options(),
                )
        finally:
            await asyncio.to_thread(self._send_session_hook, "session-end")

    def _request_json(self, path: str, payload: dict | None) -> dict[str, object]:
        if self._daemon is None:
            raise RuntimeError("daemon_not_started")
        request_payload = self._request_payload(payload)
        response = request_daemon_json(self._daemon, path, request_payload)
        if not isinstance(response, dict):
            raise ValueError("daemon_response_must_be_object")
        return cast(dict[str, object], response)

    def _request_payload(self, payload: dict | None) -> dict | None:
        request_payload = None if payload is None else dict(payload)
        if request_payload is not None:
            if self.workspace_root is not None:
                request_payload.setdefault(_REQUEST_WORKSPACE_ROOT_KEY, self.workspace_root)
            if self._session_id is not None:
                request_payload.setdefault(_REQUEST_SESSION_ID_KEY, self._session_id)
        return request_payload

    def _request_json_with_recovery(self, path: str, payload: dict | None) -> dict[str, object]:
        max_attempts = _REQUEST_RECOVERY_RETRY_COUNT + 1
        for attempt_index in range(_REQUEST_RECOVERY_RETRY_COUNT + 1):
            try:
                return self._request_json(path, payload)
            except (OSError, TimeoutError, ValueError, RuntimeError) as exc:
                if not _is_recoverable_daemon_request_error(exc):
                    raise
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
                self._daemon = ensure_daemon_started(self.workspace_root, None)
        raise RuntimeError("daemon_request_retries_exhausted")

    def _send_session_hook(self, event_name: str) -> None:
        if self._session_id is None:
            return
        try:
            self._request_json_with_recovery(
                f"/api/hooks/{event_name}",
                {
                    "session_id": self._session_id,
                    "source": "mcp-stdio",
                    "workspace_root": self.workspace_root,
                },
            )
        except (OSError, TimeoutError, ValueError) as exc:
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

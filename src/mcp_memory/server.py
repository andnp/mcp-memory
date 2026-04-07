from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast
from uuid import uuid4

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from mcp_memory.daemon import ensure_daemon_started
from mcp_memory.daemon_transport import request_daemon_json


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

logger = logging.getLogger(__name__)
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"
_REQUEST_SESSION_ID_KEY = "__session_id"
_HOOK_TRANSPORT_HEALTH_PROBE_TIMEOUT_SECONDS = 0.2
_REQUEST_RECOVERY_RETRY_COUNT = 2


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
        self._tool_path_prefix = tool_path_prefix
        self._session_id: str | None = None
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            payload = await asyncio.to_thread(self._request_json_with_recovery, self._tool_path_prefix, None)
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
            payload = await asyncio.to_thread(
                self._request_json_with_recovery,
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
            except (OSError, TimeoutError, ValueError) as exc:
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

from __future__ import annotations

import asyncio
import logging
import os
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
        self._daemon = None
        self._tool_path_prefix = tool_path_prefix
        self._session_id: str | None = None
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            payload = await asyncio.to_thread(self._request_json, self._tool_path_prefix, None)
            return [
                Tool(
                    name=tool["name"],
                    description=tool["description"],
                    inputSchema=tool["inputSchema"],
                )
                for tool in payload["tools"]
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            payload = await asyncio.to_thread(
                self._request_json,
                f"{self._tool_path_prefix}/{name}",
                arguments,
            )
            return [TextContent(type=item["type"], text=item["text"]) for item in payload["contents"]]

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

    def _request_json(self, path: str, payload: dict | None):
        if self._daemon is None:
            raise RuntimeError("daemon_not_started")
        request_payload = None if payload is None else dict(payload)
        if request_payload is not None and self.workspace_root is not None:
            request_payload.setdefault(_REQUEST_WORKSPACE_ROOT_KEY, self.workspace_root)
        return request_daemon_json(self._daemon, path, request_payload, timeout_seconds=5)

    def _send_session_hook(self, event_name: str) -> None:
        if self._session_id is None:
            return
        try:
            self._request_json(
                f"/api/hooks/{event_name}",
                {
                    "session_id": self._session_id,
                    "source": "mcp-stdio",
                    "workspace_root": self.workspace_root,
                },
            )
        except (OSError, TimeoutError, ValueError) as exc:
            logger.warning("Failed to send %s hook for session %s: %s", event_name, self._session_id, exc)

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from urllib.request import Request, urlopen

import zmq
import zmq.asyncio

from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools


DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 5.0


def request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds: float = DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS):
    socket_path = getattr(metadata, "socket_path", None)
    transport = getattr(metadata, "transport", "http")
    if isinstance(socket_path, str) and socket_path and transport in {"zmq", "hybrid"}:
        return _request_zmq_json(socket_path, path, payload, timeout_seconds=timeout_seconds)
    return _request_http_json(metadata.base_url, path, payload, timeout_seconds=timeout_seconds)


def _request_http_json(base_url: str, path: str, payload: dict | None, *, timeout_seconds: float):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    method = "GET" if body is None else "POST"
    request = Request(
        f"{base_url}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _request_zmq_json(socket_path: str, path: str, payload: dict | None, *, timeout_seconds: float):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.linger = 0
    timeout_ms = max(int(timeout_seconds * 1000), 1)
    socket.rcvtimeo = timeout_ms
    socket.sndtimeo = timeout_ms
    socket.connect(_socket_endpoint(socket_path))
    try:
        socket.send_json({"path": path, "payload": payload})
        response = socket.recv_json()
    finally:
        socket.close(0)
    if not isinstance(response, dict):
        raise ValueError("daemon_response_must_be_object")
    return response


class DaemonZmqServer:
    def __init__(
        self,
        *,
        context_factory,
        hook_handlers: dict[str, object],
        socket_path: Path,
        metadata_provider,
    ) -> None:
        self._context_factory = context_factory
        self._hook_handlers = hook_handlers
        self._socket_path = socket_path
        self._metadata_provider = metadata_provider
        self._context = zmq.asyncio.Context.instance()
        self._socket: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        _remove_stale_socket(self._socket_path)
        socket = self._context.socket(zmq.ROUTER)
        socket.linger = 0
        socket.bind(_socket_endpoint(str(self._socket_path)))
        self._socket = socket
        try:
            os.chmod(self._socket_path, 0o600)
        except OSError:
            pass
        self._task = asyncio.create_task(self._serve())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        _remove_stale_socket(self._socket_path)

    async def _serve(self) -> None:
        assert self._socket is not None
        while True:
            try:
                frames = await self._socket.recv_multipart()
            except asyncio.CancelledError:
                return
            if len(frames) < 2:
                continue
            identity = frames[0]
            payload_frame = frames[-1]
            response = await self._dispatch(payload_frame)
            await self._socket.send_multipart([identity, json.dumps(response, sort_keys=True).encode("utf-8")])

    async def _dispatch(self, payload_frame: bytes) -> dict:
        try:
            decoded = json.loads(payload_frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {"status": "error", "error": "invalid_transport_payload"}
        if not isinstance(decoded, dict):
            return {"status": "error", "error": "invalid_transport_payload"}

        path = decoded.get("path")
        payload = decoded.get("payload")
        if not isinstance(path, str) or not path:
            return {"status": "error", "error": "transport_path_required"}
        if payload is not None and not isinstance(payload, dict):
            return {"status": "error", "error": "transport_payload_must_be_object"}

        request_payload = {} if payload is None else payload
        if path == "/internal/health":
            return asdict(self._metadata_provider())
        if path == "/internal/tools":
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in get_memory_tools()
                ]
            }
        if path == "/internal/maintenance/tools":
            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in get_internal_maintenance_tools()
                ]
            }
        if path.startswith("/internal/tools/"):
            name = path.rsplit("/", 1)[-1]
            response = await call_memory_tool(self._context_factory(request_payload), name, request_payload)
            return {
                "contents": [
                    {
                        "type": content.type,
                        "text": content.text,
                    }
                    for content in response
                ]
            }
        if path.startswith("/internal/maintenance/tools/"):
            name = path.rsplit("/", 1)[-1]
            response = await call_internal_memory_tool(self._context_factory(request_payload), name, request_payload)
            return {
                "contents": [
                    {
                        "type": content.type,
                        "text": content.text,
                    }
                    for content in response
                ]
            }
        hook_handler = self._hook_handlers.get(path)
        if callable(hook_handler):
            return await hook_handler(request_payload)
        return {"status": "error", "error": "unknown_transport_path", "path": path}


def _socket_endpoint(socket_path: str) -> str:
    return f"ipc://{socket_path}"


def _remove_stale_socket(socket_path: Path) -> None:
    try:
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return
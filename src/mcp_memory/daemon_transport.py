from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Awaitable, Callable, cast

import zmq
import zmq.asyncio

from mcp_memory.daemon_dispatch import (
    dispatch_management_request,
    error_payload,
    normalize_request,
    serialize_tool,
    serialize_tool_response,
)
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools


DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 5.0


def request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds: float = DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS):
    socket_path = getattr(metadata, "socket_path", None)
    transport = getattr(metadata, "transport", "zmq")
    if transport not in {"zmq", "hybrid"}:
        raise ValueError("unsupported_daemon_transport")
    if not isinstance(socket_path, str) or not socket_path:
        raise ValueError("daemon_socket_path_required")
    return _request_zmq_json(socket_path, path, payload, timeout_seconds=timeout_seconds)


def probe_daemon_socket(socket_path: str | Path, *, timeout_seconds: float = 0.1) -> bool:
    try:
        _request_zmq_json(str(socket_path), "/internal/health", None, timeout_seconds=timeout_seconds)
    except (OSError, TimeoutError, json.JSONDecodeError, ValueError):
        return False
    return True


def remove_daemon_socket(socket_path: str | Path) -> None:
    _remove_stale_socket(Path(socket_path))


def _request_zmq_json(socket_path: str, path: str, payload: dict | None, *, timeout_seconds: float):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)
    socket.linger = 0
    timeout_ms = max(int(timeout_seconds * 1000), 1)
    socket.rcvtimeo = timeout_ms
    socket.sndtimeo = timeout_ms
    socket.connect(_socket_endpoint(socket_path))
    try:
        try:
            socket.send_json({"path": path, "payload": payload})
            response = socket.recv_json()
        except zmq.error.Again as exc:
            raise TimeoutError("daemon_request_timed_out") from exc
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
        routes_provider,
        socket_path: Path,
        metadata_provider,
    ) -> None:
        self._context_factory = context_factory
        self._hook_handlers = hook_handlers
        self._routes_provider = routes_provider
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

        normalized_path, request_payload = normalize_request(path, payload)
        try:
            if normalized_path == "/internal/health":
                return asdict(self._metadata_provider())
            if normalized_path == "/internal/tools":
                return {"tools": [serialize_tool(tool) for tool in get_memory_tools()]}
            if normalized_path == "/internal/maintenance/tools":
                return {"tools": [serialize_tool(tool) for tool in get_internal_maintenance_tools()]}
            if normalized_path.startswith("/internal/tools/"):
                name = normalized_path.rsplit("/", 1)[-1]
                response = await call_memory_tool(self._context_factory(request_payload), name, request_payload)
                return serialize_tool_response(response)
            if normalized_path.startswith("/internal/maintenance/tools/"):
                name = normalized_path.rsplit("/", 1)[-1]
                response = await call_internal_memory_tool(self._context_factory(request_payload), name, request_payload)
                return serialize_tool_response(response)
            hook_handler = self._hook_handlers.get(normalized_path)
            if callable(hook_handler):
                typed_handler = cast(Callable[[dict[str, object]], Awaitable[dict]], hook_handler)
                return await typed_handler(request_payload)
            return await self._dispatch_management_request(normalized_path, request_payload)
        except ValueError as exc:
            return error_payload(exc)

    async def _dispatch_management_request(self, path: str, payload: dict[str, object]) -> dict:
        routes = self._routes_provider()
        metadata = self._metadata_provider()

        return dispatch_management_request(routes, metadata, path, payload)


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

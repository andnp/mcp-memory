from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
import re
from typing import Awaitable, Callable, cast
from urllib.parse import parse_qs, urlsplit

import zmq
import zmq.asyncio

from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools


DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 5.0
_DEFAULT_LIST_LIMIT = 20
_DEFAULT_LOG_LIMIT = 50
_MAX_LIST_LIMIT = 200
_MEMORY_DETAIL_PATH_RE = re.compile(r"^/api/memories/(?P<memory_id>[^/]+)$")
_TASK_CANCEL_PATH_RE = re.compile(r"^/api/admin/tasks/(?P<task_id>[^/]+)/cancel$")


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

        normalized_path, request_payload = _normalize_request(path, payload)
        try:
            if normalized_path == "/internal/health":
                return asdict(self._metadata_provider())
            if normalized_path == "/internal/tools":
                return {"tools": [_serialize_tool(tool) for tool in get_memory_tools()]}
            if normalized_path == "/internal/maintenance/tools":
                return {"tools": [_serialize_tool(tool) for tool in get_internal_maintenance_tools()]}
            if normalized_path.startswith("/internal/tools/"):
                name = normalized_path.rsplit("/", 1)[-1]
                response = await call_memory_tool(self._context_factory(request_payload), name, request_payload)
                return _serialize_tool_response(response)
            if normalized_path.startswith("/internal/maintenance/tools/"):
                name = normalized_path.rsplit("/", 1)[-1]
                response = await call_internal_memory_tool(self._context_factory(request_payload), name, request_payload)
                return _serialize_tool_response(response)
            hook_handler = self._hook_handlers.get(normalized_path)
            if callable(hook_handler):
                typed_handler = cast(Callable[[dict[str, object]], Awaitable[dict]], hook_handler)
                return await typed_handler(request_payload)
            return await self._dispatch_management_request(normalized_path, request_payload)
        except ValueError as exc:
            return _error_payload(exc)

    async def _dispatch_management_request(self, path: str, payload: dict[str, object]) -> dict:
        routes = self._routes_provider()
        metadata = self._metadata_provider()

        return dispatch_management_request(routes, metadata, path, payload)


def dispatch_management_request(routes, metadata, path: str, payload: dict[str, object]) -> dict:
    if path == "/api/health":
        response = routes.service.get_health().model_dump()
        response["pid"] = metadata.pid
        response["status"] = metadata.status
        response["daemon_scope"] = metadata.daemon_scope
        response["binary_path"] = metadata.binary_path
        response["version"] = metadata.version
        response["transport"] = metadata.transport
        return response
    if path == "/api/overview":
        return routes.service.get_overview().model_dump()
    if path == "/api/tasks":
        return routes.service.list_tasks(
            status=_optional_str(payload, "status"),
            workspace_id=_optional_str(payload, "workspace_id"),
            limit=_optional_int(payload, "limit", default=_DEFAULT_LIST_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/record-thought":
        return _serialize_payload(
            routes.service.record_thought(
                _required_str(payload, "content"),
            )
        )
    if path == "/api/memories":
        return routes.service.list_memories(
            workspace_id=_optional_str(payload, "workspace_id"),
            memory_type=_optional_str(payload, "memory_type"),
            status=_optional_str(payload, "status"),
            limit=_optional_int(payload, "limit", default=_DEFAULT_LIST_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/logs":
        return routes.service.list_logs(
            level=_optional_str(payload, "level"),
            logger_name=_optional_str(payload, "logger_name"),
            source=_optional_str(payload, "source"),
            query=_optional_str(payload, "q"),
            after=_optional_float(payload, "after"),
            before=_optional_float(payload, "before"),
            limit=_optional_int(payload, "limit", default=_DEFAULT_LOG_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/logs/summary":
        return routes.service.summarize_logs(
            level=_optional_str(payload, "level"),
            logger_name=_optional_str(payload, "logger_name"),
            source=_optional_str(payload, "source"),
            query=_optional_str(payload, "q"),
            after=_optional_float(payload, "after"),
            before=_optional_float(payload, "before"),
        ).model_dump()
    if path == "/api/ai-conversations":
        return routes.service.list_ai_conversations(
            request_id=_optional_str(payload, "request_id"),
            task_name=_optional_str(payload, "task_name"),
            status=_optional_str(payload, "status"),
            limit=_optional_int(payload, "limit", default=_DEFAULT_LOG_LIMIT, minimum=1, maximum=_MAX_LIST_LIMIT),
        ).model_dump()
    if path == "/api/admin/agents/run":
        return routes.service.enqueue_background_task(
            _required_str(payload, "task_name"),
            force=_bool_value(payload, "force", default=False),
        )
    if path == "/api/admin/agents/run-all":
        return {
            "results": routes.service.enqueue_all_background_tasks(
                force=_bool_value(payload, "force", default=False)
            )
        }
    if path == "/api/admin/logs/prune":
        return routes.service.prune_logs(
            max_runtime_logs=_optional_int(payload, "max_runtime_logs"),
            max_log_age_days=_optional_int(payload, "max_log_age_days"),
        ).model_dump()
    if path == "/api/admin/search/repair":
        return _serialize_payload(routes.service.repair_search_index())
    if path == "/api/admin/links":
        return _serialize_payload(
            routes.service.create_memory_link(
                source_id=_required_str(payload, "source_id"),
                target_id=_required_str(payload, "target_id"),
                link_type=_required_str(payload, "link_type"),
                context=_required_str(payload, "context"),
            )
        )
    if path == "/api/admin/links/delete":
        return _serialize_payload(
            routes.service.delete_memory_link(
                source_id=_required_str(payload, "source_id"),
                target_id=_required_str(payload, "target_id"),
                link_type=_required_str(payload, "link_type"),
            )
        )

    memory_match = _MEMORY_DETAIL_PATH_RE.fullmatch(path)
    if memory_match is not None:
        return routes.service.get_memory_detail(memory_match.group("memory_id")).model_dump()

    cancel_match = _TASK_CANCEL_PATH_RE.fullmatch(path)
    if cancel_match is not None:
        return _serialize_payload(
            routes.service.cancel_task(
                cancel_match.group("task_id"),
                cancelled_by=_optional_str(payload, "cancelled_by") or "cli",
                reason=_optional_str(payload, "reason") or "cancelled_by_user",
            )
        )

    return {"status": "error", "error": "unknown_transport_path", "path": path}


def _serialize_tool(tool) -> dict[str, object]:
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.inputSchema,
    }


def _serialize_tool_response(response) -> dict[str, object]:
    return {
        "contents": [
            {
                "type": content.type,
                "text": content.text,
            }
            for content in response
        ]
    }


def _normalize_request(path: str, payload: dict | None) -> tuple[str, dict[str, object]]:
    split = urlsplit(path)
    request_payload = {} if payload is None else dict(payload)
    if split.query:
        for key, values in parse_qs(split.query, keep_blank_values=True).items():
            if key in request_payload:
                continue
            if len(values) == 1:
                request_payload[key] = values[0]
            else:
                request_payload[key] = values
    return split.path or path, request_payload


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_str(payload: dict[str, object], key: str) -> str:
    value = _optional_str(payload, key)
    if value is None:
        raise ValueError(f"{key}_required")
    return value


def _optional_int(
    payload: dict[str, object],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key}_must_be_integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{key}_must_be_integer") from exc
    else:
        raise ValueError(f"{key}_must_be_integer")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{key}_out_of_range")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{key}_out_of_range")
    return parsed


def _optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key}_must_be_number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{key}_must_be_number") from exc
    raise ValueError(f"{key}_must_be_number")


def _bool_value(payload: dict[str, object], key: str, *, default: bool) -> bool:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key}_must_be_boolean")


def _serialize_payload(payload) -> dict[str, object]:
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        serialized = model_dump()
        if isinstance(serialized, dict):
            return serialized
        raise ValueError("service_response_must_be_object")
    if isinstance(payload, dict):
        return payload
    raise ValueError("service_response_must_be_object")


def _error_payload(exc: ValueError) -> dict[str, object]:
    return {
        "status": "error",
        "error": str(exc),
    }


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
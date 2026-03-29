from __future__ import annotations

import asyncio
from collections import deque
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter, time
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
EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS = 30.0
_SOCKET_PROBE_TIMEOUT_SECONDS = 0.1
_EXTENDED_TIMEOUT_PATH_PREFIXES = (
    "/api/memories/search",
    "/api/admin/search/repair",
    "/api/memories/",
    "/internal/tools",
    "/internal/maintenance/tools",
)
_RECENT_TRANSPORT_SAMPLE_LIMIT = 32


@dataclass(frozen=True)
class DaemonTransportRequestSample:
    path: str
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    completed_at: float = 0.0


@dataclass(frozen=True)
class DaemonTransportDiagnosticsSnapshot:
    current_in_flight_count: int = 0
    current_constrained_in_flight_count: int = 0
    max_concurrent_requests: int = 0
    request_slots_available: int = 0
    queued_waiter_count: int = 0
    recent_completed_request_count: int = 0
    recent_queue_wait_avg_ms: float = 0.0
    recent_queue_wait_max_ms: float = 0.0
    recent_execution_avg_ms: float = 0.0
    recent_execution_max_ms: float = 0.0


def request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds: float | None = None):
    socket_path = getattr(metadata, "socket_path", None)
    transport = getattr(metadata, "transport", "zmq")
    if transport not in {"zmq", "hybrid"}:
        raise ValueError("unsupported_daemon_transport")
    if not isinstance(socket_path, str) or not socket_path:
        raise ValueError("daemon_socket_path_required")
    resolved_timeout_seconds = resolve_daemon_request_timeout_seconds(path, timeout_seconds=timeout_seconds)
    return _request_zmq_json(socket_path, path, payload, timeout_seconds=resolved_timeout_seconds)


def resolve_daemon_request_timeout_seconds(path: str, *, timeout_seconds: float | None = None) -> float:
    if timeout_seconds is not None:
        return timeout_seconds
    if _uses_extended_timeout_budget(path):
        return EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    return DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS


def _uses_extended_timeout_budget(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in _EXTENDED_TIMEOUT_PATH_PREFIXES)


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
        max_concurrent_requests: int = 8,
    ) -> None:
        self._context_factory = context_factory
        self._hook_handlers = hook_handlers
        self._routes_provider = routes_provider
        self._socket_path = socket_path
        self._metadata_provider = metadata_provider
        self._context = zmq.asyncio.Context.instance()
        self._socket: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None
        self._max_concurrent_requests = max(int(max_concurrent_requests), 1)
        self._request_semaphore = asyncio.Semaphore(self._max_concurrent_requests)
        self._inflight_tasks: set[asyncio.Task[object]] = set()
        self._active_request_count = 0
        self._active_constrained_request_count = 0
        self._queued_waiter_count = 0
        self._recent_request_samples: deque[DaemonTransportRequestSample] = deque(maxlen=_RECENT_TRANSPORT_SAMPLE_LIMIT)

    def get_diagnostics_snapshot(self) -> DaemonTransportDiagnosticsSnapshot:
        samples = tuple(self._recent_request_samples)
        queue_waits = [sample.queue_wait_ms for sample in samples]
        execution_times = [sample.execution_ms for sample in samples]
        return DaemonTransportDiagnosticsSnapshot(
            current_in_flight_count=self._active_request_count,
            current_constrained_in_flight_count=self._active_constrained_request_count,
            max_concurrent_requests=self._max_concurrent_requests,
            request_slots_available=max(self._max_concurrent_requests - self._active_constrained_request_count, 0),
            queued_waiter_count=self._queued_waiter_count,
            recent_completed_request_count=len(samples),
            recent_queue_wait_avg_ms=_average_ms(queue_waits),
            recent_queue_wait_max_ms=_max_ms(queue_waits),
            recent_execution_avg_ms=_average_ms(execution_times),
            recent_execution_max_ms=_max_ms(execution_times),
        )

    async def start(self) -> None:
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        _prepare_socket_path_for_bind(self._socket_path)
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
        if self._inflight_tasks:
            inflight = tuple(self._inflight_tasks)
            for task in inflight:
                task.cancel()
            await asyncio.gather(*inflight, return_exceptions=True)
            self._inflight_tasks.clear()
        if self._socket is not None:
            self._socket.close(0)
            self._socket = None
        _cleanup_socket_path_after_stop(self._socket_path)

    async def _serve(self) -> None:
        socket = self._socket
        assert socket is not None
        recv_task = asyncio.ensure_future(socket.recv_multipart())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    {recv_task, *self._inflight_tasks},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if recv_task in done:
                    completed_recv_task = recv_task
                    try:
                        frames = completed_recv_task.result()
                    except asyncio.CancelledError:
                        return
                    if len(frames) >= 2:
                        identity = frames[0]
                        payload_frame = frames[-1]
                        self._inflight_tasks.add(asyncio.create_task(self._dispatch_request(identity, payload_frame)))
                    recv_task = asyncio.ensure_future(socket.recv_multipart())
                else:
                    completed_recv_task = None

                completed_requests = [
                    cast(asyncio.Task[tuple[bytes, dict]], task)
                    for task in done
                    if task is not completed_recv_task
                ]
                for task in completed_requests:
                    self._inflight_tasks.discard(task)
                    if task.cancelled():
                        continue
                    try:
                        identity, response = task.result()
                    except Exception:
                        continue
                    await socket.send_multipart([identity, json.dumps(response, sort_keys=True).encode("utf-8")])
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

    async def _dispatch_request(self, identity: bytes, payload_frame: bytes) -> tuple[bytes, dict]:
        path = _normalized_transport_path(payload_frame) or "<invalid_transport_payload>"
        uses_request_semaphore = path != "/internal/health"
        queue_wait_started_at = perf_counter()
        acquired_request_slot = False
        queue_wait_ms = 0.0

        if uses_request_semaphore:
            self._queued_waiter_count += 1
            try:
                await self._request_semaphore.acquire()
                acquired_request_slot = True
                queue_wait_ms = (perf_counter() - queue_wait_started_at) * 1000.0
            finally:
                self._queued_waiter_count = max(self._queued_waiter_count - 1, 0)

        self._active_request_count += 1
        if uses_request_semaphore:
            self._active_constrained_request_count += 1

        execution_started_at = perf_counter()
        try:
            response = await self._dispatch(payload_frame)
        finally:
            execution_ms = (perf_counter() - execution_started_at) * 1000.0
            self._active_request_count = max(self._active_request_count - 1, 0)
            if uses_request_semaphore:
                self._active_constrained_request_count = max(self._active_constrained_request_count - 1, 0)
                if acquired_request_slot:
                    self._request_semaphore.release()
            self._record_request_sample(path, queue_wait_ms=queue_wait_ms, execution_ms=execution_ms)
        return identity, response

    def _record_request_sample(self, path: str, *, queue_wait_ms: float, execution_ms: float) -> None:
        self._recent_request_samples.append(
            DaemonTransportRequestSample(
                path=path,
                queue_wait_ms=round(max(queue_wait_ms, 0.0), 3),
                execution_ms=round(max(execution_ms, 0.0), 3),
                completed_at=time(),
            )
        )

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
        return await asyncio.to_thread(
            dispatch_management_request,
            routes,
            metadata,
            path,
            payload,
        )


def _socket_endpoint(socket_path: str) -> str:
    return f"ipc://{socket_path}"


def _uses_unconstrained_health_route(payload_frame: bytes) -> bool:
    normalized_path = _normalized_transport_path(payload_frame)
    return normalized_path == "/internal/health"


def _normalized_transport_path(payload_frame: bytes) -> str | None:
    try:
        decoded = json.loads(payload_frame.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None

    path = decoded.get("path")
    payload = decoded.get("payload")
    if not isinstance(path, str) or not path:
        return None
    if payload is not None and not isinstance(payload, dict):
        return None

    normalized_path, _request_payload = normalize_request(path, payload)
    return normalized_path


def _average_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _max_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(max(values), 3)


def _remove_stale_socket(socket_path: Path) -> None:
    try:
        if socket_path.exists() or socket_path.is_socket():
            socket_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _prepare_socket_path_for_bind(socket_path: Path) -> None:
    if not _socket_path_exists(socket_path):
        return
    if probe_daemon_socket(socket_path, timeout_seconds=_SOCKET_PROBE_TIMEOUT_SECONDS):
        raise RuntimeError(f"daemon_socket_already_active:{socket_path}")
    _remove_stale_socket(socket_path)


def _cleanup_socket_path_after_stop(socket_path: Path) -> None:
    if not _socket_path_exists(socket_path):
        return
    if probe_daemon_socket(socket_path, timeout_seconds=_SOCKET_PROBE_TIMEOUT_SECONDS):
        return
    _remove_stale_socket(socket_path)


def _socket_path_exists(socket_path: Path) -> bool:
    try:
        return socket_path.exists() or socket_path.is_socket()
    except OSError:
        return socket_path.exists()

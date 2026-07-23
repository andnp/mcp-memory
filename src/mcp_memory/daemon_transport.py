from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
import time as time_module
from time import perf_counter, time
from types import TracebackType
from typing import Any, Awaitable, Callable, cast

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


logger = logging.getLogger(__name__)


DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS = 5.0
EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS = 60.0
_SOCKET_PROBE_TIMEOUT_SECONDS = 0.1
_ZMQ_REQUEST_POLL_SLICE_SECONDS = 0.1
_EXTENDED_TIMEOUT_PATH_PREFIXES = (
    "/api/memories/search",
    "/api/admin/search/repair",
    "/api/memories/",
    "/api/mutation-history/",
    "/api/record-thought",
    "/internal/tools",
    "/internal/maintenance/tools",
)
_RECENT_TRANSPORT_SAMPLE_LIMIT = 32
_ACTIVE_TRANSPORT_REQUEST_SUMMARY_LIMIT = 8
_SLOW_TRANSPORT_QUEUE_WAIT_WARNING_MS = 250.0
_SLOW_TRANSPORT_EXECUTION_WARNING_MS = 1_000.0
_SLOW_TRANSPORT_TOTAL_WARNING_MS = 4_000.0
_UNCONSTRAINED_REQUEST_PATHS = frozenset(
    {
        "/internal/health",
        "/internal/tools/record_thought",
        "/api/record-thought",
    }
)


@dataclass(frozen=True)
class DaemonTransportRequestSample:
    path: str
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    completed_at: float = 0.0


@dataclass(frozen=True)
class DaemonTransportRequestDiagnostic:
    request_id: int
    path: str
    phase: str
    age_ms: float = 0.0
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    total_ms: float = 0.0
    response_status: str | None = None
    error: str | None = None


@dataclass
class _TrackedDaemonTransportRequest:
    request_id: int
    path: str
    started_at: float
    started_at_perf: float
    phase: str = "received"
    queue_wait_ms: float = 0.0
    execution_ms: float = 0.0
    response_status: str | None = None
    error: str | None = None
    warning_logged: bool = False


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
    active_requests: tuple[DaemonTransportRequestDiagnostic, ...] = ()
    recent_requests: tuple[DaemonTransportRequestDiagnostic, ...] = ()


def request_daemon_json(
    metadata,
    path: str,
    payload: dict | None,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
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


def _request_zmq_json(
    socket_path: str,
    path: str,
    payload: dict | None,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = zmq.Context()
    try:
        socket = context.socket(zmq.DEALER)
        socket.linger = 0
        socket.connect(_socket_endpoint(socket_path))
        try:
            deadline = suspend_aware_deadline(timeout_seconds)
            _send_zmq_json_before_deadline(socket, {"path": path, "payload": payload}, deadline)
            response = _recv_zmq_json_before_deadline(socket, deadline)
        finally:
            socket.close(0)
    finally:
        context.term()
    if not isinstance(response, dict):
        raise ValueError("daemon_response_must_be_object")
    return cast(dict[str, Any], response)


def suspend_aware_now() -> float:
    clock_boottime = getattr(time_module, "CLOCK_BOOTTIME", None)
    if clock_boottime is None:
        return time_module.monotonic()
    return time_module.clock_gettime(clock_boottime)


def suspend_aware_deadline(timeout_seconds: float) -> float:
    return suspend_aware_now() + max(timeout_seconds, 0.0)


def remaining_suspend_aware_seconds(deadline: float) -> float:
    return max(deadline - suspend_aware_now(), 0.0)


def _send_zmq_json_before_deadline(socket: zmq.Socket, payload: dict[str, object | None], deadline: float) -> None:
    _wait_for_socket_event_before_deadline(socket, zmq.POLLOUT, deadline)
    while True:
        try:
            socket.send_json(payload, flags=zmq.DONTWAIT)
            return
        except zmq.error.Again:
            _wait_for_socket_event_before_deadline(socket, zmq.POLLOUT, deadline)


def _recv_zmq_json_before_deadline(socket: zmq.Socket, deadline: float) -> dict[str, Any]:
    _wait_for_socket_event_before_deadline(socket, zmq.POLLIN, deadline)
    while True:
        try:
            response = socket.recv_json(flags=zmq.DONTWAIT)
            if not isinstance(response, dict):
                raise ValueError("daemon_response_must_be_object")
            return cast(dict[str, Any], response)
        except zmq.error.Again:
            _wait_for_socket_event_before_deadline(socket, zmq.POLLIN, deadline)


def _wait_for_socket_event_before_deadline(socket: zmq.Socket, event: int, deadline: float) -> None:
    while True:
        remaining_seconds = remaining_suspend_aware_seconds(deadline)
        if remaining_seconds <= 0.0:
            raise TimeoutError("daemon_request_timed_out")
        if socket.poll(_poll_timeout_milliseconds(remaining_seconds), flags=event) & event:
            return


def _poll_timeout_milliseconds(remaining_seconds: float) -> int:
    return max(min(math.ceil(remaining_seconds * 1000.0), math.ceil(_ZMQ_REQUEST_POLL_SLICE_SECONDS * 1000.0)), 1)


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
        self._recent_request_diagnostics: deque[DaemonTransportRequestDiagnostic] = deque(maxlen=_RECENT_TRANSPORT_SAMPLE_LIMIT)
        self._active_requests: dict[int, _TrackedDaemonTransportRequest] = {}
        self._next_request_id = 0

    def get_diagnostics_snapshot(self) -> DaemonTransportDiagnosticsSnapshot:
        samples = tuple(self._recent_request_samples)
        queue_waits = [sample.queue_wait_ms for sample in samples]
        execution_times = [sample.execution_ms for sample in samples]
        active_requests = tuple(self._build_active_request_diagnostics())
        recent_requests = tuple(reversed(self._recent_request_diagnostics))
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
            active_requests=active_requests,
            recent_requests=recent_requests,
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
                    cast(asyncio.Task[tuple[bytes, dict, int]], task)
                    for task in done
                    if task is not completed_recv_task
                ]
                for task in completed_requests:
                    self._inflight_tasks.discard(task)
                    if task.cancelled():
                        continue
                    try:
                        identity, response, request_id = task.result()
                    except Exception:
                        continue
                    tracked_request = self._active_requests.get(request_id)
                    if tracked_request is not None:
                        tracked_request.phase = "reply_sending"
                        tracked_request.response_status = _response_status(response)
                    try:
                        await socket.send_multipart([identity, json.dumps(response, sort_keys=True).encode("utf-8")])
                    except Exception as exc:
                        if tracked_request is not None:
                            tracked_request.phase = "reply_failed"
                            tracked_request.error = f"{type(exc).__name__}: {exc}"
                            self._log_request_warning("Daemon transport reply failed", tracked_request, exc_info=exc)
                            self._finalize_request_tracking(tracked_request)
                        continue
                    if tracked_request is not None:
                        tracked_request.phase = "reply_sent"
                        self._finalize_request_tracking(tracked_request)
        finally:
            recv_task.cancel()
            await asyncio.gather(recv_task, return_exceptions=True)

    async def _dispatch_request(self, identity: bytes, payload_frame: bytes) -> tuple[bytes, dict, int]:
        path = _normalized_transport_path(payload_frame) or "<invalid_transport_payload>"
        request_timeout_seconds = resolve_daemon_request_timeout_seconds(path)
        tracked_request = self._track_request(path)
        uses_request_semaphore = path not in _UNCONSTRAINED_REQUEST_PATHS
        queue_wait_started_at = perf_counter()
        acquired_request_slot = False

        if uses_request_semaphore:
            tracked_request.phase = "queued"
            self._queued_waiter_count += 1
            try:
                await self._request_semaphore.acquire()
                acquired_request_slot = True
                tracked_request.queue_wait_ms = (perf_counter() - queue_wait_started_at) * 1000.0
                tracked_request.phase = "admitted"
            finally:
                self._queued_waiter_count = max(self._queued_waiter_count - 1, 0)

        self._active_request_count += 1
        if uses_request_semaphore:
            self._active_constrained_request_count += 1

        execution_started_at = perf_counter()
        try:
            tracked_request.phase = "dispatching"
            response = await asyncio.wait_for(
                self._dispatch(payload_frame),
                timeout=request_timeout_seconds,
            )
            tracked_request.execution_ms = (perf_counter() - execution_started_at) * 1000.0
            tracked_request.response_status = _response_status(response)
            tracked_request.phase = "dispatched"
            return identity, response, tracked_request.request_id
        except asyncio.TimeoutError:
            tracked_request.execution_ms = (perf_counter() - execution_started_at) * 1000.0
            tracked_request.phase = "timed_out"
            tracked_request.error = "daemon_request_timed_out"
            response = _dispatch_timeout_payload(path, timeout_seconds=request_timeout_seconds)
            tracked_request.response_status = _response_status(response)
            self._log_request_warning("Daemon transport request timed out", tracked_request)
            return identity, response, tracked_request.request_id
        except asyncio.CancelledError:
            tracked_request.execution_ms = (perf_counter() - execution_started_at) * 1000.0
            tracked_request.phase = "cancelled"
            tracked_request.error = "cancelled"
            self._finalize_request_tracking(tracked_request)
            raise
        except Exception as exc:
            tracked_request.execution_ms = (perf_counter() - execution_started_at) * 1000.0
            tracked_request.phase = "dispatch_failed"
            tracked_request.error = f"{type(exc).__name__}: {exc}"
            self._log_request_warning("Daemon transport request failed before reply", tracked_request, exc_info=exc)
            # Always send a reply for handler/tool failures.  Letting the
            # dispatch task escape here means the ROUTER loop has no payload
            # to send, leaving the DEALER client waiting until its timeout.
            response = _dispatch_failure_payload()
            tracked_request.response_status = _response_status(response)
            return identity, response, tracked_request.request_id
        finally:
            self._active_request_count = max(self._active_request_count - 1, 0)
            if uses_request_semaphore:
                self._active_constrained_request_count = max(self._active_constrained_request_count - 1, 0)
                if acquired_request_slot:
                    self._request_semaphore.release()

    def _track_request(self, path: str) -> _TrackedDaemonTransportRequest:
        self._next_request_id += 1
        tracked_request = _TrackedDaemonTransportRequest(
            request_id=self._next_request_id,
            path=path,
            started_at=time(),
            started_at_perf=perf_counter(),
        )
        self._active_requests[tracked_request.request_id] = tracked_request
        return tracked_request

    def _build_active_request_diagnostics(self) -> list[DaemonTransportRequestDiagnostic]:
        active_requests = sorted(
            self._active_requests.values(),
            key=lambda tracked_request: tracked_request.started_at,
        )
        return [
            self._build_request_diagnostic(tracked_request)
            for tracked_request in active_requests[:_ACTIVE_TRANSPORT_REQUEST_SUMMARY_LIMIT]
        ]

    def _build_request_diagnostic(
        self,
        tracked_request: _TrackedDaemonTransportRequest,
        *,
        now_perf: float | None = None,
    ) -> DaemonTransportRequestDiagnostic:
        resolved_now_perf = perf_counter() if now_perf is None else now_perf
        total_ms = max((resolved_now_perf - tracked_request.started_at_perf) * 1000.0, 0.0)
        return DaemonTransportRequestDiagnostic(
            request_id=tracked_request.request_id,
            path=tracked_request.path,
            phase=tracked_request.phase,
            age_ms=round(total_ms, 3),
            queue_wait_ms=round(max(tracked_request.queue_wait_ms, 0.0), 3),
            execution_ms=round(max(tracked_request.execution_ms, 0.0), 3),
            total_ms=round(total_ms, 3),
            response_status=tracked_request.response_status,
            error=tracked_request.error,
        )

    def _finalize_request_tracking(self, tracked_request: _TrackedDaemonTransportRequest) -> None:
        diagnostic = self._build_request_diagnostic(tracked_request)
        self._record_request_sample(
            tracked_request.path,
            queue_wait_ms=tracked_request.queue_wait_ms,
            execution_ms=tracked_request.execution_ms,
        )
        self._recent_request_diagnostics.append(diagnostic)
        self._active_requests.pop(tracked_request.request_id, None)
        if tracked_request.warning_logged:
            return
        if (
            diagnostic.queue_wait_ms >= _SLOW_TRANSPORT_QUEUE_WAIT_WARNING_MS
            or diagnostic.execution_ms >= _SLOW_TRANSPORT_EXECUTION_WARNING_MS
            or diagnostic.total_ms >= _SLOW_TRANSPORT_TOTAL_WARNING_MS
        ):
            self._log_request_warning("Slow daemon transport request", tracked_request)

    def _log_request_warning(
        self,
        message: str,
        tracked_request: _TrackedDaemonTransportRequest,
        *,
        exc_info: BaseException | tuple[type[BaseException], BaseException, TracebackType | None] | None = None,
    ) -> None:
        diagnostic = self._build_request_diagnostic(tracked_request)
        tracked_request.warning_logged = True
        logger.warning(
            message,
            exc_info=cast(Any, exc_info),
            extra={
                "request_id": diagnostic.request_id,
                "path": diagnostic.path,
                "phase": diagnostic.phase,
                "queue_wait_ms": diagnostic.queue_wait_ms,
                "execution_ms": diagnostic.execution_ms,
                "total_ms": diagnostic.total_ms,
                "response_status": diagnostic.response_status,
                "error": diagnostic.error,
                "current_in_flight_count": self._active_request_count,
                "current_constrained_in_flight_count": self._active_constrained_request_count,
                "queued_waiter_count": self._queued_waiter_count,
                "max_concurrent_requests": self._max_concurrent_requests,
            },
        )

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
                return self._build_health_payload()
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
        except TimeoutError as exc:
            # Python 3.13 aliases asyncio.TimeoutError to TimeoutError.
            # Convert handler-raised timeouts into ordinary error payloads so
            # _dispatch_request only treats actual wait_for expiries as
            # transport-level daemon_request_timed_out responses.
            return {
                "status": "error",
                "error": str(exc),
            }

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

    def _build_health_payload(self) -> dict[str, object]:
        metadata = self._metadata_provider()
        if isinstance(metadata, dict):
            payload = dict(metadata)
        elif is_dataclass(metadata):
            payload = cast(dict[str, object], asdict(cast(Any, metadata)))
        else:
            raise ValueError("transport_health_metadata_must_be_mapping_or_dataclass")
        payload["transport_diagnostics"] = cast(dict[str, object], asdict(self.get_diagnostics_snapshot()))
        return payload


def _socket_endpoint(socket_path: str) -> str:
    return f"ipc://{socket_path}"


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


def _response_status(response: dict) -> str | None:
    status = response.get("status")
    if isinstance(status, str) and status:
        return status
    return None


def _dispatch_timeout_payload(path: str, *, timeout_seconds: float) -> dict[str, object]:
    return {
        "status": "error",
        "error": "daemon_request_timed_out",
        "path": path,
        "timeout_seconds": timeout_seconds,
    }


def _dispatch_failure_payload() -> dict[str, str]:
    """Return the stable client-facing error for an unexpected dispatch failure."""

    return {
        "status": "error",
        "error": "daemon_request_failed",
    }


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

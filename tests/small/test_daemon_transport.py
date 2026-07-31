from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mcp_memory.config import resolve_workspace_id
from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.daemon_dispatch import dispatch_management_request
from mcp_memory.daemon_app import _context_for_request, _handle_post_tool_use, _request_scope_for_arguments, _workspace_id_for_request_scope, create_daemon_app
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.mcp.runtime import GlobalDaemonBootstrapSpec
from mcp_memory.management.capabilities import ManagementCapabilities
from mcp_memory.server import MCPServer


def _daemon_transport_module():
    from mcp_memory import daemon_transport

    return daemon_transport


def test_dispatch_management_request_deprecates_manual_link_mutation_routes() -> None:
    routes = SimpleNamespace(service=SimpleNamespace())
    metadata = SimpleNamespace(pid=1, status="running", daemon_scope="global", binary_path="mcp-memory", version="1.0", transport="stdio")

    created = dispatch_management_request(
        routes,
        metadata,
        "/api/admin/links",
        {"source_id": "a", "target_id": "b", "link_type": "REFERENCES", "context": "ctx"},
    )
    deleted = dispatch_management_request(
        routes,
        metadata,
        "/api/admin/links/delete",
        {"source_id": "a", "target_id": "b", "link_type": "REFERENCES"},
    )

    assert created == {
        "status": "error",
        "error": "deprecated_manual_cleanup_endpoint:/api/admin/links use /api/admin/agents/run task_name=memory-curator",
    }
    assert deleted == {
        "status": "error",
        "error": "deprecated_manual_cleanup_endpoint:/api/admin/links/delete use /api/admin/agents/run task_name=memory-curator",
    }


def test_dispatch_management_request_uses_typed_management_capability() -> None:
    class _MaintenanceSpy:
        workspace_id = "workspace-a"
        dashboard_static_root = Path(".")

        def enqueue_background_task(self, task_name: str, *, force: bool = False) -> dict[str, object]:
            return {"task_name": task_name, "force": force}

    maintenance = _MaintenanceSpy()
    routes = SimpleNamespace(
        management=ManagementCapabilities.from_service(maintenance),
        service=SimpleNamespace(
            enqueue_background_task=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("transport should use the typed capability")
            )
        ),
    )
    metadata = SimpleNamespace(pid=1, status="running", daemon_scope="global", binary_path=None, version=None, transport="stdio")

    assert dispatch_management_request(
        routes,
        metadata,
        "/api/admin/agents/run",
        {"task_name": "memory-curator", "force": True},
    ) == {"task_name": "memory-curator", "force": True}


def test_resolve_daemon_request_timeout_seconds_uses_extended_budget_for_memory_and_tool_paths() -> None:
    daemon_transport = _daemon_transport_module()

    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/memories/search") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/memories/memory-123") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/record-thought") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/tools") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/tools/record_thought") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/tools/search_memory_records") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/tools/read_memory_record") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/maintenance/tools") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/internal/maintenance/tools/internal_get_work_batch") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS


def test_resolve_daemon_request_timeout_seconds_preserves_default_and_explicit_override() -> None:
    daemon_transport = _daemon_transport_module()

    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/health") == daemon_transport.DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/memories/search", timeout_seconds=7.5) == 7.5


def test_request_zmq_json_uses_fresh_client_context_per_request(monkeypatch) -> None:
    daemon_transport = _daemon_transport_module()

    created_contexts: list[FakeContext] = []

    class FakeSocket:
        def __init__(self) -> None:
            self.linger: int | None = None
            self.connected_to: str | None = None
            self.sent_payloads: list[dict[str, object | None]] = []
            self.closed_with: int | None = None
            self.poll_calls: list[tuple[int, int]] = []

        def connect(self, endpoint: str) -> None:
            self.connected_to = endpoint

        def poll(self, timeout: int, *, flags: int) -> int:
            self.poll_calls.append((timeout, flags))
            return flags

        def send_json(self, payload: dict[str, object | None], *, flags: int = 0) -> None:
            assert flags == daemon_transport.zmq.DONTWAIT
            self.sent_payloads.append(payload)

        def recv_json(self, *, flags: int = 0) -> dict[str, str]:
            assert flags == daemon_transport.zmq.DONTWAIT
            return {"status": "ok"}

        def close(self, linger: int) -> None:
            self.closed_with = linger

    class FakeContext:
        def __init__(self) -> None:
            self.socket_instances: list[FakeSocket] = []
            self.terminated = False

        def socket(self, socket_type: int) -> FakeSocket:
            assert socket_type == daemon_transport.zmq.DEALER
            socket = FakeSocket()
            self.socket_instances.append(socket)
            return socket

        def term(self) -> None:
            self.terminated = True

    def _fake_context_factory() -> FakeContext:
        context = FakeContext()
        created_contexts.append(context)
        return context

    monkeypatch.setattr("mcp_memory.daemon_transport.zmq.Context", _fake_context_factory)

    first = daemon_transport._request_zmq_json("/tmp/daemon.sock", "/internal/health", None, timeout_seconds=0.5)
    second = daemon_transport._request_zmq_json(
        "/tmp/daemon.sock",
        "/internal/tools/search_memory_records",
        {"query": "auth"},
        timeout_seconds=0.25,
    )

    assert first == {"status": "ok"}
    assert second == {"status": "ok"}
    assert len(created_contexts) == 2
    assert created_contexts[0] is not created_contexts[1]

    first_socket = created_contexts[0].socket_instances[0]
    second_socket = created_contexts[1].socket_instances[0]

    assert first_socket.connected_to == "ipc:///tmp/daemon.sock"
    assert second_socket.connected_to == "ipc:///tmp/daemon.sock"
    assert first_socket.linger == 0
    assert second_socket.linger == 0
    assert first_socket.poll_calls == [
        (100, daemon_transport.zmq.POLLOUT),
        (100, daemon_transport.zmq.POLLIN),
    ]
    assert second_socket.poll_calls == [
        (100, daemon_transport.zmq.POLLOUT),
        (100, daemon_transport.zmq.POLLIN),
    ]
    assert first_socket.sent_payloads == [{"path": "/internal/health", "payload": None}]
    assert second_socket.sent_payloads == [{"path": "/internal/tools/search_memory_records", "payload": {"query": "auth"}}]
    assert first_socket.closed_with == 0
    assert second_socket.closed_with == 0
    assert created_contexts[0].terminated is True
    assert created_contexts[1].terminated is True


def test_suspend_aware_now_prefers_boottime_and_falls_back_to_monotonic(monkeypatch) -> None:
    daemon_transport = _daemon_transport_module()

    class FakeTimeWithBoottime:
        CLOCK_BOOTTIME = 7

        def __init__(self) -> None:
            self.clock_ids: list[int] = []

        def clock_gettime(self, clock_id: int) -> float:
            self.clock_ids.append(clock_id)
            return 12.5

        def monotonic(self) -> float:
            raise AssertionError("monotonic should not be used when CLOCK_BOOTTIME is available")

    with_boottime = FakeTimeWithBoottime()
    monkeypatch.setattr("mcp_memory.daemon_transport.time_module", with_boottime)
    assert daemon_transport.suspend_aware_now() == 12.5
    assert with_boottime.clock_ids == [7]

    class FakeTimeWithoutBoottime:
        def monotonic(self) -> float:
            return 22.0

    monkeypatch.setattr("mcp_memory.daemon_transport.time_module", FakeTimeWithoutBoottime())
    assert daemon_transport.suspend_aware_now() == 22.0


def test_request_zmq_json_uses_short_poll_slices_until_deadline(monkeypatch) -> None:
    daemon_transport = _daemon_transport_module()

    created_contexts: list[FakeContext] = []

    class FakeSocket:
        def __init__(self) -> None:
            self.linger: int | None = None
            self.poll_calls: list[tuple[int, int]] = []
            self.closed_with: int | None = None

        def connect(self, endpoint: str) -> None:
            assert endpoint == "ipc:///tmp/daemon.sock"

        def poll(self, timeout: int, *, flags: int) -> int:
            self.poll_calls.append((timeout, flags))
            return 0

        def send_json(self, payload: dict[str, object | None], *, flags: int = 0) -> None:
            raise AssertionError(f"send_json should not be reached before timeout: {payload}, {flags}")

        def recv_json(self, *, flags: int = 0) -> dict[str, str]:
            raise AssertionError(f"recv_json should not be reached before timeout: {flags}")

        def close(self, linger: int) -> None:
            self.closed_with = linger

    class FakeContext:
        def __init__(self) -> None:
            self.socket_instance = FakeSocket()
            self.terminated = False

        def socket(self, socket_type: int) -> FakeSocket:
            assert socket_type == daemon_transport.zmq.DEALER
            return self.socket_instance

        def term(self) -> None:
            self.terminated = True

    def _fake_context_factory() -> FakeContext:
        context = FakeContext()
        created_contexts.append(context)
        return context

    now_values = iter([0.0, 0.0, 0.05, 0.11])

    monkeypatch.setattr("mcp_memory.daemon_transport.zmq.Context", _fake_context_factory)
    monkeypatch.setattr("mcp_memory.daemon_transport.suspend_aware_now", lambda: next(now_values))

    with pytest.raises(TimeoutError, match="daemon_request_timed_out"):
        daemon_transport._request_zmq_json("/tmp/daemon.sock", "/internal/health", None, timeout_seconds=0.1)

    assert len(created_contexts) == 1
    assert created_contexts[0].socket_instance.poll_calls == [
        (100, daemon_transport.zmq.POLLOUT),
        (50, daemon_transport.zmq.POLLOUT),
    ]
    assert created_contexts[0].socket_instance.closed_with == 0
    assert created_contexts[0].terminated is True


@pytest.mark.asyncio
async def test_daemon_transport_returns_structured_timeout_payload_for_slow_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    daemon_transport = _daemon_transport_module()
    socket_path = tmp_path / "daemon.sock"

    async def _slow_handler(_payload: dict[str, object]) -> dict[str, object]:
        await asyncio.sleep(0.05)
        return {"status": "ok"}

    monkeypatch.setattr("mcp_memory.daemon_transport.DEFAULT_DAEMON_REQUEST_TIMEOUT_SECONDS", 0.01)

    server = daemon_transport.DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/slow": _slow_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: None,
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        started_at = asyncio.get_running_loop().time()
        payload = await asyncio.to_thread(
            daemon_transport.request_daemon_json,
            metadata,
            "/api/hooks/slow",
            {},
            timeout_seconds=0.5,
        )
        elapsed = asyncio.get_running_loop().time() - started_at
    finally:
        await server.stop()

    assert payload == {
        "status": "error",
        "error": "daemon_request_timed_out",
        "path": "/api/hooks/slow",
        "timeout_seconds": 0.01,
    }
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_daemon_transport_does_not_relabel_handler_timeout_as_transport_timeout(tmp_path) -> None:
    daemon_transport = _daemon_transport_module()
    socket_path = tmp_path / "daemon.sock"

    async def _timed_out_handler(_payload: dict[str, object]) -> dict[str, object]:
        raise TimeoutError("handler_timed_out")

    server = daemon_transport.DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/timed-out": _timed_out_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: None,
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        payload = await asyncio.to_thread(
            daemon_transport.request_daemon_json,
            metadata,
            "/api/hooks/timed-out",
            {},
            timeout_seconds=0.5,
        )
    finally:
        await server.stop()

    assert payload == {
        "status": "error",
        "error": "handler_timed_out",
    }


@pytest.mark.asyncio
async def test_daemon_transport_replies_to_unexpected_handler_failure_and_accepts_next_request(tmp_path) -> None:
    daemon_transport = _daemon_transport_module()
    socket_path = tmp_path / "daemon.sock"

    async def _failing_handler(_payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("database connection details must stay server-side")

    server = daemon_transport.DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/failing": _failing_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: {"status": "ready"},
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        failed = await asyncio.to_thread(
            daemon_transport.request_daemon_json,
            metadata,
            "/api/hooks/failing",
            {},
            timeout_seconds=0.5,
        )
        health = await asyncio.to_thread(
            daemon_transport.request_daemon_json,
            metadata,
            "/internal/health",
            None,
            timeout_seconds=0.5,
        )
    finally:
        await server.stop()

    assert failed == {
        "status": "error",
        "error": "daemon_request_failed",
    }
    assert health["status"] == "ready"


def test_mcp_server_request_json_uses_transport_default_timeout(monkeypatch) -> None:
    """Verify proxy requests preserve deterministic routing metadata.

    Internal tool requests need both workspace and session identity so the daemon can attribute counts to the active run.
    """

    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-test.sock",
    )
    server._session_id = "session-123"
    captured: dict[str, object] = {}

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        captured["metadata"] = metadata
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {"status": "ok"}

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)

    result = server._request_json_with_recovery("/internal/tools/search_memory_records", {"query": "auth"})

    assert result == {"status": "ok"}
    assert captured["metadata"] is server._daemon
    assert captured["path"] == "/internal/tools/search_memory_records"
    assert captured["payload"] == {
        "query": "auth",
        "__workspace_root": "demo-workspace",
        "__session_id": "session-123",
    }
    assert captured["timeout_seconds"] is None


def test_mcp_server_request_json_with_recovery_supports_legacy_request_override(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-test.sock",
    )
    server._session_id = "session-123"

    def _legacy_request_override(path: str, payload: dict | None) -> dict[str, object]:
        assert path == "/internal/tools/search_memory_records"
        assert payload == {"query": "auth"}
        return {"status": "ok"}

    monkeypatch.setattr(server, "_request_json", _legacy_request_override)

    result = server._request_json_with_recovery("/internal/tools/search_memory_records", {"query": "auth"})

    assert result == {"status": "ok"}


def test_mcp_server_request_json_refreshes_daemon_metadata_after_timeout(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    stale_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-stale.sock",
    )
    fresh_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4343,
        pid=2,
        started_at=1.0,
        status="ready",
        socket_path="/tmp/mcp-memory-fresh.sock",
    )
    server._daemon = stale_metadata
    server._session_id = "session-123"

    request_calls: list[int] = []
    ensure_calls: list[str | None] = []

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        del timeout_seconds
        request_calls.append(metadata.pid)
        assert path == "/internal/tools/search_memory_records"
        assert payload == {
            "query": "auth",
            "__workspace_root": "demo-workspace",
            "__session_id": "session-123",
        }
        if metadata.pid == stale_metadata.pid:
            raise TimeoutError("daemon_request_timed_out")
        return {"status": "ok", "metadata_pid": metadata.pid}

    def _fake_ensure_daemon_started(workspace_root, cwd=None):
        del cwd
        ensure_calls.append(workspace_root)
        return fresh_metadata

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", _fake_ensure_daemon_started)

    result = server._request_json_with_recovery("/internal/tools/search_memory_records", {"query": "auth"})

    assert result == {"status": "ok", "metadata_pid": fresh_metadata.pid}
    assert request_calls == [stale_metadata.pid, fresh_metadata.pid]
    assert ensure_calls == ["demo-workspace"]
    assert server._daemon == fresh_metadata


def test_mcp_server_request_json_raises_after_bounded_retries(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    initial_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-initial.sock",
    )
    refreshed_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4343,
        pid=2,
        started_at=1.0,
        status="ready",
        socket_path="/tmp/mcp-memory-refreshed.sock",
    )
    server._daemon = initial_metadata

    request_calls: list[int] = []
    ensure_calls: list[str | None] = []

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        del path, payload, timeout_seconds
        request_calls.append(metadata.pid)
        raise TimeoutError("daemon_request_timed_out")

    def _fake_ensure_daemon_started(workspace_root, cwd=None):
        del cwd
        ensure_calls.append(workspace_root)
        return refreshed_metadata

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", _fake_ensure_daemon_started)

    with pytest.raises(TimeoutError, match="daemon_request_timed_out"):
        server._request_json_with_recovery("/internal/tools/search_memory_records", {"query": "auth"})

    assert request_calls == [initial_metadata.pid, refreshed_metadata.pid, refreshed_metadata.pid]
    assert ensure_calls == ["demo-workspace", "demo-workspace"]
    assert server._daemon == refreshed_metadata


def test_mcp_server_request_json_with_recovery_respects_overall_client_deadline(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    initial_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-initial.sock",
    )
    refreshed_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=4343,
        pid=2,
        started_at=1.0,
        status="ready",
        socket_path="/tmp/mcp-memory-refreshed.sock",
    )
    server._daemon = initial_metadata

    request_timeouts: list[float | None] = []
    remaining_values = iter([0.04, 0.01, 0.0])

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        del metadata, path, payload
        request_timeouts.append(timeout_seconds)
        raise TimeoutError("daemon_request_timed_out")

    def _fake_ensure_daemon_started(workspace_root, cwd=None):
        del workspace_root, cwd
        return refreshed_metadata

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", _fake_ensure_daemon_started)
    monkeypatch.setattr("mcp_memory.server.remaining_suspend_aware_seconds", lambda _deadline: next(remaining_values))

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        server._request_json_with_recovery("/internal/tools/search_memory_records", {"query": "auth"}, timeout_deadline=60.0)

    assert request_timeouts == [0.04]
    assert server._daemon == refreshed_metadata


@pytest.mark.asyncio
async def test_mcp_server_request_daemon_json_with_client_timeout_supports_legacy_recovery_override(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")

    def _legacy_recovery_override(path: str, payload: dict | None) -> dict[str, object]:
        assert path == "/internal/tools"
        assert payload is None
        return {"tools": []}

    monkeypatch.setattr(server, "_request_json_with_recovery", _legacy_recovery_override)

    payload = await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert payload == {"tools": []}


@pytest.mark.asyncio
async def test_mcp_server_request_daemon_json_with_client_timeout_bounds_blocking_recovery(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    stall = threading.Event()

    def _blocking_recovery_override(path: str, payload: dict | None) -> dict[str, object]:
        assert path == "/internal/tools"
        assert payload is None
        stall.wait(0.05)
        return {"tools": []}

    monkeypatch.setattr(server, "_request_json_with_recovery", _blocking_recovery_override)
    monkeypatch.setattr("mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)


def test_context_for_request_copies_session_id_and_workspace_root(tmp_path) -> None:
    """Verify daemon request context carries session identity alongside workspace scope.

    The tracker relies on request-scoped session ids instead of provider-parsed tool summaries.
    """

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    request_ctx = _context_for_request(
        ApplicationContext(),
        {
            "__session_id": "session-123",
            "__workspace_root": str(workspace),
        },
    )

    assert request_ctx.session_id == "session-123"
    assert request_ctx.workspace_root == workspace
    assert request_ctx.workspace_id is not None


def test_request_scope_for_arguments_extracts_request_scoped_identity(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    request_scope = _request_scope_for_arguments(
        {
            "__session_id": "session-123",
            "__workspace_root": str(workspace),
        }
    )

    assert request_scope.session_id == "session-123"
    assert request_scope.workspace_root == workspace
    assert _workspace_id_for_request_scope(request_scope) == resolve_workspace_id(workspace_root=str(workspace))


@pytest.mark.asyncio
async def test_handle_post_tool_use_uses_request_scope_without_context_overlay(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    captured: dict[str, Any] = {}

    class FakeHookReminderService:
        def __init__(self, db_manager, workspace_id: str | None) -> None:
            captured["db_manager"] = db_manager
            captured["workspace_id"] = workspace_id

        def record_post_tool_use(self, arguments: dict[str, object]) -> dict[str, object]:
            captured["arguments"] = arguments
            return {"status": "ok", "workspace_id": captured["workspace_id"]}

    monkeypatch.setattr("mcp_memory.daemon_app.HookReminderService", FakeHookReminderService)
    monkeypatch.setattr(
        "mcp_memory.daemon_app._context_for_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("hook path should not mutate context")),
    )

    db_manager = object()
    response = await _handle_post_tool_use(
        ApplicationContext(db_manager=db_manager),
        {
            "tool_name": "apply_patch",
            "__workspace_root": str(workspace),
        },
    )

    assert response == {
        "status": "ok",
        "workspace_id": resolve_workspace_id(workspace_root=str(workspace)),
    }
    assert captured["db_manager"] is db_manager
    assert captured["arguments"] == {
        "tool_name": "apply_patch",
        "__workspace_root": str(workspace),
    }


def test_create_daemon_app_uses_global_bootstrap_spec_without_workspace_identity(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    captured: dict[str, object] = {}

    def _fake_resolve_global_daemon_bootstrap_spec(
        workspace_root_override: str | None = None,
        cwd=None,
    ) -> GlobalDaemonBootstrapSpec:
        captured["workspace_root_override"] = workspace_root_override
        captured["cwd"] = cwd
        return GlobalDaemonBootstrapSpec(
            memory_path=tmp_path / "memories",
            config=Config(),
            workspace_root=workspace,
            lock_path=tmp_path / "daemon.lock",
        )

    monkeypatch.setattr(
        "mcp_memory.daemon_app.resolve_global_daemon_bootstrap_spec",
        _fake_resolve_global_daemon_bootstrap_spec,
    )

    app = create_daemon_app(workspace_root_override=str(workspace), host="127.0.0.1", port=4242)

    assert app.title == "mcp-memory daemon"
    assert captured == {
        "workspace_root_override": str(workspace),
        "cwd": None,
    }


@pytest.mark.asyncio
async def test_internal_health_includes_transport_diagnostics(tmp_path) -> None:
    daemon_transport = _daemon_transport_module()
    socket_path = tmp_path / "daemon.sock"
    server = daemon_transport.DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: DaemonMetadata(
            host="127.0.0.1",
            port=4242,
            pid=1,
            started_at=0.0,
            status="ready",
            socket_path=str(socket_path),
        ),
        max_concurrent_requests=2,
    )

    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        health = await asyncio.to_thread(
            daemon_transport.request_daemon_json,
            metadata,
            "/internal/health",
            None,
            timeout_seconds=0.5,
        )
    finally:
        await server.stop()

    assert health["status"] == "ready"
    assert health["socket_path"] == str(socket_path)
    transport_diagnostics = health["transport_diagnostics"]
    assert transport_diagnostics["max_concurrent_requests"] == 2
    assert transport_diagnostics["request_slots_available"] >= 0
    assert transport_diagnostics["queued_waiter_count"] >= 0
    assert isinstance(transport_diagnostics["active_requests"], list)
    assert isinstance(transport_diagnostics["recent_requests"], list)


def test_mcp_server_session_hook_failure_logs_transport_health_snapshot(monkeypatch, caplog) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-test.sock",
    )
    server._session_id = "session-123"
    calls: list[tuple[str, dict | None, float | None]] = []
    health_snapshot = {
        "status": "ready",
        "transport": "zmq",
        "transport_diagnostics": {
            "queued_waiter_count": 2,
            "request_slots_available": 0,
        },
    }

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        del metadata
        calls.append((path, payload, timeout_seconds))
        if path == "/api/hooks/session-start":
            raise TimeoutError("daemon_request_timed_out")
        if path == "/internal/health":
            return health_snapshot
        raise AssertionError(f"unexpected path: {path}")

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)

    with caplog.at_level(logging.WARNING, logger="mcp_memory.server"):
        server._send_session_hook("session-start")

    assert [path for path, _payload, _timeout in calls] == [
        "/api/hooks/session-start",
        "/api/hooks/session-start",
        "/api/hooks/session-start",
        "/internal/health",
    ]
    assert calls[-1][2] == 0.2
    record = caplog.records[-1]
    assert record.session_id == "session-123"
    assert record.event_name == "session-start"
    assert record.transport_health_snapshot == health_snapshot
    assert "Failed to send session-start hook for session session-123" in record.getMessage()

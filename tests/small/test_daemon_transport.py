from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.daemon_app import _context_for_request, create_daemon_app
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.mcp.runtime import GlobalDaemonBootstrapSpec
from mcp_memory.server import MCPServer


def _daemon_transport_module():
    from mcp_memory import daemon_transport

    return daemon_transport


def test_resolve_daemon_request_timeout_seconds_uses_extended_budget_for_memory_and_tool_paths() -> None:
    daemon_transport = _daemon_transport_module()

    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/memories/search") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
    assert daemon_transport.resolve_daemon_request_timeout_seconds("/api/memories/memory-123") == daemon_transport.EXTENDED_DAEMON_REQUEST_TIMEOUT_SECONDS
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
            self.rcvtimeo: int | None = None
            self.sndtimeo: int | None = None
            self.connected_to: str | None = None
            self.sent_payloads: list[dict[str, object | None]] = []
            self.closed_with: int | None = None

        def connect(self, endpoint: str) -> None:
            self.connected_to = endpoint

        def send_json(self, payload: dict[str, object | None]) -> None:
            self.sent_payloads.append(payload)

        def recv_json(self) -> dict[str, str]:
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
    assert first_socket.rcvtimeo == 500
    assert first_socket.sndtimeo == 500
    assert second_socket.rcvtimeo == 250
    assert second_socket.sndtimeo == 250
    assert first_socket.sent_payloads == [{"path": "/internal/health", "payload": None}]
    assert second_socket.sent_payloads == [{"path": "/internal/tools/search_memory_records", "payload": {"query": "auth"}}]
    assert first_socket.closed_with == 0
    assert second_socket.closed_with == 0
    assert created_contexts[0].terminated is True
    assert created_contexts[1].terminated is True


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
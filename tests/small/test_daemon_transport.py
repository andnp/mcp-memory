from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.daemon_models import DaemonMetadata
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


def test_mcp_server_request_json_uses_transport_default_timeout(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = DaemonMetadata(
        host="127.0.0.1",
        port=4242,
        pid=1,
        started_at=0.0,
        status="ready",
        socket_path="/tmp/mcp-memory-test.sock",
    )
    captured: dict[str, object] = {}

    def _fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        captured["metadata"] = metadata
        captured["path"] = path
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {"status": "ok"}

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", _fake_request_daemon_json)

    result = server._request_json("/internal/tools/search_memory_records", {"query": "auth"})

    assert result == {"status": "ok"}
    assert captured["metadata"] is server._daemon
    assert captured["path"] == "/internal/tools/search_memory_records"
    assert captured["payload"] == {
        "query": "auth",
        "__workspace_root": "demo-workspace",
    }
    assert captured["timeout_seconds"] is None


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

    assert [path for path, _payload, _timeout in calls] == ["/api/hooks/session-start", "/internal/health"]
    assert calls[1][2] == 0.2
    record = caplog.records[-1]
    assert record.session_id == "session-123"
    assert record.event_name == "session-start"
    assert record.transport_health_snapshot == health_snapshot
    assert "Failed to send session-start hook for session session-123" in record.getMessage()
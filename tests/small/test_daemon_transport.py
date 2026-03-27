from __future__ import annotations

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
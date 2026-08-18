from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from mcp_memory.config import resolve_workspace_id
from mcp_memory.daemon_app import _context_for_request
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.daemon_transport import DaemonZmqServer, request_daemon_json
from mcp_memory.mcp.runtime import create_runtime

pytestmark = pytest.mark.large


def _decode_tool_response(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(payload["contents"][0]["text"])


@pytest.mark.asyncio
async def test_zmq_server_handles_concurrent_clients_and_request_workspace_context(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))

    runtime = create_runtime(workspace_root_override=None, cwd=workspace_a)
    socket_path = tmp_path / "daemon.sock"
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=0,
        pid=os.getpid(),
        started_at=time.time(),
        status="ready",
        transport="zmq",
        socket_path=str(socket_path),
    )
    server = DaemonZmqServer(
        context_factory=lambda arguments: _context_for_request(runtime, arguments),
        hook_handlers={},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: metadata,
    )
    await server.start()
    try:
        tools_payload, first_payload, second_payload = await asyncio.gather(
            asyncio.to_thread(request_daemon_json, metadata, "/internal/tools", None, timeout_seconds=2),
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/internal/tools/record_thought",
                {"content": "alpha thought", "__workspace_root": str(workspace_a)},
                timeout_seconds=2,
            ),
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/internal/tools/record_thought",
                {"content": "beta thought", "__workspace_root": str(workspace_b)},
                timeout_seconds=2,
            ),
        )

        assert {tool["name"] for tool in tools_payload["tools"]} == {
            "record_thought",
            "record_skill_observation",
            "resolve_skill_observation",
            "search_memory_records",
            "read_memory_records",
            "read_memory_record",
        }

        first_result = _decode_tool_response(first_payload)
        second_result = _decode_tool_response(second_payload)
        workspace_a_id = resolve_workspace_id(workspace_root=str(workspace_a))
        workspace_b_id = resolve_workspace_id(workspace_root=str(workspace_b))

        assert first_result["status"] == "recorded"
        assert second_result["status"] == "recorded"

        assert runtime.journal is not None
        pending_entries = runtime.journal.get_pending(limit=10)
        assert {entry.workspace_id for entry in pending_entries} == {workspace_a_id, workspace_b_id}
    finally:
        await server.stop()
        runtime.close()

import asyncio
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner
from mcp import types
from mcp.types import TextContent

from mcp_memory.cli import main
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.tools import (
    SKILL_REVIEW_READ_ONLY_SCOPE,
    SKILL_REVIEW_WRITER_SCOPE,
    allowed_memory_tool_names,
    get_memory_tools,
)
from mcp_memory.server import MCPServer
from tests.sdk.mcp import FakeAsyncContextManager

pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_call_memory_tool_returns_placeholder_payload() -> None:
    result = await call_memory_tool(None, "record_thought", {"content": "auth"})

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    payload = json.loads(result[0].text)
    assert payload["status"] == "error"
    assert payload["error"] == "runtime_not_initialized"


def test_get_memory_tools_returns_expected_names() -> None:
    """Expose the verified disposition in the public observation schema."""
    tools = get_memory_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "record_thought",
        "record_skill_observation",
        "resolve_skill_observation",
        "get_skill_review_ledger",
        "search_memory_records",
        "read_memory_records",
        "read_memory_record",
    ]
    observation_tool = next(tool for tool in tools if tool.name == "record_skill_observation")
    assert observation_tool.input_schema["required"] == [
        "title",
        "summary",
        "content",
        "skill",
        "observation_kind",
        "privacy_classification",
    ]
    resolution_tool = next(tool for tool in tools if tool.name == "resolve_skill_observation")
    assert resolution_tool.input_schema["properties"]["resolution"]["enum"] == [
        "actioned",
        "deferred",
        "verified",
    ]

    search_tool = next(tool for tool in tools if tool.name == "search_memory_records")
    assert search_tool.description is not None
    assert "read_memory_record" in search_tool.description
    read_tool = next(tool for tool in tools if tool.name == "read_memory_record")
    assert read_tool.description is not None
    assert read_tool.input_schema == {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string"},
            "include_relationships": {
                "type": "boolean",
                "description": "Include incoming/outgoing relationship edges. Defaults to false to save tokens.",
            },
            "include_superseded": {
                "type": "boolean",
                "description": "Include superseded record breadcrumbs. Defaults to false to save tokens.",
            },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include maintenance metadata and workspace IDs. Defaults to false to save tokens.",
                },
                "summary_only": {
                    "type": "boolean",
                    "description": "Return a bounded summary projection without content.",
                },
                "content_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Optional character offset for a bounded content chunk.",
                },
                "content_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional maximum characters in the returned content chunk.",
                },
            },
        "required": ["memory_id"],
    }


def test_skill_review_scope_advertises_only_read_tools() -> None:
    """Expose only memory search and read tools to isolated skill reviews."""
    names = [tool.name for tool in get_memory_tools(SKILL_REVIEW_READ_ONLY_SCOPE)]

    assert names == [
        "get_skill_review_ledger",
        "search_memory_records",
        "read_memory_records",
        "read_memory_record",
    ]
    assert allowed_memory_tool_names(SKILL_REVIEW_READ_ONLY_SCOPE) == frozenset(names)


def test_unknown_memory_tool_scope_is_rejected() -> None:
    """Reject unrecognized scopes instead of silently widening permissions."""
    with pytest.raises(ValueError, match="unknown_memory_tool_scope"):
        get_memory_tools("untrusted")


def test_skill_review_writer_scope_advertises_only_batch_commit() -> None:
    """Expose only the atomic disposition writer to the separate writer session."""
    names = [tool.name for tool in get_memory_tools(SKILL_REVIEW_WRITER_SCOPE)]

    assert names == ["commit_skill_review"]
    assert allowed_memory_tool_names(SKILL_REVIEW_WRITER_SCOPE) == frozenset(names)
    tool = get_memory_tools(SKILL_REVIEW_WRITER_SCOPE)[0]
    assert tool.input_schema["properties"]["protocol_version"] == {"type": "integer", "const": 2}


def test_default_tools_do_not_advertise_the_writer() -> None:
    """Keep the mutating review tool behind its explicit writer scope."""
    assert "commit_skill_review" not in {tool.name for tool in get_memory_tools()}


def test_get_internal_maintenance_tools_returns_expected_names() -> None:
    tools = get_internal_maintenance_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_read_memory_records",
        "internal_peek_record",
        "internal_maintenance_search",
        "internal_list_relationships",
        "internal_bounded_adjacency",
        "internal_list_memory_records",
        "task_complete",
        "internal_task_complete",
        "internal_get_next_dedup_batch",
        "internal_get_next_curator_batch",
        "internal_get_next_ingest_batch",
        "internal_get_work_batch",
        "internal_get_compatible_work_batch",
        "internal_heartbeat_work_item",
        "internal_complete_work_item",
        "internal_defer_work_item",
        "internal_release_work_item",
        "internal_ingest_append_memory",
        "internal_ingest_create_memory",
        "internal_append_memory_content",
        "internal_archive_memory_record",
        "internal_merge_memory_into_canonical",
        "internal_split_memory_record",
        "internal_create_memory_record",
        "internal_update_memory_record",
        "internal_delete_memory_record",
        "internal_create_memory_link",
        "internal_delete_memory_link",
    ]
    search_tool = next(tool for tool in tools if tool.name == "internal_search_memory_records")
    assert search_tool.description is not None
    assert "search memory records" in search_tool.description.lower()
    completion_tool = next(tool for tool in tools if tool.name == "task_complete")
    assert completion_tool.description is not None
    assert "instead of creating journal or memory records" in completion_tool.description


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_claim_compatible_work_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    assert runtime.work_items is not None

    try:
        lightweight_specs = [
            ("memory_tagging", "tag-memory"),
            ("graph_link_review", "graph-memory"),
            ("conflict_review", "conflict-memory"),
        ]
        structural_specs = [
            ("memory_curation_review", ["curator-a", "curator-b"]),
            ("memory_dedup_review", ["dedup-a", "dedup-b"]),
        ]
        for family_key, memory_id in lightweight_specs:
            runtime.work_items.enqueue_unique(
                family_key=family_key,
                execution_lane="agentic",
                workspace_id=runtime.workspace_id,
                payload={"memory_id": memory_id, "workspace_id": runtime.workspace_id},
            )
        for family_key, seed_memory_ids in structural_specs:
            runtime.work_items.enqueue_unique(
                family_key=family_key,
                execution_lane="agentic",
                workspace_id=runtime.workspace_id,
                payload={"seed_memory_ids": seed_memory_ids, "workspace_id": runtime.workspace_id},
            )

        lightweight_result = await call_internal_memory_tool(
            runtime,
            "internal_get_compatible_work_batch",
            {
                "task_id": "compat-lightweight-task",
                "compatibility_group": "lightweight_review",
                "execution_lane": "agentic",
                "workspace_id": runtime.workspace_id,
                "limit": 5,
            },
        )
        lightweight_payload = json.loads(lightweight_result[0].text)

        assert lightweight_payload["status"] == "ok"
        assert lightweight_payload["compatibility_group"] == "lightweight_review"
        assert lightweight_payload["family_keys"] == ["memory_tagging", "graph_link_review", "conflict_review"]
        assert lightweight_payload["claimed_count"] == 3
        assert {record["family_key"] for record in lightweight_payload["records"]} == {
            "memory_tagging",
            "graph_link_review",
            "conflict_review",
        }

        structural_result = await call_internal_memory_tool(
            runtime,
            "internal_get_compatible_work_batch",
            {
                "task_id": "compat-structural-task",
                "compatibility_group": "structural_review",
                "execution_lane": "agentic",
                "workspace_id": runtime.workspace_id,
                "allowed_families": ["memory_curation_review"],
                "limit": 5,
            },
        )
        structural_payload = json.loads(structural_result[0].text)

        assert structural_payload["status"] == "ok"
        assert structural_payload["compatibility_group"] == "structural_review"
        assert structural_payload["family_keys"] == ["memory_curation_review"]
        assert structural_payload["claimed_count"] == 1
        assert [record["family_key"] for record in structural_payload["records"]] == ["memory_curation_review"]

        pending_items = runtime.work_items.list_items(status="pending", limit=10)
        assert {item.family_key for item in pending_items} == {"memory_dedup_review"}
    finally:
        runtime.close()


def test_mcp_server_initializes_with_workspace_root() -> None:
    server = MCPServer(workspace_root="demo-workspace")

    assert server.workspace_root == "demo-workspace"


def test_mcp_server_defaults_to_current_workspace_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("mcp_memory.server.resolve_workspace_root", lambda: tmp_path)
    captured: dict[str, object] = {}

    server = MCPServer()
    server._daemon = object()

    def capture_request(_metadata, _path, payload, *, timeout_seconds=None):
        del timeout_seconds
        captured["payload"] = payload
        return {"status": "ok"}

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", capture_request)

    assert server._request_json("/internal/tools", None) == {"status": "ok"}
    assert captured["payload"] == {"__workspace_root": str(tmp_path)}


def test_mcp_server_tool_requests_retry_once_after_transport_timeout(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    initial_daemon = object()
    refreshed_daemon = object()
    request_calls: list[tuple[object, str, dict | None]] = []
    ensure_calls: list[bool] = []

    def fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        request_calls.append((metadata, path, payload))
        if len(request_calls) == 1:
            raise TimeoutError("daemon_request_timed_out")
        return {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}

    def fake_ensure_daemon_started():
        ensure_calls.append(True)
        return refreshed_daemon

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_ensure_daemon_started)

    server._daemon = initial_daemon

    payload = server._request_json_with_recovery(
        "/internal/tools/record_thought",
        {"content": "retry after timeout"},
    )

    assert payload == {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}
    assert ensure_calls == [True]
    assert request_calls == [
        (
            initial_daemon,
            "/internal/tools/record_thought",
            {"content": "retry after timeout", "__workspace_root": "demo-workspace"},
        ),
        (
            refreshed_daemon,
            "/internal/tools/record_thought",
            {"content": "retry after timeout", "__workspace_root": "demo-workspace"},
        ),
    ]


def test_mcp_server_tool_requests_retry_twice_after_transient_timeouts(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    initial_daemon = object()
    refreshed_daemon_first = object()
    refreshed_daemon_second = object()
    request_calls: list[tuple[object, str, dict | None]] = []
    ensure_calls: list[bool] = []
    refreshed_daemons = iter((refreshed_daemon_first, refreshed_daemon_second))

    def fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        request_calls.append((metadata, path, payload))
        if len(request_calls) < 3:
            raise TimeoutError("daemon_request_timed_out")
        return {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}

    def fake_ensure_daemon_started():
        ensure_calls.append(True)
        return next(refreshed_daemons)

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_ensure_daemon_started)

    server._daemon = initial_daemon

    payload = server._request_json_with_recovery(
        "/internal/tools/record_thought",
        {"content": "retry after multiple timeouts"},
    )

    assert payload == {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}
    assert ensure_calls == [
        True,
        True,
    ]
    assert request_calls == [
        (
            initial_daemon,
            "/internal/tools/record_thought",
            {"content": "retry after multiple timeouts", "__workspace_root": "demo-workspace"},
        ),
        (
            refreshed_daemon_first,
            "/internal/tools/record_thought",
            {"content": "retry after multiple timeouts", "__workspace_root": "demo-workspace"},
        ),
        (
            refreshed_daemon_second,
            "/internal/tools/record_thought",
            {"content": "retry after multiple timeouts", "__workspace_root": "demo-workspace"},
        ),
    ]


def test_mcp_server_request_json_recovers_when_daemon_metadata_is_missing(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    refreshed_daemon = object()
    request_calls: list[tuple[object, str, dict | None]] = []
    ensure_calls: list[bool] = []

    def fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        request_calls.append((metadata, path, payload))
        return {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}

    def fake_ensure_daemon_started():
        ensure_calls.append(True)
        return refreshed_daemon

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_ensure_daemon_started)

    server._daemon = None

    payload = server._request_json_with_recovery(
        "/internal/tools/record_thought",
        {"content": "recover after daemon metadata reset"},
    )

    assert payload == {"contents": [{"type": "text", "text": '{"status": "recorded"}'}]}
    assert ensure_calls == [True]
    assert request_calls == [
        (
            refreshed_daemon,
            "/internal/tools/record_thought",
            {"content": "recover after daemon metadata reset", "__workspace_root": "demo-workspace"},
        ),
    ]


def test_mcp_server_tool_requests_raise_after_exhausting_retry_budget(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    initial_daemon = object()
    refreshed_daemon_first = object()
    refreshed_daemon_second = object()
    request_calls: list[tuple[object, str, dict | None]] = []
    ensure_calls: list[bool] = []
    refreshed_daemons = iter((refreshed_daemon_first, refreshed_daemon_second))

    def fake_request_daemon_json(metadata, path: str, payload: dict | None, *, timeout_seconds=None):
        request_calls.append((metadata, path, payload))
        raise TimeoutError("daemon_request_timed_out")

    def fake_ensure_daemon_started():
        ensure_calls.append(True)
        return next(refreshed_daemons)

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_request_daemon_json)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_ensure_daemon_started)

    server._daemon = initial_daemon

    with pytest.raises(TimeoutError, match="daemon_request_timed_out"):
        server._request_json_with_recovery(
            "/internal/tools/record_thought",
            {"content": "retry budget exhausted"},
        )

    assert ensure_calls == [
        True,
        True,
    ]
    assert request_calls == [
        (
            initial_daemon,
            "/internal/tools/record_thought",
            {"content": "retry budget exhausted", "__workspace_root": "demo-workspace"},
        ),
        (
            refreshed_daemon_first,
            "/internal/tools/record_thought",
            {"content": "retry budget exhausted", "__workspace_root": "demo-workspace"},
        ),
        (
            refreshed_daemon_second,
            "/internal/tools/record_thought",
            {"content": "retry budget exhausted", "__workspace_root": "demo-workspace"},
        ),
    ]


@pytest.mark.asyncio
async def test_mcp_server_list_tools_times_out_when_sync_request_stalls(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    stall = threading.Event()

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        stall.wait(0.05)
        return {"tools": []}

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)

    handler = server.server.get_request_handler("tools/list")
    assert handler is not None

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await handler.handler(cast(Any, None), types.PaginatedRequestParams())


@pytest.mark.asyncio
async def test_mcp_server_call_tool_times_out_when_sync_request_stalls(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    stall = threading.Event()
    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        stall.wait(0.05)
        return {"contents": []}

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)

    handler = server.server.get_request_handler("tools/call")
    assert handler is not None
    result = cast(
        types.CallToolResult,
        await handler.handler(
            cast(Any, None),
            types.CallToolRequestParams(name="record_thought", arguments={"content": "auth"}),
        ),
    )
    result_content = cast(TextContent, result.content[0])

    assert result.is_error is True
    assert result_content.text == "mcp_client_request_timed_out"


def test_mcp_server_client_timeout_budget_defaults_to_sixty_seconds() -> None:
    from mcp_memory import server as server_module

    assert server_module._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS == 60.0


@pytest.mark.asyncio
async def test_mcp_server_call_tool_surfaces_daemon_timeout_error_payload(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    def _timed_out_request(_path: str, _payload: dict | None) -> dict[str, object]:
        return {"status": "error", "error": "daemon_request_timed_out"}

    monkeypatch.setattr(server, "_request_json_with_recovery", _timed_out_request)

    handler = server.server.get_request_handler("tools/call")
    assert handler is not None
    result = cast(
        types.CallToolResult,
        await handler.handler(
            cast(Any, None),
            types.CallToolRequestParams(name="record_thought", arguments={"content": "auth"}),
        ),
    )
    result_content = cast(TextContent, result.content[0])

    assert result.is_error is True
    assert result_content.text == "daemon_request_timed_out"


@pytest.mark.asyncio
async def test_mcp_server_client_timeout_clears_cached_daemon_and_schedules_recovery(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    request_stall = threading.Event()
    recovery_started = threading.Event()
    allow_recovery = threading.Event()

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def blocking_recovery():
        recovery_started.set()
        allow_recovery.wait(1)
        return object()

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", blocking_recovery)

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert server._daemon is None
    assert server._daemon_recovery_task is not None
    assert await asyncio.to_thread(recovery_started.wait, 1)

    allow_recovery.set()
    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)


@pytest.mark.asyncio
async def test_mcp_server_repeated_client_timeouts_do_not_spawn_duplicate_recovery_tasks(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    request_stall = threading.Event()
    recovery_started = threading.Event()
    allow_recovery = threading.Event()
    recovery_call_count = 0

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def blocking_recovery():
        nonlocal recovery_call_count
        recovery_call_count += 1
        recovery_started.set()
        allow_recovery.wait(1)
        return object()

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", blocking_recovery)

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert await asyncio.to_thread(recovery_started.wait, 1)

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert recovery_call_count == 1

    allow_recovery.set()
    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)


@pytest.mark.asyncio
async def test_mcp_server_successful_background_recovery_updates_cached_daemon(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    request_stall = threading.Event()
    refreshed_daemon = object()

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def successful_recovery():
        return refreshed_daemon

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", successful_recovery)

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)

    assert server._daemon is refreshed_daemon


@pytest.mark.asyncio
async def test_mcp_server_successful_request_after_timeout_succeeds(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    request_stall = threading.Event()
    recovered_daemon = object()
    request_count = 0

    def sometimes_blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            request_stall.wait(0.05)
        return {"tools": []}

    def successful_recovery():
        return recovered_daemon

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", sometimes_blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", successful_recovery)

    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)

    payload = await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert payload == {"tools": []}


@pytest.mark.asyncio
async def test_mcp_server_sync_timeout_does_not_count_as_client_timeout(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    daemon = object()

    def timed_out_request(_path: str, _payload: dict | None) -> dict[str, object]:
        raise TimeoutError("daemon_request_timed_out")

    monkeypatch.setattr(server, "_request_json_with_recovery", timed_out_request)

    server._daemon = daemon

    with pytest.raises(TimeoutError, match="daemon_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert server._daemon is daemon
    assert server._daemon_recovery_task is None


@pytest.mark.asyncio
async def test_mcp_server_repeated_client_timeouts_refresh_without_stopping_daemon(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    request_stall = threading.Event()
    ensure_calls: list[bool] = []
    refreshed_daemons = [object(), object()]

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def successful_recovery():
        ensure_calls.append(True)
        return refreshed_daemons[len(ensure_calls) - 1]

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", successful_recovery)

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)

    assert ensure_calls == [True]

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    await asyncio.wait_for(cast(asyncio.Task[None], server._daemon_recovery_task), timeout=1)

    assert ensure_calls == [
        True,
        True,
    ]
    assert server._daemon is refreshed_daemons[-1]


@pytest.mark.asyncio
async def test_mcp_server_recovery_coalesces_inflight_timeout(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    request_stall = threading.Event()
    first_recovery_started = threading.Event()
    allow_first_recovery = threading.Event()
    ensure_calls: list[bool] = []
    recovered_daemon = object()

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def staged_recovery():
        ensure_calls.append(True)
        first_recovery_started.set()
        allow_first_recovery.wait(1)
        return recovered_daemon

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", staged_recovery)

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert await asyncio.to_thread(first_recovery_started.wait, 1)
    first_recovery_task = server._daemon_recovery_task

    server._daemon = object()
    with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
        await server._request_daemon_json_with_client_timeout("/internal/tools", None)

    assert server._daemon_recovery_task is first_recovery_task

    allow_first_recovery.set()
    await asyncio.wait_for(cast(asyncio.Task[None], first_recovery_task), timeout=1)

    assert ensure_calls == [True]
    assert server._daemon is recovered_daemon


@pytest.mark.asyncio
async def test_mcp_server_failed_background_recovery_logs_warning(monkeypatch, caplog: pytest.LogCaptureFixture) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()
    request_stall = threading.Event()

    def blocking_request(_path: str, _payload: dict | None) -> dict[str, object]:
        request_stall.wait(0.05)
        return {"tools": []}

    def failing_recovery():
        raise RuntimeError("recovery failed for demo-workspace")

    monkeypatch.setattr(
        "mcp_memory.server._DAEMON_BACKED_MCP_CLIENT_TIMEOUT_SECONDS",
        0.01,
    )
    monkeypatch.setattr(server, "_request_json_with_recovery", blocking_request)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", failing_recovery)

    with caplog.at_level("WARNING"):
        with pytest.raises(TimeoutError, match="mcp_client_request_timed_out"):
            await server._request_daemon_json_with_client_timeout("/internal/tools", None)

        while server._daemon_recovery_task is not None:
            await asyncio.sleep(0)

    assert server._daemon is None
    assert "Background daemon metadata recovery failed after client timeout" in caplog.text
    warning_record = next(
        record
        for record in caplog.records
        if record.message == "Background daemon metadata recovery failed after client timeout"
    )
    assert warning_record.__dict__["workspace_root"] == "demo-workspace"
    assert warning_record.__dict__["error"] == "recovery failed for demo-workspace"


@pytest.mark.asyncio
async def test_mcp_server_suspend_monitor_clears_daemon_and_starts_recovery(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    sentinel_daemon = object()
    server._daemon = sentinel_daemon

    recovery_started = threading.Event()
    allow_recovery = threading.Event()
    refreshed_daemon = object()

    def fake_recovery():
        recovery_started.set()
        allow_recovery.wait(2)
        return refreshed_daemon

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_recovery)

    # Simulate a large CLOCK_BOOTTIME jump on the second call (machine slept for 60s).
    boottime_calls = [0]
    mono_calls = [0]

    def fake_boottime() -> float:
        boottime_calls[0] += 1
        return 0.0 if boottime_calls[0] == 1 else 61.0

    def fake_monotonic() -> float:
        mono_calls[0] += 1
        return 0.0 if mono_calls[0] == 1 else 0.001

    monkeypatch.setattr("mcp_memory.server.suspend_aware_now", fake_boottime)
    monkeypatch.setattr("mcp_memory.server._time_module", type("_t", (), {"monotonic": staticmethod(fake_monotonic)})())
    monkeypatch.setattr("mcp_memory.server._SUSPEND_RESUME_CHECK_INTERVAL_SECONDS", 0.01)

    task = asyncio.create_task(server._monitor_suspend_resume())
    try:
        # Wait for recovery to start (confirms monitor fired and invalidated the daemon).
        assert await asyncio.to_thread(recovery_started.wait, 1)
        # While recovery is in-flight, daemon is cleared.
        assert server._daemon is None
        # A recovery task was scheduled.
        assert server._daemon_recovery_task is not None
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        allow_recovery.set()
        if server._daemon_recovery_task is not None:
            await asyncio.wait_for(server._daemon_recovery_task, timeout=2)


@pytest.mark.asyncio
async def test_mcp_server_suspend_monitor_does_not_trigger_without_sleep(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    sentinel_daemon = object()
    server._daemon = sentinel_daemon

    recovery_started = threading.Event()

    def fake_recovery():
        recovery_started.set()
        return object()

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_recovery)

    # Both boottime and monotonic advance by ~1s per call (no sleep occurred).
    call_count = [0]

    def fake_boottime() -> float:
        call_count[0] += 1
        return float(call_count[0])

    def fake_monotonic() -> float:
        return float(call_count[0])

    monkeypatch.setattr("mcp_memory.server.suspend_aware_now", fake_boottime)
    monkeypatch.setattr("mcp_memory.server._time_module", type("_t", (), {"monotonic": staticmethod(fake_monotonic)})())
    monkeypatch.setattr("mcp_memory.server._SUSPEND_RESUME_CHECK_INTERVAL_SECONDS", 0.01)

    task = asyncio.create_task(server._monitor_suspend_resume())
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert server._daemon is sentinel_daemon
    assert not recovery_started.is_set()


@pytest.mark.asyncio
async def test_mcp_server_tool_handlers_succeed_with_client_timeout_wrapper(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    server._daemon = object()

    def fake_request(path: str, payload: dict | None) -> dict[str, object]:
        if path == "/internal/tools":
            assert payload is None
            return {
                "tools": [
                    {
                        "name": "record_thought",
                        "description": "Record a durable memory.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"content": {"type": "string"}},
                            "required": ["content"],
                        },
                    }
                ]
            }
        assert path == "/internal/tools/record_thought"
        assert payload == {"content": "auth"}
        return {"contents": [{"text": '{"status": "recorded"}'}]}

    monkeypatch.setattr(server, "_request_json_with_recovery", fake_request)

    list_tools_handler = server.server.get_request_handler("tools/list")
    assert list_tools_handler is not None
    list_result = cast(
        types.ListToolsResult,
        await list_tools_handler.handler(cast(Any, None), types.PaginatedRequestParams()),
    )

    assert [tool.name for tool in list_result.tools] == ["record_thought"]
    assert list_result.tools[0].description == "Record a durable memory."

    call_tool_handler = server.server.get_request_handler("tools/call")
    assert call_tool_handler is not None
    call_result = cast(
        types.CallToolResult,
        await call_tool_handler.handler(
            cast(Any, None),
            types.CallToolRequestParams(name="record_thought", arguments={"content": "auth"}),
        ),
    )
    call_result_content = cast(TextContent, call_result.content[0])

    assert call_result.is_error is False
    assert call_result_content.text == '{"status": "recorded"}'


@pytest.mark.asyncio
async def test_call_memory_tool_records_real_journal_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        result = await call_memory_tool(runtime, "record_thought", {"content": "wire up mcp handlers"})
        payload = json.loads(result[0].text)

        assert payload == {"status": "recorded"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_append_and_archive(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert record is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_memory_content",
            {"memory_id": record.id, "content": "Prefer deterministic fixtures.", "tags": ["preferences"]},
        )
        archive_result = await call_internal_memory_tool(
            runtime,
            "internal_archive_memory_record",
            {"memory_id": record.id},
        )

        appended_payload = json.loads(append_result[0].text)
        archived_payload = json.loads(archive_result[0].text)
        assert appended_payload["status"] == "ok"
        updated = runtime.repository.get_memory(record.id)
        assert updated is not None and "deterministic fixtures" in updated.content
        assert "content" not in appended_payload["record"]
        assert "metadata" not in appended_payload["record"]
        assert archived_payload["record"]["status"] == "archived"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_rejects_invalid_ingest_mutations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        create_result = await call_internal_memory_tool(
            runtime,
            "internal_ingest_create_memory",
            {
                "task_id": "ingest-invalid",
                "entry_ids": [],
                "title": "Marker",
                "content": "marker",
            },
        )

        create_payload = json.loads(create_result[0].text)
        assert create_payload["status"] == "error"
        assert create_payload["error"] == "invalid_ingest_payload"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_append_workspace_ids_and_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
            metadata={"appended_entry_ids": [1]},
        )
        assert record is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_memory_content",
            {
                "memory_id": record.id,
                "content": "Prefer deterministic fixtures.",
                "workspace_ids": ["workspace-b"],
                "metadata": {"appended_entry_ids": [2], "ingest_task_id": "ingest-agentic-task"},
            },
        )

        payload = json.loads(append_result[0].text)

        assert payload["status"] == "ok"
        updated = runtime.repository.get_memory(record.id)
        assert updated is not None and "deterministic fixtures" in updated.content
        assert set(payload["record"]["workspace_ids"]) == {runtime.workspace_id, "workspace-b"}
        assert "content" not in payload["record"]
        assert "metadata" not in payload["record"]
        assert updated.metadata["appended_entry_ids"] == [1, 2]
        assert updated.metadata["ingest_task_id"] == "ingest-agentic-task"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_fetch_dedup_curator_and_ingest_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.journal is not None
        fact = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        observation = runtime.repository.create_memory(
            title="Auth observation",
            content="Observed another note about JWT enforcement.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        runtime.journal.record("JWT rollout note one", workspace_id=runtime.workspace_id)
        runtime.journal.record("JWT rollout note two", workspace_id=runtime.workspace_id)
        assert fact is not None and observation is not None

        dedup_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_dedup_batch",
            {"task_id": "dedup-batch-test", "strategy": "anomaly", "limit": 20},
        )
        curator_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_curator_batch",
            {"task_id": "curator-batch-test", "limit": 5, "exclude_memory_ids": [fact.id]},
        )
        ingest_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_ingest_batch",
            {"task_id": "agentic-ingest-batch", "batch_size": 10, "grouping_strategy": "fifo"},
        )

        dedup_payload = json.loads(dedup_result[0].text)
        curator_payload = json.loads(curator_result[0].text)
        ingest_payload = json.loads(ingest_result[0].text)

        assert dedup_payload["status"] == "ok"
        assert dedup_payload["records"]
        assert {record["id"] for record in dedup_payload["records"]} >= {fact.id, observation.id}
        assert dedup_payload["requested_strategy"] == "anomaly"
        assert dedup_payload["strategy"] == "anomaly"
        assert dedup_payload["strategy_used"] == "anomaly"
        assert dedup_payload["strategy_fallback_reason"] is None
        assert dedup_payload["candidate_count"] >= len(dedup_payload["records"])
        assert dedup_payload["sampled_memory_ids"]

        assert curator_payload["status"] == "ok"
        assert curator_payload["requested_strategy"] is None
        assert curator_payload["strategy"] == curator_payload["strategy_used"]
        assert curator_payload["strategy_fallback_reason"] is None
        assert curator_payload["strategy_selection_mode"] == "deterministic_scores"
        assert isinstance(curator_payload["strategy_selection_reason"], str)
        assert isinstance(curator_payload["strategy_selection_scores"], dict)
        assert curator_payload["strategy"] in curator_payload["strategy_selection_scores"]
        assert curator_payload["excluded_memory_ids"] == [fact.id]
        assert curator_payload["records"]
        assert all(record["id"] != fact.id for record in curator_payload["records"])
        assert curator_payload["sampled_memory_ids"]

        assert ingest_payload["status"] == "ok"
        assert ingest_payload["task_id"] == "agentic-ingest-batch"
        assert ingest_payload["requested_grouping_strategy"] == "fifo"
        assert ingest_payload["grouping_strategy_used"] == "fifo"
        assert ingest_payload["grouping_fallback_reason"] is None
        assert len(ingest_payload["claimed_entry_ids"]) == 2
        assert ingest_payload["group_count"] >= 1
        assert ingest_payload["groups"]
        assert ingest_payload["groups"][0]["entries"]
        assert ingest_payload["groups"][0]["entries"][0]["status"] == "claimed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_ingest_batch_claims_pending_entries_across_workspaces_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.journal is not None
        runtime.journal.record("cross-workspace pending note one", workspace_id=runtime.workspace_id)
        runtime.journal.record("cross-workspace pending note two", workspace_id=runtime.workspace_id)

        mismatched_runtime = replace(runtime, workspace_id="workspace-other")
        ingest_result = await call_internal_memory_tool(
            mismatched_runtime,
            "internal_get_next_ingest_batch",
            {"task_id": "cross-workspace-ingest", "batch_size": 10, "grouping_strategy": "fifo"},
        )

        ingest_payload = json.loads(ingest_result[0].text)

        assert ingest_payload["status"] == "ok"
        assert ingest_payload["task_id"] == "cross-workspace-ingest"
        assert len(ingest_payload["claimed_entry_ids"]) == 2
        assert ingest_payload["groups"]
        assert ingest_payload["groups"][0]["entries"][0]["status"] == "claimed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_ingest_batch_falls_back_from_unknown_workspace_to_global_pending(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.journal is not None
        runtime.journal.record("global fallback pending note one", workspace_id=runtime.workspace_id)
        runtime.journal.record("global fallback pending note two", workspace_id=runtime.workspace_id)

        mismatched_runtime = replace(runtime, workspace_id="workspace-other")
        ingest_result = await call_internal_memory_tool(
            mismatched_runtime,
            "internal_get_next_ingest_batch",
            {
                "task_id": "workspace-unknown-ingest",
                "batch_size": 10,
                "grouping_strategy": "fifo",
                "workspace_id": "workspace-unknown",
            },
        )

        ingest_payload = json.loads(ingest_result[0].text)

        assert ingest_payload["status"] == "ok"
        assert ingest_payload["task_id"] == "workspace-unknown-ingest"
        assert len(ingest_payload["claimed_entry_ids"]) == 2
        assert ingest_payload["groups"]
        assert ingest_payload["groups"][0]["entries"][0]["status"] == "claimed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_ingest_tools_preserve_ingest_invariants(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        existing = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
            metadata={"appended_entry_ids": [1]},
        )
        assert existing is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_to_existing_memory_for_ingest",
            {
                "memory_id": existing.id,
                "content": "Prefer deterministic fixtures.",
                "task_id": "ingest-maintenance-task",
                "entry_ids": [2],
                "workspace_ids": ["workspace-b"],
                "tags": ["testing"],
            },
        )
        create_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record_for_ingest",
            {
                "title": "Pytest fixture policy",
                "content": "- [2026-03-16 12:00] Prefer deterministic fixtures.",
                "task_id": "ingest-maintenance-task",
                "entry_ids": [3, "4"],
                "workspace_ids": [runtime.workspace_id, "workspace-b"],
                "tags": ["pytest"],
            },
        )

        append_payload = json.loads(append_result[0].text)
        create_payload = json.loads(create_result[0].text)
        appended = runtime.repository.get_memory(existing.id)
        created = runtime.repository.get_memory(create_payload["record"]["id"])
        assert appended is not None and created is not None
        assert append_payload["status"] == "ok"
        assert "deterministic fixtures" in appended.content
        assert set(append_payload["record"]["workspace_ids"]) == {runtime.workspace_id, "workspace-b"}
        assert append_payload["handled_entry_ids"] == [2]
        assert appended.metadata["appended_entry_ids"] == [1, 2]
        assert appended.metadata["appended_via_ingest"] is True
        assert appended.metadata["ingest_task_id"] == "ingest-maintenance-task"
        assert append_payload["record"]["tags"] == ["testing"]

        assert create_payload["status"] == "ok"
        assert create_payload["handled_entry_ids"] == [3, 4]
        assert created.metadata["created_via_ingest"] is True
        assert created.metadata["source_entry_ids"] == [3, 4]
        assert created.metadata["ingest_task_id"] == "ingest-maintenance-task"
        assert create_payload["record"]["tags"] == ["pytest"]
        assert create_payload["record"]["summary"]
        assert runtime.task_queue.find_open_task("summarize-memory", runtime.workspace_id or "global") is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_create_update_link_and_delete(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        created_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record",
            {
                "title": "Canonical testing preference",
                "content": "Prefer deterministic fixtures.",
                "memory_type": "fact",
                "tags": ["testing"],
            },
        )
        created_payload = json.loads(created_result[0].text)
        created_id = created_payload["record"]["id"]

        related = runtime.repository.create_memory(
            title="Related testing note",
            content="Pytest remains the standard.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
        )
        assert related is not None

        update_result = await call_internal_memory_tool(
            runtime,
            "internal_update_memory_record",
            {"memory_id": created_id, "content": "Prefer deterministic pytest fixtures.", "tags": ["testing", "pytest"]},
        )
        updated = runtime.repository.get_memory(created_id)
        assert updated is not None and updated.content == "Prefer deterministic pytest fixtures."
        create_link_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_link",
            {"source_id": created_id, "target_id": related.id, "link_type": "AMENDS", "context": "Curator linked related note."},
        )
        archive_result = await call_internal_memory_tool(
            runtime,
            "internal_archive_memory_record",
            {"memory_id": created_id},
        )
        delete_link_result = await call_internal_memory_tool(
            runtime,
            "internal_delete_memory_link",
            {"source_id": created_id, "target_id": related.id, "link_type": "AMENDS"},
        )
        delete_result = await call_internal_memory_tool(
            runtime,
            "internal_delete_memory_record",
            {"memory_id": created_id, "confirm": True},
        )

        update_payload = json.loads(update_result[0].text)
        create_link_payload = json.loads(create_link_result[0].text)
        archive_payload = json.loads(archive_result[0].text)
        delete_link_payload = json.loads(delete_link_result[0].text)
        delete_payload = json.loads(delete_result[0].text)

        assert created_payload["status"] == "ok"
        assert "content" not in update_payload["record"]
        assert update_payload["record"]["tags"] == ["pytest", "testing"]
        assert create_link_payload["status"] == "ok"
        assert archive_payload["record"]["status"] == "archived"
        assert delete_link_payload["status"] == "ok"
        assert delete_payload["status"] == "ok"
        assert runtime.repository.get_memory(created_id) is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_create_record_and_enqueue_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.task_queue is not None
        created_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record",
            {
                "title": "Canonical testing preference",
                "content": "Prefer deterministic fixtures.",
                "memory_type": "observation",
                "tags": ["testing"],
                "enqueue_summary_task": True,
            },
        )
        created_payload = json.loads(created_result[0].text)
        created_id = created_payload["record"]["id"]
        summary_task = runtime.task_queue.find_open_task_with_data_any_workspace(
            "summarize-memory",
            data_fields={"memory_id": created_id},
        )

        assert created_payload["status"] == "ok"
        assert summary_task is not None
        assert summary_task.data["memory_id"] == created_id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_merge_can_override_canonical_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
            metadata={"merged_source_ids": ["older-source"]},
        )
        source = runtime.repository.create_memory(
            title="Auth rollout detail",
            content="JWT rollout is now required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth", "rollout"],
        )
        assert canonical is not None and source is not None

        merge_result = await call_internal_memory_tool(
            runtime,
            "internal_merge_memory_into_canonical",
            {
                "canonical_memory_id": canonical.id,
                "source_memory_id": source.id,
                "title": "Canonical auth rollout fact",
                "content": "JWT rollout is required for all clients.",
                "summary": "Unified rollout requirement.",
                "tags": ["auth", "rollout"],
                "metadata": {"deduplicator_task_id": "dedup-live-task"},
                "link_context": "Merged rollout observation into canonical auth fact.",
            },
        )

        payload = json.loads(merge_result[0].text)
        updated = runtime.repository.get_memory(canonical.id)
        archived = runtime.repository.get_memory(source.id)
        links = runtime.repository.get_links(canonical.id)

        assert payload["status"] == "ok"
        assert payload["canonical"]["title"] == "Canonical auth rollout fact"
        assert payload["canonical"]["summary"] == "Unified rollout requirement."
        assert payload["canonical"]["tags"] == ["auth", "rollout"]
        assert updated is not None and updated.metadata["deduplicator_task_id"] == "dedup-live-task"
        assert updated.metadata["merged_source_ids"] == sorted(["older-source", source.id])
        assert archived is not None and archived.status == "archived"
        assert any(link.target_id == source.id and link.context == "Merged rollout observation into canonical auth fact." for link in links)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_split_memory_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        original = runtime.repository.create_memory(
            title="Oversized auth rollout",
            content="Part one. Part two. Part three.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "rollout"],
        )
        assert original is not None

        split_result = await call_internal_memory_tool(
            runtime,
            "internal_split_memory_record",
            {
                "memory_id": original.id,
                "archive_original": True,
                "parts": [
                    {"title": "Auth rollout prerequisites", "content": "Part one.", "tags": ["prereq"]},
                    {"title": "Auth rollout execution", "content": "Part two. Part three.", "tags": ["execution"]},
                ],
            },
        )

        payload = json.loads(split_result[0].text)
        first_child_id = payload["created"][0]["id"]
        second_child_id = payload["created"][1]["id"]
        first_links = runtime.repository.get_links(first_child_id)
        second_links = runtime.repository.get_links(second_child_id)
        archived_original = runtime.repository.get_memory(original.id)
        first_record = runtime.repository.get_memory(first_child_id)
        second_record = runtime.repository.get_memory(second_child_id)
        original_record = runtime.repository.get_memory(original.id)
        assert first_record is not None and second_record is not None and original_record is not None
        first_metadata = first_record.metadata
        second_metadata = second_record.metadata
        original_metadata = original_record.metadata

        assert payload["status"] == "ok"
        assert len(payload["created"]) == 2
        assert all("content" not in child and "metadata" not in child for child in payload["created"])
        assert payload["archived"]["status"] == "archived"
        assert archived_original is not None and archived_original.status == "archived"
        assert any(link.target_id == original.id and link.link_type == "DEPENDS_ON" for link in first_links)
        assert any(link.target_id == original.id and link.link_type == "DEPENDS_ON" for link in second_links)
        assert first_metadata["split_from_memory_id"] == original.id
        assert second_metadata["split_from_memory_id"] == original.id
        assert first_metadata["split_group_id"] == second_metadata["split_group_id"] == original_metadata["split_group_id"]
        assert first_metadata["split_part_index"] == 1
        assert second_metadata["split_part_index"] == 2
        assert first_metadata["split_part_count"] == 2
        assert second_metadata["split_part_count"] == 2
        assert first_metadata["split_sibling_memory_ids"] == [second_child_id]
        assert second_metadata["split_sibling_memory_ids"] == [first_child_id]
        assert original_metadata["split_child_memory_ids"] == [first_child_id, second_child_id]
        assert original_metadata["split_child_count"] == 2
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_mcp_server_run_autostarts_daemon_and_invokes_stdio(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    captured: dict[str, object] = {}
    hook_calls: list[tuple[str, dict[str, object]]] = []
    events: list[str] = []

    server = MCPServer(workspace_root="demo")

    async def fake_run(read_arg, write_arg, init_options) -> None:
        events.append("stdio-run")
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda: FakeMetadata())
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: events.append(path) or hook_calls.append((path, payload)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((read_stream, write_stream)),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert captured["read_stream"] is read_stream
    assert captured["write_stream"] is write_stream
    assert captured["init_options"] is not None
    assert [path for path, _ in hook_calls] == ["/api/hooks/session-start", "/api/hooks/session-end"]
    assert events.index("/api/hooks/session-start") < events.index("/api/hooks/session-end")
    start_payload = hook_calls[0][1]
    end_payload = hook_calls[1][1]
    assert start_payload["session_id"] == end_payload["session_id"]
    assert start_payload["source"] == "mcp-stdio"


@pytest.mark.asyncio
async def test_mcp_server_run_still_sends_session_end_hook_on_failure(monkeypatch) -> None:
    hook_calls: list[tuple[str, dict[str, object]]] = []
    server = MCPServer(workspace_root="demo")

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    async def fake_run(*_args) -> None:
        raise RuntimeError("stdio disconnected")

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda: FakeMetadata())
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: hook_calls.append((path, payload)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((object(), object())),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    with pytest.raises(RuntimeError, match="stdio disconnected"):
        await server.run()

    assert [path for path, _ in hook_calls] == ["/api/hooks/session-start", "/api/hooks/session-end"]
    assert hook_calls[0][1]["session_id"] == hook_calls[1][1]["session_id"]


@pytest.mark.asyncio
async def test_mcp_server_run_does_not_start_daemon_only_for_session_end(monkeypatch) -> None:
    hook_calls: list[str] = []
    server = MCPServer(workspace_root="demo")

    def _daemon_unavailable(*_args, **_kwargs):
        raise RuntimeError("daemon unavailable")

    async def fake_run(*_args) -> None:
        return None

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", _daemon_unavailable)
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: hook_calls.append(path) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((object(), object())),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert hook_calls == []


def test_cli_help_lists_grouped_public_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])
    help_lines = result.output.splitlines()
    command_lines = [line.strip() for line in help_lines if line.startswith("  ")]

    assert result.exit_code == 0
    assert "run" in result.output
    assert "daemon" in result.output
    assert "memory" in result.output
    assert "admin" in result.output
    assert not any(line.startswith("log") for line in command_lines)
    assert not any(line.startswith("install") for line in command_lines)
    assert not any(line.startswith("agents") for line in command_lines)
    assert not any(line.startswith("stats") for line in command_lines)
    assert not any(line.startswith("stash") for line in command_lines)
    assert not any(line.startswith("import-markdown") for line in command_lines)
    assert "daemon-status" not in result.output
    assert "daemon-stop" not in result.output
    assert "daemon-restart" not in result.output
    assert "dashboard" not in result.output
    assert not any(line.strip().startswith("logs") for line in help_lines)
    assert not any(line.strip().startswith("log-summary") for line in help_lines)
    assert not any(line.strip().startswith("log-prune") for line in help_lines)


def test_daemon_help_lists_nested_management_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["daemon", "--help"])

    assert result.exit_code == 0
    assert "status" in result.output
    assert "stop" in result.output
    assert "restart" in result.output
    assert "dashboard" not in result.output


def test_admin_log_help_lists_nested_log_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["admin", "log", "--help"])

    assert result.exit_code == 0
    assert "list" in result.output
    assert "summary" in result.output
    assert "prune" in result.output


@pytest.mark.asyncio
async def test_mcp_server_health_monitor_clears_daemon_and_starts_recovery_on_probe_failure(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    sentinel_daemon = object()
    server._daemon = sentinel_daemon

    recovery_started = threading.Event()
    allow_recovery = threading.Event()
    refreshed_daemon = object()

    def fake_recovery():
        recovery_started.set()
        allow_recovery.wait(2)
        return refreshed_daemon

    def fake_probe(metadata, path: str, payload, *, timeout_seconds=None):
        raise TimeoutError("daemon_request_timed_out")

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_probe)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_recovery)
    monkeypatch.setattr("mcp_memory.server._DAEMON_HEALTH_POLL_INTERVAL_SECONDS", 0.01)

    task = asyncio.create_task(server._monitor_daemon_health())
    try:
        assert await asyncio.to_thread(recovery_started.wait, 1)
        assert server._daemon is None
        assert server._daemon_recovery_task is not None
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        allow_recovery.set()
        if server._daemon_recovery_task is not None:
            await asyncio.wait_for(server._daemon_recovery_task, timeout=2)


@pytest.mark.asyncio
async def test_mcp_server_health_monitor_does_not_trigger_when_probe_succeeds(monkeypatch) -> None:
    server = MCPServer(workspace_root="demo-workspace")
    sentinel_daemon = object()
    server._daemon = sentinel_daemon

    recovery_started = threading.Event()

    def fake_recovery():
        recovery_started.set()
        return object()

    def fake_probe(metadata, path: str, payload, *, timeout_seconds=None):
        return {"status": "ready"}

    monkeypatch.setattr("mcp_memory.server.request_daemon_json", fake_probe)
    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", fake_recovery)
    monkeypatch.setattr("mcp_memory.server._DAEMON_HEALTH_POLL_INTERVAL_SECONDS", 0.01)

    task = asyncio.create_task(server._monitor_daemon_health())
    try:
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert server._daemon is sentinel_daemon
    assert not recovery_started.is_set()

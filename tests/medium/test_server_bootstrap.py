import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp.types import TextContent

from mcp_memory.cli import main
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools
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
    names = [tool.name for tool in get_memory_tools()]

    assert names == [
        "record_thought",
        "search_memory_records",
        "read_memory_record",
    ]


def test_get_internal_maintenance_tools_returns_expected_names() -> None:
    names = [tool.name for tool in get_internal_maintenance_tools()]

    assert names == [
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_list_memory_records",
        "internal_append_memory_content",
        "internal_archive_memory_record",
        "internal_merge_memory_into_canonical",
        "internal_create_memory_record",
        "internal_update_memory_record",
        "internal_delete_memory_record",
        "internal_create_memory_link",
        "internal_delete_memory_link",
    ]


def test_mcp_server_initializes_with_workspace_root() -> None:
    server = MCPServer(workspace_root="demo-workspace")

    assert server.workspace_root == "demo-workspace"


@pytest.mark.asyncio
async def test_call_memory_tool_records_real_journal_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        result = await call_memory_tool(runtime, "record_thought", {"content": "wire up mcp handlers"})
        payload = json.loads(result[0].text)

        assert payload["status"] == "recorded"
        assert payload["entry"]["content"] == "wire up mcp handlers"
        assert payload["entry"]["workspace_id"] == runtime.workspace_id
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
        assert "deterministic fixtures" in appended_payload["record"]["content"]
        assert archived_payload["record"]["status"] == "archived"
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
        assert update_payload["record"]["content"] == "Prefer deterministic pytest fixtures."
        assert update_payload["record"]["tags"] == ["pytest", "testing"]
        assert create_link_payload["status"] == "ok"
        assert archive_payload["record"]["status"] == "archived"
        assert delete_link_payload["status"] == "ok"
        assert delete_payload["status"] == "ok"
        assert runtime.repository.get_memory(created_id) is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_mcp_server_run_autostarts_daemon_and_invokes_stdio(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    captured: dict[str, object] = {}
    hook_calls: list[tuple[str, dict[str, object]]] = []

    server = MCPServer(workspace_root="demo")

    async def fake_run(read_arg, write_arg, init_options) -> None:
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: hook_calls.append((path, payload)) or {"status": "ok"},
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

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())
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


def test_cli_help_lists_grouped_public_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])
    help_lines = result.output.splitlines()

    assert result.exit_code == 0
    assert "run" in result.output
    assert "daemon" in result.output
    assert "log" in result.output
    assert "install" in result.output
    assert "agents" in result.output
    assert "stats" in result.output
    assert "import-markdown" in result.output
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
    assert "dashboard" in result.output


def test_log_help_lists_nested_log_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["log", "--help"])

    assert result.exit_code == 0
    assert "show" in result.output
    assert "summary" in result.output
    assert "prune" in result.output

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp.types import TextContent

from mcp_memory.cli import main
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
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
        "get_pending_thoughts",
        "get_memory_stats",
        "create_memory_record",
        "get_memory_record",
        "list_memory_records",
        "search_memory_records",
        "read_memory_record",
        "import_markdown_memory_file",
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
async def test_mcp_server_run_autostarts_daemon_and_invokes_stdio(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    captured: dict[str, object] = {}

    server = MCPServer(workspace_root="demo")

    async def fake_run(read_arg, write_arg, init_options) -> None:
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((read_stream, write_stream)),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert captured["read_stream"] is read_stream
    assert captured["write_stream"] is write_stream
    assert captured["init_options"] is not None


def test_cli_help_lists_run_daemon_and_dashboard_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "daemon" in result.output
    assert "dashboard" in result.output

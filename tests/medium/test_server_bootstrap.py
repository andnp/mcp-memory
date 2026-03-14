import json

import pytest
from click.testing import CliRunner

from mcp.types import TextContent

from mcp_memory.cli import main
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.tools import get_memory_tools
from mcp_memory.server import MCPServer
from tests.sdk.mcp import FakeAsyncContextManager


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_call_memory_tool_returns_placeholder_payload() -> None:
    result = await call_memory_tool(None, "search_memories", {"query": "auth"})

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    payload = json.loads(result[0].text)
    assert payload["status"] == "not_implemented"
    assert payload["tool"] == "search_memories"


def test_get_memory_tools_returns_empty_list() -> None:
    assert get_memory_tools() == []


def test_mcp_server_initializes_with_project_override() -> None:
    server = MCPServer(project_override="demo-project")

    assert server.project_override == "demo-project"
    assert server.ctx is None


@pytest.mark.asyncio
async def test_mcp_server_run_uses_stdio_server(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    captured: dict[str, object] = {}

    server = MCPServer()

    async def fake_run(read_arg, write_arg, init_options) -> None:
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((read_stream, write_stream)),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert captured["read_stream"] is read_stream
    assert captured["write_stream"] is write_stream
    assert captured["init_options"] is not None


def test_cli_stats_command_reports_placeholder() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["stats"])

    assert result.exit_code == 0
    assert "Stats tool coming soon" in result.output
from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp_memory.cli import main
from mcp_memory.server import MCPServer
from tests.sdk.mcp import FakeAsyncContextManager


pytestmark = pytest.mark.large


@pytest.mark.asyncio
async def test_server_run_initializes_runtime_and_invokes_stdio(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    fake_runtime = type("FakeRuntime", (), {"close": lambda self: None})()
    server = MCPServer(project_override="demo")
    captured: dict[str, object] = {}

    async def fake_run(read_arg, write_arg, init_options) -> None:
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    monkeypatch.setattr("mcp_memory.server.create_runtime", lambda project_override, cwd: fake_runtime)
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((read_stream, write_stream)),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert server.ctx is fake_runtime
    assert captured["read_stream"] is read_stream
    assert captured["write_stream"] is write_stream


def test_cli_run_invokes_server_with_project_override(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, project_override: str | None = None):
            captured["project_override"] = project_override

        async def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("mcp_memory.cli.MCPServer", FakeServer)

    result = runner.invoke(main, ["run", "--project", "demo-project"])

    assert result.exit_code == 0
    assert captured["project_override"] == "demo-project"
    assert captured["ran"] is True


def test_cli_help_lists_run_and_stats_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.output
    assert "stats" in result.output
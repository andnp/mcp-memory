from click.testing import CliRunner

from mcp_memory.cli import main


def test_cli_run_invokes_server_with_workspace_root(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(
            self,
            workspace_root: str | None = None,
            *,
            server_name: str = "mcp-memory",
            tool_path_prefix: str = "/internal/tools",
        ):
            captured["workspace_root"] = workspace_root
            captured["server_name"] = server_name
            captured["tool_path_prefix"] = tool_path_prefix

        async def run(self) -> None:
            captured["ran"] = True

    monkeypatch.setattr("mcp_memory.cli.MCPServer", FakeServer)

    result = runner.invoke(main, ["run", "--workspace-root", "demo-workspace"])

    assert result.exit_code == 0
    assert captured["workspace_root"] == "demo-workspace"
    assert captured["server_name"] == "mcp-memory"
    assert captured["tool_path_prefix"] == "/internal/tools"
    assert captured["ran"] is True

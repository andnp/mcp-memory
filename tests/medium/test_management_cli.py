from click.testing import CliRunner

from mcp_memory.cli import main


pytest_plugins: list[str] = []


def test_dashboard_command_autostarts_daemon_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

    result = runner.invoke(main, ["dashboard", "--workspace-root", "demo"])

    assert result.exit_code == 0
    assert "http://127.0.0.1:8123/" in result.output

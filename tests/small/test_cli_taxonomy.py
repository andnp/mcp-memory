import pytest
from click.testing import CliRunner

from mcp_memory.cli import main


pytestmark = pytest.mark.small


def test_daemon_start_forwards_to_existing_daemon_start_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_start_daemon(debug_enabled: bool, workspace_root: str | None, host: str, port: int | None) -> None:
        captured["debug_enabled"] = debug_enabled
        captured["workspace_root"] = workspace_root
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("mcp_memory.cli._start_daemon", fake_start_daemon)

    result = runner.invoke(
        main,
        ["--debug", "daemon", "start", "--workspace-root", "/tmp/demo", "--host", "0.0.0.0", "--port", "1234"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "debug_enabled": True,
        "workspace_root": "/tmp/demo",
        "host": "0.0.0.0",
        "port": 1234,
    }


def test_memory_stash_forwards_to_existing_stash_behavior(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_stash_thought(workspace_root: str | None, content: str) -> None:
        captured["workspace_root"] = workspace_root
        captured["content"] = content

    monkeypatch.setattr("mcp_memory.cli._stash_thought", fake_stash_thought)

    result = runner.invoke(main, ["memory", "stash", "--workspace-root", "/tmp/demo", "remember", "this"])

    assert result.exit_code == 0, result.output
    assert captured == {"workspace_root": "/tmp/demo", "content": "remember this"}


def test_admin_health_forwards_to_existing_operator_health_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_operator_health_snapshot(workspace_root: str | None, scope: str, json_output: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["scope"] = scope
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_operator_health_snapshot", fake_show_operator_health_snapshot)

    result = runner.invoke(
        main,
        ["admin", "health", "--workspace-root", "/tmp/demo", "--scope", "workspace", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "scope": "workspace",
        "json_output": True,
    }


def test_admin_overview_forwards_to_existing_stats_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_stats_command(workspace_root: str | None, watch: bool, interval: float, verbose: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["watch"] = watch
        captured["interval"] = interval
        captured["verbose"] = verbose

    monkeypatch.setattr("mcp_memory.cli._show_stats_command", fake_show_stats_command)

    result = runner.invoke(
        main,
        ["admin", "overview", "--workspace-root", "/tmp/demo", "--watch", "--interval", "1.5", "--verbose"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "watch": True,
        "interval": 1.5,
        "verbose": True,
    }
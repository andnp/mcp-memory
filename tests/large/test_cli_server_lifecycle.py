import os
from pathlib import Path
import time

import pytest
from click.testing import CliRunner

from mcp_memory.cli import main
from mcp_memory.config import resolve_daemon_metadata_path
from mcp_memory.daemon import inspect_daemon, read_daemon_metadata, stop_daemon


pytestmark = pytest.mark.large


def _wait_for_process_exit(pid: int, *, timeout_seconds: float = 10.0, interval_seconds: float = 0.05) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        proc_stat_path = Path(f"/proc/{pid}/stat")
        if proc_stat_path.exists():
            proc_stat = proc_stat_path.read_text().split()
            if len(proc_stat) >= 3 and proc_stat[2] == "Z":
                return
        time.sleep(interval_seconds)
    raise AssertionError(f"timed out waiting for process {pid} to exit")


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


def test_cli_daemon_restart_replaces_live_process(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    state_home_path = tmp_path / "state"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr("mcp_memory.cli.webbrowser.open", lambda url: True)

    runner = CliRunner()

    try:
        start_result = runner.invoke(main, ["admin", "dashboard", "open", "--workspace-root", str(workspace)])

        assert start_result.exit_code == 0, start_result.output
        first_metadata = read_daemon_metadata(resolve_daemon_metadata_path())
        assert first_metadata is not None
        initial_metadata, initial_healthy = inspect_daemon()
        assert initial_healthy is True
        assert initial_metadata is not None
        assert initial_metadata.pid == first_metadata.pid

        restart_result = runner.invoke(main, ["daemon", "restart"])

        assert restart_result.exit_code == 0, restart_result.output
        assert "Previous daemon stop:" in restart_result.output
        assert "Daemon restarted:" in restart_result.output

        second_metadata = read_daemon_metadata(resolve_daemon_metadata_path())
        assert second_metadata is not None
        assert second_metadata.pid != first_metadata.pid
        _wait_for_process_exit(first_metadata.pid)

        restarted_metadata, restarted_healthy = inspect_daemon()
        assert restarted_healthy is True
        assert restarted_metadata is not None
        assert restarted_metadata.pid == second_metadata.pid
    finally:
        stop_daemon()

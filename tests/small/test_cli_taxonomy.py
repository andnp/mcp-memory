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


def test_admin_task_list_forwards_to_existing_task_list_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_list_tasks(workspace_root: str | None, status: str | None, limit: int, json_output: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["status"] = status
        captured["limit"] = limit
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._list_tasks", fake_list_tasks)

    result = runner.invoke(
        main,
        ["admin", "task", "list", "--workspace-root", "/tmp/demo", "--status", "running", "--limit", "7", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "status": "running",
        "limit": 7,
        "json_output": True,
    }


def test_admin_task_show_forwards_to_existing_task_show_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_task(task_id: str, workspace_root: str | None, json_output: bool) -> None:
        captured["task_id"] = task_id
        captured["workspace_root"] = workspace_root
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_task", fake_show_task)

    result = runner.invoke(main, ["admin", "task", "show", "--workspace-root", "/tmp/demo", "task-123", "--json"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_id": "task-123",
        "workspace_root": "/tmp/demo",
        "json_output": True,
    }


def test_admin_task_cancel_forwards_to_existing_task_cancel_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_cancel_task(task_id: str, workspace_root: str | None, reason: str, json_output: bool) -> None:
        captured["task_id"] = task_id
        captured["workspace_root"] = workspace_root
        captured["reason"] = reason
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._cancel_task", fake_cancel_task)

    result = runner.invoke(
        main,
        ["admin", "task", "cancel", "--workspace-root", "/tmp/demo", "task-123", "--reason", "cleanup", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "task_id": "task-123",
        "workspace_root": "/tmp/demo",
        "reason": "cleanup",
        "json_output": True,
    }


def test_admin_conversation_list_forwards_to_existing_conversation_implementation(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class _FakePayload:
        def __init__(self) -> None:
            self.conversations: list[object] = []

        def model_dump(self) -> dict[str, object]:
            return {"conversations": []}

    class _FakeService:
        def list_ai_conversations(
            self,
            *,
            task_name: str | None = None,
            status: str | None = None,
            limit: int,
            request_id: str | None = None,
        ) -> _FakePayload:
            captured["task_name"] = task_name
            captured["status"] = status
            captured["limit"] = limit
            captured["request_id"] = request_id
            return _FakePayload()

    def fake_with_management_service(workspace_root: str | None, action, *, workspace_id=...) -> None:
        captured["workspace_root"] = workspace_root
        captured["workspace_id"] = workspace_id
        action(_FakeService())

    monkeypatch.setattr("mcp_memory.cli._with_management_service", fake_with_management_service)

    result = runner.invoke(
        main,
        [
            "admin",
            "conversation",
            "list",
            "--workspace-root",
            "/tmp/demo",
            "--task-name",
            "curator",
            "--status",
            "completed",
            "--limit",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "workspace_id": None,
        "task_name": "curator",
        "status": "completed",
        "limit": 7,
        "request_id": None,
    }


def test_admin_conversation_show_forwards_to_existing_conversation_implementation(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class _FakePayload:
        def __init__(self) -> None:
            self.conversations: list[object] = []

        def model_dump(self) -> dict[str, object]:
            return {"conversations": []}

    class _FakeService:
        def list_ai_conversations(
            self,
            *,
            task_name: str | None = None,
            status: str | None = None,
            limit: int,
            request_id: str | None = None,
        ) -> _FakePayload:
            captured["task_name"] = task_name
            captured["status"] = status
            captured["limit"] = limit
            captured["request_id"] = request_id
            return _FakePayload()

    def fake_with_management_service(workspace_root: str | None, action, *, workspace_id=...) -> None:
        captured["workspace_root"] = workspace_root
        captured["workspace_id"] = workspace_id
        action(_FakeService())

    monkeypatch.setattr("mcp_memory.cli._with_management_service", fake_with_management_service)

    result = runner.invoke(
        main,
        ["admin", "conversation", "show", "--workspace-root", "/tmp/demo", "request-123", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "workspace_id": None,
        "task_name": None,
        "status": None,
        "limit": 200,
        "request_id": "request-123",
    }


def test_legacy_top_level_conversation_root_is_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["conversation", "list"])

    assert result.exit_code != 0
    assert "No such command 'conversation'" in result.output


def test_admin_search_health_forwards_to_existing_search_health_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_search_health(workspace_root: str | None, json_output: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_search_health", fake_show_search_health)

    result = runner.invoke(main, ["admin", "search", "health", "--workspace-root", "/tmp/demo", "--json"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "json_output": True,
    }


def test_admin_search_repair_forwards_to_existing_search_repair_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_repair_search_index(workspace_root: str | None, json_output: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._repair_search_index", fake_repair_search_index)

    result = runner.invoke(main, ["admin", "search", "repair", "--workspace-root", "/tmp/demo", "--json"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "json_output": True,
    }


def test_legacy_top_level_search_root_is_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["search", "health"])

    assert result.exit_code != 0
    assert "No such command 'search'" in result.output


def test_admin_agent_run_forwards_to_existing_single_agent_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_enqueue_agent(agent_name: str, workspace_root: str | None, force: bool) -> None:
        captured["agent_name"] = agent_name
        captured["workspace_root"] = workspace_root
        captured["force"] = force

    monkeypatch.setattr("mcp_memory.cli._enqueue_agent", fake_enqueue_agent)

    result = runner.invoke(main, ["admin", "agent", "run", "memory-curator", "--workspace-root", "/tmp/demo", "--force"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "agent_name": "memory-curator",
        "workspace_root": "/tmp/demo",
        "force": True,
    }


def test_admin_agent_run_all_forwards_to_existing_all_agents_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_enqueue_all_agents(workspace_root: str | None, force: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["force"] = force

    monkeypatch.setattr("mcp_memory.cli._enqueue_all_agents", fake_enqueue_all_agents)

    result = runner.invoke(main, ["admin", "agent", "run", "--all", "--workspace-root", "/tmp/demo", "--force"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "force": True,
    }


def test_admin_monitor_forwards_to_existing_monitor_tui(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_monitor_tui(workspace_root: str | None, interval: float) -> None:
        captured["workspace_root"] = workspace_root
        captured["interval"] = interval

    monkeypatch.setattr("mcp_memory.cli.run_monitor_tui", fake_run_monitor_tui)

    result = runner.invoke(main, ["admin", "monitor", "--workspace-root", "/tmp/demo", "--interval", "1.5"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "interval": 1.5,
    }


def test_admin_dashboard_open_forwards_to_existing_dashboard_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_print_dashboard_url(workspace_root: str | None, *, open_browser: bool = False) -> None:
        captured["workspace_root"] = workspace_root
        captured["open_browser"] = open_browser

    monkeypatch.setattr("mcp_memory.cli._print_dashboard_url", fake_print_dashboard_url)

    result = runner.invoke(main, ["admin", "dashboard", "open", "--workspace-root", "/tmp/demo"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "open_browser": True,
    }


def test_admin_log_list_forwards_to_existing_log_list_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_logs(
        workspace_root: str | None,
        limit: int,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["limit"] = limit
        captured["level"] = level
        captured["logger_name"] = logger_name
        captured["source"] = source
        captured["query"] = query
        captured["after"] = after
        captured["before"] = before
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_logs", fake_show_logs)

    result = runner.invoke(
        main,
        [
            "admin",
            "log",
            "list",
            "--workspace-root",
            "/tmp/demo",
            "--limit",
            "5",
            "--level",
            "warning",
            "--logger",
            "demo.logger",
            "--source",
            "daemon",
            "--query",
            "oops",
            "--after",
            "1",
            "--before",
            "2",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "limit": 5,
        "level": "WARNING",
        "logger_name": "demo.logger",
        "source": "daemon",
        "query": "oops",
        "after": 1.0,
        "before": 2.0,
        "json_output": True,
    }


def test_admin_log_summary_forwards_to_existing_log_summary_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_summarize_logs(
        workspace_root: str | None,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["level"] = level
        captured["logger_name"] = logger_name
        captured["source"] = source
        captured["query"] = query
        captured["after"] = after
        captured["before"] = before
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._summarize_logs", fake_summarize_logs)

    result = runner.invoke(
        main,
        [
            "admin",
            "log",
            "summary",
            "--workspace-root",
            "/tmp/demo",
            "--level",
            "error",
            "--logger",
            "demo.logger",
            "--source",
            "daemon",
            "--query",
            "oops",
            "--after",
            "3",
            "--before",
            "4",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "level": "ERROR",
        "logger_name": "demo.logger",
        "source": "daemon",
        "query": "oops",
        "after": 3.0,
        "before": 4.0,
        "json_output": True,
    }


def test_admin_log_prune_forwards_to_existing_log_prune_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_prune_logs(
        workspace_root: str | None,
        max_runtime_logs: int | None,
        max_log_age_days: int | None,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["max_runtime_logs"] = max_runtime_logs
        captured["max_log_age_days"] = max_log_age_days
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._prune_logs", fake_prune_logs)

    result = runner.invoke(
        main,
        ["admin", "log", "prune", "--workspace-root", "/tmp/demo", "--max-runtime-logs", "11", "--max-log-age-days", "12", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "max_runtime_logs": 11,
        "max_log_age_days": 12,
        "json_output": True,
    }
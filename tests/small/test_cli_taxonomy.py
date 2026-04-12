import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import mcp_memory.cli as cli
from mcp_memory.cli import main
from mcp_memory.management.models import TaskSamplingSummaryPayload
from mcp_memory.management.frontend_build import DashboardFrontendBuildResult


pytestmark = pytest.mark.small


def test_run_forwards_to_existing_stdio_proxy_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_stdio_proxy(
        debug_enabled: bool,
        workspace_root: str | None,
        source: str,
        server_name: str,
        tool_path_prefix: str,
    ) -> None:
        captured["debug_enabled"] = debug_enabled
        captured["workspace_root"] = workspace_root
        captured["source"] = source
        captured["server_name"] = server_name
        captured["tool_path_prefix"] = tool_path_prefix

    monkeypatch.setattr("mcp_memory.cli._run_stdio_proxy", fake_run_stdio_proxy)

    result = runner.invoke(main, ["--debug", "run", "--workspace-root", "/tmp/demo"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "debug_enabled": True,
        "workspace_root": "/tmp/demo",
        "source": "stdio",
        "server_name": "mcp-memory",
        "tool_path_prefix": "/internal/tools",
    }


def test_internal_run_forwards_to_existing_internal_stdio_proxy_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run_stdio_proxy(
        debug_enabled: bool,
        workspace_root: str | None,
        source: str,
        server_name: str,
        tool_path_prefix: str,
    ) -> None:
        captured["debug_enabled"] = debug_enabled
        captured["workspace_root"] = workspace_root
        captured["source"] = source
        captured["server_name"] = server_name
        captured["tool_path_prefix"] = tool_path_prefix

    monkeypatch.setattr("mcp_memory.cli._run_stdio_proxy", fake_run_stdio_proxy)

    result = runner.invoke(main, ["--debug", "internal-run", "--workspace-root", "/tmp/demo"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "debug_enabled": True,
        "workspace_root": "/tmp/demo",
        "source": "internal-stdio",
        "server_name": "mcp-memory-internal",
        "tool_path_prefix": "/internal/maintenance/tools",
    }


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


def test_daemon_root_command_forwards_to_existing_daemon_start_helper(monkeypatch) -> None:
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
        ["--debug", "daemon", "--workspace-root", "/tmp/demo", "--host", "0.0.0.0", "--port", "1234"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "debug_enabled": True,
        "workspace_root": "/tmp/demo",
        "host": "0.0.0.0",
        "port": 1234,
    }


def test_daemon_dashboard_route_is_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["daemon", "dashboard"])

    assert result.exit_code != 0
    assert "No such command 'dashboard'" in result.output


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


def test_memory_import_markdown_forwards_to_existing_import_behavior(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_import_markdown_files(
        file_paths: tuple[str, ...],
        workspace_root: str | None,
        workspace_ids: tuple[str, ...],
        thought: bool,
    ) -> None:
        captured["file_paths"] = file_paths
        captured["workspace_root"] = workspace_root
        captured["workspace_ids"] = workspace_ids
        captured["thought"] = thought

    monkeypatch.setattr("mcp_memory.cli._import_markdown_files", fake_import_markdown_files)

    result = runner.invoke(
        main,
        [
            "memory",
            "import-markdown",
            "--workspace-root",
            "/tmp/demo",
            "--workspace-id",
            "workspace-a",
            "--workspace-id",
            "workspace-b",
            "--thought",
            "one.md",
            "two.md",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "file_paths": ("one.md", "two.md"),
        "workspace_root": "/tmp/demo",
        "workspace_ids": ("workspace-a", "workspace-b"),
        "thought": True,
    }


def test_admin_health_forwards_to_existing_operator_health_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_operator_health_snapshot(workspace_root: str | None, json_output: bool) -> None:
        captured["workspace_root"] = workspace_root
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_operator_health_snapshot", fake_show_operator_health_snapshot)

    result = runner.invoke(
        main,
        ["admin", "health", "--workspace-root", "/tmp/demo", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
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


def test_admin_quality_cleanup_forwards_to_existing_quality_cleanup_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_quality_cleanup(
        workspace_root: str | None,
        window_hours: int,
        bucket_minutes: int,
        limit: int,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["window_hours"] = window_hours
        captured["bucket_minutes"] = bucket_minutes
        captured["limit"] = limit
        captured["json_output"] = json_output

    monkeypatch.setattr("mcp_memory.cli._show_quality_cleanup_candidates", fake_quality_cleanup)

    result = runner.invoke(
        main,
        [
            "admin",
            "quality-cleanup",
            "--workspace-root",
            "/tmp/demo",
            "--window-hours",
            "48",
            "--bucket-minutes",
            "30",
            "--limit",
            "7",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "window_hours": 48,
        "bucket_minutes": 30,
        "limit": 7,
        "json_output": True,
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


def test_admin_task_recent_runs_forwards_to_existing_task_run_listing(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class _FakePayload:
        def __init__(self) -> None:
            self.runs: list[object] = []

        def model_dump(self) -> dict[str, object]:
            return {"runs": []}

    class _FakeService:
        def list_recent_agent_runs(self, *, limit: int, detail_level: str = "summary") -> _FakePayload:
            captured["limit"] = limit
            captured["detail_level"] = detail_level
            return _FakePayload()

    def fake_with_management_service(workspace_root: str | None, action, *, workspace_id=...) -> None:
        captured["workspace_root"] = workspace_root
        captured["workspace_id"] = workspace_id
        action(_FakeService())

    monkeypatch.setattr("mcp_memory.cli._with_management_service", fake_with_management_service)

    result = runner.invoke(
        main,
        ["admin", "task", "recent-runs", "--workspace-root", "/tmp/demo", "--limit", "7", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "workspace_id": None,
        "limit": 7,
        "detail_level": "full",
    }


def test_admin_task_sampling_summary_uses_management_summary_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class _FakeService:
        def get_task_sampling_summary(self, *, limit: int) -> TaskSamplingSummaryPayload:
            captured["limit"] = limit
            return TaskSamplingSummaryPayload(
                selection=[],
                grouping=[],
                selection_utility=[],
                selector_behavior=[],
            )

    def fake_with_management_service(workspace_root: str | None, action, *, workspace_id=...) -> None:
        captured["workspace_root"] = workspace_root
        captured["workspace_id"] = workspace_id
        action(_FakeService())

    monkeypatch.setattr("mcp_memory.cli._with_management_service", fake_with_management_service)

    result = runner.invoke(
        main,
        ["admin", "task", "sampling-summary", "--workspace-root", "/tmp/demo", "--limit", "9", "--json"],
    )

    assert result.exit_code == 0, result.output
    assert result.output.strip() == '{"grouping": [], "selection": [], "selection_utility": [], "selector_behavior": []}'
    assert captured == {
        "workspace_root": "/tmp/demo",
        "workspace_id": None,
        "limit": 9,
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
        workspace_id: str | None = None

        def list_ai_conversations(
            self,
            *,
            workspace_id: str | None = None,
            task_name: str | None = None,
            status: str | None = None,
            limit: int,
            request_id: str | None = None,
        ) -> _FakePayload:
            captured["service_workspace_id"] = workspace_id
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
        "service_workspace_id": None,
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
        workspace_id: str | None = None

        def list_ai_conversations(
            self,
            *,
            workspace_id: str | None = None,
            task_name: str | None = None,
            status: str | None = None,
            limit: int,
            request_id: str | None = None,
        ) -> _FakePayload:
            captured["service_workspace_id"] = workspace_id
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
        "service_workspace_id": None,
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


def test_admin_search_debug_runs_existing_debug_search_service_and_renders_summary(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_with_runtime(workspace_root: str | None, action):
        captured["workspace_root"] = workspace_root
        return action(object())

    def fake_search_memory_records_service(runtime, arguments: dict[str, object], *, caller_kind: str = "external") -> dict[str, object]:
        captured["runtime_type"] = type(runtime).__name__
        captured["arguments"] = arguments
        captured["caller_kind"] = caller_kind
        return {
            "status": "ok",
            "results": [
                {
                    "memory_id": "memory-1",
                    "title": "Ranking candidate one",
                    "summary": "Covers keyword + semantic match.",
                    "score": 0.91,
                    "ranking_debug": {
                        "matched_by_keyword": True,
                        "matched_by_semantic": True,
                        "workspace_match": True,
                        "workspace_multiplier": 1.2,
                        "semantic_score": 0.88,
                        "keyword_token_coverage": 1.0,
                        "final_score": 0.91,
                    },
                },
                {
                    "memory_id": "memory-2",
                    "title": "Ranking candidate two",
                    "summary": "Expanded through graph support.",
                    "score": 0.53,
                    "ranking_debug": {
                        "expanded_by_graph": True,
                        "graph_link_type": "DEPENDS_ON",
                        "graph_seed_id": "memory-seed-12345678",
                        "graph_support_bonus": 0.35,
                        "final_score": 0.53,
                    },
                },
            ],
            "timing_ms": {
                "total": 12.5,
                "semantic_selection": 4.0,
                "keyword_lookup": 3.0,
            },
            "adaptive_limit_enabled": True,
            "requested_limit": 5,
            "expanded_result_window": False,
            "returned_result_count": 2,
            "search_diagnostics": {
                "semantic_candidate_strategy": "speculative-bounded",
            },
        }

    monkeypatch.setattr("mcp_memory.cli._with_runtime", fake_with_runtime)
    monkeypatch.setattr("mcp_memory.cli.search_memory_records_service", fake_search_memory_records_service)

    result = runner.invoke(
        main,
        ["admin", "search", "debug", "--workspace-root", "/tmp/demo", "semantic", "latency"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
        "runtime_type": "object",
        "arguments": {"query": "semantic latency", "limit": 5, "debug": True},
        "caller_kind": "operator",
    }
    assert "Query: semantic latency" in result.output
    assert "Result count: 2" in result.output
    assert "Semantic candidate strategy: speculative-bounded" in result.output
    assert "Adaptive shaping: enabled" in result.output
    assert "Requested floor: 5" in result.output
    assert "Expanded window: no" in result.output
    assert "Total timing: 12.500ms" in result.output
    assert "semantic_selection" in result.output
    assert "keyword_lookup" in result.output
    assert "Ranked Search Results" in result.output
    assert "workspace×1.20" in result.output
    assert "sem=0.88" in result.output
    assert "graph:DEPENDS_ON@mem" in result.output


def test_admin_search_debug_json_includes_query_result_count_and_diagnostics(monkeypatch) -> None:
    runner = CliRunner()

    def fake_with_runtime(workspace_root: str | None, action):
        assert workspace_root == "/tmp/demo"
        return action(object())

    def fake_search_memory_records_service(runtime, arguments: dict[str, object], *, caller_kind: str = "external") -> dict[str, object]:
        _ = runtime, arguments, caller_kind
        return {
            "status": "ok",
            "results": [{"memory_id": "memory-1"}],
            "timing_ms": {"total": 7.25, "ranking": 1.5},
            "adaptive_limit_enabled": True,
            "requested_limit": 5,
            "expanded_result_window": False,
            "returned_result_count": 1,
            "search_diagnostics": {
                "semantic_candidate_strategy": "global",
                "timing_ms": {"total": 7.25, "ranking": 1.5},
            },
        }

    monkeypatch.setattr("mcp_memory.cli._with_runtime", fake_with_runtime)
    monkeypatch.setattr("mcp_memory.cli.search_memory_records_service", fake_search_memory_records_service)

    result = runner.invoke(
        main,
        ["admin", "search", "debug", "--workspace-root", "/tmp/demo", "--json", "latency"],
    )

    assert result.exit_code == 0, result.output
    assert '"query": "latency"' in result.output
    assert '"result_count": 1' in result.output
    assert '"semantic_candidate_strategy": "global"' in result.output
    assert '"adaptive_limit_enabled": true' in result.output
    assert '"requested_limit": 5' in result.output
    assert '"total_timing_ms": 7.25' in result.output


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


def test_admin_prefetch_model_forwards_to_existing_prefetch_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_prefetch_embedding_model(workspace_root: str | None) -> None:
        captured["workspace_root"] = workspace_root

    monkeypatch.setattr("mcp_memory.cli._prefetch_embedding_model", fake_prefetch_embedding_model)

    result = runner.invoke(main, ["admin", "prefetch-model", "--workspace-root", "/tmp/demo"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "workspace_root": "/tmp/demo",
    }


def test_admin_migrate_sqlite_to_postgres_forwards_to_existing_migration_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_migrate_sqlite_to_postgres_command(
        sqlite_path: Path | None,
        postgres_dsn: str | None,
        dry_run: bool,
        allow_non_empty_target: bool,
        json_output: bool,
    ) -> None:
        captured["sqlite_path"] = sqlite_path
        captured["postgres_dsn"] = postgres_dsn
        captured["dry_run"] = dry_run
        captured["allow_non_empty_target"] = allow_non_empty_target
        captured["json_output"] = json_output

    monkeypatch.setattr(
        "mcp_memory.cli._migrate_sqlite_to_postgres_command",
        fake_migrate_sqlite_to_postgres_command,
    )

    result = runner.invoke(
        main,
        [
            "admin",
            "migrate-sqlite-to-postgres",
            "--sqlite-path",
            "/tmp/source.db",
            "--postgres-dsn",
            "postgresql://demo",
            "--dry-run",
            "--allow-non-empty-target",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "sqlite_path": Path("/tmp/source.db"),
        "postgres_dsn": "postgresql://demo",
        "dry_run": True,
        "allow_non_empty_target": True,
        "json_output": True,
    }


def test_admin_install_forwards_to_existing_install_helper(monkeypatch) -> None:
    runner = CliRunner()

    class _FakeAction:
        def __init__(self, tool: str, component: str, status: str, path: str) -> None:
            self.tool = tool
            self.component = component
            self.status = status
            self.path = path

    class _FakeResult:
        def __init__(self) -> None:
            self.workspace_root = "/tmp/demo"
            self.actions = [_FakeAction("copilot", "hooks", "written", "/tmp/demo/.github/hooks/mcp-memory.json")]

    captured: dict[str, object] = {}

    def fake_install_integrations(
        *,
        tools: tuple[str, ...],
        components: tuple[str, ...],
        scope: str,
        workspace_root: str | None,
    ) -> _FakeResult:
        captured["tools"] = tools
        captured["components"] = components
        captured["scope"] = scope
        captured["workspace_root"] = workspace_root
        return _FakeResult()

    monkeypatch.setattr("mcp_memory.cli.install_integrations", fake_install_integrations)

    result = runner.invoke(
        main,
        [
            "admin",
            "install",
            "--tool",
            "copilot",
            "--component",
            "hooks",
            "--scope",
            "user",
            "--workspace-root",
            "/tmp/demo",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "tools": ("copilot",),
        "components": ("hooks",),
        "scope": "user",
        "workspace_root": "/tmp/demo",
    }


def test_hidden_hook_runner_forwards_to_existing_hook_helpers(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}
    echo_calls: list[tuple[object | None, bool]] = []

    def fake_load_hook_payload(stream) -> dict[str, object]:
        captured["stdin_text"] = stream.read()
        return {"hookEventName": "Stop", "source": "demo"}

    def fake_safe_forward_hook_event(
        payload: dict[str, object],
        workspace_root: str | None = None,
    ) -> tuple[dict[str, object], str | None]:
        captured["payload"] = payload
        captured["workspace_root"] = workspace_root
        return {"forwarded": True}, None

    def fake_echo(message=None, *, err: bool = False, **_kwargs) -> None:
        echo_calls.append((message, err))

    monkeypatch.setattr("mcp_memory.cli.load_hook_payload", fake_load_hook_payload)
    monkeypatch.setattr("mcp_memory.cli.safe_forward_hook_event", fake_safe_forward_hook_event)
    monkeypatch.setattr("mcp_memory.cli.click.echo", fake_echo)

    result = runner.invoke(main, ["hook-runner", "--workspace-root", "/tmp/demo"], input='{"hookEventName":"Stop"}')

    assert result.exit_code == 0, result.output
    assert captured == {
        "stdin_text": '{"hookEventName":"Stop"}',
        "payload": {"hookEventName": "Stop", "source": "demo"},
        "workspace_root": "/tmp/demo",
    }
    assert echo_calls == [('{"forwarded": true}', False)]


def test_hidden_hook_runner_preserves_invalid_json_fallback(monkeypatch) -> None:
    runner = CliRunner()
    echo_calls: list[tuple[object | None, bool]] = []

    def fake_load_hook_payload(_stream) -> dict[str, object]:
        raise json.JSONDecodeError("bad payload", "{", 0)

    def fake_echo(message=None, *, err: bool = False, **_kwargs) -> None:
        echo_calls.append((message, err))

    monkeypatch.setattr("mcp_memory.cli.load_hook_payload", fake_load_hook_payload)
    monkeypatch.setattr("mcp_memory.cli.click.echo", fake_echo)

    result = runner.invoke(main, ["hook-runner"], input="{")

    assert result.exit_code == 0, result.output
    assert len(echo_calls) == 2
    assert echo_calls[0][1] is True
    assert str(echo_calls[0][0]).startswith("mcp-memory hook-runner: bad payload")
    assert echo_calls[1] == ("{}", False)


@pytest.mark.parametrize("status", ["built", "up_to_date"])
def test_admin_dashboard_build_uses_shared_frontend_builder(monkeypatch, status: str) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_ensure_dashboard_frontend_built(*, static_root: Path) -> DashboardFrontendBuildResult:
        captured["static_root"] = static_root
        return DashboardFrontendBuildResult(
            status=status,
            frontend_root=static_root.with_name("frontend"),
            dist_index_path=static_root / "dist" / "index.html",
            built=status == "built",
        )

    monkeypatch.setattr("mcp_memory.cli.ensure_dashboard_frontend_built", fake_ensure_dashboard_frontend_built)

    result = runner.invoke(main, ["admin", "dashboard", "build"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "static_root": Path(cli.__file__).with_name("management") / "static",
    }


def test_admin_dashboard_build_returns_nonzero_for_failed_build_status(monkeypatch) -> None:
    runner = CliRunner()

    def fake_ensure_dashboard_frontend_built(*, static_root: Path) -> DashboardFrontendBuildResult:
        return DashboardFrontendBuildResult(
            status="build_failed",
            frontend_root=static_root.with_name("frontend"),
            dist_index_path=static_root / "dist" / "index.html",
            returncode=2,
            message="npm exploded politely",
        )

    monkeypatch.setattr("mcp_memory.cli.ensure_dashboard_frontend_built", fake_ensure_dashboard_frontend_built)

    result = runner.invoke(main, ["admin", "dashboard", "build"])

    assert result.exit_code == 1
    assert "dashboard_frontend_build_failed:build_failed" in result.output


def test_admin_log_list_forwards_to_existing_log_list_helper(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_show_logs(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
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
        captured["scope"] = scope
        captured["workspace_id"] = workspace_id
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
        "scope": "global",
        "workspace_id": None,
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
        scope: str,
        workspace_id: str | None,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["scope"] = scope
        captured["workspace_id"] = workspace_id
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
        "scope": "global",
        "workspace_id": None,
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
        scope: str,
        workspace_id: str | None,
        max_runtime_logs: int | None,
        max_log_age_days: int | None,
        json_output: bool,
    ) -> None:
        captured["workspace_root"] = workspace_root
        captured["scope"] = scope
        captured["workspace_id"] = workspace_id
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
        "scope": "global",
        "workspace_id": None,
        "max_runtime_logs": 11,
        "max_log_age_days": 12,
        "json_output": True,
    }


@pytest.mark.parametrize(
    ("argv", "missing_command"),
    [
        (["daemon-status"], "daemon-status"),
        (["daemon-stop"], "daemon-stop"),
        (["daemon-restart"], "daemon-restart"),
        (["import-markdown", "demo.md"], "import-markdown"),
        (["stash"], "stash"),
        (["prefetch-model"], "prefetch-model"),
        (["install"], "install"),
        (["task", "list"], "task"),
        (["log", "list"], "log"),
        (["agents", "run", "memory-curator"], "agents"),
        (["monitor"], "monitor"),
        (["dashboard"], "dashboard"),
        (["health"], "health"),
        (["stats"], "stats"),
    ],
)
def test_removed_legacy_roots_fail_with_no_such_command(argv: list[str], missing_command: str) -> None:
    runner = CliRunner()

    result = runner.invoke(main, argv)

    assert result.exit_code != 0
    assert f"No such command '{missing_command}'" in result.output


def test_removed_top_level_migrate_sqlite_to_postgres_route_fails() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["migrate-sqlite-to-postgres", "--help"])

    assert result.exit_code != 0
    assert "No such command 'migrate-sqlite-to-postgres'" in result.output
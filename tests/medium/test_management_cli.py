
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from mcp_memory.cli import main
from mcp_memory.daemon import DaemonMetadata, DaemonStopResult
from mcp_memory.management.models import (
    CacheHealthPayload,
    CacheMetricsPayload,
    CacheRecentMetricsPayload,
    ExecutionAttemptHealthPayload,
    HealthPayload,
    OperatorHealthSnapshotPayload,
    SearchHealthPayload,
)
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.provider_usage_store import ProviderUsageRepository

pytest_plugins: list[str] = []


def test_dashboard_command_autostarts_daemon_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()
    opened: list[str] = []

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda: FakeMetadata())
    monkeypatch.setattr("mcp_memory.cli.webbrowser.open", lambda url: opened.append(url) or True)

    result = runner.invoke(main, ["admin", "dashboard", "open", "--workspace-root", "demo"])

    assert result.exit_code == 0
    assert opened == ["http://127.0.0.1:8123/dashboard"]
    assert "Dashboard ready:" in result.output
    assert "http://127.0.0.1:8123/dashboard" in result.output
    assert "browser_opened=True" in result.output


def test_dashboard_command_can_open_browser(monkeypatch) -> None:
    runner = CliRunner()
    opened: list[str] = []

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda: FakeMetadata())
    monkeypatch.setattr("mcp_memory.cli.webbrowser.open", lambda url: opened.append(url) or True)

    result = runner.invoke(main, ["admin", "dashboard", "open"])

    assert result.exit_code == 0
    assert opened == ["http://127.0.0.1:8123/dashboard"]
    assert "browser_opened=True" in result.output


def test_daemon_status_command_reports_running_daemon(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        pid = 123
        started_at = 100.0
        transport_endpoint = "ipc:///tmp/mcp-memory.sock"
        daemon_scope = "global"
        transport = "zmq"
        version = "0.1.0"
        binary_path = "/tmp/mcp-memory-python"

    monkeypatch.setattr(
        "mcp_memory.cli.inspect_daemon",
        lambda: (FakeMetadata(), True),
    )

    result = runner.invoke(main, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Status:" in result.output
    assert "running" in result.output
    assert "ipc:///tmp/mcp-memory.sock" in result.output
    assert "Daemon scope:" in result.output
    assert "global" in result.output
    assert "Workspace context:" not in result.output


def test_daemon_stop_command_reports_stopped_daemon(monkeypatch) -> None:
    runner = CliRunner()

    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8123,
        pid=123,
        started_at=100.0,
        status="ready",
    )

    monkeypatch.setattr(
        "mcp_memory.cli.stop_daemon",
        lambda: DaemonStopResult(
            metadata=metadata,
            stop_reason="owner_stopped_on_stop",
            signal_sequence=("SIGTERM",),
            process_group_id=123,
            escalated_to_sigkill=False,
            stale_socket_removed=False,
        ),
    )

    result = runner.invoke(main, ["daemon", "stop"])

    assert result.exit_code == 0
    assert "Daemon stopped:" in result.output
    assert "scope=global" in result.output
    assert "signals=SIGTERM" in result.output
    assert "escalated=False" in result.output


def test_daemon_restart_command_restarts_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()

    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8124,
        pid=456,
        started_at=100.0,
        status="ready",
        transport="zmq",
        socket_path="/tmp/mcp-memory.sock",
    )

    stop_calls: list[bool] = []
    start_calls: list[bool] = []
    monkeypatch.setattr(
        "mcp_memory.cli.stop_daemon",
        lambda: stop_calls.append(True) or DaemonStopResult(
            metadata=metadata,
            stop_reason="owner_stopped_on_stop",
            signal_sequence=("SIGTERM", "SIGKILL"),
            process_group_id=456,
            escalated_to_sigkill=True,
            stale_socket_removed=True,
        ),
    )
    monkeypatch.setattr(
        "mcp_memory.cli.ensure_daemon_started",
        lambda: start_calls.append(True) or metadata,
    )

    result = runner.invoke(main, ["daemon", "restart"])

    assert result.exit_code == 0
    assert stop_calls == [True]
    assert start_calls == [True]
    assert "ipc:///tmp/mcp-memory.sock" in result.output
    assert "Previous daemon stop:" in result.output
    assert "escalated=True" in result.output


def test_install_command_writes_workspace_hook_and_gemini_configs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(
        main,
        [
            "admin",
            "install",
            "--tool",
            "copilot",
            "--tool",
            "gemini",
            "--workspace-root",
            str(workspace),
        ],
    )

    hook_path = workspace / ".github" / "hooks" / "mcp-memory.json"
    gemini_path = workspace / ".gemini" / "settings.json"
    hook_payload = json.loads(hook_path.read_text(encoding="utf-8"))
    gemini_payload = json.loads(gemini_path.read_text(encoding="utf-8"))

    assert result.exit_code == 0
    assert hook_path.exists()
    assert gemini_path.exists()
    assert set(hook_payload["hooks"]) == {"SessionStart", "PostToolUse", "Stop"}
    assert "hook-runner" in hook_payload["hooks"]["PostToolUse"][0]["command"]
    assert gemini_payload["mcp"]["allowed"] == ["mcp-memory-internal"]
    assert gemini_payload["mcpServers"]["mcp-memory-internal"]["command"] == "uv"


def test_install_command_updates_user_claude_settings_without_duplicate_hooks(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    home_dir = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    settings_path = home_dir / ".claude" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "theme": "dark",
                "hooks": {
                    "SessionStart": [
                        {
                            "type": "command",
                            "command": "existing-hook",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home_dir))

    first = runner.invoke(
        main,
        [
            "admin",
            "install",
            "--tool",
            "claude",
            "--scope",
            "user",
            "--workspace-root",
            str(workspace),
        ],
    )
    second = runner.invoke(
        main,
        [
            "admin",
            "install",
            "--tool",
            "claude",
            "--scope",
            "user",
            "--workspace-root",
            str(workspace),
        ],
    )

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    session_start_commands = [entry["command"] for entry in payload["hooks"]["SessionStart"]]

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert payload["theme"] == "dark"
    assert session_start_commands.count("existing-hook") == 1
    assert sum("hook-runner" in command for command in session_start_commands) == 1


def test_hook_runner_forwards_payload_and_prints_response(monkeypatch) -> None:
    runner = CliRunner()

    captured: dict[str, object] = {}

    def fake_forward(payload: dict, workspace_root: str | None = None):
        captured["payload"] = payload
        captured["workspace_root"] = workspace_root
        return {"systemMessage": "remember to record thought"}, None

    monkeypatch.setattr("mcp_memory.cli.safe_forward_hook_event", fake_forward)

    result = runner.invoke(
        main,
        ["hook-runner", "--workspace-root", "/tmp/demo"],
        input=json.dumps({"hookEventName": "SessionStart", "sessionId": "conv-1"}),
    )

    assert result.exit_code == 0
    assert captured["workspace_root"] == "/tmp/demo"
    assert captured["payload"] == {"hookEventName": "SessionStart", "sessionId": "conv-1"}
    assert json.loads(result.output) == {"systemMessage": "remember to record thought"}


def test_hook_runner_returns_empty_json_on_invalid_payload() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["hook-runner"], input="not-json")

    assert result.exit_code == 0
    assert "mcp-memory hook-runner:" in result.output
    assert result.output.strip().endswith("{}")


def test_logs_command_prints_runtime_logs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "daemon",
                "mcp_memory.server",
                "WARNING",
                "stored warning",
                123.0,
                "{}",
            ),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(
        main,
        ["admin", "log", "list", "--workspace-root", str(workspace), "--source", "daemon", "--query", "warning"],
    )

    assert result.exit_code == 0
    assert "Runtime Logs" in result.output
    assert "stored warning" in result.output
    assert "mcp_memory.server" in result.output


def test_logs_command_supports_json_output(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "stdio",
                "mcp_memory.server",
                "INFO",
                "json log entry",
                124.0,
                "{}",
            ),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(
        main,
        ["admin", "log", "list", "--workspace-root", str(workspace), "--source", "stdio", "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["logs"][0]["message"] == "json log entry"


def test_logs_command_reads_global_logs_across_workspaces(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=str(workspace_a), cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=str(workspace_b), cwd=workspace_b)
    try:
        assert runtime_a.db_manager is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None
        runtime_a.db_manager.get_connection().executemany(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (runtime_a.workspace_id, "daemon", "mcp_memory.server", "INFO", "workspace-a log", 10.0, "{}"),
                (runtime_b.workspace_id, "daemon", "mcp_memory.server", "INFO", "workspace-b log", 20.0, "{}"),
            ],
        )
        runtime_a.db_manager.get_connection().commit()
    finally:
        runtime_a.close()
        runtime_b.close()

    result = runner.invoke(
        main,
        ["admin", "log", "list", "--workspace-root", str(workspace_a), "--source", "daemon", "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert {row["message"] for row in payload["logs"]} == {"workspace-a log", "workspace-b log"}


def test_log_summary_command_prints_grouped_counts(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().executemany(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (runtime.workspace_id, "daemon", "mcp_memory.server", "INFO", "one", 1.0, "{}"),
                (runtime.workspace_id, "daemon", "mcp_memory.server", "ERROR", "two", 2.0, "{}"),
            ],
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "log", "summary", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Matching logs:" in result.output
    assert "By Level" in result.output
    assert "By Source" in result.output
    assert "daemon" in result.output


def test_task_list_command_shows_global_tasks_across_workspaces(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=str(workspace_a), cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=str(workspace_b), cwd=workspace_b)
    try:
        assert runtime_a.task_queue is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None
        runtime_a.task_queue.enqueue("memory-curator", task_id="task-a", workspace_id=runtime_a.workspace_id)
        runtime_a.task_queue.enqueue("deduplicator", task_id="task-b", workspace_id=runtime_b.workspace_id)
    finally:
        runtime_a.close()
        runtime_b.close()

    result = runner.invoke(
        main,
        ["admin", "task", "list", "--workspace-root", str(workspace_a), "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert {task["id"] for task in payload["tasks"]} == {"task-a", "task-b"}


def test_conversation_list_command_reads_global_conversations_across_workspaces(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=str(workspace_a), cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=str(workspace_b), cwd=workspace_b)
    try:
        assert runtime_a.db_manager is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None
        ProviderUsageRepository(runtime_a.db_manager, workspace_id=runtime_a.workspace_id).record_conversation(
            request_id="req-a",
            attempt=1,
            task_name="memory-curator",
            task_id="task-a",
            provider_key="gemini-cli",
            provider_name="Gemini CLI",
            model_name="gemini-3-flash-preview",
            subprocess_pid=1111,
            prompt_text="prompt a",
            response_text="response a",
            parsed=None,
            status="completed",
            error_text=None,
            started_at=10.0,
            completed_at=12.0,
        )
        ProviderUsageRepository(runtime_b.db_manager, workspace_id=runtime_b.workspace_id).record_conversation(
            request_id="req-b",
            attempt=1,
            task_name="deduplicator",
            task_id="task-b",
            provider_key="copilot-mini",
            provider_name="Copilot CLI",
            model_name="gpt-5-mini",
            subprocess_pid=2222,
            prompt_text="prompt b",
            response_text="response b",
            parsed=None,
            status="completed",
            error_text=None,
            started_at=20.0,
            completed_at=24.0,
        )
    finally:
        runtime_a.close()
        runtime_b.close()

    result = runner.invoke(
        main,
        ["admin", "conversation", "list", "--workspace-root", str(workspace_a), "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert {row["request_id"] for row in payload["conversations"]} == {"req-a", "req-b"}


def test_log_prune_command_supports_json_output(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().executemany(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (runtime.workspace_id, "daemon", "mcp_memory.server", "INFO", "first", time.time(), "{}"),
                (runtime.workspace_id, "daemon", "mcp_memory.server", "INFO", "second", time.time() + 1, "{}"),
            ],
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(
        main,
        ["admin", "log", "prune", "--workspace-root", str(workspace), "--max-runtime-logs", "1", "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["deleted"] == 1
    assert payload["max_runtime_logs"] == 1


def test_run_command_records_logs_without_polluting_stdio(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    async def fake_run(self) -> None:
        logging.getLogger("mcp_memory.server").info("stdio bootstrap log")

    monkeypatch.setattr("mcp_memory.cli.MCPServer.run", fake_run)

    result = runner.invoke(main, ["run", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert result.output == ""

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        payload = runtime.db_manager.get_connection().execute(
            "SELECT source, message FROM runtime_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        runtime.close()

    assert payload is not None
    assert payload["source"] == "stdio"
    assert payload["message"] == "stdio bootstrap log"


def test_prefetch_model_command_caches_embedding_model(monkeypatch) -> None:
    runner = CliRunner()
    closed: list[bool] = []

    class FakeEmbedder:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        def cache_model(self) -> bool:
            return True

        def status(self):
            from mcp_memory.embeddings import EmbedderStatus

            return EmbedderStatus(
                model_name=self.model_name,
                backend="sentence-transformer",
                model_cached=True,
            )

    class FakeRuntime:
        embedder = FakeEmbedder()

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("mcp_memory.cli.create_runtime", lambda workspace_root_override=None: FakeRuntime())

    result = runner.invoke(main, ["admin", "prefetch-model"])

    assert result.exit_code == 0
    assert "Embedding model cached:" in result.output
    assert "cached=True" in result.output
    assert closed == [True]


def test_import_markdown_command_imports_record(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    markdown_file = workspace / "epic-01.md"
    markdown_file.write_text(
        "---\n"
        "type: plan\n"
        "status: active\n"
        "tags: [sqlite, migration]\n"
        "created_at: '2026-03-01T10:30:00+00:00'\n"
        "---\n\n"
        "Ship the relational bootstrap.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        main,
        ["memory", "import-markdown", str(markdown_file), "--workspace-root", str(workspace)],
    )

    assert result.exit_code == 0
    assert "Imported memory:" in result.output


def test_import_markdown_command_supports_multiple_paths_and_globs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    first = workspace / "alpha.md"
    second = workspace / "beta.md"
    first.write_text("---\ntype: fact\n---\n\nAlpha memory\n", encoding="utf-8")
    second.write_text("---\ntype: plan\n---\n\nBeta memory\n", encoding="utf-8")

    result = runner.invoke(
        main,
        [
            "memory",
            "import-markdown",
            str(first),
            str(workspace / "*.md"),
            "--workspace-root",
            str(workspace),
        ],
    )

    assert result.exit_code == 0
    assert result.output.count("Imported memory:") == 2
    assert "Imported total: 2" in result.output


def test_agents_run_command_enqueues_background_agent(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda: FakeMetadata())

    result = runner.invoke(
        main,
        ["admin", "agent", "run", "sweeper", "--workspace-root", str(workspace)],
    )

    assert result.exit_code == 0
    assert "sweeper" in result.output
    assert "task_id=" in result.output


def test_stats_command_prints_memory_and_agent_metrics(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        runtime.repository.create_memory(
            title="Stats fact",
            content="First line.\nSecond line.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        read_memory = runtime.repository.create_memory(
            title="Most read fact",
            content="Read me twice.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        assert read_memory is not None
        runtime.repository.record_access(read_memory.id, 1.0, "2026-03-15T10:00:00+00:00", increment_read_count=True)
        runtime.repository.record_access(read_memory.id, 2.0, "2026-03-15T10:05:00+00:00", increment_read_count=True)
        task = runtime.task_queue.enqueue(
            "defragmenter",
            task_id="defrag-stats",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        runtime.task_queue.enqueue(
            "summarize-memory",
            task_id="summary-deferred",
            workspace_id=runtime.workspace_id,
            available_at=time.time() + 120.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(
            task.id,
            completed_at=14.0,
            run_result={
                "lines_compressed": 3,
                "strategy_used": "cold-storage",
                "candidate_count": 5,
            },
        )
        runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (runtime.workspace_id, "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.5, time.time(), None),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace), "--verbose"])

    assert result.exit_code == 0
    assert "Stats: global" in result.output
    assert "Execution Attempts" in result.output
    assert "Embedding Repair Backlog" in result.output
    assert "Memory Metrics" in result.output
    assert "Next Pending Tasks" in result.output
    assert "Background Agents" in result.output
    assert "AI Provider Usage" in result.output
    assert "Top Read Memories" in result.output
    assert "Agent Details" in result.output
    assert "Recent Agent Runs" in result.output
    assert "Total lines compressed" in result.output
    assert "Most read fact" in result.output
    assert "defragmenter" in result.output
    assert "gemini-cli" in result.output
    assert "memory-curator" in result.output
    assert "lines_compressed=3" in result.output
    assert "strategy_used=cold-storage" in result.output
    assert "summarize-memory" in result.output


def test_stats_command_default_output_is_more_compact(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Search Health" in result.output
    assert "Execution Attempts" in result.output
    assert "Embedding Repair Backlog" in result.output
    assert "Memory Metrics" in result.output
    assert "Next Pending Tasks" in result.output
    assert "Background Agents" in result.output
    assert "AI Provider Usage" in result.output
    assert "Top Read Memories" in result.output
    assert "Agent Details" not in result.output
    assert "Recent Agent Runs" not in result.output


def test_health_command_supports_global_json_snapshot(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=str(workspace_a), cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=str(workspace_b), cwd=workspace_b)
    try:
        assert runtime_a.db_manager is not None
        assert runtime_a.repository is not None
        assert runtime_a.task_queue is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None

        runtime_a.repository.create_memory(
            title="CLI health memory",
            content="Fresh edit for operator snapshot.",
            workspace_ids=[runtime_a.workspace_id],
            memory_type="fact",
        )
        runtime_a.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime_a.workspace_id,
                "daemon",
                "mcp_memory.tests",
                "ERROR",
                "cli health error",
                time.time(),
                "{}",
            ),
        )
        runtime_a.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime_a.workspace_id,
                "daemon",
                "mcp_memory.core.provider_policy",
                "WARNING",
                "Provider routing exhausted all configured routes",
                time.time(),
                '{"extra":{"task_name":"graph-linker"}}',
            ),
        )
        task = runtime_a.task_queue.enqueue(
            "graph-linker",
            task_id="cli-health-task",
            workspace_id=runtime_a.workspace_id,
            available_at=0.0,
        )
        assert runtime_a.task_queue.claim_next(now=10.0) is not None
        runtime_a.task_queue.complete(task.id, completed_at=11.0, run_result={"updated": 1})
        retry_task = runtime_a.task_queue.enqueue(
            "graph-linker",
            task_id="cli-health-retry-task",
            workspace_id=runtime_a.workspace_id,
            available_at=0.0,
        )
        assert runtime_a.task_queue.claim_next(now=12.0) is not None
        runtime_a.task_queue.fail(retry_task.id, "retry me", retry_delay_seconds=15.0, failed_at=13.0)
        runtime_a.db_manager.get_connection().execute(
            "INSERT INTO provider_policy_events (workspace_id, task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime_a.workspace_id,
                "graph-linker",
                task.id,
                "route_exhausted",
                "provider_routing_exhausted",
                None,
                None,
                None,
                None,
                '["gemini-cheap","copilot-mini"]',
                None,
                None,
                None,
                0,
                time.time(),
            ),
        )
        runtime_a.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text, reason_category, reason_code, retry_delay_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime_a.workspace_id,
                "graph-linker",
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                "skipped",
                0.0,
                time.time(),
                "quota reset pending",
                "upstream",
                "provider_quota_exhausted",
                90.0,
            ),
        )
        ProviderUsageRepository(runtime_a.db_manager, workspace_id=runtime_b.workspace_id).record_conversation(
            request_id="cli-health-conversation",
            attempt=1,
            task_name="deduplicator",
            task_id="task-b",
            provider_key="copilot-mini",
            provider_name="Copilot CLI",
            model_name="gpt-5-mini",
            subprocess_pid=2222,
            prompt_text="prompt",
            response_text="response",
            parsed=None,
            status="error",
            error_text="provider unavailable",
            started_at=time.time() - 20.0,
            completed_at=time.time() - 10.0,
        )
        runtime_a.db_manager.get_connection().commit()
    finally:
        runtime_a.close()
        runtime_b.close()

    result = runner.invoke(
        main,
        ["admin", "health", "--workspace-root", str(workspace_a), "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["status"] == "warn"
    assert "scope" not in payload
    assert "recent_error_logs" in payload["alerts"]
    assert "recent_task_retries" in payload["alerts"]
    assert "provider_policy_churn" in payload["alerts"]
    assert payload["logs"]["by_level"]["ERROR"] == 1
    assert payload["warnings"]["total"] == 1
    assert payload["warnings"]["recent"][0]["logger_name"] == "mcp_memory.core.provider_policy"
    assert payload["conversations"]["by_status"]["error"] == 1
    assert payload["tasks"]["recent_status_counts"]["completed"] == 1
    assert payload["tasks"]["recent_status_counts"]["retry"] == 1
    assert payload["tasks"]["recent_retry_count"] == 1
    assert payload["tasks"]["recent_retries"][0]["task_id"] == "cli-health-retry-task"
    provider_policy_stats = {stat["key"]: stat["value"] for stat in payload["provider_policy"]["stats"]}
    assert provider_policy_stats["provider_policy_route_exhaustion_count"] == 1.0
    assert provider_policy_stats["provider_policy_admission_skip_count"] == 1.0
    assert any(item["title"] == "CLI health memory" for item in payload["memory_activity"]["recent"])


def test_health_command_help_omits_workspace_scope_flag() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["admin", "health", "--help"])

    assert result.exit_code == 0
    assert "--scope" not in result.output


def test_health_command_human_output_renders_warning_and_provider_policy_sections(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "daemon",
                "mcp_memory.core.provider_policy",
                "WARNING",
                "Provider routing exhausted all configured routes",
                time.time(),
                '{"extra":{"task_name":"graph-linker"}}',
            ),
        )
        task = runtime.task_queue.enqueue(
            "graph-linker",
            task_id="cli-health-human-retry",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.fail(task.id, "retry me", retry_delay_seconds=30.0, failed_at=11.0)
        runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_policy_events (workspace_id, task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "graph-linker",
                task.id,
                "route_exhausted",
                "provider_routing_exhausted",
                None,
                None,
                None,
                None,
                '["gemini-cheap","copilot-mini"]',
                None,
                None,
                None,
                0,
                time.time(),
            ),
        )
        runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text, reason_category, reason_code, retry_delay_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "graph-linker",
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                "skipped",
                0.0,
                time.time(),
                "quota reset pending",
                "upstream",
                "provider_quota_exhausted",
                90.0,
            ),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "health", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "scope=" not in result.output
    assert "Workspace context:" not in result.output
    assert "Recent Warnings" in result.output
    assert "Provider Policy Churn" in result.output
    assert "Recent retried task runs" in result.output
    assert "Route exhaustion" in result.output


def test_health_command_human_output_renders_cache_section(monkeypatch) -> None:
    runner = CliRunner()

    snapshot = OperatorHealthSnapshotPayload(
        generated_at=100.0,
        health=HealthPayload(
            status="ok",
            storage_backend="postgres",
            runtime_active=True,
            client_count=1,
            task_queue_enabled=True,
            cache=CacheHealthPayload(
                enabled=True,
                mode="readonly",
                state="active",
                path="/tmp/shared_read_cache.sqlite3",
                metrics=CacheMetricsPayload(
                    search_requests=8,
                    fresh_exact_search_hits=3,
                    read_requests=4,
                    validated_read_hits=2,
                    warmed_projection_rows=5,
                    fresh_exact_search_hit_rate=0.375,
                    validated_read_hit_rate=0.5,
                    recent=CacheRecentMetricsPayload(
                        window_minutes=15,
                        search_requests=4,
                        fresh_exact_search_hits=2,
                        read_requests=2,
                        validated_read_hits=1,
                        warmed_projection_rows=3,
                        fresh_exact_search_hit_rate=0.5,
                        validated_read_hit_rate=0.5,
                    ),
                ),
            ),
            search=SearchHealthPayload(),
            execution_attempts=ExecutionAttemptHealthPayload(),
        ),
    )

    monkeypatch.setattr(
        "mcp_memory.cli._with_management_service",
        lambda workspace_root, action, workspace_id=...: action(
            SimpleNamespace(get_operator_health_snapshot=lambda: snapshot)
        ),
    )

    result = runner.invoke(main, ["admin", "health"])

    assert result.exit_code == 0
    assert "Cache" in result.output
    assert "readonly" in result.output
    assert "active" in result.output
    assert "/tmp/shared_read_cache.sqlite3" in result.output
    assert "Search requests" in result.output
    assert "Fresh exact hits" in result.output
    assert "Warmed projection rows" in result.output
    assert "Recent window" in result.output
    assert "Recent search requests" in result.output
    assert "Recent validated read hits" in result.output


def test_stats_command_top_reads_show_memory_status(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None
        active = runtime.repository.create_memory(
            title="Active top read",
            content="Active record.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        archived = runtime.repository.create_memory(
            title="Archived top read",
            content="Archived record.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        assert active is not None and archived is not None
        runtime.repository.record_access(active.id, 1.0, "2026-03-15T10:00:00+00:00", increment_read_count=True)
        runtime.repository.record_access(archived.id, 1.0, "2026-03-15T10:00:00+00:00", increment_read_count=True)
        runtime.repository.record_access(archived.id, 2.0, "2026-03-15T10:05:00+00:00", increment_read_count=True)
        runtime.repository.update_memory(archived.id, status="archived")
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Top Read Memories" in result.output
    assert "Status" in result.output
    assert "Archived top read" in result.output
    assert "archived" in result.output
    assert "Active top read" in result.output
    assert "active" in result.output


def test_search_health_and_repair_commands_surface_resilience_state(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None
        runtime.repository.create_memory(
            title="Search repair fact",
            content="This record should be embedded during repair.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
    finally:
        runtime.close()

    health_result = runner.invoke(main, ["admin", "search", "health", "--workspace-root", str(workspace)])
    repair_result = runner.invoke(main, ["admin", "search", "repair", "--workspace-root", str(workspace)])

    assert health_result.exit_code == 0
    assert "Search Health" in health_result.output
    assert "Semantic enabled" in health_result.output
    assert repair_result.exit_code == 0
    assert "Search index rebuilt:" in repair_result.output


def test_stats_command_shows_running_background_agents(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        runtime.task_queue.enqueue(
            "graph-linker",
            task_id="graph-linker-running",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace), "--verbose"])

    assert result.exit_code == 0
    assert "Background Agents" in result.output
    assert "graph-linker" in result.output
    assert "running=1" in result.output
    assert "next=" in result.output


def test_stats_command_does_not_require_daemon_start(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        "mcp_memory.cli.ensure_daemon_started",
        lambda: (_ for _ in ()).throw(AssertionError("stats should not start daemon")),
    )

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace)])

    assert result.exit_code == 0


def test_stats_command_aggregates_globally_across_workspaces(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=str(workspace_a), cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=str(workspace_b), cwd=workspace_b)
    try:
        assert runtime_b.task_queue is not None
        assert runtime_b.workspace_id is not None
        task = runtime_b.task_queue.enqueue(
            "defragmenter",
            task_id="global-stats-task",
            workspace_id=runtime_b.workspace_id,
            available_at=0.0,
        )
        assert runtime_b.task_queue.claim_next(now=10.0) is not None
        runtime_b.task_queue.complete(task.id, completed_at=14.0, run_result={"lines_compressed": 7})
    finally:
        runtime_a.close()
        runtime_b.close()

    result = runner.invoke(main, ["admin", "overview", "--workspace-root", str(workspace_a), "--verbose"])

    assert result.exit_code == 0
    assert "Stats: global" in result.output
    assert "defragmenter" in result.output
    assert "lines_compressed=7" in result.output


def test_stats_command_watch_mode_refreshes_without_daemon_start(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        "mcp_memory.cli.ensure_daemon_started",
        lambda: (_ for _ in ()).throw(AssertionError("stats watch should not start daemon")),
    )

    sleep_calls: list[float] = []

    def stop_after_first_refresh(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr("mcp_memory.cli.time.sleep", stop_after_first_refresh)

    result = runner.invoke(
        main,
        ["admin", "overview", "--workspace-root", str(workspace), "--watch", "--interval", "2"],
    )

    assert result.exit_code == 0
    assert sleep_calls == [2.0]
    assert "Watching every 2.0s" in result.output
    assert "Stopped stats watch." in result.output


def test_monitor_command_runs_tui(monkeypatch) -> None:
    runner = CliRunner()
    called: dict[str, object] = {}

    monkeypatch.setattr(
        "mcp_memory.cli.run_monitor_tui",
        lambda workspace_root, interval_seconds: called.update(
            {"workspace_root": workspace_root, "interval_seconds": interval_seconds}
        ),
    )

    result = runner.invoke(main, ["admin", "monitor", "--workspace-root", "demo", "--interval", "1.5"])

    assert result.exit_code == 0
    assert called == {"workspace_root": "demo", "interval_seconds": 1.5}


def test_task_list_and_cancel_commands_show_running_task_metadata(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        task = runtime.task_queue.enqueue(
            "memory-curator",
            task_id="cancel-cli-task",
            workspace_id=runtime.workspace_id,
            data={"strategy": "anomaly", "grouping_strategy": "fifo"},
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.set_running_process(task.id, subprocess_pid=4444, request_id="req-cli-1", updated_at=11.0)
    finally:
        runtime.close()

    list_result = runner.invoke(
        main,
        ["admin", "task", "list", "--workspace-root", str(workspace), "--status", "running", "--json"],
    )
    cancel_result = runner.invoke(
        main,
        ["admin", "task", "cancel", "cancel-cli-task", "--workspace-root", str(workspace), "--reason", "manual_cancel"],
    )

    list_payload = json.loads(list_result.output)
    assert list_result.exit_code == 0
    assert list_payload["tasks"][0]["task_name"] == "memory-curator"
    assert list_payload["tasks"][0]["strategy"] == "anomaly"
    assert list_payload["tasks"][0]["grouping_strategy"] == "fifo"
    assert list_payload["tasks"][0]["subprocess_pid"] == 4444
    assert list_payload["tasks"][0]["active_request_id"] == "req-cli-1"
    assert cancel_result.exit_code == 0
    assert "cancellation_requested" in cancel_result.output or "cancelled" in cancel_result.output


def test_task_recent_runs_and_sampling_summary_commands(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        dedup = runtime.task_queue.enqueue(
            "deduplicator",
            task_id="recent-run-dedup",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        ingest = runtime.task_queue.enqueue(
            "ingest-system1",
            task_id="recent-run-ingest",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(
            dedup.id,
            completed_at=12.0,
            run_result={
                "merged": 1,
                "requested_strategy": "semantic",
                "strategy_used": "semantic",
                "strategy_selection_mode": "deterministic_scores",
                "strategy_selection_reason": "selected=semantic",
                "strategy_selection_scores": {"semantic": 0.8, "anomaly": 0.2},
                "candidate_count": 8,
                "sampled_memory_ids": ["memory-1", "memory-2"],
            },
        )
        assert runtime.task_queue.claim_next(now=13.0) is not None
        runtime.task_queue.complete(
            ingest.id,
            completed_at=14.0,
            run_result={
                "meaningful_actions": 1,
                "requested_grouping_strategy": "fifo",
                "grouping_strategy_used": "fifo",
                "group_count": 2,
            },
        )
    finally:
        runtime.close()

    recent_result = runner.invoke(main, ["admin", "task", "recent-runs", "--workspace-root", str(workspace), "--json"])
    summary_result = runner.invoke(main, ["admin", "task", "sampling-summary", "--workspace-root", str(workspace)])
    summary_json_result = runner.invoke(main, ["admin", "task", "sampling-summary", "--workspace-root", str(workspace), "--json"])

    recent_payload = json.loads(recent_result.output)
    assert recent_result.exit_code == 0
    semantic_run = next(run for run in recent_payload["runs"] if run["result_metadata"]["strategy_used"] == "semantic")
    assert semantic_run["result_metadata"]["strategy_selection_mode"] == "deterministic_scores"
    assert semantic_run["result_metadata"]["strategy_selection_reason"] == "selected=semantic"
    assert semantic_run["result_metadata"]["strategy_selection_scores"] == {"semantic": 0.8, "anomaly": 0.2}
    assert any(run["result_metadata"]["grouping_strategy_used"] == "fifo" for run in recent_payload["runs"])
    assert summary_result.exit_code == 0
    assert "Selection Strategy Usage" in summary_result.output
    assert "Ingest Grouping Strategy Usage" in summary_result.output
    assert "Selector Behavior by Recent Runs" in summary_result.output
    assert "deterministic_scores" in summary_result.output
    assert "semantic" in summary_result.output
    assert "fifo" in summary_result.output
    summary_payload = json.loads(summary_json_result.output)
    assert summary_json_result.exit_code == 0
    assert summary_payload["selection"][0]["name"] == "semantic"
    assert summary_payload["grouping"][0]["name"] == "fifo"
    assert summary_payload["selector_behavior"][0]["reason_family"] == "deterministic_signals"
    assert summary_payload["selector_behavior"][0]["strategy_selection_mode"] == "deterministic_scores"


def test_task_recent_runs_json_preserves_ingest_audit_metadata(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        task = runtime.task_queue.enqueue(
            "ingest-system1",
            task_id="recent-run-ingest-audit",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(
            task.id,
            completed_at=12.0,
            run_result={
                "claimed_entry_ids": [101, 102],
                "processed_entry_ids": [101],
                "released_entry_ids": [102],
                "meaningful_actions": 1,
                "tool_calls_executed": 2,
                "mutations": 1,
                "provider_reported_tool_calls": 3,
                "provider_reported_mutations": 2,
                "touched_memory_ids": ["memory-created", "memory-existing"],
                "appended_memory_ids": ["memory-existing"],
                "matched_memory_ids": ["memory-existing"],
                "provider_reported_touched_memory_ids": ["memory-created", "memory-existing"],
                "provider_reported_matched_memory_ids": ["memory-existing"],
                "entry_dispositions": [
                    {
                        "entry_id": 101,
                        "disposition": "created",
                        "finalization_status": "recoverable",
                        "memory_id": "memory-created",
                    },
                    {
                        "entry_id": 102,
                        "disposition": "released_unhandled",
                        "finalization_status": "released",
                    },
                ],
                "provider_reported_entry_outcomes": [
                    {
                        "entry_id": 101,
                        "disposition": "created",
                        "memory_id": "memory-created",
                        "reason": "Captured the concrete thought.",
                    }
                ],
            },
        )
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "task", "recent-runs", "--workspace-root", str(workspace), "--json"])

    payload = json.loads(result.output)
    ingest_run = next(run for run in payload["runs"] if run["task_id"] == "recent-run-ingest-audit")
    assert result.exit_code == 0
    assert ingest_run["ingest_audit"]["claimed_count"] == 2
    assert ingest_run["ingest_audit"]["handled_count"] == 1
    assert ingest_run["ingest_audit"]["released_count"] == 1
    assert ingest_run["ingest_audit"]["mutations"] == 1
    assert ingest_run["ingest_audit"]["provider_reported_mutations"] == 2
    assert ingest_run["ingest_audit"]["touched_memory_ids"] == ["memory-created", "memory-existing"]
    assert ingest_run["ingest_audit"]["appended_memory_ids"] == ["memory-existing"]
    assert ingest_run["ingest_audit"]["matched_memory_ids"] == ["memory-existing"]
    assert ingest_run["ingest_audit"]["entry_dispositions"][1]["disposition"] == "released_unhandled"
    assert ingest_run["ingest_audit"]["provider_reported_entry_outcomes"][0]["memory_id"] == "memory-created"
    assert ingest_run["result"]["provider_reported_tool_calls"] == 3


def test_task_recent_runs_human_output_shows_compact_ingest_hints(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        task = runtime.task_queue.enqueue(
            "ingest-system1",
            task_id="recent-run-ingest-hints",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(
            task.id,
            completed_at=11.0,
            run_result={
                "claimed_entry_ids": [201, 202],
                "processed_entry_ids": [201],
                "released_entry_ids": [202],
                "mutations": 1,
                "touched_memory_ids": ["memory-1"],
            },
        )
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "task", "recent-runs", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Recent Agent Runs" in result.output
    assert "handled=1" in result.output
    assert "released=1" in result.output
    assert "mut=1" in result.output
    assert "touched=1" in result.output


def test_task_show_returns_full_run_result_json(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        task = runtime.task_queue.enqueue(
            "ingest-system1",
            task_id="task-show-ingest",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(
            task.id,
            completed_at=12.0,
            run_result={
                "claimed_entry_ids": [301],
                "processed_entry_ids": [301],
                "meaningful_actions": 1,
                "strategy_selection_mode": "deterministic_scores",
                "strategy_selection_reason": "selected=semantic",
                "strategy_selection_scores": {"semantic": 0.75, "anomaly": 0.2},
                "entry_dispositions": [
                    {
                        "entry_id": 301,
                        "disposition": "created",
                        "finalization_status": "recoverable",
                        "memory_id": "memory-301",
                    }
                ],
                "touched_memory_ids": ["memory-301"],
            },
        )
    finally:
        runtime.close()

    result = runner.invoke(main, ["admin", "task", "show", "task-show-ingest", "--workspace-root", str(workspace), "--json"])
    human_result = runner.invoke(main, ["admin", "task", "show", "task-show-ingest", "--workspace-root", str(workspace)])

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert human_result.exit_code == 0
    assert "deterministic_scores" in human_result.output
    assert "selected=semantic" in human_result.output
    assert "semantic:0.75" in human_result.output
    assert payload["task"]["id"] == "task-show-ingest"
    assert payload["runs"][0]["task_id"] == "task-show-ingest"
    assert payload["runs"][0]["result_metadata"]["strategy_selection_mode"] == "deterministic_scores"
    assert payload["runs"][0]["result_metadata"]["strategy_selection_reason"] == "selected=semantic"
    assert payload["runs"][0]["result_metadata"]["strategy_selection_scores"] == {"semantic": 0.75, "anomaly": 0.2}
    assert payload["runs"][0]["result"]["entry_dispositions"][0]["memory_id"] == "memory-301"
    assert payload["runs"][0]["ingest_audit"]["entry_dispositions"][0]["disposition"] == "created"


def test_conversation_commands_render_and_return_json(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().execute(
            "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "req-cli-2",
                1,
                runtime.workspace_id,
                "deduplicator",
                "task-2",
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                5555,
                "prompt text",
                "response text",
                '{"ok": true}',
                "success",
                None,
                1.0,
                2.0,
                1.0,
            ),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    list_result = runner.invoke(
        main,
        ["admin", "conversation", "list", "--workspace-root", str(workspace), "--task-name", "deduplicator", "--json"],
    )
    show_result = runner.invoke(
        main,
        ["admin", "conversation", "show", "req-cli-2", "--workspace-root", str(workspace)],
    )
    json_result = runner.invoke(
        main,
        ["admin", "conversation", "list", "--workspace-root", str(workspace), "--json"],
    )

    list_payload = json.loads(list_result.output)
    assert list_result.exit_code == 0
    assert list_payload["conversations"][0]["request_id"] == "req-cli-2"
    assert list_payload["conversations"][0]["subprocess_pid"] == 5555
    assert show_result.exit_code == 0
    assert "Prompt" in show_result.output
    assert "Response" in show_result.output
    assert "prompt text" in show_result.output
    payload = json.loads(json_result.output)
    assert json_result.exit_code == 0
    assert payload["conversations"][0]["request_id"] == "req-cli-2"


def test_conversation_list_labels_running_rows_as_heartbeat(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None
        runtime.db_manager.get_connection().execute(
            "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "req-cli-running",
                1,
                runtime.workspace_id,
                "memory-curator",
                "task-running",
                "copilot-strong:agentic",
                "Copilot CLI Agentic",
                "claude-haiku-4.5",
                7777,
                "prompt text",
                "response text",
                None,
                "running",
                None,
                1.0,
                5.0,
                4.0,
            ),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(
        main,
        ["admin", "conversation", "list", "--workspace-root", str(workspace), "--status", "running"],
    )

    assert result.exit_code == 0
    assert "AI Conversations" in result.output
    assert "Last Update" in result.output
    assert "heartbeat" in result.output

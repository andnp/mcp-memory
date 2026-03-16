
import json
import logging
from pathlib import Path
import time

from click.testing import CliRunner

from mcp_memory.cli import main
from mcp_memory.mcp.runtime import create_runtime


pytest_plugins: list[str] = []


def test_dashboard_command_autostarts_daemon_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        transport_endpoint = "ipc:///tmp/mcp-memory.sock"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

    result = runner.invoke(main, ["dashboard", "--workspace-root", "demo"])

    assert result.exit_code == 0
    assert "Daemon transport ready:" in result.output
    assert "ipc:///tmp/mcp-memory.sock" in result.output


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
        lambda workspace_root, cwd=None: ("workspace-a", FakeMetadata(), True),
    )

    result = runner.invoke(main, ["daemon", "status"])

    assert result.exit_code == 0
    assert "Status:" in result.output
    assert "running" in result.output
    assert "ipc:///tmp/mcp-memory.sock" in result.output
    assert "Daemon scope:" in result.output
    assert "global" in result.output


def test_daemon_stop_command_reports_stopped_daemon(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        pid = 123
        workspace_id = "global"

    monkeypatch.setattr("mcp_memory.cli.stop_daemon", lambda workspace_root, cwd=None: FakeMetadata())

    result = runner.invoke(main, ["daemon", "stop"])

    assert result.exit_code == 0
    assert "Daemon stopped:" in result.output
    assert "scope=global" in result.output


def test_daemon_restart_command_restarts_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        pid = 456
        transport_endpoint = "ipc:///tmp/mcp-memory.sock"

    stop_calls: list[tuple[str | None, object | None]] = []
    start_calls: list[tuple[str | None, object | None]] = []
    monkeypatch.setattr(
        "mcp_memory.cli.stop_daemon",
        lambda workspace_root, cwd=None: stop_calls.append((workspace_root, cwd)) or None,
    )
    monkeypatch.setattr(
        "mcp_memory.cli.ensure_daemon_started",
        lambda workspace_root, cwd=None: start_calls.append((workspace_root, cwd)) or FakeMetadata(),
    )

    result = runner.invoke(main, ["daemon", "restart", "--workspace-root", "demo"])

    assert result.exit_code == 0
    assert stop_calls == [("demo", None)]
    assert start_calls == [("demo", None)]
    assert "ipc:///tmp/mcp-memory.sock" in result.output


def test_install_command_writes_workspace_hook_and_gemini_configs(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(
        main,
        [
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
        ["log", "show", "--workspace-root", str(workspace), "--source", "daemon", "--query", "warning"],
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
        ["log", "show", "--workspace-root", str(workspace), "--source", "stdio", "--json"],
    )

    payload = json.loads(result.output)
    assert result.exit_code == 0
    assert payload["logs"][0]["message"] == "json log entry"


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

    result = runner.invoke(main, ["log", "summary", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Matching logs:" in result.output
    assert "By Level" in result.output
    assert "By Source" in result.output
    assert "daemon" in result.output


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
        ["log", "prune", "--workspace-root", str(workspace), "--max-runtime-logs", "1", "--json"],
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

    result = runner.invoke(main, ["prefetch-model"])

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
        ["import-markdown", str(markdown_file), "--workspace-root", str(workspace)],
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

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

    result = runner.invoke(
        main,
        ["agents", "run", "sweeper", "--workspace-root", str(workspace)],
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
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(task.id, completed_at=14.0, run_result={"lines_compressed": 3})
        runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (runtime.workspace_id, "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.5, time.time(), None),
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace), "--verbose"])

    assert result.exit_code == 0
    assert "Search Health" in result.output
    assert "Memory Metrics" in result.output
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


def test_stats_command_default_output_is_more_compact(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Search Health" in result.output
    assert "Memory Metrics" in result.output
    assert "Background Agents" in result.output
    assert "AI Provider Usage" in result.output
    assert "Top Read Memories" in result.output
    assert "Agent Details" not in result.output
    assert "Recent Agent Runs" not in result.output


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

    health_result = runner.invoke(main, ["search", "health", "--workspace-root", str(workspace)])
    repair_result = runner.invoke(main, ["search", "repair", "--workspace-root", str(workspace)])

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

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace), "--verbose"])

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
        lambda workspace_root, cwd=None: (_ for _ in ()).throw(AssertionError("stats should not start daemon")),
    )

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace)])

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

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace_a), "--verbose"])

    assert result.exit_code == 0
    assert "Stats scope:" in result.output
    assert "global" in result.output
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
        lambda workspace_root, cwd=None: (_ for _ in ()).throw(AssertionError("stats watch should not start daemon")),
    )

    sleep_calls: list[float] = []

    def stop_after_first_refresh(seconds: float) -> None:
        sleep_calls.append(seconds)
        raise KeyboardInterrupt

    monkeypatch.setattr("mcp_memory.cli.time.sleep", stop_after_first_refresh)

    result = runner.invoke(
        main,
        ["stats", "--workspace-root", str(workspace), "--watch", "--interval", "2"],
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

    result = runner.invoke(main, ["monitor", "--workspace-root", "demo", "--interval", "1.5"])

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
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.set_running_process(task.id, subprocess_pid=4444, request_id="req-cli-1", updated_at=11.0)
    finally:
        runtime.close()

    list_result = runner.invoke(
        main,
        ["task", "list", "--workspace-root", str(workspace), "--status", "running", "--json"],
    )
    cancel_result = runner.invoke(
        main,
        ["task", "cancel", "cancel-cli-task", "--workspace-root", str(workspace), "--reason", "manual_cancel"],
    )

    list_payload = json.loads(list_result.output)
    assert list_result.exit_code == 0
    assert list_payload["tasks"][0]["task_name"] == "memory-curator"
    assert list_payload["tasks"][0]["subprocess_pid"] == 4444
    assert list_payload["tasks"][0]["active_request_id"] == "req-cli-1"
    assert cancel_result.exit_code == 0
    assert "cancellation_requested" in cancel_result.output or "cancelled" in cancel_result.output


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
        ["conversation", "list", "--workspace-root", str(workspace), "--task-name", "deduplicator", "--json"],
    )
    show_result = runner.invoke(
        main,
        ["conversation", "show", "req-cli-2", "--workspace-root", str(workspace)],
    )
    json_result = runner.invoke(
        main,
        ["conversation", "list", "--workspace-root", str(workspace), "--json"],
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

import json
from click.testing import CliRunner
from pathlib import Path

from mcp_memory.cli import main
from mcp_memory.mcp.runtime import create_runtime


pytest_plugins: list[str] = []


def test_dashboard_command_autostarts_daemon_and_prints_url(monkeypatch) -> None:
    runner = CliRunner()

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

    result = runner.invoke(main, ["dashboard", "--workspace-root", "demo"])

    assert result.exit_code == 0
    assert "http://127.0.0.1:8123/" in result.output


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

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        runtime.repository.create_memory(
            title="Stats fact",
            content="First line.\nSecond line.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        task = runtime.task_queue.enqueue(
            "defragmenter",
            task_id="defrag-stats",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(task.id, completed_at=14.0, run_result={"lines_compressed": 3})
    finally:
        runtime.close()

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Memory Metrics" in result.output
    assert "Background Agents" in result.output
    assert "Agent Details" in result.output
    assert "Recent Agent Runs" in result.output
    assert "Total lines compressed" in result.output
    assert "defragmenter" in result.output
    assert "lines_compressed=3" in result.output


def test_stats_command_shows_running_background_agents(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.cli.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())

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

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert "Background Agents" in result.output
    assert "graph-linker" in result.output
    assert "running=1" in result.output
    assert "next=" in result.output


def test_stats_command_autostarts_daemon(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    calls: list[tuple[str | None, object | None]] = []

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(
        "mcp_memory.cli.ensure_daemon_started",
        lambda workspace_root, cwd=None: calls.append((workspace_root, cwd)) or FakeMetadata(),
    )

    result = runner.invoke(main, ["stats", "--workspace-root", str(workspace)])

    assert result.exit_code == 0
    assert calls == [(str(workspace), None)]

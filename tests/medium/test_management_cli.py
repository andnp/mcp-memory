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
    assert "Total lines compressed" in result.output
    assert "defragmenter" in result.output


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
    assert "Running" in result.output


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

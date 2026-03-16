from __future__ import annotations

from pathlib import Path
import time

import pytest

from textual.widgets import DataTable, Static

from mcp_memory.cli_tui import MemoryMonitorApp
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_memory_monitor_app_renders_runtime_snapshot(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        assert runtime.db_manager is not None
        assert runtime.workspace_id is not None

        read_memory = runtime.repository.create_memory(
            title="Monitor read memory",
            content="Read me from the TUI.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        assert read_memory is not None
        runtime.repository.record_access(read_memory.id, 1.0, "2026-03-16T03:00:00+00:00", increment_read_count=True)

        task = runtime.task_queue.enqueue(
            "defragmenter",
            task_id="monitor-defrag-task",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=10.0) is not None
        runtime.task_queue.complete(task.id, completed_at=12.0, run_result={"lines_compressed": 5})

        runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "memory-curator",
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                "success",
                0.25,
                time.time(),
                None,
            ),
        )
        runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                runtime.workspace_id,
                "daemon",
                "mcp_memory.monitor",
                "INFO",
                "monitor log row",
                time.time(),
                "{}",
            ),
        )
        runtime.db_manager.get_connection().commit()

        app = MemoryMonitorApp(runtime=runtime, interval_seconds=60.0)
        async with app.run_test() as pilot:
            await pilot.pause()

            summary = app.query_one("#summary", Static)
            agent_table = app.query_one("#agent-runs", DataTable)
            provider_table = app.query_one("#provider-usage", DataTable)
            task_table = app.query_one("#task-list", DataTable)
            log_table = app.query_one("#recent-logs", DataTable)
            top_reads_table = app.query_one("#top-read-memories", DataTable)

            assert runtime.workspace_id in str(summary.content)
            assert agent_table.row_count >= 1
            assert provider_table.row_count >= 1
            assert task_table.row_count >= 1
            assert log_table.row_count >= 1
            assert top_reads_table.row_count >= 1
    finally:
        runtime.close()
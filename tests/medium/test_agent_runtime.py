import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    FACT_CHECKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    bootstrap_background_tasks,
    build_runtime_task_worker,
    handle_fact_checker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
)
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord
from mcp_memory.mcp.runtime import create_runtime
from tests.sdk.providers import FakeAIProvider


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_ingest_handler_creates_relational_memories_and_summary_tasks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(project_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("track sqlite queue status updates")
        runtime.journal.record("track sqlite queue worker retries")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.project_name,
            data={"workspace_id": runtime.project_name},
            available_at=0.0,
            task_id="ingest-test",
        )

        result = await handle_ingest_system1_task(runtime, task)
        records = runtime.repository.list_memories(workspace_id=runtime.project_name)

        assert len(result["created_memory_ids"]) == 1
        assert len(result["processed_entry_ids"]) == 2
        assert len(records) == 1
        assert records[0].type == "observation"
        assert records[0].metadata["ingest_task_id"] == "ingest-test"
        summary_task = runtime.task_queue.find_open_task(
            SUMMARIZE_MEMORY_TASK_NAME,
            runtime.project_name or "global",
        )
        assert summary_task is not None
        assert summary_task.data["memory_id"] == records[0].id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_summarize_handler_uses_provider_and_falls_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(project_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None

    try:
        record = runtime.repository.create_memory(
            title="Queue summary",
            content="First sentence. Second sentence. Third sentence.",
            workspace_ids=[runtime.project_name or "global"],
            summary="stale",
        )
        task = runtime.task_queue.enqueue(
            SUMMARIZE_MEMORY_TASK_NAME,
            data={"memory_id": record.id},
            available_at=0.0,
            task_id=f"summary:{record.id}",
        )

        provider = FakeAIProvider(responses=[{"summary": "Provider-generated summary."}])
        provider_result = await handle_summarize_memory_task(runtime, task, provider)
        updated = runtime.repository.get_memory(record.id)
        assert provider_result["summary"] == "Provider-generated summary."
        assert updated is not None
        assert updated.summary == "Provider-generated summary."

        failing_provider = FakeAIProvider(error=RuntimeError("offline"))
        fallback_result = await handle_summarize_memory_task(runtime, task, failing_provider)
        assert fallback_result["summary"] == "First sentence. Second sentence."
    finally:
        runtime.close()


def test_bootstrap_background_tasks_is_idempotent(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    ctx = ApplicationContext(project_name="workspace-a", db_manager=db_manager, task_queue=queue)

    bootstrap_background_tasks(ctx)
    bootstrap_background_tasks(ctx)

    assert queue.count_by_status() == {"pending": 3}
    assert queue.find_open_task(PROJECT_MANAGER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(FACT_CHECKER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(SWEEPER_TASK_NAME, "workspace-a") is not None


def test_project_manager_fact_checker_and_sweeper_tasks_update_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(project_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None
    assert runtime.db_manager is not None

    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        stale_plan = runtime.repository.create_memory(
            title="Old plan",
            content="This plan is stale.",
            workspace_ids=[runtime.project_name or "global"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        healthy_memory = runtime.repository.create_memory(
            title="Healthy ext link",
            content="Tracks a valid file.",
            workspace_ids=[runtime.project_name or "global"],
        )
        broken_memory = runtime.repository.create_memory(
            title="Broken ext link",
            content="Tracks a missing file.",
            workspace_ids=[runtime.project_name or "global"],
        )
        valid_file = workspace / "README.md"
        valid_file.write_text("ok", encoding="utf-8")
        runtime.repository.add_link(healthy_memory.id, f"ext:{valid_file.name}", "REFERENCES")
        runtime.repository.add_link(broken_memory.id, "ext:missing.txt", "REFERENCES")

        conn = runtime.db_manager.get_connection()
        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=8)).timestamp()
        conn.execute(
            "INSERT INTO tasks (id, task_name, workspace_id, data, status, priority, retries_count, max_retries, created_at, updated_at, available_at, completed_at, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-completed-task",
                "sweeper",
                runtime.project_name,
                "{}",
                "completed",
                100,
                0,
                3,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                None,
            ),
        )
        conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (?, ?, ?, ?)",
            ("processed note", runtime.project_name, cutoff_timestamp, "processed"),
        )
        conn.commit()

        project_result = handle_project_manager_task(
            runtime,
            TaskRecord(
                id="project-manager-task",
                task_name=PROJECT_MANAGER_TASK_NAME,
                data={"workspace_id": runtime.project_name},
                workspace_id=runtime.project_name,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        fact_result = handle_fact_checker_task(
            runtime,
            TaskRecord(
                id="fact-checker-task",
                task_name=FACT_CHECKER_TASK_NAME,
                data={
                    "workspace_id": runtime.project_name,
                    "workspace_root": str(workspace),
                },
                workspace_id=runtime.project_name,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        sweep_result = handle_sweeper_task(
            runtime,
            TaskRecord(
                id="sweeper-task",
                task_name=SWEEPER_TASK_NAME,
                data={"workspace_id": runtime.project_name},
                workspace_id=runtime.project_name,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        assert project_result["updated"] == 1
        assert fact_result["degraded"] == 1
        assert sweep_result["deleted_tasks"] == 1
        assert sweep_result["deleted_journal_entries"] == 1
        assert runtime.repository.get_memory(stale_plan.id).status == "stale"
        assert runtime.repository.get_memory(healthy_memory.id).status == "active"
        assert runtime.repository.get_memory(broken_memory.id).status == "degraded"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_processes_enqueued_ingest_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(project_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("first queue thought")
        runtime.journal.record("second queue thought")
        runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.project_name,
            data={"workspace_id": runtime.project_name},
            available_at=0.0,
        )
        worker = build_runtime_task_worker(runtime)
        await worker.start()

        for _ in range(50):
            if runtime.task_queue.count_by_status().get("completed", 0) >= 1:
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        assert runtime.repository.list_memories(workspace_id=runtime.project_name)
    finally:
        runtime.close()
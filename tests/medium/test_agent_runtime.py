import json
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    bootstrap_background_tasks,
    build_runtime_task_worker,
    handle_defragmenter_task,
    handle_conflict_detector_task,
    handle_memory_curator_task,
    handle_deduplicator_task,
    handle_fact_checker_task,
    handle_graph_linker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
    handle_taxonomist_task,
)
from mcp_memory.core.task_handlers.maintenance import CURATOR_MAX_MEMORY_CHARS, _select_curator_seed_records
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.core.journal import System1Journal
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
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("track sqlite queue status updates")
        runtime.journal.record("track sqlite queue worker retries")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-test",
        )

        result = await handle_ingest_system1_task(runtime, task)
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert len(result["created_memory_ids"]) == 1
        assert len(result["claimed_entry_ids"]) == 2
        assert len(result["deleted_entry_ids"]) == 2
        assert result["released_entry_ids"] == []
        assert result["meaningful_actions"] == 1
        assert runtime.journal.count_by_status() == {}
        assert len(records) == 1
        assert records[0].type == "observation"
        assert records[0].metadata["ingest_task_id"] == "ingest-test"
        summary_task = runtime.task_queue.find_open_task(
            SUMMARIZE_MEMORY_TASK_NAME,
            runtime.workspace_id or "global",
        )
        assert summary_task is not None
        assert summary_task.data["memory_id"] == records[0].id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_can_append_directly_into_existing_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        runtime.journal.record("The user also prefers deterministic fixtures for pytest.")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-append-test",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer deterministic fixtures for pytest.",
                        }
                    ]
                }
            ]
        )
        result = await handle_ingest_system1_task(runtime, task, provider)
        updated = runtime.repository.get_memory(target.id)
        memories = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert result["created_memory_ids"] == [target.id]
        assert len(result["deleted_entry_ids"]) == 1
        assert result["released_entry_ids"] == []
        assert updated is not None
        assert "deterministic fixtures" in updated.content
        assert len(memories) == 1
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_append_accumulates_workspace_ids_across_runtimes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True, exist_ok=True)
    workspace_b.mkdir(parents=True, exist_ok=True)

    runtime_a = create_runtime(workspace_root_override=None, cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=None, cwd=workspace_b)
    assert runtime_a.repository is not None
    assert runtime_b.repository is not None
    assert runtime_b.journal is not None
    assert runtime_b.task_queue is not None

    try:
        target = runtime_a.repository.create_memory(
            title="Shared testing preferences",
            content="Prefer pytest integration coverage.",
            workspace_ids=[runtime_a.workspace_id or "workspace-a"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None

        runtime_b.journal.record(
            "The user applies the same testing preference in this workspace.",
            workspace_id=runtime_b.workspace_id,
        )
        task = runtime_b.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime_b.workspace_id,
            data={"workspace_id": runtime_b.workspace_id},
            available_at=0.0,
            task_id="cross-workspace-append",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer pytest integration coverage.",
                        }
                    ]
                }
            ]
        )

        result = await handle_ingest_system1_task(runtime_b, task, provider)
        updated = runtime_b.repository.get_memory(target.id)

        assert result["created_memory_ids"] == [target.id]
        assert updated is not None
        assert set(updated.workspace_ids) == {runtime_a.workspace_id, runtime_b.workspace_id}
    finally:
        runtime_a.close()
        runtime_b.close()


@pytest.mark.asyncio
async def test_ingest_handler_can_use_internal_tools_before_appending(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        runtime.journal.record("The user also prefers deterministic fixtures for pytest.")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-tool-loop-test",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_search_memory_records",
                            "arguments": {
                                "query": "pytest deterministic fixtures",
                                "workspace_id": runtime.workspace_id,
                                "limit": 5,
                            },
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer deterministic fixtures for pytest.",
                        }
                    ]
                },
            ]
        )

        result = await handle_ingest_system1_task(runtime, task, provider)
        updated = runtime.repository.get_memory(target.id)

        assert result["created_memory_ids"] == [target.id]
        assert len(result["deleted_entry_ids"]) == 1
        assert updated is not None
        assert "deterministic fixtures" in updated.content
        assert provider.call_count == 2
        assert "internal_search_memory_records" in provider.prompts[1]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_releases_claimed_entries_when_actions_are_ignore_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        entry = runtime.journal.record("ignore-only thought", workspace_id=runtime.workspace_id)
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-ignore-only",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "ignore",
                            "entry_indices": [0],
                        }
                    ]
                }
            ]
        )

        result = await handle_ingest_system1_task(runtime, task, provider)

        assert result["created_memory_ids"] == []
        assert result["deleted_entry_ids"] == []
        assert result["released_entry_ids"] == [entry.id]
        assert result["meaningful_actions"] == 0
        assert [pending.id for pending in runtime.journal.get_pending(workspace_id=runtime.workspace_id)] == [entry.id]
        assert runtime.repository.list_memories(workspace_id=runtime.workspace_id) == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_summarize_handler_uses_provider_and_falls_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None

    try:
        record = runtime.repository.create_memory(
            title="Queue summary",
            content="First sentence. Second sentence. Third sentence.",
            workspace_ids=[runtime.workspace_id or "global"],
            summary="stale",
        )
        assert record is not None
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

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)

    bootstrap_background_tasks(ctx)
    bootstrap_background_tasks(ctx)

    assert queue.count_by_status() == {"pending": 9}
    assert queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None) is not None
    assert queue.find_open_task(FACT_CHECKER_TASK_NAME, None) is not None
    assert queue.find_open_task(GRAPH_LINKER_TASK_NAME, None) is not None
    assert queue.find_open_task(CONFLICT_DETECTOR_TASK_NAME, None) is not None
    assert queue.find_open_task(DEFRAGMENTER_TASK_NAME, None) is not None
    assert queue.find_open_task(DEDUPLICATOR_TASK_NAME, None) is not None
    assert queue.find_open_task(TAXONOMIST_TASK_NAME, None) is not None
    assert queue.find_open_task(SWEEPER_TASK_NAME, None) is not None
    assert queue.find_open_task(CURATOR_TASK_NAME, None) is not None


def test_bootstrap_background_tasks_enqueues_ingest_when_pending_thoughts_exist(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    journal = System1Journal(db_manager)
    journal.record("capture pending context", workspace_id="workspace-a")
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        task_queue=queue,
        journal=journal,
    )

    bootstrap_background_tasks(ctx)

    ingest_task = queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, None)
    assert ingest_task is not None
    assert ingest_task.data["trigger"] == "system1_debounce"
    assert ingest_task.available_at > ingest_task.created_at


def test_bootstrap_background_tasks_pulls_ingest_forward_at_threshold(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    journal = System1Journal(db_manager)
    for index in range(20):
        journal.record(f"capture pending context {index}", workspace_id="workspace-a")
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        task_queue=queue,
        journal=journal,
    )

    bootstrap_background_tasks(ctx)

    ingest_task = queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, None)
    assert ingest_task is not None
    assert ingest_task.data["trigger"] == "system1_threshold"
    assert ingest_task.available_at <= ingest_task.updated_at


def test_bootstrap_background_tasks_respects_persistent_task_cadence(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 200.0)

    task = queue.enqueue(
        PROJECT_MANAGER_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="project-manager-seed",
    )
    claimed = queue.claim_next(now=100.0)
    assert claimed is not None
    queue.complete(task.id, completed_at=120.0, run_result={"updated": 1})

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    bootstrap_background_tasks(ctx)

    scheduled = queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None)
    assert scheduled is not None
    assert scheduled.available_at == pytest.approx(120.0 + RECURRING_TASK_INTERVAL_SECONDS[PROJECT_MANAGER_TASK_NAME])


def test_project_manager_fact_checker_and_sweeper_tasks_update_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None
    assert runtime.db_manager is not None

    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        stale_plan = runtime.repository.create_memory(
            title="Old plan",
            content="This plan is stale.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        healthy_memory = runtime.repository.create_memory(
            title="Healthy ext link",
            content="Tracks a valid file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        broken_memory = runtime.repository.create_memory(
            title="Broken ext link",
            content="Tracks a missing file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        assert stale_plan is not None
        assert healthy_memory is not None
        assert broken_memory is not None
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
                runtime.workspace_id,
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
            ("processed note", runtime.workspace_id, cutoff_timestamp, "processed"),
        )
        conn.commit()

        project_result = handle_project_manager_task(
            runtime,
            TaskRecord(
                id="project-manager-task",
                task_name=PROJECT_MANAGER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
                    "workspace_id": runtime.workspace_id,
                    "workspace_root": str(workspace),
                },
                workspace_id=runtime.workspace_id,
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
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
        refreshed_stale = runtime.repository.get_memory(stale_plan.id)
        refreshed_healthy = runtime.repository.get_memory(healthy_memory.id)
        refreshed_broken = runtime.repository.get_memory(broken_memory.id)
        assert refreshed_stale is not None and refreshed_stale.status == "stale"
        assert refreshed_healthy is not None and refreshed_healthy.status == "active"
        assert refreshed_broken is not None and refreshed_broken.status == "degraded"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_and_conflict_detector_create_links(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        authority_fact = runtime.repository.create_memory(
            title="Auth contract",
            content="Bearer tokens are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        related_plan = runtime.repository.create_memory(
            title="Auth rollout",
            content="Ship the auth contract to all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "rollout"],
        )
        conflicting_fact = runtime.repository.create_memory(
            title="Auth contract",
            content="Bearer tokens are optional.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        assert authority_fact is not None and related_plan is not None and conflicting_fact is not None

        graph_result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
        conflict_result = await handle_conflict_detector_task(
            runtime,
            TaskRecord(
                id="conflict-detector-task",
                task_name=CONFLICT_DETECTOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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

        outgoing_from_plan = runtime.repository.get_links(related_plan.id, direction="outgoing")
        outgoing_from_fact = runtime.repository.get_links(authority_fact.id, direction="outgoing")
        incoming_to_fact = runtime.repository.get_links(authority_fact.id, direction="incoming")

        assert graph_result["created"] >= 1
        assert any(link.target_id == authority_fact.id for link in outgoing_from_plan)
        assert conflict_result["created"] == 2
        assert any(link.link_type == "CONTRADICTS" for link in outgoing_from_fact)
        assert any(link.link_type == "CONTRADICTS" for link in incoming_to_fact)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_skips_provider_when_fallback_is_sufficient(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Auth rollout plan",
            content="Roll out auth across services.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "rollout"],
        )
        second = runtime.repository.create_memory(
            title="Auth api contract",
            content="Document the auth api contract.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        third = runtime.repository.create_memory(
            title="Auth frontend tasks",
            content="Update the auth frontend screens.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "ui"],
        )
        assert first is not None and second is not None and third is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": first.id,
                            "target_id": third.id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider result",
                        }
                    ]
                }
            ]
        )

        result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-fallback-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        assert result["created"] >= 2
        assert provider.call_count == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_escalates_to_provider_with_internal_tool_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        records = []
        for index in range(13):
            record = runtime.repository.create_memory(
                title=f"topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None
            records.append(record)

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": records[0].id,
                            "target_id": records[1].id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider detected dependency",
                        }
                    ]
                }
            ]
        )

        result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-ai-tool-loop-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        assert result["created"] == 1
        assert provider.call_count == 1
        assert provider.prompts
        assert "Available internal tools:" in provider.prompts[0]
        assert "internal_search_memory_records" in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert "Use DEPENDS_ON when one memory relies on" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_conflict_detector_escalates_to_provider_when_fallback_is_sparse(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        records = []
        for index in range(16):
            record = runtime.repository.create_memory(
                    title=f"topic{index}",
                    content=f"body{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                    tags=[f"tag{index}"],
            )
            assert record is not None
            records.append(record)

        provider = FakeAIProvider(
            responses=[
                {
                    "conflicts": [
                        {
                            "left_id": records[0].id,
                            "right_id": records[1].id,
                            "context": "provider detected semantic conflict",
                        }
                    ]
                }
            ]
        )

        result = await handle_conflict_detector_task(
            runtime,
            TaskRecord(
                id="conflict-detector-ai-task",
                task_name=CONFLICT_DETECTOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        assert result["created"] == 2
        assert provider.call_count == 1
        assert provider.prompts
        assert "Available internal tools:" in provider.prompts[0]
        assert "internal_search_memory_records" in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert "materially incompatible claims" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_defragmenter_and_taxonomist_update_memory_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Auth journal 1",
            content="Captured auth rollout note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["unit_test", "auth"],
        )
        second = runtime.repository.create_memory(
            title="Auth journal 2",
            content="Captured auth rollout follow-up.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["tests", "auth"],
        )
        stable = runtime.repository.create_memory(
            title="Stable fact",
            content="Keep this as-is.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["Auth"],
        )
        assert first is not None and second is not None and stable is not None

        defrag_result = await handle_defragmenter_task(
            runtime,
            TaskRecord(
                id="defragmenter-task",
                task_name=DEFRAGMENTER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
        tax_result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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

        reflections = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="reflection",
            limit=10,
        )
        updated_first = runtime.repository.get_memory(first.id)
        updated_second = runtime.repository.get_memory(second.id)
        updated_stable = runtime.repository.get_memory(stable.id)

        assert defrag_result["created"] == 1
        assert defrag_result["archived"] == 2
        assert reflections
        source_memory_ids = reflections[0].metadata["source_memory_ids"]
        assert isinstance(source_memory_ids, list)
        assert set(source_memory_ids) == {first.id, second.id}
        assert updated_first is not None and updated_first.status == "archived"
        assert updated_second is not None and updated_second.status == "archived"
        assert tax_result["updated"] >= 1
        assert updated_stable is not None and updated_stable.tags == ["auth"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_defragmenter_skips_provider_for_small_groups(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Short note one",
            content="Brief auth note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        second = runtime.repository.create_memory(
            title="Short note two",
            content="Another brief auth note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        assert first is not None and second is not None

        provider = FakeAIProvider(responses=[{"title": "Provider title", "content": "Provider content"}])
        result = await handle_defragmenter_task(
            runtime,
            TaskRecord(
                id="defragmenter-small-task",
                task_name=DEFRAGMENTER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        reflections = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="reflection",
            limit=10,
        )
        assert result["created"] == 1
        assert provider.call_count == 0
        assert reflections[0].title in {"Reflection: Short note one", "Reflection: Short note two"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_skips_provider_when_deterministic_normalization_suffices(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Tagged fact",
            content="A fact with messy tags.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["Tests", "Authn"],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"tags": ["provider"]}])
        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-deterministic-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        updated = runtime.repository.get_memory(record.id)
        assert result["updated"] == 1
        assert provider.call_count == 0
        assert updated is not None and updated.tags == ["auth", "testing"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_uses_provider_for_untagged_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"tags": ["Authn", "Tests"]}])
        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-provider-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        updated = runtime.repository.get_memory(record.id)
        assert result["updated"] == 1
        assert provider.call_count == 1
        assert updated is not None and updated.tags == ["auth", "testing"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_merges_related_facts_and_absorbs_observations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical_fact = runtime.repository.create_memory(
            title="User testing preferences",
            content="The user prefers pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing", "preferences"],
        )
        duplicate_fact = runtime.repository.create_memory(
            title="Testing preferences",
            content="Use pytest for integration tests and avoid brittle mocks.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        new_observation = runtime.repository.create_memory(
            title="Fresh testing preference",
            content="The user likes deterministic pytest fixtures for test coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["testing", "preference"],
        )
        assert canonical_fact is not None and duplicate_fact is not None and new_observation is not None

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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

        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )
        refreshed_canonical = runtime.repository.get_memory(canonical_fact.id)
        refreshed_duplicate = runtime.repository.get_memory(duplicate_fact.id)
        refreshed_observation = runtime.repository.get_memory(new_observation.id)
        fact_links = runtime.repository.get_links(active_facts[0].id, direction="outgoing") if active_facts else []

        assert result["merged"] >= 1
        assert result["absorbed_observations"] == 1
        assert len(active_facts) == 1
        assert refreshed_canonical is not None
        assert refreshed_duplicate is not None
        assert {refreshed_canonical.status, refreshed_duplicate.status} == {"active", "archived"}
        assert refreshed_observation is not None and refreshed_observation.status == "archived"
        assert any(
            link.target_id in {canonical_fact.id, duplicate_fact.id} and link.target_id != active_facts[0].id and link.link_type == "SUPERSEDES"
            for link in fact_links
        )
        assert any(link.target_id == new_observation.id and link.link_type == "SUPERSEDES" for link in fact_links)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_skips_provider_for_short_non_subset_merges(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical = runtime.repository.create_memory(
            title="JWT auth note",
            content="JWT required. Rotate keys daily.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="JWT auth policy",
            content="Bearer auth required. Log token use.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[{"title": "Provider merged title", "content": "Provider merged content"}]
        )
        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-short-merge-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )

        assert result["merged"] >= 1
        assert provider.call_count == 0
        assert len(active_facts) == 1
        assert active_facts[0].title != "Provider merged title"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_skips_provider_for_subset_merge(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical = runtime.repository.create_memory(
            title="JWT requirement",
            content="JWTs are required for all clients. Bearer tokens are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="JWT requirement summary",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[{"title": "Provider merged title", "content": "Provider merged content"}]
        )
        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-subset-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        updated_canonical = runtime.repository.get_memory(canonical.id)
        updated_duplicate = runtime.repository.get_memory(duplicate.id)
        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )

        assert result["merged"] >= 1
        assert provider.call_count == 0
        assert updated_canonical is not None
        assert updated_duplicate is not None
        assert len(active_facts) == 1
        assert active_facts[0].title != "Provider merged title"
        assert {updated_canonical.status, updated_duplicate.status} == {"active", "archived"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_can_use_internal_tools_to_merge_memories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs must be required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_merge_memory_into_canonical",
                            "arguments": {
                                "canonical_memory_id": canonical.id,
                                "source_memory_id": duplicate.id,
                            },
                        }
                    ]
                },
                {"summary": "Merged duplicate auth fact into canonical memory.", "actions_taken": 1},
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        updated_canonical = runtime.repository.get_memory(canonical.id)
        updated_duplicate = runtime.repository.get_memory(duplicate.id)

        assert result["summary"] == "Merged duplicate auth fact into canonical memory."
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert provider.prompts
        assert "Seed memories (compact view):" in provider.prompts[0]
        assert "inputSchema" not in provider.prompts[0]
        assert "read_count" not in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert updated_canonical is not None and "JWTs must be required" in updated_canonical.content
        assert updated_duplicate is not None and updated_duplicate.status == "archived"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_can_use_internal_tools_to_split_oversized_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        oversized = runtime.repository.create_memory(
            title="Oversized rollout memory",
            content="Oversized rollout detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "rollout"],
        )
        assert oversized is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_split_memory_record",
                            "arguments": {
                                "memory_id": oversized.id,
                                "archive_original": True,
                                "parts": [
                                    {
                                        "title": "Rollout prerequisites",
                                        "content": "Step one and step two.",
                                        "tags": ["prereq"],
                                    },
                                    {
                                        "title": "Rollout execution",
                                        "content": "Step three and step four.",
                                        "tags": ["execution"],
                                    },
                                ],
                            },
                        }
                    ]
                },
                {"summary": "Split oversized rollout memory into two focused facts.", "actions_taken": 1},
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-split-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        split_children = runtime.repository.list_memories(workspace_id=runtime.workspace_id, limit=10)
        refreshed_original = runtime.repository.get_memory(oversized.id)

        assert result["summary"] == "Split oversized rollout memory into two focused facts."
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert "internal_split_memory_record" in result["tool_names_used"]
        assert refreshed_original is not None and refreshed_original.status == "archived"
        assert any(record.title == "Rollout prerequisites" for record in split_children)
        assert any(record.title == "Rollout execution" for record in split_children)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_prompt_truncates_large_seed_summaries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        huge_summary = "Important architecture note. " * 80
        record = runtime.repository.create_memory(
            title="Very long architecture reflection title that should be shortened before being sent to Gemini for curator work",
            content=huge_summary,
            summary=huge_summary,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="reflection",
            tags=["architecture", "long", "summary", "curator", "debug", "prompt", "extra-tag"],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-truncate-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        prompt = provider.prompts[0]
        assert result["summary"] == "No-op"
        assert huge_summary[:400] not in prompt
        assert "Very long architecture reflection title that should be shortened before being s" in prompt
        assert "Very long architecture reflection title that should be shortened before being sent to Gemini for curator work" not in prompt
        assert "…" in prompt
        assert len(prompt) < 5000
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_prompt_flags_oversized_seed_memories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        oversized_content = "Oversized architecture detail. " * 220
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content=oversized_content,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None
        assert len(record.content.strip()) > CURATOR_MAX_MEMORY_CHARS

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-oversized-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        prompt = provider.prompts[0]
        marker = "Seed memories (compact view):\n"
        suffix = "\n\nAvailable internal tools:"
        start = prompt.index(marker) + len(marker)
        end = prompt.index(suffix, start)
        seed_payload = json.loads(prompt[start:end])

        assert f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized." in prompt
        assert seed_payload[0]["id"] == record.id
        assert seed_payload[0]["oversized_for_curator"] is True
        assert seed_payload[0]["content_size_chars"] == len(record.content.strip())
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_prompt_serializes_seed_memories_as_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Auth rollout note",
            content="JWT rollout needs client coordination.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth", "rollout"],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-json-prompt-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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
            provider,
        )

        prompt = provider.prompts[0]
        marker = "Seed memories (compact view):\n"
        suffix = "\n\nAvailable internal tools:"
        start = prompt.index(marker) + len(marker)
        end = prompt.index(suffix, start)
        seed_json = prompt[start:end]
        seed_payload = json.loads(seed_json)

        assert isinstance(seed_payload, list)
        assert seed_payload
        assert seed_payload[0]["id"] == record.id
        assert seed_payload[0]["title"] == "Auth rollout note"
        assert seed_payload[0]["content_size_chars"] == len(record.content.strip())
        assert seed_payload[0]["oversized_for_curator"] is False
    finally:
        runtime.close()


def test_memory_curator_size_anomaly_pass_can_surface_largest_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        largest = runtime.repository.create_memory(
            title="Largest fact",
            content="L" * (CURATOR_MAX_MEMORY_CHARS - 100),
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["largest"],
        )
        assert largest is not None
        for index in range(9):
            created = runtime.repository.create_memory(
                title=f"Observation {index}",
                content="short observation",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["routine"],
            )
            assert created is not None

        seed_records = _select_curator_seed_records(
            runtime,
            TaskRecord(
                id="aaa",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
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

        assert len(seed_records) == 8
        assert largest.id in {record.id for record in seed_records}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_processes_enqueued_ingest_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("first queue thought")
        runtime.journal.record("second queue thought")
        runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
        )
        worker = build_runtime_task_worker(runtime)
        await worker.start()

        for _ in range(50):
            if runtime.task_queue.count_by_status().get("completed", 0) >= 1:
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        assert runtime.repository.list_memories(workspace_id=runtime.workspace_id)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_uses_configured_provider_for_ingest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("provider-backed ingest thought", workspace_id=runtime.workspace_id)
        runtime.ai_provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "create",
                            "entry_indices": [0],
                            "title": "Provider-backed ingest thought",
                            "content": "Provider-backed ingest content.",
                        }
                    ]
                }
            ]
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="provider-backed-ingest",
        )
        worker = build_runtime_task_worker(runtime)

        await worker.start()
        for _ in range(50):
            if runtime.task_queue.get_task(task.id).status == "completed":
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        assert runtime.ai_provider.call_count >= 1
        assert runtime.ai_provider.prompts
        assert any(
            "Analyze these system1 journal entries" in prompt
            for prompt in runtime.ai_provider.prompts
        )
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)
        assert records
        assert records[0].title == "Provider-backed ingest thought"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_drains_multiple_ingest_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None

    try:
        for index in range(45):
            runtime.journal.record(f"sqlite queue batch item {index}")

        runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="drain-ingest-test",
        )

        worker = build_runtime_task_worker(runtime)
        await worker.start()
        for _ in range(150):
            if runtime.journal.count_by_status().get("pending", 0) == 0:
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        pending_counts = runtime.journal.count_by_status()
        assert pending_counts.get("pending", 0) == 5
        assert pending_counts.get("claimed", 0) == 0
        assert pending_counts.get("processed", 0) == 0
        delayed_tasks = runtime.task_queue.list_tasks(
            status="pending",
            workspace_id=runtime.workspace_id,
            limit=5,
        )
        assert len(delayed_tasks) == 1
        assert delayed_tasks[0].available_at > delayed_tasks[0].updated_at
    finally:
        runtime.close()


class _SemanticFakeEmbedder:
    model_name = "semantic-fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["jwt", "session", "auth", "cookie"]):
                vectors.append([1.0, 0.0])
            elif any(token in lowered for token in ["sqlite", "wal", "database"]):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


@pytest.mark.asyncio
async def test_ingest_handler_clusters_semantically_related_thoughts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)
    if runtime.relational_search is not None:
        runtime.relational_search._embedder = runtime.embedder  # noqa: SLF001
        runtime.relational_search._vector_store = runtime.vector_store  # noqa: SLF001

    try:
        first = runtime.journal.record("jwt refresh rollout")
        second = runtime.journal.record("session cookie migration")
        third = runtime.journal.record("sqlite wal tuning")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="semantic-ingest-test",
        )

        result = await handle_ingest_system1_task(runtime, task)
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert len(result["created_memory_ids"]) == 2
        assert len(records) == 2
        source_sets = []
        for record in records:
            source_entry_ids = record.metadata["source_entry_ids"]
            assert isinstance(source_entry_ids, list)
            source_sets.append(set(source_entry_ids))
        assert {first.id, second.id} in source_sets
        assert {third.id} in source_sets
    finally:
        runtime.close()
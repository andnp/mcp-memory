import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    CONFLICT_DETECTOR_TASK_NAME,
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
    handle_fact_checker_task,
    handle_graph_linker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
    handle_taxonomist_task,
)
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
        assert len(result["processed_entry_ids"]) == 2
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

    assert queue.count_by_status() == {"pending": 7}
    assert queue.find_open_task(PROJECT_MANAGER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(FACT_CHECKER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(GRAPH_LINKER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(CONFLICT_DETECTOR_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(DEFRAGMENTER_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(TAXONOMIST_TASK_NAME, "workspace-a") is not None
    assert queue.find_open_task(SWEEPER_TASK_NAME, "workspace-a") is not None


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

    ingest_task = queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, "workspace-a")
    assert ingest_task is not None
    assert ingest_task.data["trigger"] == "bootstrap_pending_thoughts"


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

    scheduled = queue.find_open_task(PROJECT_MANAGER_TASK_NAME, "workspace-a")
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

        assert runtime.journal.count_by_status().get("pending", 0) == 0
        summary = runtime.task_queue.summarize_task_runs([SYSTEM1_INGEST_TASK_NAME], workspace_id=runtime.workspace_id)[0]
        assert summary.total_runs >= 3
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
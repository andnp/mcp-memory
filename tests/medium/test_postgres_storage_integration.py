from __future__ import annotations

import asyncio

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.storage.postgres_provider_usage_store import PostgresProviderUsageRepository
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_migrations import POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC


pytestmark = pytest.mark.medium


def test_postgres_integration_bootstraps_schema_and_exercises_runtime_primitives(postgres_storage_config) -> None:
    state = ensure_postgres_schema(postgres_storage_config)

    assert state.schema_metadata_present is True
    assert state.schema_version == POSTGRES_SCHEMA_VERSION

    with PostgresConnectionManager(postgres_storage_config) as manager:
        journal = PostgresSystem1Journal(manager)
        work_items = PostgresWorkItemRepository(manager)
        repair_queue = PostgresEmbeddingRepairQueue(manager)
        vector_store = PostgresVectorStore(manager)

        entry = journal.record("hello from postgres", workspace_id="workspace-a")
        pending = journal.get_pending(workspace_id="workspace-a")

        work_item, created = work_items.enqueue_unique(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            payload={"memory_id": "memory-1"},
            workspace_id="workspace-a",
            priority=10,
            available_at=5.0,
            idempotency_key="memory-tagging:memory-1",
        )
        claimed_items = work_items.claim_batch(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            lease_owner="worker-a",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )
        completed_item = work_items.complete_item(work_item.id, completed_at=20.0)

        repair_item, repair_created = repair_queue.enqueue_unique(
            memory_id="memory-1",
            workspace_id="workspace-a",
            model_name="mini-embed",
            memory_updated_at="2026-03-28T00:00:00+00:00",
            available_at=5.0,
        )
        claimed_repairs = repair_queue.claim_batch(
            lease_owner="repair-worker",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )
        completed_repair = repair_queue.complete_item(repair_item.id, completed_at=30.0)

        vector_store.upsert(
            source_kind="memory",
            source_id="memory-1",
            workspace_id="workspace-a",
            model_name="mini-embed",
            embedding=[1.0, 0.0],
        )
        vector_store.upsert(
            source_kind="memory",
            source_id="memory-2",
            workspace_id="workspace-a",
            model_name="mini-embed",
            embedding=[0.0, 1.0],
        )
        ranked = vector_store.search(
            source_kind="memory",
            model_name="mini-embed",
            query_embedding=[0.9, 0.1],
            workspace_id="workspace-a",
            limit=2,
        )

    assert entry.content == "hello from postgres"
    assert [item.content for item in pending] == ["hello from postgres"]
    assert created is True
    assert [item.id for item in claimed_items] == [work_item.id]
    assert completed_item.status == "completed"
    assert repair_created is True
    assert [item.id for item in claimed_repairs] == [repair_item.id]
    assert completed_repair.status == "completed"
    assert [memory_id for memory_id, _score in ranked] == ["memory-1", "memory-2"]


def test_postgres_integration_task_queue_lifecycle_and_task_runs(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)

        task, created = queue.enqueue_unique(
            "ingest-system1",
            data={"memory_id": "memory-1"},
            workspace_id="workspace-a",
            priority=10,
            max_retries=2,
            available_at=5.0,
        )
        duplicate, duplicate_created = queue.enqueue_unique(
            "ingest-system1",
            data={"memory_id": "memory-2"},
            workspace_id="workspace-a",
            priority=99,
            max_retries=2,
            available_at=6.0,
        )
        claimed = queue.claim_next(now=10.0, workspace_id="workspace-a")
        retried = queue.fail(task.id, "temporary failure", retry_delay_seconds=5.0, failed_at=12.0, execution_epoch=1)
        reclaimed = queue.claim_next(now=17.0, workspace_id="workspace-a")
        completed = queue.complete(task.id, completed_at=20.0, run_result={"lines_compressed": 3}, execution_epoch=2)
        task_runs = queue.list_task_runs(task_id=task.id)
        summaries = queue.summarize_task_runs(["ingest-system1"], workspace_id="workspace-a")

        assert created is True
        assert duplicate_created is False
        assert duplicate.id == task.id
        assert claimed is not None
        assert claimed.execution_epoch == 1
        assert retried.status == "pending"
        assert reclaimed is not None
        assert reclaimed.execution_epoch == 2
        assert completed.status == "completed"
        assert [run.status for run in task_runs] == ["completed", "retry"]
        assert summaries[0].completed_runs == 1
        assert summaries[0].retry_runs == 1
        assert summaries[0].total_lines_compressed == 3


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_completes_real_queue_task(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        provider_usage = PostgresProviderUsageRepository(manager, workspace_id=None)
        seen: list[dict[str, str]] = []

        def handle_ingest(_ctx: ApplicationContext, task) -> None:
            seen.append(task.data)

        task = queue.enqueue(
            "ingest-system1",
            data={"memory_id": "123"},
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-worker-task",
        )
        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                provider_usage=provider_usage,
                workspace_id="workspace-a",
            ),
            handlers={"ingest-system1": handle_ingest},
            poll_interval_seconds=0.01,
            abandoned_recovery_interval_seconds=0.05,
        )

        await worker.start()
        try:
            for _ in range(100):
                if queue.get_task(task.id).status == "completed":
                    break
                await asyncio.sleep(0.01)
        finally:
            await worker.stop(0.1)

        completed = queue.get_task(task.id)
        task_runs = queue.list_task_runs(task_id=task.id)

        assert completed.status == "completed"
        assert seen == [{"memory_id": "123"}]
        assert [run.status for run in task_runs] == ["completed"]
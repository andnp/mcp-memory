from __future__ import annotations

import pytest

from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_migrations import POSTGRES_SCHEMA_VERSION
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
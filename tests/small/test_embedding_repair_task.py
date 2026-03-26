from __future__ import annotations

import asyncio

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.management.health_reporting import build_search_health
from mcp_memory.core.task_handlers.constants import EMBEDDING_REPAIR_TASK_NAME
from mcp_memory.core.task_handlers.embedding_repair import handle_embedding_repair_task
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.work_item_store import SQLiteWorkItemRepository


pytestmark = pytest.mark.small


class _QueueAwareFakeEmbedder:
    model_name = "queue-fake-mini"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["auth", "permission", "token", "security", "identity"]):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


@pytest.mark.asyncio
async def test_search_can_wait_for_queue_backed_embedding_repairs(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    work_items = SQLiteWorkItemRepository(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        config=__import__("mcp_memory.config", fromlist=["Config"]).Config(),
        embedder=_QueueAwareFakeEmbedder(),
        vector_store=vector_store,
        db_manager=db_manager,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=0.5,
    )
    ctx = ApplicationContext(
        workspace_id="workspace-alpha",
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        embedder=service._embedder,
        vector_store=vector_store,
    )

    record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    worker = RuntimeTaskWorker(
        ctx,
        handlers={EMBEDDING_REPAIR_TASK_NAME: handle_embedding_repair_task},
        poll_interval_seconds=0.01,
        retry_delay_seconds=0.0,
    )
    await worker.start()
    try:
        results = await asyncio.to_thread(
            service.search_memories,
            "permissions security",
            "workspace-alpha",
            5,
        )
    finally:
        await worker.stop(1.0)

    embedder = service._embedder
    assert embedder is not None
    assert results
    assert results[0].memory_id == record.id
    stored = vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name=embedder.model_name,
    )
    health = service.get_health()
    assert stored is not None
    assert stored.workspace_id is None
    assert embedding_repair_queue.list_items(status="running") == []
    assert embedding_repair_queue.list_items(status="completed") == []
    assert health.background_repair_enabled is True
    assert health.repair_wait_count == 1
    assert health.partial_semantic_search_count == 0
    assert health.queued_repair_backlog_count == 0
    assert health.running_repair_count == 0
    assert build_search_health(service).repair_wait_count == 1


def test_search_without_worker_keeps_best_effort_results_and_queues_repairs(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    work_items = SQLiteWorkItemRepository(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        config=__import__("mcp_memory.config", fromlist=["Config"]).Config(),
        embedder=_QueueAwareFakeEmbedder(),
        vector_store=vector_store,
        db_manager=db_manager,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=0.01,
    )

    record = repository.create_memory(
        title="Permission rollout note",
        content="Permission rollout note with lexical search terms.",
        summary="Permission rollout summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    results = service.search_memories("permission rollout", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id
    queued_items = embedding_repair_queue.list_items(limit=10)
    health = service.get_health()
    assert queued_items
    queued_task = task_queue.find_open_task(EMBEDDING_REPAIR_TASK_NAME, None)
    assert queued_task is not None
    assert health.background_repair_enabled is True
    assert health.repair_wait_count == 1
    assert health.partial_semantic_search_count == 1
    assert health.last_partial_semantic_at is not None
    assert health.queued_repair_backlog_count >= 1
    assert health.last_repair_candidate_count >= 1
    assert health.last_repair_pending_count >= 1


def test_search_queues_repairs_in_specialized_backlog_store(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        config=__import__("mcp_memory.config", fromlist=["Config"]).Config(),
        embedder=_QueueAwareFakeEmbedder(),
        vector_store=vector_store,
        db_manager=db_manager,
        task_queue=task_queue,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=0.01,
    )

    record = repository.create_memory(
        title="Identity policy backlog",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)

    embedder = service._embedder
    assert embedder is not None
    queued_items = embedding_repair_queue.list_items(limit=10)
    assert len(queued_items) == 1
    assert queued_items[0].memory_id == record.id
    assert queued_items[0].model_name == embedder.model_name


def test_embedding_repair_queue_prune_completed_removes_old_rows(db_manager) -> None:
    queue = SQLiteEmbeddingRepairQueue(db_manager)
    item, created = queue.enqueue_unique(
        memory_id="memory-1",
        workspace_id=None,
        model_name="demo-model",
        memory_updated_at="2026-03-24T00:00:00+00:00",
        available_at=0.0,
    )
    assert created is True
    claimed = queue.claim_batch(lease_owner="task-1", limit=1)
    assert [entry.id for entry in claimed] == [item.id]
    queue.complete_item(item.id, completed_at=10.0)

    pruned = queue.prune_completed(older_than_seconds=0.0, limit=10, now=20.0)

    assert pruned == 1
    assert queue.list_items(limit=10) == []
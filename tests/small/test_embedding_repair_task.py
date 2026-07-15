from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.management.health_reporting import build_search_health
from mcp_memory.core.task_handlers.constants import EMBEDDING_REPAIR_TASK_NAME
from mcp_memory.core.task_handlers.embedding_repair import handle_embedding_repair_task
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.embeddings import EmbeddingRecord, SQLiteVectorStore
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


@dataclass
class _IntegrityScanRow:
    source_kind: str
    source_id: str
    model_name: str
    updated_at: float
    issues: list[str]


class _IntegrityScanVectorStore:
    def __init__(self, rows: list[_IntegrityScanRow]) -> None:
        self._rows = list(rows)
        self._records: dict[tuple[str, str, str], EmbeddingRecord] = {
            (row.source_kind, row.source_id, row.model_name): EmbeddingRecord(
                source_kind=row.source_kind,
                source_id=row.source_id,
                workspace_id=None,
                model_name=row.model_name,
                embedding=[9.0, 9.0],
                updated_at=row.updated_at,
            )
            for row in rows
            if row.source_kind == "memory"
        }
        self.upserted_ids: list[str] = []

    def scan_integrity(self, *, active_model_name: str | None = None, expected_dimension: int | None = None):
        _ = expected_dimension
        invalid_rows = [
            {
                "source_kind": row.source_kind,
                "source_id": row.source_id,
                "workspace_id": None,
                "model_name": row.model_name,
                "dimension": None if row.issues else 2,
                "payload_type": "object" if "invalid_payload_type" in row.issues else "array",
                "updated_at": row.updated_at,
                "issues": list(row.issues),
            }
            for row in self._rows
            if row.issues
        ]
        active_rows = [
            {
                "source_kind": row.source_kind,
                "source_id": row.source_id,
                "workspace_id": None,
                "model_name": row.model_name,
                "dimension": None if "invalid_payload_type" in row.issues else (3 if "dimension_mismatch" in row.issues else 2),
                "payload_type": "object" if "invalid_payload_type" in row.issues else "array",
                "updated_at": row.updated_at,
                "issues": list(row.issues),
            }
            for row in self._rows
            if row.model_name == active_model_name
        ]
        source_kind_counts: dict[str, int] = {}
        for row in active_rows:
            source_kind = str(row["source_kind"])
            source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1
        return {
            "scanned_row_count": len(self._rows),
            "invalid_row_count": len(invalid_rows),
            "mixed_dimension_group_count": 1,
            "mixed_dimension_groups": [
                {
                    "model_name": active_model_name,
                    "dimensions": [2, 3],
                    "row_count": len(active_rows),
                    "source_kind_counts": source_kind_counts,
                }
            ],
            "invalid_rows": invalid_rows,
            "active_model_name": active_model_name,
            "active_model_row_count": len(active_rows),
            "active_model_rows": active_rows,
        }

    def delete(self, *, source_kind: str, source_id: str, model_name: str | None = None) -> int:
        key = (source_kind, source_id, model_name or "")
        if key in self._records:
            self._records.pop(key)
            return 1
        return 0

    def get(self, *, source_kind: str, source_id: str, model_name: str):
        return self._records.get((source_kind, source_id, model_name))

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
        memory_updated_at: str | None = None,
    ) -> bool:
        _ = memory_updated_at
        self._records[(source_kind, source_id, model_name)] = EmbeddingRecord(
            source_kind=source_kind,
            source_id=source_id,
            workspace_id=workspace_id,
            model_name=model_name,
            embedding=list(embedding),
            updated_at=time.time(),
        )
        self.upserted_ids.append(source_id)
        return True


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


def test_versioned_embedding_repair_rejects_a_stale_write(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    record = repository.create_memory(
        title="Versioned memory",
        content="first version",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-04-11T00:00:00+00:00",
        created_at="2026-04-11T00:00:00+00:00",
    )
    assert record is not None

    assert vector_store.upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id=None,
        model_name="versioned-model",
        embedding=[1.0, 0.0],
        memory_updated_at=record.updated_at,
    ) is True
    newer = repository.update_memory(record.id, content="second version")
    assert newer is not None
    assert newer.updated_at != record.updated_at
    assert vector_store.upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id=None,
        model_name="versioned-model",
        embedding=[0.0, 1.0],
        memory_updated_at=newer.updated_at,
    ) is True

    assert vector_store.upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id=None,
        model_name="versioned-model",
        embedding=[9.0, 9.0],
        memory_updated_at=record.updated_at,
    ) is False
    stored = vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name="versioned-model",
    )
    assert stored is not None
    assert stored.embedding == [0.0, 1.0]
    assert stored.memory_updated_at == newer.updated_at


@pytest.mark.asyncio
async def test_embedding_repair_task_scans_integrity_and_repairs_missing_or_invalid_memory_rows(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    embedder = _QueueAwareFakeEmbedder()

    invalid_record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-04-11T00:00:00+00:00",
        created_at="2026-04-11T00:00:00+00:00",
    )
    missing_record = repository.create_memory(
        title="Permission rollout",
        content="Permission rollout note with lexical search terms.",
        summary="Permission rollout summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-04-11T01:00:00+00:00",
        created_at="2026-04-11T01:00:00+00:00",
    )
    stale_record = repository.create_memory(
        title="Security audit",
        content="Security audit findings for token scope drift.",
        summary="Security audit summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-04-11T02:00:00+00:00",
        created_at="2026-04-11T02:00:00+00:00",
    )
    assert invalid_record is not None and missing_record is not None and stale_record is not None
    vector_store = _IntegrityScanVectorStore(
        [
            _IntegrityScanRow(
                source_kind="memory",
                source_id=invalid_record.id,
                model_name=embedder.model_name,
                updated_at=10.0,
                issues=["dimension_mismatch"],
            ),
            _IntegrityScanRow(
                source_kind="memory",
                source_id=stale_record.id,
                model_name=embedder.model_name,
                updated_at=10.0,
                issues=[],
            ),
            _IntegrityScanRow(
                source_kind="thought",
                source_id="thought-invalid",
                model_name=embedder.model_name,
                updated_at=20.0,
                issues=["invalid_payload_type"],
            ),
        ]
    )

    task = TaskRecord(
        id="task-embedding-repair",
        task_name=EMBEDDING_REPAIR_TASK_NAME,
        data={"batch_size": 10, "max_batches_per_run": 10, "integrity_scan_limit": 10},
        workspace_id=None,
        status="running",
        priority=1,
        retries_count=0,
        max_retries=0,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )
    ctx = ApplicationContext(
        repository=repository,
        embedder=embedder,
        vector_store=vector_store,
        embedding_repair_queue=embedding_repair_queue,
    )

    result = await handle_embedding_repair_task(ctx, task)

    assert result["repaired"] == 3
    assert result["claimed_work_item_count"] == 3
    assert result["pruned_completed"] == 3
    assert set(vector_store.upserted_ids) == {invalid_record.id, missing_record.id, stale_record.id}
    assert embedding_repair_queue.list_items(limit=10) == []
    assert result["integrity_scan"] == {
        "supported": True,
        "active_model_name": embedder.model_name,
        "active_model_dimension": 2,
        "memory_scan_limit": 10,
        "memory_scanned_count": 3,
        "embedding_row_scanned_count": 3,
        "invalid_row_count": 2,
        "mixed_dimension_group_count": 1,
        "mixed_dimension_groups": [
            {
                "model_name": embedder.model_name,
                "dimensions": [2, 3],
                "row_count": 3,
                "source_kind_counts": {"memory": 2, "thought": 1},
            }
        ],
        "missing_memory_embedding_count": 1,
        "invalid_memory_embedding_count": 1,
        "stale_memory_embedding_count": 1,
        "invalid_thought_embedding_count": 1,
        "skipped_thought_rows": 1,
        "deleted_invalid_memory_rows": 1,
        "enqueued_memory_repairs": 3,
        "already_queued_memory_repairs": 0,
    }

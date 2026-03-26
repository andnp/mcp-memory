from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC, WORK_FAMILY_MEMORY_EMBEDDING_REPAIR

if TYPE_CHECKING:
    from mcp_memory.context import ApplicationContext


DEFAULT_EMBEDDING_REPAIR_BATCH_SIZE = 32
DEFAULT_EMBEDDING_REPAIR_MAX_BATCHES_PER_RUN = 8
DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT = 256


async def handle_embedding_repair_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    from mcp_memory.relational.search import _embedding_is_stale, _memory_embedding_text

    repository = getattr(ctx, "repository", None)
    embedder = getattr(ctx, "embedder", None)
    vector_store = getattr(ctx, "vector_store", None)
    work_items = getattr(ctx, "work_items", None)
    embedding_repair_queue = getattr(ctx, "embedding_repair_queue", None)
    if repository is None or embedder is None or vector_store is None:
        return {
            "repaired": 0,
            "claimed_work_item_count": 0,
            "batches_processed": 0,
            "reason": "semantic_search_not_initialized",
        }
    if embedding_repair_queue is None and work_items is None:
        return {
            "repaired": 0,
            "claimed_work_item_count": 0,
            "batches_processed": 0,
            "reason": "repair_queue_not_initialized",
        }

    batch_size = max(int(task.data.get("batch_size", DEFAULT_EMBEDDING_REPAIR_BATCH_SIZE)), 1)
    max_batches_per_run = max(int(task.data.get("max_batches_per_run", DEFAULT_EMBEDDING_REPAIR_MAX_BATCHES_PER_RUN)), 1)
    repaired = 0
    claimed_work_item_count = 0
    batches_processed = 0

    while batches_processed < max_batches_per_run:
        if embedding_repair_queue is not None:
            claimed_items = embedding_repair_queue.claim_batch(
                lease_owner=task.id,
                limit=batch_size,
                workspace_id=task.workspace_id,
            )
        else:
            assert work_items is not None
            claimed_items = work_items.claim_batch(
                family_key=WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
                execution_lane=EXECUTION_LANE_DETERMINISTIC,
                lease_owner=task.id,
                limit=batch_size,
                workspace_id=task.workspace_id,
            )
        if not claimed_items:
            break

        batches_processed += 1
        claimed_work_item_count += len(claimed_items)
        repair_pairs: list[tuple[Any, Any]] = []

        for work_item in claimed_items:
            if embedding_repair_queue is not None:
                memory_id = work_item.memory_id
                queued_model_name = work_item.model_name
                queued_updated_at = work_item.memory_updated_at
                completer = embedding_repair_queue.complete_item
            else:
                assert work_items is not None
                payload = work_item.payload
                memory_id = payload.get("memory_id")
                queued_model_name = payload.get("model_name")
                queued_updated_at = payload.get("memory_updated_at")
                completer = work_items.complete_item

            if not isinstance(memory_id, str) or not memory_id.strip():
                completer(work_item.id)
                continue

            if not isinstance(queued_model_name, str) or queued_model_name != embedder.model_name:
                completer(work_item.id)
                continue

            record = repository.get_memory(memory_id)
            if record is None or record.status == "archived":
                completer(work_item.id)
                continue

            if isinstance(queued_updated_at, str) and queued_updated_at != (record.updated_at or ""):
                completer(work_item.id)
                continue

            existing = vector_store.get(
                source_kind="memory",
                source_id=record.id,
                model_name=embedder.model_name,
            )
            if existing is not None and not _embedding_is_stale(existing.updated_at, record.updated_at):
                completer(work_item.id)
                continue

            repair_pairs.append((record, work_item))

        if not repair_pairs:
            continue

        embeddings = embedder.embed([_memory_embedding_text(record) for record, _ in repair_pairs])
        for (record, work_item), embedding in zip(repair_pairs, embeddings, strict=False):
            vector_store.upsert(
                source_kind="memory",
                source_id=record.id,
                workspace_id=None,
                model_name=embedder.model_name,
                embedding=embedding,
            )
            if embedding_repair_queue is not None:
                embedding_repair_queue.complete_item(work_item.id)
            else:
                assert work_items is not None
                work_items.complete_item(work_item.id)
            repaired += 1

    pruned_completed = 0
    if embedding_repair_queue is not None:
        pruned_completed = embedding_repair_queue.prune_completed(limit=DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT)

    return {
        "repaired": repaired,
        "claimed_work_item_count": claimed_work_item_count,
        "batches_processed": batches_processed,
        "pruned_completed": pruned_completed,
    }
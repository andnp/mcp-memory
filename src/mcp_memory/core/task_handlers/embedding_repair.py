from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.relational.search import _embedding_is_stale
from mcp_memory.core.ports.work_items import EXECUTION_LANE_DETERMINISTIC, WORK_FAMILY_MEMORY_EMBEDDING_REPAIR
from searchkernel.ingestion import EmbeddingInput, embed_and_upsert

if TYPE_CHECKING:
    from mcp_memory.context import ApplicationContext


DEFAULT_EMBEDDING_REPAIR_BATCH_SIZE = 32
DEFAULT_EMBEDDING_REPAIR_MAX_BATCHES_PER_RUN = 8
DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT = 256
DEFAULT_EMBEDDING_REPAIR_INTEGRITY_SCAN_LIMIT = 10_000


def _normalize_scan_summary(summary: Any) -> dict[str, Any]:
    return summary if isinstance(summary, dict) else {}


def _summary_int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    return int(value) if isinstance(value, int | float | str) else 0


def _active_embedder_dimension(embedder: Any) -> int | None:
    configured_dimension = getattr(embedder, "embedding_dimension", None)
    if isinstance(configured_dimension, int) and configured_dimension > 0:
        return configured_dimension

    internal_dimension = getattr(embedder, "_dimensions", None)
    if isinstance(internal_dimension, int) and internal_dimension > 0:
        return internal_dimension

    probe_embeddings = embedder.embed(["embedding integrity scan probe"])
    if not probe_embeddings:
        return None
    probe = probe_embeddings[0]
    return len(probe) if isinstance(probe, list) and probe else None


def _enqueue_memory_embedding_repair(
    *,
    work_items: Any,
    embedding_repair_queue: Any,
    memory_id: str,
    workspace_id: str | None,
    model_name: str,
    memory_updated_at: str,
) -> bool:
    if embedding_repair_queue is not None:
        _, created = embedding_repair_queue.enqueue_unique(
            memory_id=memory_id,
            workspace_id=workspace_id,
            model_name=model_name,
            memory_updated_at=memory_updated_at,
        )
        return bool(created)

    assert work_items is not None
    _, created = work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        workspace_id=workspace_id,
        idempotency_key=f"{WORK_FAMILY_MEMORY_EMBEDDING_REPAIR}:{model_name}:{memory_id}:{memory_updated_at}",
        payload={
            "memory_id": memory_id,
            "model_name": model_name,
            "memory_updated_at": memory_updated_at,
        },
    )
    return bool(created)


def _run_integrity_scan(
    *,
    repository: Any,
    embedder: Any,
    vector_store: Any,
    work_items: Any,
    embedding_repair_queue: Any,
    scan_limit: int,
) -> dict[str, Any]:
    scan_integrity = getattr(vector_store, "scan_integrity", None)
    list_memories = getattr(repository, "list_memories", None)
    delete_embedding = getattr(vector_store, "delete", None)
    if not callable(scan_integrity) or not callable(list_memories):
        return {
            "supported": False,
            "reason": "integrity_scan_unsupported",
        }

    active_dimension = _active_embedder_dimension(embedder)
    if active_dimension is None:
        return {
            "supported": False,
            "reason": "active_embedder_dimension_unavailable",
        }

    active_memories = [
        memory
        for memory in repository.list_memories(status="active", limit=scan_limit)
        if memory.status != "archived"
    ]
    scan_summary = _normalize_scan_summary(scan_integrity(
        active_model_name=embedder.model_name,
        expected_dimension=active_dimension,
    ))
    active_model_rows = [
        row
        for row in scan_summary.get("active_model_rows", [])
        if isinstance(row, dict)
    ] if isinstance(scan_summary.get("active_model_rows", []), list) else []
    active_memory_rows = {
        (
            None if row.get("workspace_id") is None else str(row["workspace_id"]),
            str(row["source_id"]),
        ): row
        for row in active_model_rows
        if row.get("source_kind") == "memory"
    }
    active_memory_rows_by_source_id = {
        str(row["source_id"]): row
        for row in active_model_rows
        if row.get("source_kind") == "memory"
    }

    invalid_rows = [
        row
        for row in scan_summary.get("invalid_rows", [])
        if isinstance(row, dict)
    ] if isinstance(scan_summary.get("invalid_rows", []), list) else []
    invalid_thought_rows = [
        row
        for row in invalid_rows
        if row.get("model_name") == embedder.model_name
        and row.get("source_kind") == "thought"
    ]

    missing_memory_count = 0
    invalid_memory_count = 0
    stale_memory_count = 0
    deleted_invalid_memory_rows = 0
    enqueued_memory_repairs = 0
    already_queued_memory_repairs = 0

    for record in active_memories:
        workspace_id = next(
            (item for item in record.workspace_ids if item),
            None,
        )
        row = active_memory_rows.get((workspace_id, record.id))
        if row is None:
            row = active_memory_rows_by_source_id.get(record.id)
        should_enqueue = False
        if row is None:
            missing_memory_count += 1
            should_enqueue = True
        else:
            issues = row.get("issues", [])
            normalized_issues = [str(issue) for issue in issues] if isinstance(issues, list | tuple) else []
            if normalized_issues:
                invalid_memory_count += 1
                should_enqueue = True
                if callable(delete_embedding):
                    deleted_count = delete_embedding(
                        source_kind="memory",
                        source_id=record.id,
                        model_name=embedder.model_name,
                    )
                    if isinstance(deleted_count, int | float | str):
                        deleted_invalid_memory_rows += int(deleted_count)
            else:
                row_updated_at = row.get("updated_at")
                if isinstance(row_updated_at, (int, float)) and _embedding_is_stale(float(row_updated_at), record.updated_at):
                    stale_memory_count += 1
                    should_enqueue = True

        if not should_enqueue:
            continue

        created = _enqueue_memory_embedding_repair(
            work_items=work_items,
            embedding_repair_queue=embedding_repair_queue,
            memory_id=record.id,
            workspace_id=workspace_id,
            model_name=embedder.model_name,
            memory_updated_at=record.updated_at or "",
        )
        if created:
            enqueued_memory_repairs += 1
        else:
            already_queued_memory_repairs += 1

    return {
        "supported": True,
        "active_model_name": embedder.model_name,
        "active_model_dimension": active_dimension,
        "memory_scan_limit": scan_limit,
        "memory_scanned_count": len(active_memories),
        "embedding_row_scanned_count": _summary_int(scan_summary, "scanned_row_count"),
        "invalid_row_count": _summary_int(scan_summary, "invalid_row_count"),
        "mixed_dimension_group_count": _summary_int(scan_summary, "mixed_dimension_group_count"),
        "mixed_dimension_groups": list(scan_summary.get("mixed_dimension_groups", [])),
        "missing_memory_embedding_count": missing_memory_count,
        "invalid_memory_embedding_count": invalid_memory_count,
        "stale_memory_embedding_count": stale_memory_count,
        "invalid_thought_embedding_count": len(invalid_thought_rows),
        "skipped_thought_rows": len(invalid_thought_rows),
        "deleted_invalid_memory_rows": deleted_invalid_memory_rows,
        "enqueued_memory_repairs": enqueued_memory_repairs,
        "already_queued_memory_repairs": already_queued_memory_repairs,
    }


async def handle_embedding_repair_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    from mcp_memory.relational.search import _memory_embedding_text

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

    integrity_scan = _run_integrity_scan(
        repository=repository,
        embedder=embedder,
        vector_store=vector_store,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        scan_limit=max(int(task.data.get("integrity_scan_limit", DEFAULT_EMBEDDING_REPAIR_INTEGRITY_SCAN_LIMIT)), 1),
    )

    if embedding_repair_queue is None and work_items is None:
        return {
            "repaired": 0,
            "claimed_work_item_count": 0,
            "batches_processed": 0,
            "reason": "repair_queue_not_initialized",
            "integrity_scan": integrity_scan,
        }

    batch_size = max(int(task.data.get("batch_size", DEFAULT_EMBEDDING_REPAIR_BATCH_SIZE)), 1)
    max_batches_per_run = max(int(task.data.get("max_batches_per_run", DEFAULT_EMBEDDING_REPAIR_MAX_BATCHES_PER_RUN)), 1)
    repaired = 0
    claimed_work_item_count = 0
    batches_processed = 0

    while batches_processed < max_batches_per_run:
        claim_workspace_id = task.workspace_id if task.workspace_id is not None else "*"
        if embedding_repair_queue is not None:
            claimed_items = embedding_repair_queue.claim_batch(
                lease_owner=task.id,
                limit=batch_size,
                workspace_id=claim_workspace_id,
            )
        else:
            assert work_items is not None
            claimed_items = work_items.claim_batch(
                family_key=WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
                execution_lane=EXECUTION_LANE_DETERMINISTIC,
                lease_owner=task.id,
                limit=batch_size,
                workspace_id=claim_workspace_id,
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

        embedding_result = embed_and_upsert(
            [
                EmbeddingInput(
                    source_kind="memory",
                    source_id=record.id,
                    workspace_id=next(
                        (item for item in record.workspace_ids if item),
                        None,
                    ),
                    text=_memory_embedding_text(record),
                    source_updated_at=record.updated_at,
                )
                for record, _ in repair_pairs
            ],
            provider=embedder,
            sink=vector_store,
            batch_size=len(repair_pairs),
        )
        for _, work_item in repair_pairs:
            if embedding_repair_queue is not None:
                embedding_repair_queue.complete_item(work_item.id)
            else:
                assert work_items is not None
                work_items.complete_item(work_item.id)
        repaired += embedding_result.stored

    pruned_completed = 0
    if embedding_repair_queue is not None:
        pruned_completed = embedding_repair_queue.prune_completed(limit=DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT)

    return {
        "repaired": repaired,
        "claimed_work_item_count": claimed_work_item_count,
        "batches_processed": batches_processed,
        "pruned_completed": pruned_completed,
        "integrity_scan": integrity_scan,
    }

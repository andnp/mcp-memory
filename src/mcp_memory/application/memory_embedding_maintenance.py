from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, cast

from searchkernel.ingestion import EmbeddingInput, embed_and_upsert

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.core.ports.work_items import (
    EXECUTION_LANE_DETERMINISTIC,
    WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
)
from mcp_memory.core.search_repair import EmbeddingRepairScheduler
from mcp_memory.embeddings import is_fallback_embedding_model

logger = logging.getLogger(__name__)

INLINE_EMBEDDING_REPAIR_LIMIT = 64
DEFAULT_BACKGROUND_REPAIR_BATCH_SIZE = 32
DEFAULT_BACKGROUND_REPAIR_MAX_BATCHES_PER_RUN = 8
DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT = 256
DEFAULT_EMBEDDING_REPAIR_INTEGRITY_SCAN_LIMIT = 10_000


@dataclass(slots=True)
class EmbeddingMaintenanceHealth:
    semantic_enabled: bool
    available: bool
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    background_repair_enabled: bool = False
    background_repair_wait_seconds: float = 0.0
    queued_repair_backlog_count: int = 0
    running_repair_count: int = 0
    oldest_queued_repair_age_seconds: float | None = None
    repair_wait_count: int = 0
    partial_semantic_search_count: int = 0
    last_partial_semantic_at: str | None = None
    last_repair_wait_seconds: float = 0.0
    last_repair_candidate_count: int = 0
    last_repair_pending_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    last_recovery_at: str | None = None
    last_integrity_check_at: str | None = None
    integrity_check_error: str | None = None


class MemoryEmbeddingMaintenance:
    """Application-owned orchestration for memory embedding persistence and repair."""

    def __init__(
        self,
        repository: Any,
        config: Config,
        *,
        embedder: Any | None = None,
        vector_store: Any | None = None,
        db_manager: Any | None = None,
        task_queue: Any | None = None,
        work_items: Any | None = None,
        embedding_repair_queue: Any | None = None,
        background_repair_wait_seconds: float = 0.0,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._db_manager = db_manager
        self._work_items = work_items
        self._repair_queue = embedding_repair_queue
        wait_seconds = max(background_repair_wait_seconds, 0.0)
        self._scheduler = (
            EmbeddingRepairScheduler(
                config=config,
                task_queue=task_queue,
                work_items=work_items,
                embedding_repair_queue=embedding_repair_queue,
                wait_seconds=wait_seconds,
            )
            if task_queue is not None and (embedding_repair_queue is not None or work_items is not None)
            else None
        )
        semantic_enabled = embedder is not None and vector_store is not None
        self._health = EmbeddingMaintenanceHealth(
            semantic_enabled=semantic_enabled,
            available=semantic_enabled,
            background_repair_enabled=self._scheduler is not None and wait_seconds > 0,
            background_repair_wait_seconds=wait_seconds,
        )

    @classmethod
    def from_context(cls, ctx: Any) -> MemoryEmbeddingMaintenance:
        return cls(
            getattr(ctx, "repository", None),
            getattr(ctx, "config", None) or Config(),
            embedder=getattr(ctx, "embedder", None),
            vector_store=getattr(ctx, "vector_store", None),
            db_manager=getattr(ctx, "db_manager", None),
            task_queue=getattr(ctx, "task_queue", None),
            work_items=getattr(ctx, "work_items", None),
            embedding_repair_queue=getattr(ctx, "embedding_repair_queue", None),
        )

    def get_health(self) -> EmbeddingMaintenanceHealth:
        self._refresh_background_repair_health_snapshot()
        return replace(self._health)

    def mark_semantic_failure(self, exc: Exception, *, fallback: bool) -> None:
        self._health.available = False
        self._health.degraded = fallback or self._health.degraded
        self._health.last_error = str(exc)
        self._health.last_failure_at = _utc_now()
        if fallback:
            self._health.fallback_count += 1

    def mark_semantic_recovered(self) -> None:
        self._health.available = self._health.semantic_enabled
        self._health.degraded = False
        self._health.last_error = None
        self._health.last_failure_at = None
        self._health.last_recovery_at = _utc_now()
        self._health.integrity_check_error = None

    def blocked_embedding_persistence_reason(self) -> str | None:
        if self._embedder is None or self._vector_store is None:
            return None
        if not is_fallback_embedding_model(self._embedder.model_name):
            return None
        get_write_policy_state = getattr(self._vector_store, "get_write_policy_state", None)
        if not callable(get_write_policy_state):
            return None
        policy_state = get_write_policy_state()
        if getattr(policy_state, "fallback_persistence_policy", None) != "blocked":
            return None
        return (
            "Fallback/hash embeddings cannot be persisted in Postgres/shared mode; "
            f"blocked model_name {self._embedder.model_name!r}"
        )

    def run_startup_health_check(self) -> EmbeddingMaintenanceHealth:
        check_timestamp = _utc_now()
        self._health.last_integrity_check_at = check_timestamp
        self._health.integrity_check_error = None
        if not self._health.semantic_enabled:
            self._health.available = False
            self._health.degraded = False
            return self.get_health()
        try:
            self._run_integrity_check_once()
            self.mark_semantic_recovered()
            self._health.last_recovery_at = check_timestamp
        except (OSError, sqlite3.Error, ValueError) as exc:
            recovered = self._retry_after_reopen(
                "startup health check",
                lambda: self._run_integrity_check_once() or True,
            )
            if not recovered:
                self.mark_semantic_failure(exc, fallback=False)
                self._health.integrity_check_error = str(exc)
        return self.get_health()

    def rebuild_semantic_index(self, *, limit: int = 10_000) -> dict[str, int | bool | str | None]:
        if not self._health.semantic_enabled or self._embedder is None or self._vector_store is None:
            return {
                "semantic_enabled": False,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "semantic_search_disabled",
            }
        blocked_reason = self.blocked_embedding_persistence_reason()
        if blocked_reason is not None:
            self.mark_semantic_failure(ValueError(blocked_reason), fallback=True)
            return {
                "semantic_enabled": True,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "fallback_embedding_persistence_blocked",
            }
        delete_by_model = getattr(self._vector_store, "delete_by_model", None)
        if not callable(delete_by_model):
            return {
                "semantic_enabled": True,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "vector_store_rebuild_unsupported",
            }
        candidates = [
            record
            for record in self._repository.list_memories(limit=limit)
            if record.status != "archived"
        ]
        delete_by_model(source_kind="memory", model_name=self._embedder.model_name)
        self._ensure_memory_embeddings(candidates)
        self._health.rebuild_count += 1
        self.mark_semantic_recovered()
        self._health.integrity_check_error = None
        return {
            "semantic_enabled": True,
            "rebuilt": True,
            "records_indexed": len(candidates),
            "reason": None,
        }

    def ensure_searchable_memory_embeddings(
        self,
        candidates: list[MemoryRecord],
        *,
        semantic_timing_ms: dict[str, float] | None = None,
    ) -> None:
        blocked_reason = self.blocked_embedding_persistence_reason()
        if blocked_reason is not None:
            raise ValueError(blocked_reason)
        stale_check_started = time.perf_counter()
        stale_or_missing = self.stale_or_missing_embedding_candidates(candidates)
        _record_timing_ms(semantic_timing_ms, "semantic_embedding_stale_check", stale_check_started)
        if not stale_or_missing:
            return
        if self._scheduler is not None and self._health.background_repair_wait_seconds > 0:
            refresh_started = time.perf_counter()
            self.queue_memory_embedding_repairs(stale_or_missing)
            self.wait_for_background_repairs(stale_or_missing)
            _record_timing_ms(semantic_timing_ms, "semantic_embedding_refresh", refresh_started)
            return
        refresh_started = time.perf_counter()
        self._ensure_memory_embeddings(stale_or_missing)
        _record_timing_ms(semantic_timing_ms, "semantic_embedding_refresh", refresh_started)

    def stale_or_missing_embedding_candidates(self, candidates: list[MemoryRecord]) -> list[MemoryRecord]:
        assert self._embedder is not None
        assert self._vector_store is not None
        get_memory_updated_at_map = getattr(self._vector_store, "get_memory_updated_at_map", None)
        memory_updated_at_by_id: dict[str, str | None] | None = None
        if callable(get_memory_updated_at_map):
            memory_updated_at_by_id = cast(dict[str, str | None], get_memory_updated_at_map(
                source_kind="memory",
                model_name=self._embedder.model_name,
                source_ids=[candidate.id for candidate in candidates],
            ))
        get_updated_at_map = getattr(self._vector_store, "get_updated_at_map", None)
        updated_at_by_id: dict[str, float] | None = None
        if callable(get_updated_at_map):
            updated_at_by_id = cast(dict[str, float], get_updated_at_map(
                source_kind="memory",
                model_name=self._embedder.model_name,
                source_ids=[candidate.id for candidate in candidates],
            ))
        stale_or_missing: list[MemoryRecord] = []
        for candidate in candidates:
            if memory_updated_at_by_id is not None:
                if memory_updated_at_by_id.get(candidate.id) == candidate.updated_at:
                    continue
                if memory_updated_at_by_id.get(candidate.id) is not None:
                    stale_or_missing.append(candidate)
                    continue
            existing_updated_at = updated_at_by_id.get(candidate.id) if updated_at_by_id is not None else None
            if updated_at_by_id is None:
                existing = self._vector_store.get(
                    source_kind="memory",
                    source_id=candidate.id,
                    model_name=self._embedder.model_name,
                )
                existing_updated_at = None if existing is None else existing.updated_at
            if existing_updated_at is None or _embedding_is_stale(existing_updated_at, candidate.updated_at):
                stale_or_missing.append(candidate)
        return stale_or_missing

    def _ensure_memory_embeddings(self, stale_or_missing: list[MemoryRecord]) -> None:
        assert self._embedder is not None
        assert self._vector_store is not None
        if not stale_or_missing:
            return
        if len(stale_or_missing) > INLINE_EMBEDDING_REPAIR_LIMIT:
            stale_or_missing = _prioritize_embedding_repairs(stale_or_missing)[:INLINE_EMBEDDING_REPAIR_LIMIT]
        embed_and_upsert(
            [_embedding_input(candidate) for candidate in stale_or_missing],
            provider=self._embedder,
            sink=self._vector_store,
            batch_size=len(stale_or_missing),
        )

    def queue_memory_embedding_repairs(self, candidates: Sequence[MemoryRecord]) -> None:
        if self._scheduler is not None and self._embedder is not None:
            self._scheduler.schedule(candidates, self._embedder.model_name)

    def wait_for_background_repairs(self, candidates: Sequence[MemoryRecord]) -> None:
        if self._scheduler is None:
            return
        result = self._scheduler.wait_for(candidates, self.stale_or_missing_embedding_candidates)
        self._health.repair_wait_count += 1
        self._health.last_repair_wait_seconds = result.elapsed_seconds
        self._health.last_repair_candidate_count = len(candidates)
        self._health.last_repair_pending_count = result.pending_count
        if result.pending_count:
            self._health.partial_semantic_search_count += 1
            self._health.last_partial_semantic_at = _utc_now()
        self._refresh_background_repair_health_snapshot()

    async def repair_task(self, task: Any) -> dict[str, Any]:
        if self._repository is None or self._embedder is None or self._vector_store is None:
            return {
                "repaired": 0,
                "claimed_work_item_count": 0,
                "batches_processed": 0,
                "reason": "semantic_search_not_initialized",
            }
        integrity_scan = self._run_integrity_scan(
            scan_limit=max(int(task.data.get("integrity_scan_limit", DEFAULT_EMBEDDING_REPAIR_INTEGRITY_SCAN_LIMIT)), 1),
        )
        if self._repair_queue is None and self._work_items is None:
            return {
                "repaired": 0,
                "claimed_work_item_count": 0,
                "batches_processed": 0,
                "reason": "repair_queue_not_initialized",
                "integrity_scan": integrity_scan,
            }
        batch_size = max(int(task.data.get("batch_size", DEFAULT_BACKGROUND_REPAIR_BATCH_SIZE)), 1)
        max_batches_per_run = max(int(task.data.get("max_batches_per_run", DEFAULT_BACKGROUND_REPAIR_MAX_BATCHES_PER_RUN)), 1)
        repaired = 0
        claimed_work_item_count = 0
        batches_processed = 0
        while batches_processed < max_batches_per_run:
            claim_workspace_id = task.workspace_id if task.workspace_id is not None else "*"
            claimed_items = self._claim_batch(task.id, batch_size, claim_workspace_id)
            if not claimed_items:
                break
            batches_processed += 1
            claimed_work_item_count += len(claimed_items)
            repair_pairs: list[tuple[Any, Any]] = []
            for work_item in claimed_items:
                memory_id, queued_model_name, queued_updated_at = self._queued_item_values(work_item)
                if not isinstance(memory_id, str) or not memory_id.strip():
                    self._complete_item(work_item.id)
                    continue
                if queued_model_name != self._embedder.model_name:
                    self._complete_item(work_item.id)
                    continue
                record = self._repository.get_memory(memory_id)
                if record is None or record.status == "archived":
                    self._complete_item(work_item.id)
                    continue
                if isinstance(queued_updated_at, str) and queued_updated_at != (record.updated_at or ""):
                    self._complete_item(work_item.id)
                    continue
                existing = self._vector_store.get(
                    source_kind="memory", source_id=record.id, model_name=self._embedder.model_name
                )
                if existing is not None and not _embedding_is_stale(existing.updated_at, record.updated_at):
                    self._complete_item(work_item.id)
                    continue
                repair_pairs.append((record, work_item))
            if not repair_pairs:
                continue
            embedding_result = embed_and_upsert(
                [_embedding_input(record) for record, _ in repair_pairs],
                provider=self._embedder,
                sink=self._vector_store,
                batch_size=len(repair_pairs),
            )
            for _, work_item in repair_pairs:
                self._complete_item(work_item.id)
            repaired += embedding_result.stored
        pruned_completed = 0
        if self._repair_queue is not None:
            pruned_completed = self._repair_queue.prune_completed(limit=DEFAULT_EMBEDDING_REPAIR_PRUNE_LIMIT)
        return {
            "repaired": repaired,
            "claimed_work_item_count": claimed_work_item_count,
            "batches_processed": batches_processed,
            "pruned_completed": pruned_completed,
            "integrity_scan": integrity_scan,
        }

    def _claim_batch(self, task_id: str, limit: int, workspace_id: str):
        if self._repair_queue is not None:
            return self._repair_queue.claim_batch(lease_owner=task_id, limit=limit, workspace_id=workspace_id)
        assert self._work_items is not None
        return self._work_items.claim_batch(
            family_key=WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            lease_owner=task_id,
            limit=limit,
            workspace_id=workspace_id,
        )

    def _queued_item_values(self, work_item: Any) -> tuple[Any, Any, Any]:
        if self._repair_queue is not None:
            return work_item.memory_id, work_item.model_name, work_item.memory_updated_at
        payload = work_item.payload
        return payload.get("memory_id"), payload.get("model_name"), payload.get("memory_updated_at")

    def _complete_item(self, item_id: str) -> None:
        if self._repair_queue is not None:
            self._repair_queue.complete_item(item_id)
        else:
            assert self._work_items is not None
            self._work_items.complete_item(item_id)

    def _refresh_background_repair_health_snapshot(self) -> None:
        if not self._health.semantic_enabled or self._scheduler is None:
            self._health.queued_repair_backlog_count = 0
            self._health.running_repair_count = 0
            self._health.oldest_queued_repair_age_seconds = None
            return
        snapshot = self._scheduler.backlog_snapshot()
        self._health.queued_repair_backlog_count = snapshot.queued_count
        self._health.running_repair_count = snapshot.running_count
        self._health.oldest_queued_repair_age_seconds = snapshot.oldest_queued_age_seconds

    def _run_integrity_check_once(self) -> None:
        if self._embedder is None or self._vector_store is None:
            return
        connection = None
        if self._db_manager is not None:
            get_connection = getattr(self._db_manager, "get_connection", None)
            if callable(get_connection):
                connection = get_connection()
        if isinstance(connection, sqlite3.Connection):
            quick_check = connection.execute("PRAGMA quick_check").fetchone()
            if quick_check is not None and str(quick_check[0]).lower() != "ok":
                raise sqlite3.DatabaseError(f"quick_check_failed:{quick_check[0]}")
            query_embedding = [0.0]
        else:
            query_embedding = [0.0] * max(_active_embedder_dimension(self._embedder) or 1, 1)
        self._vector_store.search(
            source_kind="memory",
            model_name=self._embedder.model_name,
            query_embedding=query_embedding,
            limit=1,
        )

    def _retry_after_reopen(self, reason: str, operation):
        close = None if self._db_manager is None else getattr(self._db_manager, "close", None)
        if not callable(close):
            return None
        try:
            close()
            result = operation()
        except (OSError, sqlite3.Error, ValueError) as retry_exc:
            logger.warning("Semantic search recovery after %s failed: %s", reason, retry_exc)
            return None
        self.mark_semantic_recovered()
        logger.info("Semantic search recovered after %s by reopening the database connection", reason)
        return result

    def _run_integrity_scan(self, *, scan_limit: int) -> dict[str, Any]:
        scan_integrity = getattr(self._vector_store, "scan_integrity", None)
        if not callable(scan_integrity) or self._repository is None:
            return {"supported": False, "reason": "integrity_scan_unsupported"}
        assert self._embedder is not None
        assert self._vector_store is not None
        active_dimension = _active_embedder_dimension(self._embedder)
        if active_dimension is None:
            return {"supported": False, "reason": "active_embedder_dimension_unavailable"}
        active_memories = [
            memory
            for memory in self._repository.list_memories(status="active", limit=scan_limit)
            if memory.status != "archived"
        ]
        scan_summary = _normalize_scan_summary(scan_integrity(
            active_model_name=self._embedder.model_name,
            expected_dimension=active_dimension,
        ))
        active_model_rows = [
            row for row in scan_summary.get("active_model_rows", []) if isinstance(row, dict)
        ]
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
            row for row in scan_summary.get("invalid_rows", []) if isinstance(row, dict)
        ]
        invalid_thought_rows = [
            row
            for row in invalid_rows
            if row.get("model_name") == self._embedder.model_name
            and row.get("source_kind") == "thought"
        ]
        counts = {"missing": 0, "invalid": 0, "stale": 0, "deleted": 0, "enqueued": 0, "already": 0}
        for record in active_memories:
            workspace_id = next((item for item in record.workspace_ids if item), None)
            row = active_memory_rows.get((workspace_id, record.id)) or active_memory_rows_by_source_id.get(record.id)
            should_enqueue = False
            if row is None:
                counts["missing"] += 1
                should_enqueue = True
            else:
                issues = row.get("issues", [])
                normalized_issues = [str(issue) for issue in issues] if isinstance(issues, list | tuple) else []
                if normalized_issues:
                    counts["invalid"] += 1
                    should_enqueue = True
                    delete_embedding = getattr(self._vector_store, "delete", None)
                    if callable(delete_embedding):
                        deleted_count = delete_embedding(
                            source_kind="memory",
                            source_id=record.id,
                            model_name=self._embedder.model_name,
                        )
                        if isinstance(deleted_count, int | float | str):
                            counts["deleted"] += int(deleted_count)
                elif isinstance(row.get("updated_at"), (int, float)) and _embedding_is_stale(
                    float(row["updated_at"]), record.updated_at
                ):
                    counts["stale"] += 1
                    should_enqueue = True
            if not should_enqueue:
                continue
            created = _enqueue_memory_embedding_repair(
                work_items=self._work_items,
                embedding_repair_queue=self._repair_queue,
                memory_id=record.id,
                workspace_id=workspace_id,
                model_name=self._embedder.model_name,
                memory_updated_at=record.updated_at or "",
            )
            counts["enqueued" if created else "already"] += 1
        return {
            "supported": True,
            "active_model_name": self._embedder.model_name,
            "active_model_dimension": active_dimension,
            "memory_scan_limit": scan_limit,
            "memory_scanned_count": len(active_memories),
            "embedding_row_scanned_count": _summary_int(scan_summary, "scanned_row_count"),
            "invalid_row_count": _summary_int(scan_summary, "invalid_row_count"),
            "mixed_dimension_group_count": _summary_int(scan_summary, "mixed_dimension_group_count"),
            "mixed_dimension_groups": list(scan_summary.get("mixed_dimension_groups", [])),
            "missing_memory_embedding_count": counts["missing"],
            "invalid_memory_embedding_count": counts["invalid"],
            "stale_memory_embedding_count": counts["stale"],
            "invalid_thought_embedding_count": len(invalid_thought_rows),
            "skipped_thought_rows": len(invalid_thought_rows),
            "deleted_invalid_memory_rows": counts["deleted"],
            "enqueued_memory_repairs": counts["enqueued"],
            "already_queued_memory_repairs": counts["already"],
        }


def _embedding_input(record: MemoryRecord) -> EmbeddingInput:
    return EmbeddingInput(
        source_kind="memory",
        source_id=record.id,
        workspace_id=next((item for item in record.workspace_ids if item), None),
        text=_memory_embedding_text(record),
        source_updated_at=record.updated_at,
    )


def _memory_embedding_text(record: MemoryRecord) -> str:
    return "\n".join(
        part
        for part in [record.title, record.summary or "", record.content, ", ".join(record.tags)]
        if part
    )


def _embedding_is_stale(embedding_updated_at: float, memory_updated_at: str | None) -> bool:
    if not memory_updated_at:
        return False
    try:
        updated_at = datetime.fromisoformat(memory_updated_at)
    except (TypeError, ValueError, OverflowError):
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return embedding_updated_at + 1e-6 < updated_at.timestamp()


def _prioritize_embedding_repairs(candidates: Sequence[MemoryRecord]) -> list[MemoryRecord]:
    return sorted(candidates, key=lambda record: (record.updated_at or "", record.id), reverse=True)


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _record_timing_ms(timing_ms: dict[str, float] | None, key: str, started_at: float) -> None:
    if timing_ms is None:
        return
    timing_ms[key] = round(timing_ms.get(key, 0.0) + (time.perf_counter() - started_at) * 1000.0, 3)

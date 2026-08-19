from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.maintenance import (
    ArchivedMemoryGcResult,
    DanglingLinkReconciliationResult,
    LineageMemoryRecord,
    MaintenanceHousekeepingPort,
    MaintenanceHousekeepingTransaction,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_STALE_PLAN_DAYS,
    DEFAULT_SWEEP_RETENTION_DAYS,
)
from mcp_memory.core.task_handlers.workspace_resolution import resolve_task_workspace_id as _resolve_workspace_id

_LINEAGE_METADATA_WARNING_BYTES = 2_000
_LINEAGE_METADATA_LIST_WARNING_COUNT = 10
_RELATIONSHIP_DENSITY_WARNING_COUNT = 20
_LINEAGE_HOTSPOT_EXAMPLE_LIMIT = 10
_DANGLING_LINK_EXAMPLE_LIMIT = 10
_ARCHIVED_MEMORY_GC_EXAMPLE_LIMIT = 10
_DEFAULT_ARCHIVED_MEMORY_RETENTION_DAYS = 90
_DEFAULT_MEMORY_GC_BATCH_SIZE = 100
_DEFAULT_DANGLING_LINK_BATCH_SIZE = 100
_MEMORY_GC_MODES = frozenset({"report-only", "delete"})


def gc_archived_memories(
    housekeeping: MaintenanceHousekeepingPort,
    *,
    cutoff: datetime,
    batch_size: int = _DEFAULT_MEMORY_GC_BATCH_SIZE,
    mode: str = "report-only",
) -> ArchivedMemoryGcResult:
    """Report or delete one bounded batch of safe-to-remove archived memories."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if mode not in _MEMORY_GC_MODES:
        raise ValueError(f"unsupported memory GC mode: {mode!r}")
    return housekeeping.execute(
        lambda transaction: transaction.gc_archived_memories(
            cutoff=cutoff,
            batch_size=batch_size,
            mode=mode,
        )
    )


def reconcile_dangling_links(
    housekeeping: MaintenanceHousekeepingPort,
    *,
    batch_size: int = _DEFAULT_DANGLING_LINK_BATCH_SIZE,
) -> DanglingLinkReconciliationResult:
    """Delete links whose source or target memory no longer exists."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return housekeeping.execute(
        lambda transaction: transaction.reconcile_dangling_links(batch_size=batch_size)
    )


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}
    housekeeping = _require_housekeeping(ctx)
    cutoff = (datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)).isoformat()
    workspace_id = _resolve_workspace_id(ctx, task)
    updated = housekeeping.execute(
        lambda transaction: transaction.mark_stale_plans(cutoff, workspace_id)
    )
    return {"updated": updated}


def handle_fact_checker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"degraded": 0, "restored": 0}
    housekeeping = _require_housekeeping(ctx)
    workspace_root_override = task.data.get("workspace_root")

    def reconcile_external_links(transaction: MaintenanceHousekeepingTransaction) -> tuple[set[str], set[str]]:
        degraded_ids: set[str] = set()
        healthy_ids: set[str] = set()
        for link in transaction.list_external_links():
            workspace_root = _resolve_workspace_root(ctx, link.workspace_id, workspace_root_override)
            file_path = Path(link.target_id)
            if not file_path.is_absolute() and workspace_root is not None:
                file_path = workspace_root / link.target_id
            if file_path.exists():
                if link.status == "degraded":
                    healthy_ids.add(link.memory_id)
            else:
                degraded_ids.add(link.memory_id)

        restored_ids = healthy_ids - degraded_ids
        if degraded_ids:
            transaction.update_memory_statuses(sorted(degraded_ids), status="degraded")
        if restored_ids:
            transaction.update_memory_statuses(
                sorted(restored_ids),
                status="active",
                current_status="degraded",
            )
        return degraded_ids, restored_ids

    degraded_ids, restored_ids = housekeeping.execute(reconcile_external_links)
    return {"degraded": len(degraded_ids), "restored": len(restored_ids)}


def handle_sweeper_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {
            "deleted_tasks": 0,
            "deleted_journal_entries": 0,
            "gc_metadata_records": 0,
            "dangling_links": _empty_dangling_link_result(),
            "memory_gc": _empty_archived_memory_gc_result(),
            "dangling_links_deleted": 0,
            "memory_gc_scanned": 0,
            "memory_gc_eligible": 0,
            "memory_gc_deleted": 0,
        }

    housekeeping = _require_housekeeping(ctx)
    cutoff = datetime.now(UTC) - timedelta(days=DEFAULT_SWEEP_RETENTION_DAYS)
    cutoff_timestamp = cutoff.timestamp()
    maintenance_config = getattr(ctx.config, "maintenance", None)
    link_batch_size = int(
        task.data.get(
            "dangling_link_gc_batch_size",
            getattr(maintenance_config, "dangling_link_gc_batch_size", _DEFAULT_DANGLING_LINK_BATCH_SIZE),
        )
    )
    memory_gc_batch_size = int(
        task.data.get(
            "memory_gc_batch_size",
            getattr(maintenance_config, "memory_gc_batch_size", _DEFAULT_MEMORY_GC_BATCH_SIZE),
        )
    )
    memory_gc_mode = str(
        task.data.get(
            "memory_gc_mode",
            getattr(maintenance_config, "memory_gc_mode", "report-only"),
        )
    )
    memory_retention_days = int(
        task.data.get(
            "archived_memory_retention_days",
            getattr(
                maintenance_config,
                "archived_memory_retention_days",
                _DEFAULT_ARCHIVED_MEMORY_RETENTION_DAYS,
            ),
        )
    )
    if link_batch_size <= 0 or memory_gc_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if memory_gc_mode not in _MEMORY_GC_MODES:
        raise ValueError(f"unsupported memory GC mode: {memory_gc_mode!r}")
    archived_memory_cutoff = datetime.now(UTC) - timedelta(days=memory_retention_days)

    def sweep(transaction: MaintenanceHousekeepingTransaction) -> dict[str, Any]:
        lineage_hotspots = _scan_lineage_hotspots(transaction.list_lineage_memories())
        dangling_links = transaction.reconcile_dangling_links(batch_size=link_batch_size)
        memory_gc = transaction.gc_archived_memories(
            cutoff=archived_memory_cutoff,
            batch_size=memory_gc_batch_size,
            mode=memory_gc_mode,
        )
        deleted_tasks = transaction.delete_completed_tasks(cutoff_timestamp)
        if ctx.journal is not None:
            deleted_recoverable_entry_ids = ctx.journal.purge_expired_recoverable(now=cutoff_timestamp)
        else:
            deleted_recoverable_entry_ids = transaction.purge_recoverable_journal_entries(cutoff_timestamp)
        if deleted_recoverable_entry_ids:
            _cleanup_deleted_thought_embeddings(ctx, deleted_recoverable_entry_ids)
        deleted_journal_entries = (
            transaction.purge_processed_journal_entries(cutoff_timestamp)
            + len(deleted_recoverable_entry_ids)
        )
        gc_metadata_records = transaction.gc_dead_metadata_keys(_DEAD_METADATA_KEYS)
        return {
            "deleted_tasks": deleted_tasks,
            "deleted_journal_entries": deleted_journal_entries,
            "gc_metadata_records": gc_metadata_records,
            "lineage_hotspots": lineage_hotspots,
            "dangling_links": dangling_links,
            "memory_gc": memory_gc,
            "dangling_links_deleted": dangling_links["deleted"],
            "memory_gc_scanned": memory_gc["scanned"],
            "memory_gc_eligible": memory_gc["eligible"],
            "memory_gc_deleted": memory_gc["deleted"],
        }

    return housekeeping.execute(sweep)


def _require_housekeeping(ctx: ApplicationContext) -> MaintenanceHousekeepingPort:
    if ctx.housekeeping is None:
        raise RuntimeError("maintenance_housekeeping_not_initialized")
    return ctx.housekeeping


def _empty_dangling_link_result() -> DanglingLinkReconciliationResult:
    return {"scanned": 0, "deleted": 0, "deleted_link_examples": []}


def _empty_archived_memory_gc_result() -> ArchivedMemoryGcResult:
    return {
        "mode": "report-only",
        "cutoff": "",
        "scanned": 0,
        "eligible": 0,
        "skipped_protected": 0,
        "skipped_linked": 0,
        "deleted": 0,
        "eligible_memory_examples": [],
        "deleted_memory_examples": [],
    }


_DEAD_METADATA_KEYS = (
    "ingest_task_id",
    "appended_via_ingest",
    "created_via_ingest",
    "defragmenter_task_id",
    "deduplicator_task_id",
    "split_group_id",
    "split_child_memory_ids",
    "split_from_memory_title",
    "split_sibling_memory_ids",
    "split_part_index",
    "split_part_count",
    "split_child_count",
)


def _scan_lineage_hotspots(rows: Sequence[LineageMemoryRecord]) -> dict[str, Any]:
    active_split_original_count = 0
    oversized_metadata_count = 0
    high_relationship_density_count = 0
    examples: list[dict[str, Any]] = []

    for row in rows:
        metadata = _decode_metadata(row.metadata)
        metadata_bytes = len(json.dumps(metadata, sort_keys=True)) if metadata else 0
        reasons: list[str] = []
        if row.status == "active" and isinstance(metadata.get("split_child_memory_ids"), list):
            active_split_original_count += 1
            reasons.append("active_split_original")
        if _metadata_exceeds_lineage_budget(metadata, metadata_bytes=metadata_bytes):
            oversized_metadata_count += 1
            reasons.append("oversized_lineage_metadata")
        if row.relationship_count >= _RELATIONSHIP_DENSITY_WARNING_COUNT:
            high_relationship_density_count += 1
            reasons.append("high_relationship_density")
        if reasons and len(examples) < _LINEAGE_HOTSPOT_EXAMPLE_LIMIT:
            examples.append(
                {
                    "memory_id": row.memory_id,
                    "reasons": reasons,
                    "metadata_bytes": metadata_bytes,
                    "relationship_count": row.relationship_count,
                }
            )
    return {
        "active_split_original_records": active_split_original_count,
        "oversized_lineage_metadata_records": oversized_metadata_count,
        "high_relationship_density_records": high_relationship_density_count,
        "examples": examples,
    }


def _metadata_exceeds_lineage_budget(metadata: dict[str, object], *, metadata_bytes: int) -> bool:
    if metadata_bytes >= _LINEAGE_METADATA_WARNING_BYTES:
        return True
    for key in ("split_child_memory_ids", "split_sibling_memory_ids", "merged_source_ids"):
        value = metadata.get(key)
        if isinstance(value, list) and len(value) >= _LINEAGE_METADATA_LIST_WARNING_COUNT:
            return True
    return False


def _decode_metadata(raw_metadata: object) -> dict[str, object]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if raw_metadata is None:
        return {}
    try:
        decoded = json.loads(str(raw_metadata) or "{}")
    except (TypeError, ValueError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return dict(decoded)


def _cleanup_deleted_thought_embeddings(ctx: ApplicationContext, deleted_ids: list[int]) -> None:
    vector_store = getattr(ctx, "vector_store", None)
    embedder = getattr(ctx, "embedder", None)
    if vector_store is None:
        return
    model_name = None if embedder is None else embedder.model_name
    for entry_id in deleted_ids:
        vector_store.delete(
            source_kind="thought",
            source_id=str(entry_id),
            model_name=model_name,
        )


def _resolve_workspace_root(
    ctx: ApplicationContext,
    workspace_id: str | None,
    workspace_root_override: object | None = None,
) -> Path | None:
    if isinstance(workspace_root_override, str) and workspace_root_override.strip():
        override_path = Path(workspace_root_override).expanduser()
        if override_path.exists():
            return override_path
    if workspace_id is None:
        return None
    if ctx.workspace_root is not None and ctx.workspace_id == workspace_id:
        return ctx.workspace_root
    path = Path(workspace_id).expanduser()
    if path.exists():
        return path
    return None

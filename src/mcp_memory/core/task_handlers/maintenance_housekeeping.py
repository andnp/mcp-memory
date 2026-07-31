from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sqlite3
from typing import Any, ContextManager, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_STALE_PLAN_DAYS,
    DEFAULT_SWEEP_RETENTION_DAYS,
)
from mcp_memory.core.task_handlers.workspace_resolution import resolve_task_workspace_id as _resolve_workspace_id
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.storage.session import CursorLike, DbConnectionLike


BackendConnection = sqlite3.Connection | DbConnectionLike
_LINEAGE_METADATA_WARNING_BYTES = 2_000
_LINEAGE_METADATA_LIST_WARNING_COUNT = 10
_RELATIONSHIP_DENSITY_WARNING_COUNT = 20
_LINEAGE_HOTSPOT_EXAMPLE_LIMIT = 10


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}

    cutoff = (datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)).isoformat()
    workspace_id = _resolve_workspace_id(ctx, task)
    with _open_backend_connection(ctx) as connection:
        sqlite_mode = _uses_sqlite_backend(ctx)
        if workspace_id is None:
            updated = _execute_write(
                connection,
                sqlite_mode,
                """
                UPDATE memories
                SET status = 'stale'
                WHERE type = 'plan' AND status = 'active' AND updated_at < {}
                """,
                (cutoff,),
            )
        else:
            updated = _execute_write(
                connection,
                sqlite_mode,
                """
                UPDATE memories
                SET status = 'stale'
                WHERE id IN (
                    SELECT memories.id
                    FROM memories
                    JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                    WHERE memories.type = 'plan'
                      AND memories.status = 'active'
                      AND memories.updated_at < {}
                      AND memory_workspaces.workspace_id = {}
                )
                """,
                (cutoff, workspace_id),
            )
        connection.commit()
    return {"updated": updated}


def handle_fact_checker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"degraded": 0, "restored": 0}

    degraded_ids: set[str] = set()
    healthy_ids: set[str] = set()
    workspace_root_override = task.data.get("workspace_root")
    with _open_backend_connection(ctx) as connection:
        sqlite_mode = _uses_sqlite_backend(ctx)
        rows = _fetchall_rows(
            connection,
            sqlite_mode,
            """
            SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
            FROM memories
            JOIN links ON links.source_id = memories.id
            LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
            WHERE links.target_id LIKE 'ext:%'
            """,
        )

        for row in rows:
            current_status = None if row[1] is None else str(row[1])
            workspace_root = _resolve_workspace_root(
                ctx,
                None if row[3] is None else str(row[3]),
                workspace_root_override,
            )
            target_path = str(row[2])[4:]
            file_path = Path(target_path)
            if not file_path.is_absolute() and workspace_root is not None:
                file_path = workspace_root / target_path

            if file_path.exists():
                if current_status == "degraded":
                    healthy_ids.add(str(row[0]))
            else:
                degraded_ids.add(str(row[0]))

        restored_ids = healthy_ids - degraded_ids
        if degraded_ids:
            _execute_memory_status_update(
                connection,
                sqlite_mode,
                memory_ids=degraded_ids,
                status="degraded",
            )
        if restored_ids:
            _execute_memory_status_update(
                connection,
                sqlite_mode,
                memory_ids=restored_ids,
                status="active",
                current_status="degraded",
            )
        connection.commit()

    return {"degraded": len(degraded_ids), "restored": len(restored_ids)}


def handle_sweeper_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"deleted_tasks": 0, "deleted_journal_entries": 0, "gc_metadata_records": 0}

    cutoff = datetime.now(UTC) - timedelta(days=DEFAULT_SWEEP_RETENTION_DAYS)
    cutoff_timestamp = cutoff.timestamp()
    with _open_backend_connection(ctx) as connection:
        sqlite_mode = _uses_sqlite_backend(ctx)
        lineage_hotspots = _scan_lineage_hotspots(connection, sqlite_mode)
        deleted_tasks = _execute_write(
            connection,
            sqlite_mode,
            "DELETE FROM tasks WHERE status = 'completed' AND updated_at < {}",
            (cutoff_timestamp,),
        )
        deleted_recoverable_entry_ids = _purge_recoverable_journal_entries(
            ctx,
            cutoff_timestamp,
            connection=connection,
            sqlite_mode=sqlite_mode,
        )
        if deleted_recoverable_entry_ids:
            _cleanup_deleted_thought_embeddings(ctx, deleted_recoverable_entry_ids)
        deleted_journal_entries = _execute_write(
            connection,
            sqlite_mode,
            "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < {}",
            (cutoff_timestamp,),
        ) + len(deleted_recoverable_entry_ids)
        gc_metadata_records = _gc_dead_metadata_keys(connection, sqlite_mode)
        connection.commit()
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_journal_entries": deleted_journal_entries,
        "gc_metadata_records": gc_metadata_records,
        "lineage_hotspots": lineage_hotspots,
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


def _gc_dead_metadata_keys(
    connection: BackendConnection,
    sqlite_mode: bool,
) -> int:
    """Strip dead internal maintenance keys from memory metadata."""
    if sqlite_mode:
        return _gc_dead_metadata_keys_sqlite(connection)
    removal_expr = " - ".join(
        ["metadata"] + [f"'{key}'" for key in _DEAD_METADATA_KEYS]
    )
    key_array = ", ".join(f"'{key}'" for key in _DEAD_METADATA_KEYS)
    query = f"""
        UPDATE memories
        SET metadata = {removal_expr}
        WHERE metadata IS NOT NULL
          AND metadata != '{{}}'::jsonb
          AND metadata ?| ARRAY[{key_array}]
    """
    cursor = cast(CursorLike, connection.cursor())
    try:
        cursor.execute(query)
        return int(getattr(cursor, "rowcount", 0) or 0)
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _gc_dead_metadata_keys_sqlite(connection: BackendConnection) -> int:
    if not isinstance(connection, sqlite3.Connection):
        return 0
    rows = connection.execute(
        """
        SELECT id, metadata
        FROM memories
        WHERE metadata IS NOT NULL AND metadata != '{}'
        """
    ).fetchall()
    updated = 0
    for memory_id, raw_metadata in rows:
        metadata = _decode_metadata(raw_metadata)
        if not metadata:
            continue
        pruned = {key: value for key, value in metadata.items() if key not in _DEAD_METADATA_KEYS}
        if pruned == metadata:
            continue
        connection.execute(
            "UPDATE memories SET metadata = ? WHERE id = ?",
            (json.dumps(pruned, sort_keys=True), memory_id),
        )
        updated += 1
    return updated


def _scan_lineage_hotspots(
    connection: BackendConnection,
    sqlite_mode: bool,
) -> dict[str, Any]:
    metadata_rows = _fetchall_rows(
        connection,
        sqlite_mode,
        """
        SELECT id, status, metadata
        FROM memories
        WHERE status != 'archived'
        """,
    )
    relationship_rows = _fetchall_rows(
        connection,
        sqlite_mode,
        """
        SELECT memories.id,
               COALESCE(outgoing.link_count, 0) AS outgoing_count,
               COALESCE(incoming.link_count, 0) AS incoming_count
        FROM memories
        LEFT JOIN (
            SELECT source_id AS memory_id, COUNT(*) AS link_count
            FROM links
            GROUP BY source_id
        ) outgoing ON outgoing.memory_id = memories.id
        LEFT JOIN (
            SELECT target_id AS memory_id, COUNT(*) AS link_count
            FROM links
            GROUP BY target_id
        ) incoming ON incoming.memory_id = memories.id
        WHERE memories.status != 'archived'
        """,
    )
    relationship_counts = {
        str(memory_id): _coerce_int(outgoing) + _coerce_int(incoming)
        for memory_id, outgoing, incoming in relationship_rows
    }

    active_split_original_count = 0
    oversized_metadata_count = 0
    high_relationship_density_count = 0
    examples: list[dict[str, Any]] = []

    for memory_id_raw, status_raw, raw_metadata in metadata_rows:
        memory_id = str(memory_id_raw)
        metadata = _decode_metadata(raw_metadata)
        metadata_bytes = len(json.dumps(metadata, sort_keys=True)) if metadata else 0
        relationship_count = relationship_counts.get(memory_id, 0)
        reasons: list[str] = []

        if status_raw == "active" and isinstance(metadata.get("split_child_memory_ids"), list):
            active_split_original_count += 1
            reasons.append("active_split_original")

        if _metadata_exceeds_lineage_budget(metadata, metadata_bytes=metadata_bytes):
            oversized_metadata_count += 1
            reasons.append("oversized_lineage_metadata")

        if relationship_count >= _RELATIONSHIP_DENSITY_WARNING_COUNT:
            high_relationship_density_count += 1
            reasons.append("high_relationship_density")

        if reasons and len(examples) < _LINEAGE_HOTSPOT_EXAMPLE_LIMIT:
            examples.append(
                {
                    "memory_id": memory_id,
                    "reasons": reasons,
                    "metadata_bytes": metadata_bytes,
                    "relationship_count": relationship_count,
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


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


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


def _purge_recoverable_journal_entries(
    ctx: ApplicationContext,
    cutoff_timestamp: float,
    *,
    connection: BackendConnection | None = None,
    sqlite_mode: bool | None = None,
) -> list[int]:
    if ctx.journal is not None:
        return ctx.journal.purge_expired_recoverable(now=cutoff_timestamp)

    if connection is None or sqlite_mode is None:
        return []

    rows = _fetchall_rows(
        connection,
        sqlite_mode,
        """
        SELECT id
        FROM system1_journal
        WHERE status = 'recoverable'
          AND recoverable_until IS NOT NULL
          AND recoverable_until <= {}
        ORDER BY timestamp ASC
        """,
        (cutoff_timestamp,),
    )
    entry_ids = [int(cast(str | int, row[0])) for row in rows]
    if not entry_ids:
        return []

    placeholders = _placeholder_list(len(entry_ids), sqlite_mode=sqlite_mode)
    _execute_write(
        connection,
        sqlite_mode,
        f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'recoverable'",
        tuple(entry_ids),
    )
    return entry_ids


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


def _uses_sqlite_backend(ctx: ApplicationContext) -> bool:
    db_manager = ctx.db_manager
    return db_manager is not None and hasattr(db_manager, "get_connection")


def _open_backend_connection(ctx: ApplicationContext) -> ContextManager[BackendConnection]:
    if ctx.db_manager is None:
        raise RuntimeError("db_manager_not_initialized")
    if _uses_sqlite_backend(ctx):
        return nullcontext(cast(BackendConnection, ctx.db_manager.get_connection()))
    return cast(ContextManager[BackendConnection], ctx.db_manager.open_connection())


def _format_query(query_template: str, *, sqlite_mode: bool) -> str:
    placeholder = "?" if sqlite_mode else "%s"
    formatted = query_template.format(*([placeholder] * query_template.count("{}")))
    if sqlite_mode:
        return formatted
    return formatted.replace("%", "%%").replace("%%s", "%s")


def _placeholder_list(count: int, *, sqlite_mode: bool) -> str:
    placeholder = "?" if sqlite_mode else "%s"
    return ",".join([placeholder] * count)


def _execute_write(
    connection: BackendConnection,
    sqlite_mode: bool,
    query_template: str,
    params: tuple[object, ...] = (),
) -> int:
    query = _format_query(query_template, sqlite_mode=sqlite_mode)
    if isinstance(connection, sqlite3.Connection):
        cursor = connection.execute(query, params)
        return int(getattr(cursor, "rowcount", 0) or 0)
    cursor = cast(CursorLike, connection.cursor())
    try:
        cursor.execute(query, params)
        return int(getattr(cursor, "rowcount", 0) or 0)
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _fetchall_rows(
    connection: BackendConnection,
    sqlite_mode: bool,
    query_template: str,
    params: tuple[object, ...] = (),
) -> list[tuple[object, ...]]:
    query = _format_query(query_template, sqlite_mode=sqlite_mode)
    if isinstance(connection, sqlite3.Connection):
        rows = connection.execute(query, params).fetchall()
    else:
        cursor = cast(CursorLike, connection.cursor())
        try:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
    return [tuple(row) for row in rows]


def _execute_memory_status_update(
    connection: BackendConnection,
    sqlite_mode: bool,
    *,
    memory_ids: set[str],
    status: str,
    current_status: str | None = None,
) -> None:
    if not memory_ids:
        return
    placeholders = _placeholder_list(len(memory_ids), sqlite_mode=sqlite_mode)
    params: list[object] = [status, *sorted(memory_ids)]
    query = f"UPDATE memories SET status = {{}} WHERE id IN ({placeholders})"
    if current_status is not None:
        query += " AND status = {}"
        params.append(current_status)
    _execute_write(connection, sqlite_mode, query, tuple(params))

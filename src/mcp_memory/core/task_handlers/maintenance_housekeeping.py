from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_STALE_PLAN_DAYS,
    DEFAULT_SWEEP_RETENTION_DAYS,
)
from mcp_memory.core.tasks import TaskRecord


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}

    cutoff = (datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)).isoformat()
    workspace_id = _resolve_workspace_id(ctx, task)
    conn = ctx.db_manager.get_connection()
    if workspace_id is None:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE type = 'plan' AND status = 'active' AND updated_at < ?
            """,
            (cutoff,),
        )
    else:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE id IN (
                SELECT memories.id
                FROM memories
                JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                WHERE memories.type = 'plan'
                  AND memories.status = 'active'
                  AND memories.updated_at < ?
                  AND memory_workspaces.workspace_id = ?
            )
            """,
            (cutoff, workspace_id),
        )
    conn.commit()
    return {"updated": cursor.rowcount}


def handle_fact_checker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"degraded": 0, "restored": 0}

    conn = ctx.db_manager.get_connection()
    rows = conn.execute(
        """
        SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
        FROM memories
        JOIN links ON links.source_id = memories.id
        LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
        WHERE links.target_id LIKE 'ext:%'
        """
    ).fetchall()

    degraded_ids: set[str] = set()
    healthy_ids: set[str] = set()
    workspace_root_override = task.data.get("workspace_root")
    for row in rows:
        workspace_root = _resolve_workspace_root(
            ctx,
            row["workspace_id"],
            workspace_root_override,
        )
        target_path = row["target_id"][4:]
        file_path = Path(target_path)
        if not file_path.is_absolute() and workspace_root is not None:
            file_path = workspace_root / target_path

        if file_path.exists():
            healthy_ids.add(row["id"])
        else:
            degraded_ids.add(row["id"])

    restored_ids = healthy_ids - degraded_ids
    if degraded_ids:
        placeholders = ",".join("?" for _ in degraded_ids)
        conn.execute(
            f"UPDATE memories SET status = 'degraded' WHERE id IN ({placeholders})",
            list(degraded_ids),
        )
    if restored_ids:
        placeholders = ",".join("?" for _ in restored_ids)
        conn.execute(
            f"UPDATE memories SET status = 'active' WHERE id IN ({placeholders}) AND status = 'degraded'",
            list(restored_ids),
        )
    conn.commit()
    return {"degraded": len(degraded_ids), "restored": len(restored_ids)}


def handle_sweeper_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"deleted_tasks": 0, "deleted_journal_entries": 0}

    cutoff = datetime.now(UTC) - timedelta(days=DEFAULT_SWEEP_RETENTION_DAYS)
    cutoff_timestamp = cutoff.timestamp()
    conn = ctx.db_manager.get_connection()
    deleted_tasks = conn.execute(
        "DELETE FROM tasks WHERE status = 'completed' AND updated_at < ?",
        (cutoff_timestamp,),
    ).rowcount
    deleted_recoverable_entry_ids = _purge_recoverable_journal_entries(ctx, cutoff_timestamp)
    if deleted_recoverable_entry_ids:
        _cleanup_deleted_thought_embeddings(ctx, deleted_recoverable_entry_ids)
    deleted_journal_entries = conn.execute(
        "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < ?",
        (cutoff_timestamp,),
    ).rowcount + len(deleted_recoverable_entry_ids)
    conn.commit()
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_journal_entries": deleted_journal_entries,
    }


def _purge_recoverable_journal_entries(ctx: ApplicationContext, cutoff_timestamp: float) -> list[int]:
    if ctx.journal is not None:
        return ctx.journal.purge_expired_recoverable(now=cutoff_timestamp)

    if ctx.db_manager is None:
        return []

    conn = ctx.db_manager.get_connection()
    rows = conn.execute(
        """
        SELECT id
        FROM system1_journal
        WHERE status = 'recoverable'
          AND recoverable_until IS NOT NULL
          AND recoverable_until <= ?
        ORDER BY timestamp ASC
        """,
        (cutoff_timestamp,),
    ).fetchall()
    entry_ids = [int(row[0]) for row in rows]
    if not entry_ids:
        return []

    placeholders = ",".join("?" for _ in entry_ids)
    conn.execute(
        f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'recoverable'",
        entry_ids,
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


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    return None


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

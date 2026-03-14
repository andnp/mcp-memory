from __future__ import annotations

import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from pathlib import Path
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.consolidation import _group_related_entries
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import TaskRecord


SYSTEM1_INGEST_TASK_NAME = "ingest-system1"
SUMMARIZE_MEMORY_TASK_NAME = "summarize-memory"
PROJECT_MANAGER_TASK_NAME = "project-manager"
FACT_CHECKER_TASK_NAME = "fact-checker"
SWEEPER_TASK_NAME = "sweeper"
SYSTEM1_INGEST_THRESHOLD = 3
DEFAULT_INGEST_BATCH_SIZE = 20
DEFAULT_SWEEP_RETENTION_DAYS = 7
DEFAULT_STALE_PLAN_DAYS = 60


def build_runtime_task_worker(
    ctx: ApplicationContext,
    provider: Any = None,
) -> RuntimeTaskWorker:
    return RuntimeTaskWorker(
        ctx,
        handlers=build_default_task_handlers(provider),
        poll_interval_seconds=0.05,
    )


def build_default_task_handlers(
    provider: Any = None,
) -> dict[str, Callable[[ApplicationContext, TaskRecord], Any]]:
    return {
        SYSTEM1_INGEST_TASK_NAME: lambda ctx, task: handle_ingest_system1_task(ctx, task, provider),
        SUMMARIZE_MEMORY_TASK_NAME: lambda ctx, task: handle_summarize_memory_task(ctx, task, provider),
        PROJECT_MANAGER_TASK_NAME: handle_project_manager_task,
        FACT_CHECKER_TASK_NAME: handle_fact_checker_task,
        SWEEPER_TASK_NAME: handle_sweeper_task,
    }


def bootstrap_background_tasks(ctx: ApplicationContext) -> None:
    task_queue = getattr(ctx, "task_queue", None)
    if task_queue is None:
        return

    workspace_id = getattr(ctx, "project_name", None)
    for task_name in (
        PROJECT_MANAGER_TASK_NAME,
        FACT_CHECKER_TASK_NAME,
        SWEEPER_TASK_NAME,
    ):
        task_queue.enqueue_unique(task_name=task_name, workspace_id=workspace_id)


async def handle_ingest_system1_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.journal is None or ctx.repository is None:
        return {"created_memory_ids": [], "processed_entry_ids": []}

    entries = ctx.journal.get_pending(
        limit=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE))
    )
    if not entries:
        return {"created_memory_ids": [], "processed_entry_ids": []}

    workspace_id = _resolve_workspace_id(ctx, task)
    created_ids: list[str] = []
    processed_ids: list[int] = []

    actions = None
    if provider is not None:
        try:
            actions = await _analyze_ingest_actions(provider, entries)
        except Exception:
            actions = None

    if actions:
        created_ids, processed_ids = _execute_ingest_actions(
            ctx,
            task,
            workspace_id,
            entries,
            actions,
        )

    if not processed_ids:
        created_ids, processed_ids = _fallback_ingest_entries(
            ctx,
            task,
            workspace_id,
            entries,
        )

    if processed_ids:
        ctx.journal.mark_processed(processed_ids)

    return {
        "created_memory_ids": created_ids,
        "processed_entry_ids": processed_ids,
    }


async def handle_summarize_memory_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"memory_id": None, "summary": None}

    memory_id = str(task.data.get("memory_id", "")).strip()
    if not memory_id:
        return {"memory_id": None, "summary": None}

    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"memory_id": memory_id, "summary": None}

    summary = None
    if provider is not None:
        try:
            summary_response = await provider.ask(
                "Write a concise 2-sentence summary as JSON: "
                '{"summary": "..."}\n\n'
                f"Title: {record.title}\nContent: {record.content}"
            )
            maybe_summary = summary_response.get("summary")
            if isinstance(maybe_summary, str) and maybe_summary.strip():
                summary = maybe_summary.strip()
        except Exception:
            summary = None

    if summary is None:
        summary = _build_summary(record.content)

    updated = ctx.repository.update_memory(memory_id, summary=summary)
    return {
        "memory_id": memory_id,
        "summary": updated.summary if updated is not None else summary,
    }


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}

    cutoff = (
        datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)
    ).isoformat()
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
    deleted_journal_entries = conn.execute(
        "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < ?",
        (cutoff_timestamp,),
    ).rowcount
    conn.commit()
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_journal_entries": deleted_journal_entries,
    }


async def _analyze_ingest_actions(provider: Any, entries) -> list[dict[str, Any]]:
    entry_text = "\n".join(f"[{index}] {entry.content}" for index, entry in enumerate(entries))
    prompt = (
        "Analyze these system1 journal entries and return JSON with actions.\n"
        'Allowed actions: {"type": "create"|"ignore", "entry_indices": [...], '
        '"title": "...", "content": "..."}.\n\n'
        f"Entries:\n{entry_text}"
    )
    response = provider.ask(prompt)
    if isawaitable(response):
        response = await response
    actions = response.get("actions", [])
    if not isinstance(actions, list):
        raise ValueError("provider returned invalid actions")
    return actions


def _execute_ingest_actions(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
    actions: list[dict[str, Any]],
) -> tuple[list[str], list[int]]:
    assert ctx.repository is not None
    created_ids: list[str] = []
    processed_ids: list[int] = []
    for action in actions:
        entry_indices = action.get("entry_indices", [])
        selected_entries = [
            entries[index]
            for index in entry_indices
            if isinstance(index, int) and 0 <= index < len(entries)
        ]
        if not selected_entries:
            continue

        action_type = str(action.get("type", "")).strip()
        if action_type == "ignore":
            processed_ids.extend(entry.id for entry in selected_entries)
            continue

        if action_type != "create":
            continue

        content = str(action.get("content", "")).strip() or _format_entries(selected_entries)
        title = str(action.get("title", "")).strip() or _build_title(selected_entries)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            workspace_ids=[workspace_id],
            tags=["auto-ingested", "system1"],
            memory_type="observation",
            metadata={
                "source_entry_ids": [entry.id for entry in selected_entries],
                "ingest_task_id": task.id,
            },
        )
        created_ids.append(record.id)
        processed_ids.extend(entry.id for entry in selected_entries)
        _enqueue_summary_task(ctx, workspace_id, record.id)

    return created_ids, processed_ids


def _fallback_ingest_entries(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
) -> tuple[list[str], list[int]]:
    assert ctx.repository is not None
    created_ids: list[str] = []
    processed_ids: list[int] = []
    for group in _group_related_entries(entries):
        record = ctx.repository.create_memory(
            title=_build_title(group),
            content=_format_entries(group),
            workspace_ids=[workspace_id],
            tags=["auto-ingested", "system1"],
            memory_type="observation",
            metadata={
                "source_entry_ids": [entry.id for entry in group],
                "ingest_task_id": task.id,
            },
        )
        created_ids.append(record.id)
        processed_ids.extend(entry.id for entry in group)
        _enqueue_summary_task(ctx, workspace_id, record.id)
    return created_ids, processed_ids


def _enqueue_summary_task(
    ctx: ApplicationContext,
    workspace_id: str,
    memory_id: str,
) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.enqueue(
            task_name=SUMMARIZE_MEMORY_TASK_NAME,
            task_id=f"{SUMMARIZE_MEMORY_TASK_NAME}:{memory_id}",
            workspace_id=workspace_id,
            data={"memory_id": memory_id},
        )
    except sqlite3.IntegrityError:
        return


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    if ctx.project_name:
        return ctx.project_name
    return "global"


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
    if ctx.config is not None:
        for project in ctx.config.projects:
            if project.name == workspace_id:
                return Path(project.path)
    path = Path(workspace_id).expanduser()
    if path.exists():
        return path
    return None


def _build_title(entries) -> str:
    if not entries:
        return "System 1 Note"
    words = entries[0].content.strip().split()
    return " ".join(words[:6]).strip().title() or "System 1 Note"


def _format_entries(entries) -> str:
    lines: list[str] = []
    for entry in entries:
        timestamp = datetime.fromtimestamp(entry.timestamp, tz=UTC)
        lines.append(f"- [{timestamp.strftime('%Y-%m-%d %H:%M')}] {entry.content}")
    return "\n".join(lines)


def _build_summary(content: str) -> str:
    stripped = content.strip()
    if not stripped:
        return ""
    sentences = [segment.strip() for segment in stripped.split(".") if segment.strip()]
    if len(sentences) >= 2:
        return ". ".join(sentences[:2]) + "."
    if len(stripped) <= 220:
        return stripped
    return stripped[:217].rstrip() + "..."
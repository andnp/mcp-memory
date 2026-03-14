from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from inspect import isawaitable
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_INGEST_BATCH_SIZE,
    SUMMARIZE_MEMORY_TASK_NAME,
)
from mcp_memory.core.tasks import TaskRecord


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


def _group_related_entries(entries, threshold: float = 0.3):
    if not entries:
        return []

    word_sets = [set(entry.content.lower().split()) for entry in entries]
    used: set[int] = set()
    groups = []

    for index, entry in enumerate(entries):
        if index in used:
            continue
        group = [entry]
        used.add(index)

        for candidate_index in range(index + 1, len(entries)):
            if candidate_index in used:
                continue
            intersection = len(word_sets[index] & word_sets[candidate_index])
            union = len(word_sets[index] | word_sets[candidate_index])
            if union > 0 and intersection / union >= threshold:
                group.append(entries[candidate_index])
                used.add(candidate_index)

        groups.append(group)

    return groups


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
        assert record is not None
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
        assert record is not None
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
    if ctx.workspace_id:
        return ctx.workspace_id
    return "workspace-unknown"


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
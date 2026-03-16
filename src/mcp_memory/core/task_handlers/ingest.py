from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
import re
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.embeddings import cosine_similarity
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_INGEST_BATCH_SIZE,
    SUMMARIZE_MEMORY_TASK_NAME,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord


SEMANTIC_CLUSTER_SIZE = 5
SEMANTIC_SIMILARITY_THRESHOLD = 0.3
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")


async def handle_ingest_system1_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.journal is None or ctx.repository is None:
        return _build_ingest_result(created_ids=[], claimed_ids=[], deleted_ids=[], released_ids=[], meaningful_actions=0)

    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        task.data.get("journal_workspace_id", task.workspace_id),
    )
    workspace_id = _resolve_workspace_id(ctx, task)

    entries = ctx.journal.claim_pending(
        task_id=task.id,
        limit=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
        workspace_id=journal_workspace_id,
    )
    if not entries:
        return _build_ingest_result(created_ids=[], claimed_ids=[], deleted_ids=[], released_ids=[], meaningful_actions=0)

    claimed_ids = [entry.id for entry in entries]
    created_ids: list[str] = []
    meaningful_actions = 0
    grouped_entries = _build_ingest_groups(ctx, entries, workspace_id)

    try:
        for group in grouped_entries:
            actions = None
            if provider is not None:
                try:
                    actions = await _analyze_ingest_actions(ctx, provider, workspace_id, group)
                except Exception:
                    actions = None

            group_created_ids: list[str] = []
            group_handled_ids: list[int] = []
            group_meaningful_actions = 0
            if actions:
                group_created_ids, group_handled_ids, group_meaningful_actions = _execute_ingest_actions(
                    ctx,
                    task,
                    workspace_id,
                    group,
                    actions,
                )

            if not group_handled_ids:
                group_created_ids, group_handled_ids, group_meaningful_actions = _fallback_ingest_entries(
                    ctx,
                    task,
                    workspace_id,
                    group,
                )

            created_ids.extend(group_created_ids)
            meaningful_actions += group_meaningful_actions

        if meaningful_actions <= 0:
            released_ids = ctx.journal.release_claims(task.id)
            return _build_ingest_result(
                created_ids=created_ids,
                claimed_ids=claimed_ids,
                deleted_ids=[],
                released_ids=released_ids,
                meaningful_actions=meaningful_actions,
            )

        deleted_ids = ctx.journal.delete_claims(task.id)
        if deleted_ids:
            _cleanup_deleted_thought_embeddings(ctx, deleted_ids)

        return _build_ingest_result(
            created_ids=created_ids,
            claimed_ids=claimed_ids,
            deleted_ids=deleted_ids,
            released_ids=[],
            meaningful_actions=meaningful_actions,
        )
    except BaseException:
        ctx.journal.release_claims(task.id)
        raise


async def _analyze_ingest_actions(
    ctx: ApplicationContext,
    provider: Any,
    workspace_id: str,
    entries,
) -> list[dict[str, Any]]:
    entry_text = "\n".join(f"[{index}] {entry.content}" for index, entry in enumerate(entries))
    prompt = (
        "Analyze these system1 journal entries and return JSON with actions.\n"
        'Allowed actions: {"type": "create"|"ignore"|"append", "entry_indices": [...], '
        '"target_memory_id": "...", "title": "...", "content": "..."}.\n'
        f"Active workspace_id: {workspace_id}. "
        "If a thought clearly belongs in an existing canonical memory, prefer append and identify the target_memory_id. "
        "Use internal maintenance tools to search and read existing memories before choosing a target whenever append might apply.\n\n"
        f"Entries:\n{entry_text}"
    )
    response = await run_internal_tool_loop(
        ctx,
        provider,
        prompt=prompt,
        allowed_tool_names=[
            "internal_search_memory_records",
            "internal_read_memory_record",
            "internal_list_memory_records",
        ],
    )
    actions = response.response.get("actions", [])
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


def _build_ingest_groups(
    ctx: ApplicationContext,
    entries,
    workspace_id: str,
):
    embedder = getattr(ctx, "embedder", None)
    vector_store = getattr(ctx, "vector_store", None)
    if embedder is None or vector_store is None:
        return _group_related_entries(entries)

    entry_map = {entry.id: entry for entry in entries}
    embeddings = embedder.embed([entry.content for entry in entries])
    by_id = {}
    for entry, embedding in zip(entries, embeddings, strict=False):
        by_id[entry.id] = embedding
        vector_store.upsert(
            source_kind="thought",
            source_id=str(entry.id),
            workspace_id=workspace_id,
            model_name=embedder.model_name,
            embedding=embedding,
        )

    pending_ids = [entry.id for entry in entries]
    remaining = set(pending_ids)
    groups = []
    for entry in entries:
        if entry.id not in remaining:
            continue
        seed_embedding = by_id[entry.id]
        scored = [
            (
                candidate_id,
                _entry_similarity(
                    entry_map[entry.id].content,
                    entry_map[candidate_id].content,
                    cosine_similarity(seed_embedding, by_id[candidate_id]),
                ),
            )
            for candidate_id in pending_ids
            if candidate_id in remaining
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        group_ids = [
            candidate_id
            for candidate_id, score in scored
            if candidate_id == entry.id or score >= SEMANTIC_SIMILARITY_THRESHOLD
        ][:SEMANTIC_CLUSTER_SIZE]
        for group_id in group_ids:
            remaining.discard(group_id)
        groups.append([entry_map[group_id] for group_id in group_ids])

    return groups or _group_related_entries(entries)


def _entry_similarity(left: str, right: str, semantic_similarity: float) -> float:
    left_tokens = {token.lower() for token in TOKEN_PATTERN.findall(left)}
    right_tokens = {token.lower() for token in TOKEN_PATTERN.findall(right)}
    lexical_similarity = 0.0
    if left_tokens and right_tokens:
        lexical_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(semantic_similarity, lexical_similarity)


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


def _execute_ingest_actions(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
    actions: list[dict[str, Any]],
) -> tuple[list[str], list[int], int]:
    assert ctx.repository is not None
    created_ids: list[str] = []
    handled_ids: list[int] = []
    meaningful_actions = 0
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
            handled_ids.extend(entry.id for entry in selected_entries)
            continue

        if action_type == "append":
            target_memory_id = str(action.get("target_memory_id", "")).strip()
            target = None if not target_memory_id else ctx.repository.get_memory(target_memory_id)
            if target is None:
                continue
            updated = _append_entries_to_existing_memory(ctx, target, selected_entries, task)
            append_workspace_ids = _resolve_entry_workspace_ids(selected_entries, workspace_id)
            missing_workspace_ids = [item for item in append_workspace_ids if item not in updated.workspace_ids]
            if missing_workspace_ids:
                refreshed = ctx.repository.append_workspace_ids(updated.id, missing_workspace_ids)
                if refreshed is not None:
                    updated = refreshed
            handled_ids.extend(entry.id for entry in selected_entries)
            created_ids.append(updated.id)
            meaningful_actions += 1
            continue

        if action_type != "create":
            continue

        content = str(action.get("content", "")).strip() or _format_entries(selected_entries)
        title = str(action.get("title", "")).strip() or _build_title(selected_entries)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            workspace_ids=_resolve_entry_workspace_ids(selected_entries, workspace_id),
            tags=["auto-ingested", "system1"],
            memory_type="observation",
            metadata={
                "source_entry_ids": [entry.id for entry in selected_entries],
                "ingest_task_id": task.id,
            },
        )
        assert record is not None
        created_ids.append(record.id)
        handled_ids.extend(entry.id for entry in selected_entries)
        meaningful_actions += 1
        _enqueue_summary_task(ctx, _primary_workspace_id(record.workspace_ids), record.id)

    return created_ids, handled_ids, meaningful_actions


def _append_entries_to_existing_memory(
    ctx: ApplicationContext,
    target,
    entries,
    task: TaskRecord,
):
    assert ctx.repository is not None
    addition = _format_entries(entries)
    merged_content = target.content if addition in target.content else f"{target.content.rstrip()}\n\n{addition}".strip()
    metadata = dict(target.metadata)
    appended_entry_ids = metadata.get("appended_entry_ids", [])
    if not isinstance(appended_entry_ids, list):
        appended_entry_ids = []
    metadata["appended_entry_ids"] = sorted({*map(int, [item for item in appended_entry_ids if isinstance(item, int)]), *[entry.id for entry in entries]})
    metadata["ingest_task_id"] = task.id
    updated = ctx.repository.update_memory(
        target.id,
        content=merged_content,
        tags=sorted({*target.tags, "system1-appended"}),
        metadata=metadata,
    )
    assert updated is not None
    return updated


def _fallback_ingest_entries(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
) -> tuple[list[str], list[int], int]:
    assert ctx.repository is not None
    workspace_ids = _resolve_entry_workspace_ids(entries, workspace_id)
    record = ctx.repository.create_memory(
        title=_build_title(entries),
        content=_format_entries(entries),
        workspace_ids=workspace_ids,
        tags=["auto-ingested", "system1"],
        memory_type="observation",
        metadata={
            "source_entry_ids": [entry.id for entry in entries],
            "ingest_task_id": task.id,
        },
    )
    assert record is not None
    _enqueue_summary_task(ctx, _primary_workspace_id(workspace_ids), record.id)
    return [record.id], [entry.id for entry in entries], 1


def _build_ingest_result(
    *,
    created_ids: list[str],
    claimed_ids: list[int],
    deleted_ids: list[int],
    released_ids: list[int],
    meaningful_actions: int,
) -> dict[str, Any]:
    return {
        "created_memory_ids": created_ids,
        "claimed_entry_ids": claimed_ids,
        "deleted_entry_ids": deleted_ids,
        "released_entry_ids": released_ids,
        "meaningful_actions": meaningful_actions,
        "processed_entry_ids": deleted_ids,
    }


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
    return "workspace-unknown"


def _resolve_entry_workspace_ids(entries, fallback_workspace_id: str) -> list[str]:
    workspace_ids = sorted(
        {
            entry.workspace_id.strip()
            for entry in entries
            if isinstance(entry.workspace_id, str) and entry.workspace_id.strip()
        }
    )
    if workspace_ids:
        return workspace_ids
    return [fallback_workspace_id]


def _primary_workspace_id(workspace_ids: list[str]) -> str | None:
    if not workspace_ids:
        return None
    return workspace_ids[0]


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

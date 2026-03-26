from __future__ import annotations

from datetime import UTC, datetime
from random import Random
import re
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_provenance import build_ingest_appended_metadata, build_ingest_created_metadata
from mcp_memory.core.task_handlers.ingest_support import (
    _build_ingest_result,
    _build_semantic_entry_dispositions,
    _finalize_claimed_ingest_entries,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.embeddings import cosine_similarity


SEMANTIC_CLUSTER_SIZE = 5
SEMANTIC_SIMILARITY_THRESHOLD = 0.3
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
FIFO_GROUPING_STRATEGY = "fifo"


async def process_ingest_batch(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any,
    *,
    workspace_id: str,
    grouping_strategy: str,
    analyze_ingest_actions,
    entries,
) -> dict[str, Any]:
    assert ctx.journal is not None
    claimed_ids = [entry.id for entry in entries]
    created_ids: list[str] = []
    handled_ids: list[int] = []
    meaningful_actions = 0
    semantic_entry_dispositions: list[dict[str, Any]] = []
    grouped_entries = _build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task.id,
        grouping_strategy=grouping_strategy,
    )

    for group in grouped_entries:
        actions = None
        tool_mutations = 0
        if provider is not None:
            try:
                actions, tool_mutations = await analyze_ingest_actions(ctx, provider, workspace_id, group)
            except Exception:
                actions = None
                tool_mutations = 0

        group_created_ids: list[str] = []
        group_handled_ids: list[int] = []
        group_meaningful_actions = 0
        group_entry_dispositions: list[dict[str, Any]] = []
        if actions:
            group_created_ids, group_handled_ids, group_meaningful_actions, group_entry_dispositions = _execute_ingest_actions(
                ctx,
                task,
                workspace_id,
                group,
                actions,
            )

        if not group_handled_ids:
            group_created_ids, group_handled_ids, group_meaningful_actions, group_entry_dispositions = _fallback_ingest_entries(
                ctx,
                task,
                workspace_id,
                group,
            )

        created_ids.extend(group_created_ids)
        handled_ids.extend(group_handled_ids)
        semantic_entry_dispositions.extend(group_entry_dispositions)
        meaningful_actions += group_meaningful_actions + tool_mutations

    if meaningful_actions <= 0:
        released_ids = ctx.journal.release_claims(task.id)
        return {
            "created_ids": created_ids,
            "claimed_ids": claimed_ids,
            "recoverable_ids": [],
            "released_ids": released_ids,
            "meaningful_actions": meaningful_actions,
            "entry_dispositions": semantic_entry_dispositions,
        }

    batch_claimed_ids, batch_recoverable_ids, batch_released_ids = _finalize_claimed_ingest_entries(
        ctx,
        task_id=task.id,
        handled_entry_ids=handled_ids,
    )
    return {
        "created_ids": created_ids,
        "claimed_ids": batch_claimed_ids,
        "recoverable_ids": batch_recoverable_ids,
        "released_ids": batch_released_ids,
        "meaningful_actions": meaningful_actions,
        "entry_dispositions": _build_ingest_result(
            created_ids=created_ids,
            claimed_ids=batch_claimed_ids,
            deleted_ids=[],
            recoverable_ids=batch_recoverable_ids,
            released_ids=batch_released_ids,
            meaningful_actions=meaningful_actions,
            semantic_entry_dispositions=semantic_entry_dispositions,
        )["entry_dispositions"],
    }


def build_ingest_groups(
    ctx: ApplicationContext,
    entries,
    workspace_id: str,
    *,
    task_id: str | None = None,
    grouping_strategy: str = FIFO_GROUPING_STRATEGY,
):
    return _build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task_id,
        grouping_strategy=grouping_strategy,
    )


def entry_similarity(left: str, right: str, semantic_similarity: float) -> float:
    return _entry_similarity(left, right, semantic_similarity)


def resolve_entry_workspace_ids(entries, fallback_workspace_id: str) -> list[str]:
    return _resolve_entry_workspace_ids(entries, fallback_workspace_id)


def format_entries(entries) -> str:
    return _format_entries(entries)


def _group_related_entries(entries, threshold: float = 0.3, *, seed_entries=None):
    if not entries:
        return []

    ordered_seed_entries = list(entries if seed_entries is None else seed_entries)
    word_sets = {entry.id: set(entry.content.lower().split()) for entry in entries}
    entry_by_id = {entry.id: entry for entry in entries}
    used: set[int] = set()
    groups = []

    for entry in ordered_seed_entries:
        if entry.id in used:
            continue
        group = [entry_by_id[entry.id]]
        used.add(entry.id)

        for candidate in entries:
            if candidate.id in used or candidate.id == entry.id:
                continue
            intersection = len(word_sets[entry.id] & word_sets[candidate.id])
            union = len(word_sets[entry.id] | word_sets[candidate.id])
            if union > 0 and intersection / union >= threshold:
                group.append(candidate)
                used.add(candidate.id)

        groups.append(group)

    return groups


def _build_ingest_groups(
    ctx: ApplicationContext,
    entries,
    workspace_id: str,
    *,
    task_id: str | None = None,
    grouping_strategy: str = FIFO_GROUPING_STRATEGY,
):
    embedder = getattr(ctx, "embedder", None)
    vector_store = getattr(ctx, "vector_store", None)
    seed_entries = _seed_entries_for_grouping(entries, task_id=task_id, grouping_strategy=grouping_strategy)
    if embedder is None or vector_store is None:
        return _group_related_entries(entries, seed_entries=seed_entries)

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
    for entry in seed_entries:
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

    return groups or _group_related_entries(entries, seed_entries=seed_entries)


def _entry_similarity(left: str, right: str, semantic_similarity: float) -> float:
    left_tokens = {token.lower() for token in TOKEN_PATTERN.findall(left)}
    right_tokens = {token.lower() for token in TOKEN_PATTERN.findall(right)}
    lexical_similarity = 0.0
    if left_tokens and right_tokens:
        lexical_similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return max(semantic_similarity, lexical_similarity)


def _seed_entries_for_grouping(entries, *, task_id: str | None, grouping_strategy: str):
    if grouping_strategy == FIFO_GROUPING_STRATEGY or not entries:
        return list(entries)
    rng = Random(task_id or "ingest-grouping")
    seeded_entries = list(entries)
    rng.shuffle(seeded_entries)
    return seeded_entries


def _execute_ingest_actions(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
    actions: list[dict[str, Any]],
) -> tuple[list[str], list[int], int, list[dict[str, Any]]]:
    assert ctx.repository is not None
    created_ids: list[str] = []
    handled_ids: list[int] = []
    meaningful_actions = 0
    entry_dispositions: list[dict[str, Any]] = []
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
            entry_dispositions.extend(
                _build_semantic_entry_dispositions(
                    selected_entries,
                    disposition="ignored",
                    reason="provider_marked_ignore",
                )
            )
            continue

        if action_type == "append":
            target_memory_id = str(action.get("target_memory_id", "")).strip()
            target = None if not target_memory_id else ctx.repository.get_memory(target_memory_id)
            if target is None:
                continue
            updated = _append_entries_to_existing_memory(ctx, target, selected_entries, task)
            requested_summary = action.get("summary")
            if isinstance(requested_summary, str) and requested_summary.strip():
                refreshed = ctx.repository.update_memory(updated.id, summary=requested_summary.strip())
                if refreshed is not None:
                    updated = refreshed
            append_workspace_ids = _resolve_entry_workspace_ids(selected_entries, workspace_id)
            missing_workspace_ids = [item for item in append_workspace_ids if item not in updated.workspace_ids]
            if missing_workspace_ids:
                refreshed = ctx.repository.append_workspace_ids(updated.id, missing_workspace_ids)
                if refreshed is not None:
                    updated = refreshed
            handled_ids.extend(entry.id for entry in selected_entries)
            created_ids.append(updated.id)
            entry_dispositions.extend(
                _build_semantic_entry_dispositions(
                    selected_entries,
                    disposition="appended",
                    memory_id=updated.id,
                    memory_title=updated.title,
                )
            )
            meaningful_actions += 1
            continue

        if action_type != "create":
            continue

        content = str(action.get("content", "")).strip() or _format_entries(selected_entries)
        title = str(action.get("title", "")).strip() or _build_title(selected_entries)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            summary=(
                str(action.get("summary")).strip()
                if isinstance(action.get("summary"), str) and str(action.get("summary")).strip()
                else None
            ),
            workspace_ids=_resolve_entry_workspace_ids(selected_entries, workspace_id),
            tags=[],
            memory_type="observation",
            metadata=build_ingest_created_metadata(
                task_id=task.id,
                entry_ids=[entry.id for entry in selected_entries],
            ),
        )
        assert record is not None
        created_ids.append(record.id)
        handled_ids.extend(entry.id for entry in selected_entries)
        entry_dispositions.extend(
            _build_semantic_entry_dispositions(
                selected_entries,
                disposition="created",
                memory_id=record.id,
                memory_title=record.title,
            )
        )
        meaningful_actions += 1

    return created_ids, handled_ids, meaningful_actions, entry_dispositions


def _append_entries_to_existing_memory(
    ctx: ApplicationContext,
    target,
    entries,
    task: TaskRecord,
):
    assert ctx.repository is not None
    addition = _format_entries(entries)
    merged_content = target.content if addition in target.content else f"{target.content.rstrip()}\n\n{addition}".strip()
    metadata = build_ingest_appended_metadata(
        task_id=task.id,
        entry_ids=[entry.id for entry in entries],
        metadata=target.metadata,
    )
    updated = ctx.repository.update_memory(
        target.id,
        content=merged_content,
        tags=sorted(set(target.tags)),
        metadata=metadata,
    )
    assert updated is not None
    return updated


def _fallback_ingest_entries(
    ctx: ApplicationContext,
    task: TaskRecord,
    workspace_id: str,
    entries,
) -> tuple[list[str], list[int], int, list[dict[str, Any]]]:
    assert ctx.repository is not None
    workspace_ids = _resolve_entry_workspace_ids(entries, workspace_id)
    record = ctx.repository.create_memory(
        title=_build_title(entries),
        content=_format_entries(entries),
        workspace_ids=workspace_ids,
        tags=[],
        memory_type="observation",
        metadata=build_ingest_created_metadata(
            task_id=task.id,
            entry_ids=[entry.id for entry in entries],
        ),
    )
    assert record is not None
    return (
        [record.id],
        [entry.id for entry in entries],
        1,
        _build_semantic_entry_dispositions(
            entries,
            disposition="created",
            memory_id=record.id,
            memory_title=record.title,
        ),
    )


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
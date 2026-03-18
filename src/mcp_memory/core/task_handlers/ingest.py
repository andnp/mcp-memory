from __future__ import annotations

import json
from random import Random
import sqlite3
from datetime import UTC, datetime
import re
from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.embeddings import cosine_similarity
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_INGEST_BATCH_SIZE,
    SUMMARIZE_MEMORY_TASK_NAME,
    SUMMARIZE_MEMORY_PRIORITY,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord


SEMANTIC_CLUSTER_SIZE = 5
SEMANTIC_SIMILARITY_THRESHOLD = 0.3
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
FIFO_GROUPING_STRATEGY = "fifo"
SEMANTIC_SEEDED_GROUPING_STRATEGY = "semantic-seeded"
LEXICAL_SEEDED_GROUPING_STRATEGY = "lexical-seeded"
INGEST_GROUPING_STRATEGIES = (
    FIFO_GROUPING_STRATEGY,
    SEMANTIC_SEEDED_GROUPING_STRATEGY,
    LEXICAL_SEEDED_GROUPING_STRATEGY,
)
INGEST_APPEND_TOOL_NAME = "internal_ingest_append_memory"
INGEST_CREATE_TOOL_NAME = "internal_ingest_create_memory"


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
    pending_count_before_run = ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    workspace_id = _resolve_workspace_id(ctx, task)
    grouping_strategy_requested = _requested_grouping_strategy(task.data)
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=grouping_strategy_requested,
    )
    if pending_count_before_run <= 0:
        return {
            **_build_ingest_result(created_ids=[], claimed_ids=[], deleted_ids=[], released_ids=[], meaningful_actions=0),
            "requested_grouping_strategy": grouping_strategy_requested,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
            "reason": "no_pending_entries",
        }

    provider = _resolve_ingest_execution_provider(
        ctx,
        task,
        provider,
        journal_workspace_id=journal_workspace_id,
        workspace_id=workspace_id,
        pending_count_before_run=pending_count_before_run,
    )

    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        try:
            agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                _build_ingest_agent_prompt(
                    task,
                    workspace_id=workspace_id,
                    batch_size=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
                    grouping_strategy=grouping_strategy_used,
                )
            )
            normalized = _normalize_ingest_agentic_result(agentic_result)
            meaningful_actions = max(normalized["meaningful_actions"], normalized["mutations"])
            if (
                pending_count_before_run > 0
                and normalized["tool_calls_executed"] <= 0
                and meaningful_actions <= 0
                and not normalized["created_memory_ids"]
            ):
                ctx.journal.release_claims(task.id)
                raise RuntimeError(
                    "ingest_agentic_no_tool_calls_with_pending_entries: "
                    f"pending_count={pending_count_before_run}"
                )
            if meaningful_actions <= 0:
                released_ids = ctx.journal.release_claims(task.id)
                return {
                    **_build_ingest_result(
                        created_ids=normalized["created_memory_ids"],
                        claimed_ids=released_ids,
                        deleted_ids=[],
                        released_ids=released_ids,
                        meaningful_actions=0,
                    ),
                        "requested_grouping_strategy": grouping_strategy_requested,
                        "grouping_strategy_used": grouping_strategy_used,
                        "grouping_fallback_reason": grouping_fallback_reason,
                    "summary": normalized["summary"],
                    "execution_mode": "agentic_mcp",
                    "tool_calls_executed": normalized["tool_calls_executed"],
                    "mutations": normalized["mutations"],
                    "tool_names_used": normalized["tool_names_used"],
                }

            deleted_ids = ctx.journal.delete_claims(task.id)
            if deleted_ids:
                _cleanup_deleted_thought_embeddings(ctx, deleted_ids)
            return {
                **_build_ingest_result(
                    created_ids=normalized["created_memory_ids"],
                    claimed_ids=deleted_ids,
                    deleted_ids=deleted_ids,
                    released_ids=[],
                    meaningful_actions=meaningful_actions,
                ),
                "requested_grouping_strategy": grouping_strategy_requested,
                "grouping_strategy_used": grouping_strategy_used,
                "grouping_fallback_reason": grouping_fallback_reason,
                "summary": normalized["summary"],
                "execution_mode": "agentic_mcp",
                "tool_calls_executed": normalized["tool_calls_executed"],
                "mutations": normalized["mutations"],
                "tool_names_used": normalized["tool_names_used"],
            }
        except BaseException:
            ctx.journal.release_claims(task.id)
            raise

    entries = ctx.journal.claim_pending(
        task_id=task.id,
        limit=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
        workspace_id=journal_workspace_id,
    )
    if not entries:
        return {
            **_build_ingest_result(created_ids=[], claimed_ids=[], deleted_ids=[], released_ids=[], meaningful_actions=0),
            "requested_grouping_strategy": grouping_strategy_requested,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
        }

    claimed_ids = [entry.id for entry in entries]
    created_ids: list[str] = []
    meaningful_actions = 0
    grouped_entries = _build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task.id,
        grouping_strategy=grouping_strategy_used,
    )

    try:
        for group in grouped_entries:
            actions = None
            tool_mutations = 0
            if provider is not None:
                try:
                    actions, tool_mutations = await _analyze_ingest_actions(ctx, provider, workspace_id, group)
                except Exception:
                    actions = None
                    tool_mutations = 0

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
            meaningful_actions += group_meaningful_actions + tool_mutations

        if meaningful_actions <= 0:
            released_ids = ctx.journal.release_claims(task.id)
            return _build_ingest_result(
                created_ids=created_ids,
                claimed_ids=claimed_ids,
                deleted_ids=[],
                released_ids=released_ids,
                meaningful_actions=meaningful_actions,
            ) | {
                "requested_grouping_strategy": grouping_strategy_requested,
                "grouping_strategy_used": grouping_strategy_used,
                "grouping_fallback_reason": grouping_fallback_reason,
            }

        deleted_ids = ctx.journal.delete_claims(task.id)
        if deleted_ids:
            _cleanup_deleted_thought_embeddings(ctx, deleted_ids)

        return _build_ingest_result(
            created_ids=created_ids,
            claimed_ids=claimed_ids,
            deleted_ids=deleted_ids,
            released_ids=[],
            meaningful_actions=meaningful_actions,
        ) | {
            "requested_grouping_strategy": grouping_strategy_requested,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
        }
    except BaseException:
        ctx.journal.release_claims(task.id)
        raise


async def _analyze_ingest_actions(
    ctx: ApplicationContext,
    provider: Any,
    workspace_id: str,
    entries,
) -> tuple[list[dict[str, Any]], int]:
    entry_text = "\n".join(f"[{index}] {entry.content}" for index, entry in enumerate(entries))
    prompt = (
        "Analyze these system1 journal entries and return JSON with actions.\n"
        'Allowed actions: {"type": "create"|"ignore"|"append", "entry_indices": [...], '
        '"target_memory_id": "...", "title": "...", "content": "...", "summary": "..."}.\n'
        f"Active workspace_id: {workspace_id}. "
        "If a thought clearly belongs in an existing canonical memory, prefer append and identify the target_memory_id. "
        "When you already understand the resulting memory well, include a concise summary so the handler can update it without another background task. "
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
    actions = _extract_ingest_actions(response.response)
    if not isinstance(actions, list):
        raise ValueError("provider returned invalid actions")
    return actions, response.mutating_tool_calls


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


def _build_ingest_agent_prompt(
    task: TaskRecord,
    *,
    workspace_id: str,
    batch_size: int,
    grouping_strategy: str,
) -> str:
    return (
        "You are the ingest-system1 maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly.\n"
        f"Start with internal_get_next_ingest_batch using task_id='{task.id}', batch_size={batch_size}, and grouping_strategy='{grouping_strategy}'.\n"
        "Only process journal entries claimed for this task.\n"
        "Use internal_search_memory_records, internal_read_memory_record, and internal_list_memory_records to find append targets before mutating memories.\n"
        f"When a thought clearly belongs in an existing canonical memory, prefer {INGEST_APPEND_TOOL_NAME} with task_id='{task.id}', the claimed entry_ids, relevant workspace_ids, the content to append, and a concise summary when you already understand the updated memory.\n"
        f"When a new memory is warranted, use {INGEST_CREATE_TOOL_NAME} with task_id='{task.id}', the claimed entry_ids, a focused title/content payload, relevant workspace_ids, and a concise summary when you can provide one cheaply.\n"
        "Use the generic append/create tools only when a non-ingest workflow truly requires them.\n"
        f"Use workspace_id '{workspace_id}' when you need a fallback workspace.\n"
        "Do not delete or release journal claims yourself; the handler finalizes claimed entries after your run based on actual memory mutations.\n"
        'When finished, output final JSON only in the form {"summary": "...", "created_memory_ids": ["..."], "meaningful_actions": N}.\n'
    )


def _requested_grouping_strategy(values: dict[str, Any]) -> str | None:
    raw_value = values.get("grouping_strategy")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _resolve_grouping_strategy(
    ctx: ApplicationContext,
    *,
    requested_strategy: str | None,
) -> tuple[str, str | None]:
    embedder = getattr(ctx, "embedder", None)
    vector_store = getattr(ctx, "vector_store", None)
    default_strategy = (
        SEMANTIC_SEEDED_GROUPING_STRATEGY
        if embedder is not None and vector_store is not None
        else LEXICAL_SEEDED_GROUPING_STRATEGY
    )
    if requested_strategy is None:
        return default_strategy, None
    if requested_strategy not in INGEST_GROUPING_STRATEGIES:
        return default_strategy, "unknown_requested_grouping_strategy"
    if requested_strategy == SEMANTIC_SEEDED_GROUPING_STRATEGY and (embedder is None or vector_store is None):
        return LEXICAL_SEEDED_GROUPING_STRATEGY, "semantic_grouping_unavailable"
    return requested_strategy, None


def _resolve_ingest_execution_provider(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any,
    *,
    journal_workspace_id,
    workspace_id: str,
    pending_count_before_run: int,
):
    if provider is None or ctx.journal is None:
        return provider
    escalation_config = None if ctx.config is None else ctx.config.ingest_escalation
    if escalation_config is None or not escalation_config.enabled or not escalation_config.deterministic_first:
        return provider

    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(supports_agentic) and not supports_agentic():
        return provider

    if not _is_routed_ingest_provider(ctx, provider):
        return provider
    if pending_count_before_run >= escalation_config.agentic_pending_count_threshold:
        return provider

    preview_entries = ctx.journal.get_pending(workspace_id=journal_workspace_id)[: escalation_config.preview_entry_limit]
    novelty_score = _estimate_ingest_novelty(ctx, preview_entries, workspace_id=workspace_id)
    if novelty_score < escalation_config.novelty_threshold:
        return None
    return provider


def _is_routed_ingest_provider(ctx: ApplicationContext, provider: Any) -> bool:
    profile_key = getattr(provider, "_budget_key", None)
    registry = getattr(ctx, "ai_provider_registry", None) or {}
    return isinstance(profile_key, str) and profile_key in registry


def _estimate_ingest_novelty(ctx: ApplicationContext, entries, *, workspace_id: str) -> float:
    if ctx.repository is None or not entries:
        return 1.0
    candidates = ctx.repository.list_memories(workspace_id=workspace_id, status="active", limit=25)
    if not candidates:
        return 1.0
    max_similarities: list[float] = []
    for entry in entries:
        entry_text = entry.content.strip()
        best_similarity = 0.0
        for candidate in candidates:
            candidate_text = _memory_similarity_text(candidate)
            best_similarity = max(best_similarity, _entry_similarity(entry_text, candidate_text, 0.0))
        max_similarities.append(best_similarity)
    if not max_similarities:
        return 1.0
    average_similarity = sum(max_similarities) / len(max_similarities)
    return max(0.0, min(1.0, 1.0 - average_similarity))


def _memory_similarity_text(record) -> str:
    summary = record.summary or ""
    lead_line = record.content.strip().splitlines()[0] if record.content.strip() else ""
    return " ".join(part for part in [record.title, summary, lead_line] if part).strip()


def _seed_entries_for_grouping(entries, *, task_id: str | None, grouping_strategy: str):
    if grouping_strategy == FIFO_GROUPING_STRATEGY or not entries:
        return list(entries)
    rng = Random(task_id or "ingest-grouping")
    seeded_entries = list(entries)
    rng.shuffle(seeded_entries)
    return seeded_entries


def _normalize_ingest_agentic_result(agentic_result: Any) -> dict[str, Any]:
    parsed = agentic_result.parsed if isinstance(getattr(agentic_result, "parsed", None), dict) else {}
    response_payload = parsed
    response_text = parsed.get("response")
    if not {"summary", "created_memory_ids", "meaningful_actions"} <= set(response_payload) and isinstance(response_text, str):
        nested = _extract_embedded_json_object(response_text)
        if isinstance(nested, dict):
            response_payload = nested
    raw_tool_stats = parsed.get("stats")
    tool_stats = raw_tool_stats if isinstance(raw_tool_stats, dict) else {}
    raw_tool_payload = tool_stats.get("tools")
    tool_payload = raw_tool_payload if isinstance(raw_tool_payload, dict) else {}
    return {
        "summary": _coerce_text_summary(getattr(agentic_result, "summary", None)) or _coerce_text_summary(response_payload.get("summary")),
        "created_memory_ids": _coerce_string_list(response_payload.get("created_memory_ids")),
        "meaningful_actions": _coerce_non_negative_int(response_payload.get("meaningful_actions")),
        "tool_calls_executed": _coerce_non_negative_int(tool_payload.get("totalCalls")),
        "mutations": _count_mutating_agentic_tool_calls(tool_payload.get("byName")),
        "tool_names_used": _extract_agentic_tool_names(tool_payload.get("byName")),
    }


def _extract_ingest_actions(response: dict[str, Any]) -> list[dict[str, Any]] | object:
    actions = response.get("actions")
    if isinstance(actions, list):
        return actions
    results = response.get("results")
    if isinstance(results, list):
        return results
    return actions


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
    workspace_id: str | None,
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
            priority=SUMMARIZE_MEMORY_PRIORITY,
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


def _coerce_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


def _coerce_text_summary(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_agentic_tool_names(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(name) for name, payload in value.items() if isinstance(name, str) and isinstance(payload, dict))


def _count_mutating_agentic_tool_calls(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    read_only_tool_names = {
        "mcp_mcp-memory-internal_internal_get_next_ingest_batch",
        "mcp_mcp-memory-internal_internal_search_memory_records",
        "mcp_mcp-memory-internal_internal_read_memory_record",
        "mcp_mcp-memory-internal_internal_list_memory_records",
    }
    total = 0
    for name, payload in value.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        total += _coerce_non_negative_int(payload.get("count"))
    return total

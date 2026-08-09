from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from random import Random
import re
from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_claim_lifecycle import (
    _finalize_claimed_ingest_entries as _finalize_claimed_ingest_entries,
    _journal_entry_payload,
    _record_ingest_tool_invocation,
    _recorded_ingest_entry_dispositions as _recorded_ingest_entry_dispositions,
    _recorded_ingest_handled_entry_ids as _recorded_ingest_handled_entry_ids,
    _recorded_ingest_run_metadata as _recorded_ingest_run_metadata,
    _recorded_ingest_tool_usage as _recorded_ingest_tool_usage,
    _recorded_ingest_touched_memory_ids as _recorded_ingest_touched_memory_ids,
    _reset_recorded_ingest_handled_entry_ids as _reset_recorded_ingest_handled_entry_ids,
)
from mcp_memory.core.ingress_evidence import IngressBatchEvidence, IngressSourceSnapshot
from mcp_memory.core.ingress_identity import SourceEntryIdentity, batch_id, source_fingerprint
from mcp_memory.core.ingress_source import normalize_source_snapshot
from mcp_memory.core.ingest_provenance import build_ingest_appended_metadata, build_ingest_created_metadata
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.agentic_guardrails import build_ingest_guardrails
from mcp_memory.core.task_handlers.agentic_result_support import (
    build_tool_usage_summary,
    coerce_non_negative_int,
    coerce_text_summary,
    count_mutating_agentic_tool_calls,
    extract_agentic_tool_names,
    extract_embedded_json_object,
)
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    prefer_deterministic_agentic_counts,
    reset_agentic_tool_tracking,
)
from mcp_memory.core.task_handlers.constants import DEFAULT_INGEST_BATCH_SIZE
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.task_handlers.workspace_resolution import resolve_task_or_context_workspace_id
from mcp_memory.core.ports.tasks import TaskRecord
from searchkernel.ingestion import embed_in_batches
from searchkernel.utils.similarity import cosine_similarity_lists


# ---------------------------------------------------------------------------
# ingest_support: result building and normalization helpers
# ---------------------------------------------------------------------------

_INGEST_READ_ONLY_TOOL_NAMES = {
    "mcp_mcp-memory-internal_task_complete",
    "mcp_mcp-memory-internal_internal_get_next_ingest_batch",
    "mcp_mcp-memory-internal_internal_search_memory_records",
    "mcp_mcp-memory-internal_internal_read_memory_record",
    "mcp_mcp-memory-internal_internal_list_memory_records",
    "mcp_mcp-memory-internal_internal_task_complete",
}


def _build_ingest_result(
    *,
    created_ids: list[str],
    claimed_ids: list[int],
    deleted_ids: list[int],
    recoverable_ids: list[int],
    released_ids: list[int],
    meaningful_actions: int,
    semantic_entry_dispositions: list[dict[str, Any]] | None = None,
    recorded_touched_memory_ids: list[str] | None = None,
) -> dict[str, Any]:
    entry_dispositions = _finalize_entry_dispositions(
        claimed_ids=claimed_ids,
        recoverable_ids=recoverable_ids,
        released_ids=released_ids,
        semantic_entry_dispositions=semantic_entry_dispositions or [],
    )
    touched_memory_ids = _merge_string_lists(
        _memory_ids_for_dispositions(entry_dispositions, {"appended", "created", "matched_existing"}),
        recorded_touched_memory_ids or [],
    )
    return {
        "created_memory_ids": created_ids,
        "claimed_entry_ids": claimed_ids,
        "deleted_entry_ids": deleted_ids,
        "recoverable_entry_ids": recoverable_ids,
        "released_entry_ids": released_ids,
        "meaningful_actions": meaningful_actions,
        "processed_entry_ids": recoverable_ids,
        "entry_dispositions": entry_dispositions,
        "touched_memory_ids": touched_memory_ids,
        "appended_memory_ids": _memory_ids_for_dispositions(entry_dispositions, {"appended"}),
        "matched_memory_ids": _memory_ids_for_dispositions(entry_dispositions, {"appended", "matched_existing"}),
    }


def _normalize_ingest_agentic_result(agentic_result: Any) -> dict[str, Any]:
    parsed = agentic_result.parsed if isinstance(getattr(agentic_result, "parsed", None), dict) else {}
    response_payload = parsed
    response_text = parsed.get("response")
    if not _looks_like_ingest_agentic_final_json(response_payload) and isinstance(response_text, str):
        nested = _extract_embedded_json_object(response_text)
        if isinstance(nested, dict):
            response_payload = nested
    created_memory_ids = _coerce_string_list(response_payload.get("created_memory_ids"))
    entry_outcomes = _coerce_ingest_entry_outcomes(response_payload.get("entry_outcomes"))
    cluster_outcomes = _coerce_ingest_cluster_outcomes(response_payload.get("cluster_outcomes"))
    response_has_touched_memory_ids = "touched_memory_ids" in response_payload
    response_has_matched_memory_ids = "matched_memory_ids" in response_payload
    touched_memory_ids = _coerce_string_list(response_payload.get("touched_memory_ids"))
    matched_memory_ids = _coerce_string_list(response_payload.get("matched_memory_ids"))
    if not response_has_touched_memory_ids:
        touched_memory_ids = _merge_string_lists(
            created_memory_ids,
            _memory_ids_from_entry_outcomes(entry_outcomes),
        )
    if not response_has_matched_memory_ids:
        matched_memory_ids = _memory_ids_from_entry_outcomes(entry_outcomes, {"appended", "matched_existing"})
    return {
        "summary": _coerce_text_summary(getattr(agentic_result, "summary", None)) or _coerce_text_summary(response_payload.get("summary")),
        "created_memory_ids": created_memory_ids,
        "entry_outcomes": entry_outcomes,
        "cluster_outcomes": cluster_outcomes,
        "touched_memory_ids": touched_memory_ids,
        "matched_memory_ids": matched_memory_ids,
        "meaningful_actions": _coerce_non_negative_int(response_payload.get("meaningful_actions")),
    } | build_tool_usage_summary(parsed, read_only_tool_names=_INGEST_READ_ONLY_TOOL_NAMES)


def _build_semantic_entry_dispositions(
    entries,
    *,
    disposition: str,
    memory_id: str | None = None,
    memory_title: str | None = None,
    reason: str | None = None,
) -> list[dict[str, Any]]:
    return [
        _build_semantic_entry_disposition(
            entry_id=entry.id,
            disposition=disposition,
            memory_id=memory_id,
            memory_title=memory_title,
            reason=reason,
        )
        for entry in entries
    ]


def _build_semantic_entry_disposition(
    *,
    entry_id: int,
    disposition: str,
    memory_id: object = None,
    memory_title: object = None,
    reason: object = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "entry_id": entry_id,
        "disposition": disposition,
    }
    if isinstance(memory_id, str) and memory_id.strip():
        payload["memory_id"] = memory_id.strip()
    if isinstance(memory_title, str) and memory_title.strip():
        payload["memory_title"] = memory_title.strip()
    if isinstance(reason, str) and reason.strip():
        payload["reason"] = reason.strip()
    return payload


def _coerce_non_negative_int(value: object) -> int:
    return coerce_non_negative_int(value)


def _coerce_text_summary(value: object) -> str | None:
    return coerce_text_summary(value)


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


def _looks_like_ingest_agentic_final_json(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    ingest_contract_keys = {
        "summary",
        "created_memory_ids",
        "entry_outcomes",
        "touched_memory_ids",
        "matched_memory_ids",
        "meaningful_actions",
    }
    return any(key in value for key in ingest_contract_keys)


def _coerce_ingest_entry_outcomes(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        entry_id = item_dict.get("entry_id")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            continue
        raw_disposition = item_dict.get("disposition", item_dict.get("outcome"))
        if not isinstance(raw_disposition, str) or not raw_disposition.strip():
            continue
        normalized.append(
            _build_semantic_entry_disposition(
                entry_id=entry_id,
                disposition=raw_disposition.strip(),
                memory_id=item_dict.get("memory_id"),
                memory_title=item_dict.get("memory_title"),
                reason=item_dict.get("reason"),
            )
        )
    return normalized


def _coerce_ingest_cluster_outcomes(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        entry_ids = [
            entry_id
            for entry_id in item_dict.get("entry_ids", [])
            if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
        ]
        memory_ids = _coerce_string_list(item_dict.get("memory_ids"))
        disposition = item_dict.get("disposition")
        if not entry_ids or not isinstance(disposition, str) or not disposition.strip():
            continue
        payload: dict[str, Any] = {
            "entry_ids": sorted(set(entry_ids)),
            "disposition": disposition.strip(),
        }
        if memory_ids:
            payload["memory_ids"] = memory_ids
        reason = item_dict.get("reason")
        if isinstance(reason, str) and reason.strip():
            payload["reason"] = reason.strip()
        normalized.append(payload)
    return normalized


def _memory_ids_from_entry_outcomes(
    entry_outcomes: list[dict[str, Any]],
    included_dispositions: set[str] | None = None,
) -> list[str]:
    dispositions = included_dispositions or {"created", "appended", "matched_existing"}
    return _memory_ids_for_dispositions(entry_outcomes, dispositions)


def _merge_string_lists(*values: list[str]) -> list[str]:
    return sorted({item for value in values for item in value})


def _extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    return extract_embedded_json_object(text)


def _extract_agentic_tool_names(value: object) -> list[str]:
    return extract_agentic_tool_names(value)


def _count_mutating_agentic_tool_calls(value: object) -> int:
    return count_mutating_agentic_tool_calls(value, read_only_tool_names=_INGEST_READ_ONLY_TOOL_NAMES)


def _finalize_entry_dispositions(
    *,
    claimed_ids: list[int],
    recoverable_ids: list[int],
    released_ids: list[int],
    semantic_entry_dispositions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    recoverable_entry_id_set = set(recoverable_ids)
    released_entry_id_set = set(released_ids)
    semantic_by_entry_id: dict[int, dict[str, Any]] = {}
    for disposition in semantic_entry_dispositions:
        entry_id = disposition.get("entry_id")
        if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0:
            semantic_by_entry_id[entry_id] = disposition

    finalized: list[dict[str, Any]] = []
    for entry_id in claimed_ids:
        finalization_status = "recoverable" if entry_id in recoverable_entry_id_set else "released"
        semantic_disposition = semantic_by_entry_id.get(entry_id)
        if semantic_disposition is None:
            finalized.append(
                {
                    "entry_id": entry_id,
                    "disposition": "released_unhandled" if entry_id in released_entry_id_set else "recoverable_unclassified",
                    "finalization_status": finalization_status,
                }
            )
            continue

        finalized_disposition = {
            **semantic_disposition,
            "finalization_status": finalization_status,
        }
        finalized.append(finalized_disposition)
    return finalized


def _memory_ids_for_dispositions(entry_dispositions: list[dict[str, Any]], included_dispositions: set[str]) -> list[str]:
    return sorted(
        {
            memory_id.strip()
            for disposition in entry_dispositions
            if disposition.get("disposition") in included_dispositions
            for memory_id in [disposition.get("memory_id")]
            if isinstance(memory_id, str) and memory_id.strip()
        }
    )

# ---------------------------------------------------------------------------
# ingest_batch_support: batch grouping and processing
# ---------------------------------------------------------------------------

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
    batch_sequence: int | None = None,
) -> dict[str, Any]:
    assert ctx.journal is not None
    claimed_ids = [entry.id for entry in entries]
    _persist_ingress_batch_evidence(
        ctx,
        task,
        provider,
        workspace_id=workspace_id,
        grouping_strategy=grouping_strategy,
        entries=entries,
        batch_sequence=batch_sequence,
    )
    if getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off") == "enforce":
        ctx.journal.move_claims_to_recoverable(task.id)
        raise RuntimeError("atomic_boundary_required")
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


def _persist_ingress_batch_evidence(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any,
    *,
    workspace_id: str,
    grouping_strategy: str,
    entries,
    batch_sequence: int | None,
    provider_route: str | None = None,
    execution_mode: str | None = None,
    grouping_fallback_reason: str | None = None,
) -> str | None:
    mode = getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off")
    if mode == "off":
        return None

    sequence = batch_sequence if batch_sequence is not None else int(task.data.get("batch_sequence", 1))
    if isinstance(sequence, bool) or sequence < 1:
        raise ValueError("batch_sequence must be a positive integer")
    repository = getattr(ctx, "ingress_batch_evidence", None)
    try:
        source = normalize_source_snapshot(
            [
                {
                    "entry_id": str(entry.id),
                    "timestamp": datetime.fromtimestamp(entry.timestamp, tz=UTC).isoformat(),
                    "workspace_ids": (entry.workspace_id or workspace_id,),
                    "content_digest": sha256(entry.content.encode("utf-8")).hexdigest(),
                }
                for entry in entries
            ]
        )
        entries_by_id = {str(item.id): item for item in entries}
        source_entries = tuple(
            IngressSourceSnapshot(
                entry_id=str(entry["entry_id"]),
                workspace_ids=tuple(cast(list[str], entry["workspace_ids"])),
                timestamp=str(entry["timestamp"]),
                content_digest=str(entry["content_digest"]),
                snapshot={"content": entries_by_id[str(entry["entry_id"])].content},
            )
            for entry in cast(list[dict[str, object]], source["entries"])
        )
        source_identities = tuple(
            SourceEntryIdentity(
                entry_id=entry.entry_id,
                timestamp=entry.timestamp,
                workspace_ids=entry.workspace_ids,
                content_digest=entry.content_digest,
            )
            for entry in source_entries
        )
        fingerprint = source_fingerprint(source_identities)
        evidence = IngressBatchEvidence(
            batch_id=batch_id([entry.entry_id for entry in source_entries], fingerprint),
            task_id=task.id,
            execution_epoch=task.execution_epoch,
            batch_sequence=sequence,
            claimed_entry_ids=tuple(entry.entry_id for entry in source_entries),
            source_fingerprint=fingerprint,
            source_entries=source_entries,
            grouping_strategy=grouping_strategy,
            grouping_fallback_reason=(
                grouping_fallback_reason
                if grouping_fallback_reason is not None
                else task.data.get("grouping_fallback_reason")
            ),
            provider_route=provider_route or _ingress_provider_route(provider),
            execution_mode=execution_mode or ("provider_analysis" if provider is not None else "deterministic_fallback"),
            policy_version=str(task.data.get("policy_version", "ingress-v1")),
            schema_version=str(task.data.get("schema_version", "1")),
            claimed_at=datetime.now(UTC).isoformat(),
        )
        if repository is None:
            raise RuntimeError("ingress_evidence_unavailable")
        existing = repository.get(evidence.batch_id)
        if existing is not None:
            return evidence.batch_id
        repository.save(evidence)
        return evidence.batch_id
    except Exception:
        if mode == "enforce":
            ctx.journal.move_claims_to_recoverable(task.id)
            raise
    return None


def _ingress_provider_route(provider: Any) -> str:
    if provider is None:
        return "none"
    route = getattr(provider, "_budget_key", None) or getattr(provider, "_provider_key", None)
    if isinstance(route, str) and route.strip():
        return route.strip()
    return type(provider).__name__


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
    embeddings = embed_in_batches(
        [entry.content for entry in entries],
        provider=embedder,
        batch_size=max(len(entries), 1),
    )
    by_id = {}
    for entry, embedding in zip(entries, embeddings, strict=True):
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
                    cosine_similarity_lists(seed_embedding, by_id[candidate_id]),
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

# ---------------------------------------------------------------------------
# ingest_claim_batch_support: grouping strategy resolution and batch payload
# ---------------------------------------------------------------------------

SEMANTIC_SEEDED_GROUPING_STRATEGY = "semantic-seeded"
LEXICAL_SEEDED_GROUPING_STRATEGY = "lexical-seeded"
INGEST_GROUPING_STRATEGIES = (
    FIFO_GROUPING_STRATEGY,
    SEMANTIC_SEEDED_GROUPING_STRATEGY,
    LEXICAL_SEEDED_GROUPING_STRATEGY,
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


def build_next_ingest_batch_payload(ctx: ApplicationContext, arguments: dict[str, Any]) -> dict[str, Any]:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    from mcp_memory.mcp.validation import optional_positive_int, optional_string, require_string

    task_id = require_string(arguments, "task_id")
    requested_workspace_id = optional_string(arguments, "workspace_id")
    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        requested_workspace_id,
    )
    workspace_id = requested_workspace_id or ctx.workspace_id or "workspace-unknown"
    batch_size = optional_positive_int(arguments, "batch_size", 20)
    grouping_strategy_requested = optional_string(arguments, "grouping_strategy")
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=grouping_strategy_requested,
    )

    entries = ctx.journal.claim_pending(
        task_id=task_id,
        limit=batch_size,
        workspace_id=journal_workspace_id,
    )
    persisted_batch_id = None
    if getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off") != "off":
        task = None
        try:
            task_queue = getattr(ctx, "task_queue", None)
            if task_queue is None:
                raise RuntimeError("task_queue_unavailable")
            task = task_queue.get_task(task_id)
            repository = getattr(ctx, "ingress_batch_evidence", None)
            configured_sequence = task.data.get("batch_sequence")
            if configured_sequence is None:
                if repository is None:
                    raise RuntimeError("ingress_evidence_unavailable")
                batch_sequence = len(repository.list_for_execution(task.id, task.execution_epoch)) + 1
            else:
                batch_sequence = None
            persisted_batch_id = _persist_ingress_batch_evidence(
                ctx,
                task,
                None,
                workspace_id=workspace_id,
                grouping_strategy=grouping_strategy_used,
                grouping_fallback_reason=grouping_fallback_reason,
                entries=entries,
                batch_sequence=batch_sequence,
                provider_route="agentic_mcp",
                execution_mode="agentic_mcp",
            )
        except Exception:
            if getattr(getattr(ctx, "config", None), "ingress_evidence_mode", "off") == "enforce":
                if task is None:
                    ctx.journal.move_claims_to_recoverable(task_id)
                raise
    groups = build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task_id,
        grouping_strategy=grouping_strategy_used,
    )
    pending_remaining = ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_get_next_ingest_batch",
        mutation=False,
    )
    payload = {
        "status": "ok",
        "task_id": task_id,
        "requested_grouping_strategy": grouping_strategy_requested,
        "grouping_strategy_used": grouping_strategy_used,
        "grouping_fallback_reason": grouping_fallback_reason,
        "claimed_entry_ids": [entry.id for entry in entries],
        "pending_remaining": pending_remaining,
        "has_more": pending_remaining > 0,
        "group_count": len(groups),
        "groups": [
            {
                "group_index": index,
                "entries": [_journal_entry_payload(entry) for entry in group],
            }
            for index, group in enumerate(groups)
        ],
    }
    if persisted_batch_id is not None:
        payload["batch_id"] = persisted_batch_id
    return payload

# ---------------------------------------------------------------------------
# ingest_run_support: run accumulator
# ---------------------------------------------------------------------------

@dataclass
class IngestRunAccumulator:
    created_ids: list[str] = field(default_factory=list)
    claimed_ids: list[int] = field(default_factory=list)
    recoverable_ids: list[int] = field(default_factory=list)
    released_ids: list[int] = field(default_factory=list)
    semantic_entry_dispositions: list[dict[str, Any]] = field(default_factory=list)
    meaningful_actions: int = 0
    batches_processed: int = 0

    def absorb_batch_result(self, batch_result: dict[str, Any]) -> None:
        self.batches_processed += 1
        self.created_ids.extend(batch_result["created_ids"])
        self.claimed_ids.extend(batch_result["claimed_ids"])
        self.recoverable_ids.extend(batch_result["recoverable_ids"])
        self.released_ids.extend(batch_result["released_ids"])
        self.semantic_entry_dispositions.extend(batch_result["entry_dispositions"])
        self.meaningful_actions += int(batch_result["meaningful_actions"])

    def build_handler_result(
        self,
        *,
        requested_grouping_strategy: str | None,
        grouping_strategy_used: str,
        grouping_fallback_reason: str | None,
        pending_remaining: int,
    ) -> dict[str, Any]:
        return _build_ingest_result(
            created_ids=self.created_ids,
            claimed_ids=sorted(set(self.claimed_ids)),
            deleted_ids=[],
            recoverable_ids=sorted(set(self.recoverable_ids)),
            released_ids=sorted(set(self.released_ids)),
            meaningful_actions=self.meaningful_actions,
            semantic_entry_dispositions=self.semantic_entry_dispositions,
        ) | {
            "requested_grouping_strategy": requested_grouping_strategy,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
            "batches_processed": self.batches_processed,
            "pending_remaining": pending_remaining,
        }


def should_continue_ingest_run(*, batch_meaningful_actions: int, pending_remaining: int) -> bool:
    if batch_meaningful_actions <= 0:
        return False
    return pending_remaining > 0

# ---------------------------------------------------------------------------
# ingest_preflight_support: preflight state
# ---------------------------------------------------------------------------

DEFAULT_INGEST_MAX_BATCHES_PER_RUN = 8


@dataclass(frozen=True)
class IngestPreflightState:
    journal_workspace_id: object
    workspace_id: str
    batch_size: int
    max_batches_per_run: int
    pending_count_before_run: int
    requested_grouping_strategy: str | None
    grouping_strategy_used: str
    grouping_fallback_reason: str | None
    provider: Any


def build_ingest_preflight_state(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any,
) -> IngestPreflightState:
    assert ctx.journal is not None
    journal = ctx.journal
    journal_workspace_id = resolve_pending_workspace_id(
        journal,
        task.data.get("journal_workspace_id", task.workspace_id),
    )
    batch_size = int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE))
    max_batches_per_run = max(1, int(task.data.get("max_batches_per_run", DEFAULT_INGEST_MAX_BATCHES_PER_RUN)))
    pending_count_before_run = journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    workspace_id = resolve_task_or_context_workspace_id(
        ctx,
        task,
        fallback="workspace-unknown",
    ) or "workspace-unknown"
    requested_grouping_strategy = _requested_grouping_strategy(task.data)
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=requested_grouping_strategy,
    )
    resolved_provider = _resolve_ingest_execution_provider(
        ctx,
        task,
        provider,
        journal_workspace_id=journal_workspace_id,
        workspace_id=workspace_id,
        pending_count_before_run=pending_count_before_run,
    )
    return IngestPreflightState(
        journal_workspace_id=journal_workspace_id,
        workspace_id=workspace_id,
        batch_size=batch_size,
        max_batches_per_run=max_batches_per_run,
        pending_count_before_run=pending_count_before_run,
        requested_grouping_strategy=requested_grouping_strategy,
        grouping_strategy_used=grouping_strategy_used,
        grouping_fallback_reason=grouping_fallback_reason,
        provider=resolved_provider,
    )


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
            best_similarity = max(best_similarity, entry_similarity(entry_text, candidate_text, 0.0))
        max_similarities.append(best_similarity)
    if not max_similarities:
        return 1.0
    average_similarity = sum(max_similarities) / len(max_similarities)
    return max(0.0, min(1.0, 1.0 - average_similarity))


def _memory_similarity_text(record) -> str:
    summary = record.summary or ""
    lead_line = record.content.strip().splitlines()[0] if record.content.strip() else ""
    return " ".join(part for part in [record.title, summary, lead_line] if part).strip()

# ---------------------------------------------------------------------------
# ingest_agentic_support: agentic ingest pass
# ---------------------------------------------------------------------------

INGEST_APPEND_TOOL_NAME = "internal_ingest_append_memory"
INGEST_CREATE_TOOL_NAME = "internal_ingest_create_memory"


async def run_agentic_ingest_pass(
    ctx: ApplicationContext,
    task: TaskRecord,
    run_agent: Callable[[str], Awaitable[Any]],
    *,
    workspace_id: str,
    journal_workspace_id,
    batch_size: int,
    grouping_strategy: str,
    max_batches_per_run: int,
    pending_count_before_run: int,
) -> dict[str, Any] | None:
    _reset_recorded_ingest_handled_entry_ids(ctx, task.id)
    reset_agentic_tool_tracking(ctx, task.id, execution_epoch=task.execution_epoch)
    try:
        agentic_result = await run_agent(
            build_ingest_agent_prompt(
                task,
                workspace_id=workspace_id,
                batch_size=batch_size,
                grouping_strategy=grouping_strategy,
                max_batches_per_run=max_batches_per_run,
            )
        )
    except BaseException:
        finalize_agentic_tool_tracking(ctx, task.id)
        raise
    provider_reported = _normalize_ingest_agentic_result(agentic_result)
    normalized = prefer_deterministic_agentic_counts(
        provider_reported,
        deterministic_counts=finalize_agentic_tool_tracking(ctx, task.id),
    )
    recorded_run_metadata = _recorded_ingest_run_metadata(ctx, task.id)
    authoritative_tool_usage = {
        "tool_calls_executed": normalized["tool_calls_executed"],
        "mutations": normalized["mutations"],
        "tool_names_used": normalized["tool_names_used"],
    }
    meaningful_actions = max(normalized["meaningful_actions"], authoritative_tool_usage["mutations"])
    if (
        pending_count_before_run > 0
        and authoritative_tool_usage["tool_calls_executed"] <= 0
        and meaningful_actions <= 0
        and not normalized["created_memory_ids"]
    ):
        if ctx.journal is not None:
            ctx.journal.release_claims(task.id)
        return None

    semantic_entry_dispositions = recorded_run_metadata["entry_dispositions"]
    handled_entry_ids = recorded_run_metadata["handled_entry_ids"]
    touched_memory_ids = recorded_run_metadata["touched_memory_ids"]
    meaningful_actions = max(meaningful_actions, 1 if handled_entry_ids else 0)

    claimed_ids, recoverable_ids, released_ids = _finalize_claimed_ingest_entries(
        ctx,
        task_id=task.id,
        handled_entry_ids=handled_entry_ids,
    )
    return {
        **_build_ingest_result(
            created_ids=normalized["created_memory_ids"],
            claimed_ids=claimed_ids,
            deleted_ids=[],
            recoverable_ids=recoverable_ids,
            released_ids=released_ids,
            meaningful_actions=meaningful_actions,
            semantic_entry_dispositions=semantic_entry_dispositions,
            recorded_touched_memory_ids=touched_memory_ids,
        ),
        "summary": normalized["summary"],
        "execution_mode": "agentic_mcp",
        "tool_calls_executed": authoritative_tool_usage["tool_calls_executed"],
        "mutations": authoritative_tool_usage["mutations"],
        "tool_names_used": authoritative_tool_usage["tool_names_used"],
        "provider_reported_tool_calls": provider_reported["tool_calls_executed"],
        "provider_reported_mutations": provider_reported["mutations"],
        "provider_reported_tool_names_used": provider_reported["tool_names_used"],
        "provider_reported_entry_outcomes": provider_reported["entry_outcomes"],
        "provider_reported_cluster_outcomes": provider_reported["cluster_outcomes"],
        "provider_reported_touched_memory_ids": provider_reported["touched_memory_ids"],
        "provider_reported_matched_memory_ids": provider_reported["matched_memory_ids"],
        "pending_remaining": 0
        if ctx.journal is None
        else ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0),
    }


def build_ingest_agent_prompt(
    task: TaskRecord,
    *,
    workspace_id: str,
    batch_size: int,
    grouping_strategy: str,
    max_batches_per_run: int,
) -> str:
    workspace_fallback_line = (
        f"Use workspace_id '{workspace_id}' when you need a fallback workspace for created or updated memories.\n"
        if workspace_id.strip() != "workspace-unknown"
        else ""
    )
    return (
        "You are the ingest-system1 maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly.\n"
        "You are not a simple promotion script; your job is to integrate claimed System 1 thoughts while leaving the surrounding System 2 neighborhood tidier when safe and clearly beneficial.\n"
        f"Start with internal_get_next_ingest_batch using task_id='{task.id}', batch_size={batch_size}, and grouping_strategy='{grouping_strategy}'.\n"
        f"Aim to drain the queue for this task in one run by repeating internal_get_next_ingest_batch after each handled batch until has_more is false, no meaningful mutation is possible, or you have already processed {max_batches_per_run} batches in this run.\n"
        f"{build_ingest_guardrails()}\n"
        "A good memory is focused and durable: capture one finding, decision, anomaly, or reusable lesson with enough concrete evidence/context that another agent can trust it later.\n"
        "Good memory anatomy: a specific title, a summary that states the real conclusion instead of a generic 'Covers ...' / 'Added ...' phrase, concrete identifiers preserved in content, and a small set of useful tags when you create a new record.\n"
        "Bad memory patterns: routine progress logs, mixed unrelated topics, vague summaries, and tiny split-like fragments that are not useful on their own.\n"
        "Anti-bucket rule: do not append into a memory whose current title/scope is materially narrower than the new evidence. If the new evidence would broaden the topic into a catch-all bucket, prefer a new focused sibling memory or refactor the target first.\n"
        "Append only when the source entry and target memory clearly share the same subsystem, decision thread, or durable operational invariant. Generic overlap like 'infra', 'global scope', 'cleanup', or 'runtime' is not enough.\n"
        "Before appending, sanity-check that you can honestly complete the sentence 'This belongs in the target memory because both are fundamentally about ___." " If that blank would be vague or hand-wavy, do not append there.\n"
        "Standing ingest jobs: append into the right canonical memory, create a new narrow memory when novelty warrants it, lightly rewrite or resummarize touched memories when the batch reveals a clearer durable shape, split bloated targets that would become mixed-topic blobs, merge or archive stale leftovers when consolidation makes them obsolete, and clean up links when structure is obviously misleading or incomplete.\n"
        "Treat one claimed batch as a small maintenance campaign, not a one-thought-to-one-memory conveyor belt. Multiple claimed thoughts may belong in one focused memory, and one thought may justify refactoring an existing cluster before the best durable landing spot is clear.\n"
        "Only process journal entries claimed for this task.\n"
        "Use internal_search_memory_records, internal_read_memory_record, and internal_list_memory_records to find append targets before mutating memories.\n"
        "Copy the batch_id from each internal_get_next_ingest_batch response into every internal_ingest_append_memory or internal_ingest_create_memory mutation for that batch.\n"
        f"Tool mapping: prefer {INGEST_APPEND_TOOL_NAME} and {INGEST_CREATE_TOOL_NAME} whenever a mutation should consume claimed entry_ids directly. Use internal_update_memory_record for title/summary/content cleanup on touched memories, internal_split_memory_record for decompositions, internal_merge_memory_into_canonical for canonicalization, internal_archive_memory_record for safe cleanup, and internal_create_memory_link or internal_delete_memory_link when structural edge cleanup clearly improves retrieval. Include task_id='{task.id}' on those adjacent cleanup mutations so the run's telemetry stays attributable to the current ingest task.\n"
        f"When a thought clearly belongs in an existing canonical memory, prefer {INGEST_APPEND_TOOL_NAME} with task_id='{task.id}', the claimed entry_ids, relevant workspace_ids, the content to append, and a concise summary when you already understand the updated memory.\n"
        f"When a new memory is warranted, use {INGEST_CREATE_TOOL_NAME} with task_id='{task.id}', the claimed entry_ids, a focused title/content payload, 3-6 concrete tags when they are obvious, relevant workspace_ids, and a concise summary when you can provide one cheaply.\n"
        "When the best existing target is close but too narrow, prefer light refactoring first: rewrite/split/link the target or create a sibling memory before appending more detail.\n"
        "Prefer a narrowly named sibling memory over stuffing more detail into a broad operational bucket. Link related siblings when that helps retrieval.\n"
        "Use the generic append/create tools only when a non-ingest workflow truly requires them. After create/append, do any adjacent tidy-up work that is now clearly justified rather than leaving an obvious cleanup for a later run.\n"
        f"{workspace_fallback_line}"
        "Do not delete or release journal claims yourself; the handler finalizes claimed entries after your run based on actual memory mutations.\n"
        "Do not end after the first successful mutation if more claimed work remains and additional safe tidy maintenance is still obvious.\n"
        "Do not create memories that only log task completion, queue progress, tool usage, repo state snapshots, temporary runtime-health snapshots, one-off validation summaries, or other routine status traces; durable memory should capture findings, decisions, incidents, or reusable observations instead.\n"
        "Preserve concrete symbols, file paths, thresholds, IDs, error strings, config keys, and commit refs when they appear in the source entries or supporting memories.\n"
        "When uncertain, prefer narrow concrete observations over broad abstraction.\n"
        "When multiple claimed entries only make sense together, keep them together: it is valid for one cluster of entries to justify a coordinated create+rewrite+link cleanup sequence, or for one entry to stay unchanged because its meaning depends on a sibling entry that you handled elsewhere in the same batch.\n"
        f"When your full ingest pass is complete, call task_complete with task_id='{task.id}', task_name='ingest-system1', and a short operational summary before your final JSON response.\n"
        "When finished, output final JSON only. Include explicit per-entry outcomes for every claimed entry you handled or intentionally left unchanged.\n"
        'Use an additive contract like {"summary": "...", "created_memory_ids": ["..."], "touched_memory_ids": ["..."], "matched_memory_ids": ["..."], "meaningful_actions": N, "entry_outcomes": [{"entry_id": 123, "disposition": "created"|"appended"|"matched_existing"|"ignored"|"no_mutation", "memory_id": "...", "reason": "..."}]}.\n'
        'When multi-thought dependencies matter, also include "cluster_outcomes": [{"entry_ids": [123, 124], "disposition": "created_cluster"|"appended_cluster"|"refactored_cluster"|"linked_cluster"|"ignored_cluster", "memory_ids": ["..."], "reason": "..."}] to explain the grouped reasoning.\n'
        "Do not make vague claims like 'matched existing canonical memories' unless the final JSON includes structured entry_outcomes with entry_id and memory_id for each such match.\n"
        "Provide a reason whenever an entry outcome is ignored, no_mutation, or matched_existing.\n"
    )

# ---------------------------------------------------------------------------
# Main handlers
# ---------------------------------------------------------------------------

async def handle_ingest_system1_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.journal is None or ctx.repository is None:
        return _build_ingest_result(
            created_ids=[],
            claimed_ids=[],
            deleted_ids=[],
            recoverable_ids=[],
            released_ids=[],
            meaningful_actions=0,
        )
    journal = ctx.journal
    preflight = build_ingest_preflight_state(ctx, task, provider)
    if preflight.pending_count_before_run <= 0:
        return {
            **_build_ingest_result(
                created_ids=[],
                claimed_ids=[],
                deleted_ids=[],
                recoverable_ids=[],
                released_ids=[],
                meaningful_actions=0,
            ),
            "requested_grouping_strategy": preflight.requested_grouping_strategy,
            "grouping_strategy_used": preflight.grouping_strategy_used,
            "grouping_fallback_reason": preflight.grouping_fallback_reason,
            "reason": "no_pending_entries",
        }
    provider = preflight.provider

    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        try:
            agentic_pass_result = await run_agentic_ingest_pass(
                ctx,
                task,
                cast(Any, run_agent),
                workspace_id=preflight.workspace_id,
                journal_workspace_id=preflight.journal_workspace_id,
                batch_size=preflight.batch_size,
                grouping_strategy=preflight.grouping_strategy_used,
                max_batches_per_run=preflight.max_batches_per_run,
                pending_count_before_run=preflight.pending_count_before_run,
            )
            if agentic_pass_result is not None:
                return {
                    **agentic_pass_result,
                    "requested_grouping_strategy": preflight.requested_grouping_strategy,
                    "grouping_strategy_used": preflight.grouping_strategy_used,
                    "grouping_fallback_reason": preflight.grouping_fallback_reason,
                }
        except BaseException:
            journal.release_claims(task.id)
            raise

    run_state = IngestRunAccumulator()

    try:
        while run_state.batches_processed < preflight.max_batches_per_run:
            entries = journal.claim_pending(
                task_id=task.id,
                limit=preflight.batch_size,
                workspace_id=preflight.journal_workspace_id,
            )
            if not entries:
                break

            batch_result = await process_ingest_batch(
                ctx,
                task,
                provider,
                workspace_id=preflight.workspace_id,
                grouping_strategy=preflight.grouping_strategy_used,
                analyze_ingest_actions=_analyze_ingest_actions,
                entries=entries,
                batch_sequence=run_state.batches_processed + 1,
            )
            run_state.absorb_batch_result(batch_result)
            pending_remaining = journal.count_by_status(workspace_id=preflight.journal_workspace_id).get("pending", 0)
            if not should_continue_ingest_run(
                batch_meaningful_actions=int(batch_result["meaningful_actions"]),
                pending_remaining=pending_remaining,
            ):
                break

        return run_state.build_handler_result(
            requested_grouping_strategy=preflight.requested_grouping_strategy,
            grouping_strategy_used=preflight.grouping_strategy_used,
            grouping_fallback_reason=preflight.grouping_fallback_reason,
            pending_remaining=journal.count_by_status(workspace_id=preflight.journal_workspace_id).get("pending", 0),
        )
    except BaseException:
        journal.release_claims(task.id)
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
        f"{build_ingest_guardrails()}\n"
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
    normalized_actions = [dict(cast(dict[str, Any], action)) for action in actions if isinstance(action, dict)]
    return normalized_actions, response.mutating_tool_calls


_build_ingest_agent_prompt = build_ingest_agent_prompt


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
def _extract_ingest_actions(response: dict[str, Any]) -> list[dict[str, Any]] | object:
    actions = response.get("actions")
    if isinstance(actions, list):
        return actions
    results = response.get("results")
    if isinstance(results, list):
        return results
    return actions

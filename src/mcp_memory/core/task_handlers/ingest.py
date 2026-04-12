from __future__ import annotations

from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.agentic_guardrails import build_ingest_guardrails
from mcp_memory.core.task_handlers.ingest_agentic_support import (
    INGEST_APPEND_TOOL_NAME as INGEST_APPEND_TOOL_NAME,
    INGEST_CREATE_TOOL_NAME as INGEST_CREATE_TOOL_NAME,
    build_ingest_agent_prompt,
    run_agentic_ingest_pass,
)
from mcp_memory.core.task_handlers.ingest_batch_support import build_ingest_groups
from mcp_memory.core.task_handlers.ingest_batch_support import entry_similarity
from mcp_memory.core.task_handlers.ingest_batch_support import process_ingest_batch
from mcp_memory.core.task_handlers.ingest_claim_batch_support import (
    _requested_grouping_strategy,
    _resolve_grouping_strategy,
)
from mcp_memory.core.task_handlers.ingest_support import (
    _build_ingest_result,
    _normalize_ingest_agentic_result as _normalize_ingest_agentic_result,
)
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_INGEST_BATCH_SIZE,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.task_handlers.workspace_resolution import resolve_task_or_context_workspace_id
from mcp_memory.core.tasks import TaskRecord

DEFAULT_INGEST_MAX_BATCHES_PER_RUN = 8


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

    journal_workspace_id = resolve_pending_workspace_id(
        journal,
        task.data.get("journal_workspace_id", task.workspace_id),
    )
    max_batches_per_run = max(1, int(task.data.get("max_batches_per_run", DEFAULT_INGEST_MAX_BATCHES_PER_RUN)))
    pending_count_before_run = journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    workspace_id = _resolve_workspace_id(ctx, task)
    grouping_strategy_requested = _requested_grouping_strategy(task.data)
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=grouping_strategy_requested,
    )
    if pending_count_before_run <= 0:
        return {
            **_build_ingest_result(
                created_ids=[],
                claimed_ids=[],
                deleted_ids=[],
                recoverable_ids=[],
                released_ids=[],
                meaningful_actions=0,
            ),
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
            agentic_pass_result = await run_agentic_ingest_pass(
                ctx,
                task,
                cast(Any, run_agent),
                workspace_id=workspace_id,
                journal_workspace_id=journal_workspace_id,
                batch_size=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
                grouping_strategy=grouping_strategy_used,
                max_batches_per_run=max_batches_per_run,
                pending_count_before_run=pending_count_before_run,
            )
            if agentic_pass_result is not None:
                return {
                    **agentic_pass_result,
                    "requested_grouping_strategy": grouping_strategy_requested,
                    "grouping_strategy_used": grouping_strategy_used,
                    "grouping_fallback_reason": grouping_fallback_reason,
                }
        except BaseException:
            journal.release_claims(task.id)
            raise

    created_ids: list[str] = []
    claimed_ids: list[int] = []
    recoverable_ids: list[int] = []
    released_ids: list[int] = []
    meaningful_actions = 0
    semantic_entry_dispositions: list[dict[str, Any]] = []
    batches_processed = 0

    try:
        while batches_processed < max_batches_per_run:
            entries = journal.claim_pending(
                task_id=task.id,
                limit=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
                workspace_id=journal_workspace_id,
            )
            if not entries:
                break

            batch_result = await process_ingest_batch(
                ctx,
                task,
                provider,
                workspace_id=workspace_id,
                grouping_strategy=grouping_strategy_used,
                analyze_ingest_actions=_analyze_ingest_actions,
                entries=entries,
            )
            batches_processed += 1
            created_ids.extend(batch_result["created_ids"])
            claimed_ids.extend(batch_result["claimed_ids"])
            recoverable_ids.extend(batch_result["recoverable_ids"])
            released_ids.extend(batch_result["released_ids"])
            semantic_entry_dispositions.extend(batch_result["entry_dispositions"])
            meaningful_actions += batch_result["meaningful_actions"]

            if batch_result["meaningful_actions"] <= 0:
                break
            if journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0) <= 0:
                break

        return _build_ingest_result(
            created_ids=created_ids,
            claimed_ids=sorted(set(claimed_ids)),
            deleted_ids=[],
            recoverable_ids=sorted(set(recoverable_ids)),
            released_ids=sorted(set(released_ids)),
            meaningful_actions=meaningful_actions,
            semantic_entry_dispositions=semantic_entry_dispositions,
        ) | {
            "requested_grouping_strategy": grouping_strategy_requested,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
            "batches_processed": batches_processed,
            "pending_remaining": journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0),
        }
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
_build_ingest_groups = build_ingest_groups


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
def _extract_ingest_actions(response: dict[str, Any]) -> list[dict[str, Any]] | object:
    actions = response.get("actions")
    if isinstance(actions, list):
        return actions
    results = response.get("results")
    if isinstance(results, list):
        return results
    return actions
def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str:
    return resolve_task_or_context_workspace_id(
        ctx,
        task,
        fallback="workspace-unknown",
    ) or "workspace-unknown"

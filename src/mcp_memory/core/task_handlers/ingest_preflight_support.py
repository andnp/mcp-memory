from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.constants import DEFAULT_INGEST_BATCH_SIZE
from mcp_memory.core.task_handlers.ingest_claim_batch_support import (
    _requested_grouping_strategy,
    _resolve_grouping_strategy,
)
from mcp_memory.core.task_handlers.workspace_resolution import resolve_task_or_context_workspace_id
from mcp_memory.core.tasks import TaskRecord


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
    from mcp_memory.core.task_handlers.ingest_batch_support import entry_similarity

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
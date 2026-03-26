from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.agentic_guardrails import build_ingest_guardrails
from mcp_memory.core.task_handlers.ingest_batch_support import build_ingest_groups
from mcp_memory.core.task_handlers.ingest_batch_support import entry_similarity
from mcp_memory.core.task_handlers.ingest_batch_support import process_ingest_batch
from mcp_memory.core.task_handlers.ingest_support import (
    _build_ingest_result,
    _finalize_claimed_ingest_entries,
    _normalize_ingest_agentic_result,
    _recorded_ingest_entry_dispositions,
    _recorded_ingest_handled_entry_ids,
    _recorded_ingest_touched_memory_ids,
    _recorded_ingest_tool_usage,
    _reset_recorded_ingest_handled_entry_ids,
)
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_INGEST_BATCH_SIZE,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord


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
            _reset_recorded_ingest_handled_entry_ids(ctx, task.id)
            agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                _build_ingest_agent_prompt(
                    task,
                    workspace_id=workspace_id,
                    batch_size=int(task.data.get("batch_size", DEFAULT_INGEST_BATCH_SIZE)),
                    grouping_strategy=grouping_strategy_used,
                    max_batches_per_run=max_batches_per_run,
                )
            )
            normalized = _normalize_ingest_agentic_result(agentic_result)
            authoritative_tool_usage = _recorded_ingest_tool_usage(ctx, task.id)
            meaningful_actions = max(normalized["meaningful_actions"], authoritative_tool_usage["mutations"])
            if (
                pending_count_before_run > 0
                and authoritative_tool_usage["tool_calls_executed"] <= 0
                and meaningful_actions <= 0
                and not normalized["created_memory_ids"]
            ):
                journal.release_claims(task.id)
            else:
                semantic_entry_dispositions = _recorded_ingest_entry_dispositions(ctx, task.id)
                handled_entry_ids = _recorded_ingest_handled_entry_ids(ctx, task.id)
                touched_memory_ids = _recorded_ingest_touched_memory_ids(ctx, task.id)
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
                    "requested_grouping_strategy": grouping_strategy_requested,
                    "grouping_strategy_used": grouping_strategy_used,
                    "grouping_fallback_reason": grouping_fallback_reason,
                    "summary": normalized["summary"],
                    "execution_mode": "agentic_mcp",
                    "tool_calls_executed": authoritative_tool_usage["tool_calls_executed"],
                    "mutations": authoritative_tool_usage["mutations"],
                    "tool_names_used": authoritative_tool_usage["tool_names_used"],
                    "provider_reported_tool_calls": normalized["tool_calls_executed"],
                    "provider_reported_mutations": normalized["mutations"],
                    "provider_reported_tool_names_used": normalized["tool_names_used"],
                    "provider_reported_entry_outcomes": normalized["entry_outcomes"],
                    "provider_reported_cluster_outcomes": normalized["cluster_outcomes"],
                    "provider_reported_touched_memory_ids": normalized["touched_memory_ids"],
                    "provider_reported_matched_memory_ids": normalized["matched_memory_ids"],
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
        "Before appending, sanity-check that you can honestly complete the sentence 'This belongs in the target memory because both are fundamentally about ___.' If that blank would be vague or hand-wavy, do not append there.\n"
        "Standing ingest jobs: append into the right canonical memory, create a new narrow memory when novelty warrants it, lightly rewrite or resummarize touched memories when the batch reveals a clearer durable shape, split bloated targets that would become mixed-topic blobs, merge or archive stale leftovers when consolidation makes them obsolete, and clean up links when structure is obviously misleading or incomplete.\n"
        "Treat one claimed batch as a small maintenance campaign, not a one-thought-to-one-memory conveyor belt. Multiple claimed thoughts may belong in one focused memory, and one thought may justify refactoring an existing cluster before the best durable landing spot is clear.\n"
        "Only process journal entries claimed for this task.\n"
        "Use internal_search_memory_records, internal_read_memory_record, and internal_list_memory_records to find append targets before mutating memories.\n"
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
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    if isinstance(ctx.workspace_id, str) and ctx.workspace_id.strip():
        return ctx.workspace_id.strip()
    return "workspace-unknown"

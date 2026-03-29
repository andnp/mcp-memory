from __future__ import annotations

from typing import Any, Awaitable, Callable

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_claim_lifecycle import (
    _finalize_claimed_ingest_entries,
    _recorded_ingest_run_metadata,
    _reset_recorded_ingest_handled_entry_ids,
)
from mcp_memory.core.task_handlers.agentic_guardrails import build_ingest_guardrails
from mcp_memory.core.task_handlers.ingest_support import _build_ingest_result, _normalize_ingest_agentic_result
from mcp_memory.core.tasks import TaskRecord


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
    agentic_result = await run_agent(
        build_ingest_agent_prompt(
            task,
            workspace_id=workspace_id,
            batch_size=batch_size,
            grouping_strategy=grouping_strategy,
            max_batches_per_run=max_batches_per_run,
        )
    )
    normalized = _normalize_ingest_agentic_result(agentic_result)
    recorded_run_metadata = _recorded_ingest_run_metadata(ctx, task.id)
    authoritative_tool_usage = recorded_run_metadata["tool_usage"]
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
        "provider_reported_tool_calls": normalized["tool_calls_executed"],
        "provider_reported_mutations": normalized["mutations"],
        "provider_reported_tool_names_used": normalized["tool_names_used"],
        "provider_reported_entry_outcomes": normalized["entry_outcomes"],
        "provider_reported_cluster_outcomes": normalized["cluster_outcomes"],
        "provider_reported_touched_memory_ids": normalized["touched_memory_ids"],
        "provider_reported_matched_memory_ids": normalized["matched_memory_ids"],
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

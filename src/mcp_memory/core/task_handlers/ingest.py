from __future__ import annotations

from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.agentic_guardrails import build_ingest_guardrails
from mcp_memory.core.task_handlers.ingest_agentic_support import (
    INGEST_APPEND_TOOL_NAME as INGEST_APPEND_TOOL_NAME,
    INGEST_CREATE_TOOL_NAME as INGEST_CREATE_TOOL_NAME,
    build_ingest_agent_prompt,
    run_agentic_ingest_pass,
)
from mcp_memory.core.task_handlers.ingest_batch_support import build_ingest_groups
from mcp_memory.core.task_handlers.ingest_batch_support import process_ingest_batch
from mcp_memory.core.task_handlers.ingest_preflight_support import (
    build_ingest_preflight_state,
)
from mcp_memory.core.task_handlers.ingest_run_support import (
    IngestRunAccumulator,
    should_continue_ingest_run,
)
from mcp_memory.core.task_handlers.ingest_support import (
    _build_ingest_result,
    _normalize_ingest_agentic_result as _normalize_ingest_agentic_result,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord


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
_build_ingest_groups = build_ingest_groups
def _extract_ingest_actions(response: dict[str, Any]) -> list[dict[str, Any]] | object:
    actions = response.get("actions")
    if isinstance(actions, list):
        return actions
    results = response.get("results")
    if isinstance(results, list):
        return results
    return actions

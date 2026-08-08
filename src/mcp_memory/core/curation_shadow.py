"""Run the curator as a direct MCP-tool agent."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_investigation import CURATOR_AGENT_TOOLS
from mcp_memory.core.curation_models import CampaignHypothesis
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.direct_mutation_evidence import DirectMutationOutcome, productive_mutation_count
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    reset_agentic_tool_tracking,
    validate_agentic_tool_tracking_snapshot,
)
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import complete_work_item, release_work_item
from mcp_memory.core.ports.tasks import TaskRecord


async def run_curator_direct_mcp(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    provider: Any,
    mutation_budget: CurationMutationBudget | None = None,
    campaign_hypothesis: CampaignHypothesis | None = None,
    seed_batch: Any,
    sampled_records: list[Any],
    seed_records: list[Any],
    claimed_work_item: Any,
    work_item_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run one curator session with direct access to the mutation MCP tools."""
    agentic_provider = _curator_agentic_provider(ctx, task, provider)
    if agentic_provider is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            execution_mode="curation_direct_mcp",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="campaign_agentic_provider_not_configured",
        )

    opener = getattr(agentic_provider, "open_agent_session", None)
    if not callable(opener):
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            execution_mode="curation_direct_mcp",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="campaign_agentic_session_not_configured",
        )

    budget = mutation_budget or CurationMutationBudget()
    session = None
    reset_agentic_tool_tracking(ctx, task.id, execution_epoch=task.execution_epoch)
    try:
        session = await cast(Callable[..., Awaitable[Any]], opener)(
            allowed_tool_names=CURATOR_AGENT_TOOLS,
        )
        result = await session.run_agent(
            _direct_curator_prompt(
                task=task,
                seed_records=seed_records,
                campaign_hypothesis=campaign_hypothesis,
                mutation_budget=budget,
            )
        )
    except Exception:
        if _work_item_is_running(ctx, claimed_work_item):
            release_work_item(ctx, claimed_work_item.id)
        raise
    finally:
        tool_snapshot = finalize_agentic_tool_tracking(ctx, task.id)
        if session is not None:
            await session.close()

    tool_calls = int(getattr(tool_snapshot, "total_calls", 0))
    actual_mutations = int(getattr(tool_snapshot, "mutating_calls", 0))
    tool_call_ledger = list(getattr(tool_snapshot, "tool_call_ledger", []))
    ledger_validation = validate_agentic_tool_tracking_snapshot(
        tool_snapshot,
        allowed_tool_names=CURATOR_AGENT_TOOLS,
    )
    ledger_valid = bool(ledger_validation["valid"])
    direct_evidence = _direct_evidence_for_task(ctx, task)
    verified_mutations = productive_mutation_count(direct_evidence)
    mutations = verified_mutations if direct_evidence else actual_mutations if ledger_valid else 0
    provider_metadata = _direct_provider_usage_metadata(ctx, task)
    outcome = "applied" if mutations else "no_op"
    if direct_evidence and not ledger_valid:
        outcome = "ledger_invalid"
    elif direct_evidence:
        outcome = _project_direct_evidence_outcome(direct_evidence, actual_mutations)
    elif not ledger_valid:
        outcome = "ledger_invalid"
    if claimed_work_item is not None:
        complete_work_item(ctx, claimed_work_item.id)
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=_direct_curator_summary(result),
        execution_mode="curation_direct_mcp",
        claimed_work_item_count=1 if claimed_work_item is not None else 0,
        tool_calls_executed=tool_calls,
        mutations=mutations,
        actual_mutation_count=actual_mutations,
        verified_mutation_count=verified_mutations,
        tool_call_ledger=tool_call_ledger,
        tool_call_ledger_validation=ledger_validation,
        **provider_metadata,
        curation_outcome=outcome,
        curation_no_op_reason=(
            "tool_call_ledger_invalid" if not ledger_valid else None if mutations else "agent_no_mutations"
        ),
        curation_campaign_result={
            "outcome": outcome,
            "mutation_count": mutations,
            "productive_mutation_count": mutations,
            "verified_mutation_count": verified_mutations,
            "actual_mutation_count": actual_mutations,
            "tool_calls_executed": tool_calls,
            "tool_names_used": list(getattr(tool_snapshot, "tool_names_used", [])),
            "tool_call_ledger": tool_call_ledger,
            "tool_call_ledger_validation": ledger_validation,
            **provider_metadata,
        },
    )


def _direct_evidence_for_task(ctx: ApplicationContext, task: TaskRecord) -> list[Any]:
    repository = getattr(ctx, "direct_mutation_evidence", None)
    reader = getattr(repository, "list_for_execution", None)
    if not callable(reader):
        return []
    return list(cast(Iterable[Any], reader(task.id, task.execution_epoch)))


def _project_direct_evidence_outcome(evidence: list[Any], actual_mutations: int) -> str:
    outcomes = {str(getattr(item, "outcome", "")) for item in evidence}
    if not actual_mutations:
        return "no_op"
    if DirectMutationOutcome.LEDGER_INVALID in outcomes:
        return DirectMutationOutcome.LEDGER_INVALID.value
    if DirectMutationOutcome.MUTATION_EVIDENCE_INVALID in outcomes:
        return DirectMutationOutcome.MUTATION_EVIDENCE_INVALID.value
    if DirectMutationOutcome.APPLIED_UNVERIFIED in outcomes:
        return DirectMutationOutcome.APPLIED_UNVERIFIED.value
    if DirectMutationOutcome.APPLIED_VERIFIED in outcomes:
        return DirectMutationOutcome.APPLIED_VERIFIED.value
    return DirectMutationOutcome.NO_OP.value


def _direct_curator_prompt(
    *,
    task: TaskRecord,
    seed_records: list[Any],
    campaign_hypothesis: CampaignHypothesis | None,
    mutation_budget: CurationMutationBudget,
) -> str:
    records = [
        {
            "memory_id": str(record.id),
            "title": str(record.title)[:200],
            "summary": str(record.summary or "")[:500],
            "memory_type": str(record.type),
            "status": str(record.status),
        }
        for record in seed_records
    ]
    context = {
        "task_id": task.id,
        "campaign_hypothesis": campaign_hypothesis,
        "mutation_budget": {
            "max_accepted_mutations": mutation_budget.max_accepted_mutations,
            "max_proposed_actions": mutation_budget.max_proposed_actions,
        },
        "seed_records": records,
        "available_tools": list(CURATOR_AGENT_TOOLS),
    }
    return (
        "You are the memory curator. Work directly through the listed MCP tools. "
        "There is no planning or submission stage: do not create a plan, do not record "
        "retention decisions, and do not wait for approval. Read records and relationships "
        "as needed, then invoke a mutation tool immediately when a focused improvement is "
        "justified. Omit records that need no change. Never invent memory IDs. Include the "
        f"task_id {task.id!r} in every mutation call. Stay within the mutation budget. "
        "Use archive instead of delete. Return a short JSON summary after tool work with "
        "summary and mutations_attempted fields.\n"
        + json.dumps(context, sort_keys=True, default=str)
    )


def _direct_curator_summary(result: Any) -> str | None:
    parsed = getattr(result, "parsed", None)
    if isinstance(parsed, dict):
        summary = parsed.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary.strip()[:2000]
    raw_text = getattr(result, "raw_text", None)
    return raw_text.strip()[:2000] if isinstance(raw_text, str) and raw_text.strip() else None


def _supports_agentic_provider(provider: Any) -> bool:
    return callable(getattr(provider, "run_agent", None))


def _work_item_is_running(ctx: ApplicationContext, work_item: Any) -> bool:
    if work_item is None or ctx.work_items is None:
        return False
    current = ctx.work_items.get_item(work_item.id)
    return current is not None and str(current.status) == "running"


def _curator_agentic_provider(ctx: ApplicationContext, task: TaskRecord, provider: Any) -> Any:
    candidate = provider if _supports_agentic_provider(provider) else getattr(ctx, "ai_agent_provider", None)
    if not _supports_agentic_provider(candidate):
        return None
    with_usage_context = getattr(candidate, "with_usage_context", None)
    if callable(with_usage_context):
        return with_usage_context(
            task_name=task.task_name,
            task_id=task.id,
            execution_epoch=task.execution_epoch,
            workspace_id=task.workspace_id,
        )
    return candidate


def _direct_provider_usage_metadata(ctx: ApplicationContext, task: TaskRecord) -> dict[str, Any]:
    provider_usage = getattr(ctx, "provider_usage", None)
    list_conversations = getattr(provider_usage, "list_conversations", None)
    if not callable(list_conversations):
        return {"provider_call_count": 0, "provider_calls_used": 0}
    try:
        conversations = cast(Iterable[Any], list_conversations(
            workspace_id=task.workspace_id,
            task_id=task.id,
            limit=200,
        ))
    except TypeError:
        conversations = cast(Iterable[Any], list_conversations(
            workspace_id=task.workspace_id,
            task_name=task.task_name,
            limit=200,
        ))
    selected: dict[str, Any] = {}
    for conversation in conversations:
        if getattr(conversation, "task_id", task.id) != task.id:
            continue
        request_id = getattr(conversation, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            continue
        previous = selected.get(request_id)
        if previous is None or _conversation_order(conversation) > _conversation_order(previous):
            selected[request_id] = conversation

    metadata: dict[str, Any] = {
        "provider_call_count": len(selected),
        "provider_calls_used": len(selected),
    }
    token_fields = (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    for field_name in token_fields:
        values = [getattr(item, field_name, None) for item in selected.values()]
        present_values = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
        if present_values:
            metadata[field_name] = sum(present_values)
    sources = [getattr(item, "token_usage_source", None) for item in selected.values()]
    source = next((value for value in reversed(sources) if isinstance(value, str) and value), None)
    if source is not None:
        metadata["token_usage_source"] = source
    return metadata


def _conversation_order(conversation: Any) -> tuple[int, float, int]:
    return (
        int(getattr(conversation, "attempt", 0) or 0),
        float(getattr(conversation, "completed_at", 0.0) or 0.0),
        int(getattr(conversation, "id", 0) or 0),
    )

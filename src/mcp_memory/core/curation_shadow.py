"""Run the curator as a direct MCP-tool agent."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_investigation import CURATOR_AGENT_TOOLS
from mcp_memory.core.curation_models import CampaignHypothesis
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    reset_agentic_tool_tracking,
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
    reset_agentic_tool_tracking(ctx, task.id)
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
    mutations = int(getattr(tool_snapshot, "mutating_calls", 0))
    outcome = "applied" if mutations else "no_op"
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
        curation_outcome=outcome,
        curation_no_op_reason=None if mutations else "agent_no_mutations",
        curation_campaign_result={
            "outcome": outcome,
            "mutation_count": mutations,
            "productive_mutation_count": mutations,
            "tool_calls_executed": tool_calls,
            "tool_names_used": list(getattr(tool_snapshot, "tool_names_used", [])),
        },
    )


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

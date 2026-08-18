"""Run the curator as a direct MCP-tool agent."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_investigation import CURATOR_AGENT_TOOLS
from mcp_memory.core.curation_models import CampaignHypothesis
from mcp_memory.core.curation_quality import (
    CurationQualityEvidence,
    CurationQualitySampler,
    quality_productive_mutation_count,
)
from mcp_memory.core.curation_quality_inputs import mutations_from_direct_evidence
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.direct_mutation_evidence import DirectMutationOutcome, productive_mutation_count
from mcp_memory.core.ports.curation import (
    CurationRepository,
    CurationRun,
    CurationRunOutcome,
    CurationRunState,
)
from mcp_memory.core.ports.memory import MemoryRepositoryPort
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
    from mcp_memory.core.task_handlers.agentic_tool_tracking import (
        finalize_agentic_tool_tracking,
        reset_agentic_tool_tracking,
        validate_agentic_tool_tracking_snapshot,
    )
    from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
    from mcp_memory.core.task_handlers.maintenance_work_items import complete_work_item, release_work_item

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

    runtime_errors = list(getattr(tool_snapshot, "runtime_errors", []))
    if runtime_errors:
        raise RuntimeError(runtime_errors[0])

    tool_calls = int(getattr(tool_snapshot, "total_calls", 0))
    actual_mutations = int(getattr(tool_snapshot, "mutating_calls", 0))
    tool_call_ledger = list(getattr(tool_snapshot, "tool_call_ledger", []))
    ledger_validation = validate_agentic_tool_tracking_snapshot(
        tool_snapshot,
        allowed_tool_names=CURATOR_AGENT_TOOLS,
    )
    direct_evidence = _direct_evidence_for_task(ctx, task)
    quality_evaluation = _evaluate_direct_quality(
        ctx,
        task,
        direct_evidence,
        campaign_hypothesis=campaign_hypothesis,
    )
    quality_evidence = quality_evaluation.evidence
    verified_mutations = productive_mutation_count(direct_evidence)
    quality_productive_mutations = quality_productive_mutation_count(quality_evidence)
    mutations = verified_mutations
    provider_metadata = _direct_provider_usage_metadata(ctx, task)
    quality_evidence_status = quality_evaluation.status
    quality_evidence_reason = quality_evaluation.reason
    if actual_mutations and not direct_evidence:
        outcome = DirectMutationOutcome.APPLIED_UNVERIFIED.value
        quality_evidence_status = "not_observed"
        quality_evidence_reason = "direct_mutation_evidence_missing"
    elif direct_evidence:
        outcome = _project_direct_evidence_outcome(direct_evidence, actual_mutations)
    else:
        outcome = "no_op"
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
        curation_no_op_reason=None if mutations else "agent_no_mutations",
        curation_campaign_result={
            "outcome": outcome,
            "mutation_count": mutations,
            "productive_mutation_count": quality_productive_mutations,
            "verified_mutation_count": verified_mutations,
            "actual_mutation_count": actual_mutations,
            "tool_calls_executed": tool_calls,
            "tool_names_used": list(getattr(tool_snapshot, "tool_names_used", [])),
            "tool_call_ledger": tool_call_ledger,
            "tool_call_ledger_validation": ledger_validation,
            "quality_evidence": [
                item.model_dump(mode="json") for item in quality_evidence
            ],
            "quality_outcomes": _quality_outcome_counts(quality_evidence),
            "quality_evidence_status": quality_evidence_status,
            "quality_evidence_reason": quality_evidence_reason,
            **provider_metadata,
        },
        quality_evidence=[item.model_dump(mode="json") for item in quality_evidence],
        quality_evidence_status=quality_evidence_status,
        quality_evidence_reason=quality_evidence_reason,
    )


def _direct_evidence_for_task(ctx: ApplicationContext, task: TaskRecord) -> list[Any]:
    repository = getattr(ctx, "direct_mutation_evidence", None)
    reader = getattr(repository, "list_for_execution", None)
    if not callable(reader):
        return []
    return list(cast(Iterable[Any], reader(task.id, task.execution_epoch)))


def _evaluate_direct_quality(
    ctx: ApplicationContext,
    task: TaskRecord,
    evidence: list[Any],
    *,
    campaign_hypothesis: CampaignHypothesis | None,
) -> _DirectQualityEvaluation:
    """Record quality observations without rejecting curator execution."""
    if not evidence:
        return _DirectQualityEvaluation((), "not_applicable")
    quality_repository = getattr(ctx, "curation_quality", None)
    if quality_repository is None:
        return _DirectQualityEvaluation((), "unavailable", "quality_repository_unavailable")
    if ctx.db_manager is None:
        return _DirectQualityEvaluation((), "unavailable", "database_unavailable")
    search_repository = cast(MemoryRepositoryPort | None, ctx.repository)
    if search_repository is None:
        return _DirectQualityEvaluation((), "unavailable", "search_unavailable")
    sampler = CurationQualitySampler(
        db_manager=ctx.db_manager,
        search=search_repository,
        repository=quality_repository,
        candidate_repository=getattr(ctx, "curation", None),
        sample_rate=1.0,
    )
    run = _direct_quality_run(task)
    persisted_run = _persist_direct_quality_run(ctx, run)
    if persisted_run is None:
        return _DirectQualityEvaluation((), "unavailable", "curation_repository_unavailable")
    quality_evidence = sampler.evaluate(
        run=run,
        mutations=[mutations_from_direct_evidence(item) for item in evidence],
        campaign_hypothesis=campaign_hypothesis,
    )
    if not quality_evidence:
        _terminalize_direct_quality_run(ctx, persisted_run, CurationRunOutcome.APPLIED)
        return _DirectQualityEvaluation((), "not_observed", "no_sampled_mutations")
    run_outcome = CurationRunOutcome.APPLIED
    _terminalize_direct_quality_run(ctx, persisted_run, run_outcome)
    return _DirectQualityEvaluation(quality_evidence, "recorded", run_outcome=run_outcome)


def _persist_direct_quality_run(ctx: ApplicationContext, run: CurationRun) -> CurationRun | None:
    repository = cast(CurationRepository | None, getattr(ctx, "curation", None))
    getter = getattr(repository, "get_run", None)
    creator = getattr(repository, "create_run", None)
    if not callable(getter) or not callable(creator):
        return None
    existing = cast(CurationRun | None, getter(run.run_id))
    if existing is not None:
        return existing
    try:
        return cast(CurationRun, creator(run))
    except Exception:
        existing = cast(CurationRun | None, getter(run.run_id))
        if existing is None:
            raise
        return existing


def _terminalize_direct_quality_run(
    ctx: ApplicationContext,
    run: CurationRun,
    outcome: CurationRunOutcome,
) -> None:
    if run.state is CurationRunState.TERMINAL:
        return
    repository = cast(CurationRepository | None, getattr(ctx, "curation", None))
    terminalizer = getattr(repository, "terminalize_run", None)
    if callable(terminalizer):
        terminalizer(run.run_id, run.state, outcome)


def _direct_quality_run(task: TaskRecord) -> CurationRun:
    run_id = uuid5(NAMESPACE_URL, f"mcp-memory:direct-quality-run:{task.id}:{task.execution_epoch}")
    return CurationRun(
        run_id=run_id,
        task_id=_optional_task_uuid(task.id),
        execution_epoch=task.execution_epoch,
        frontier_key=f"direct:{task.id}",
        context_fingerprint=f"direct:{task.id}:{task.execution_epoch}",
        policy_version=str(task.data.get("policy_version", "direct-quality-v1")),
        selector_strategy=str(task.data.get("strategy", "direct")),
        state=CurationRunState.PLANNING,
    )


def _optional_task_uuid(task_id: str) -> UUID | None:
    try:
        return UUID(task_id)
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class _DirectQualityEvaluation:
    """Quality telemetry that never changes the curator execution outcome."""

    evidence: tuple[CurationQualityEvidence, ...]
    status: str
    reason: str | None = None
    run_outcome: CurationRunOutcome = CurationRunOutcome.APPLIED


def _quality_outcome_counts(evidence: Iterable[CurationQualityEvidence]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence:
        counts[item.status] = counts.get(item.status, 0) + 1
    return counts


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
        "Retention/durability rubric: distinguish durable content (reusable facts, decisions, "
        "deadlines, releases, incidents, or historical context), transient content, and mixed "
        "content. Treat dates as semantic only when they carry deadline, release, incident, "
        "historical, or decision meaning; preserve meaningful dates and qualifiers. Flag work "
        "logs, status updates, task-complete summaries, and execution residue as advisory cleanup "
        "candidates. Never archive or delete solely due to age, date, or access. When rewriting "
        "or splitting, preserve every durable claim and its meaningful qualifiers. "
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

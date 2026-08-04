"""Curator rollout routing through the verified curation campaign."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_context import AcceptedMaintenanceRead, CurationReadBudget
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_feedback import (
    _feedback_termination_reason,
    _quality_feedback_payload,
)
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier, CurationHarnessConfig
from mcp_memory.core.curation_investigation import (
    CurationInvestigationLimits,
    CurationInvestigationResult,
    READ_ONLY_CURATOR_INVESTIGATION_TOOLS,
    run_curator_investigation,
)
from mcp_memory.core.curation_planner import (
    CurationPlanSubmissionBuffer,
    IncrementalCurationPlanState,
    SessionCurationPlanner,
)
from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.curation_models import (
    CampaignHypothesis,
    CurationAction,
    CurationPlan,
    CurationRunOutcome,
    RetentionDecision,
)
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.curator_support import (
    curator_quality_feedback,
    curator_seed_payload_item,
)
from mcp_memory.core.task_handlers.maintenance_work_items import release_work_item
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.mutation_history import ProtectionMode, is_protection_active
from mcp_memory.curation_quality_store import (
    PostgresCurationQualityStore,
    SQLiteCurationQualityStore,
)


CURATOR_MAX_FEEDBACK_ITERATIONS = 2
CURATOR_MAX_CONTEXT_CHARACTERS = 24_000


async def run_curator_verified_campaign(
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
    investigation_limits: CurationInvestigationLimits | None = None,
) -> dict[str, Any]:
    """Plan and execute the canonical verified curator campaign path."""
    action_store = getattr(ctx, "curation_action_store", None)
    if action_store is None or ctx.curation is None or ctx.relational_search is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise RuntimeError("verified campaign storage is unavailable")

    agentic_provider = _curator_agentic_provider(ctx, task, provider)
    if agentic_provider is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            execution_mode="curation_verified_campaign",
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
            execution_mode="curation_verified_campaign",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="campaign_agentic_session_not_configured",
        )

    session = None
    plan_state = IncrementalCurationPlanState()
    session = await cast(Callable[..., Awaitable[Any]], opener)(
        allowed_tool_names=READ_ONLY_CURATOR_INVESTIGATION_TOOLS,
        tools=_incremental_curation_tools(plan_state),
    )
    investigation = (
        await run_curator_investigation(
            agentic_provider,
            seed_records=seed_records,
            task_id=task.id,
            limits=investigation_limits,
            session=session,
        )
    )
    exploratory_records, exploratory_reads = _investigated_reads(
        ctx, investigation, seed_records
    )
    context_records = [*seed_records, *exploratory_records]

    planner = SessionCurationPlanner(
        session,
        plan_state=plan_state,
        provider_key=str(getattr(agentic_provider, "_provider_key", "curation-session")),
        provider_name=str(getattr(agentic_provider, "_provider_name", "curation-session")),
        model_name=str(getattr(agentic_provider, "_model_name", "curation-session")),
    )
    support_records = seed_records[len(sampled_records) :] if len(seed_records) > len(sampled_records) else []
    frontier_args = {
        "family": "curator",
        "strategy": seed_batch.strategy_used,
        "seed_reads": (
            _record_read(
                record,
                edges=_record_edges(ctx, record),
                selection_reason=seed_batch.strategy_selection_reason,
                selection_signals=_strategy_signals(seed_batch),
                selection_scores=seed_batch.strategy_selection_scores,
                quality_feedback=curator_quality_feedback(ctx, record),
            )
            for record in sampled_records
        ),
        "support_reads": (_record_read(record, edges=_record_edges(ctx, record)) for record in support_records),
        "exploratory_reads": exploratory_reads,
        "task_id": _task_uuid(task.id),
        "campaign_hypothesis": campaign_hypothesis,
    }
    frontier = (
        CurationFrontier.claimed(claimed_work_item.id, **frontier_args)
        if claimed_work_item is not None
        else CurationFrontier.direct(**frontier_args)
    )
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(
        ctx, agentic_provider, context_records
    )
    memory_types = {UUID(record.id): record.type for record in context_records}
    configured_budget = mutation_budget or CurationMutationBudget()
    plan_state.max_proposed_actions = configured_budget.max_proposed_actions
    cumulative_proposed = 0
    cumulative_accepted = 0
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            read_budget=CurationReadBudget(
                max_context_characters=CURATOR_MAX_CONTEXT_CHARACTERS,
            ),
            mutation_budget=configured_budget,
            execute_accepted_actions=True,
            require_authoritative_disclosure_context=require_policy,
        ),
        memory_types=memory_types,
        protections_by_memory=protections,
        sensitive_fields_by_memory=sensitive_fields,
        executor=CurationExecutor(action_store),
        verifier=CurationVerifier(ctx.curation, ctx.relational_search),
        quality_sampler=(
            None
            if ctx.db_manager is None
            else CurationQualitySampler(
                db_manager=ctx.db_manager,
                search=ctx.relational_search,
                repository=(
                    SQLiteCurationQualityStore(ctx.db_manager)
                    if hasattr(ctx.db_manager, "get_connection")
                    else PostgresCurationQualityStore(ctx.db_manager)
                ),
                candidate_repository=ctx.curation,
                sample_rate=1.0,
            )
        ),
    )
    try:
        result = await harness.run(frontier)
        cumulative_tool_calls = _authoritative_read_tool_calls(result)
        correction_turns = 0
        termination_reason = "initial_result"
        while (
            session is not None
            and correction_turns < CURATOR_MAX_FEEDBACK_ITERATIONS
        ):
            cumulative_proposed += _proposed_action_count(result)
            cumulative_accepted += int(
                getattr(result.result, "verified_action_count", len(result.result.receipts))
            )
            termination_reason = _feedback_termination_reason(result)
            if termination_reason is not None:
                break
            remaining_budget = _remaining_mutation_budget(
                configured_budget,
                cumulative_proposed=cumulative_proposed,
                cumulative_accepted=cumulative_accepted,
            )
            if remaining_budget.max_accepted_mutations == 0 or remaining_budget.max_proposed_actions == 0:
                termination_reason = "cumulative_budget_exhausted"
                break
            feedback = _quality_feedback_payload(result)
            cast(Any, planner).set_quality_feedback(feedback)
            harness._config.mutation_budget = remaining_budget
            investigation = await run_curator_investigation(
                agentic_provider,
                seed_records=seed_records,
                task_id=task.id,
                limits=investigation_limits,
                session=session,
            )
            newly_read_records, newly_read_context = _investigated_reads(
                ctx, investigation, seed_records
            )
            exploratory_records = list(
                {
                    record.id: record
                    for record in [*exploratory_records, *newly_read_records]
                }.values()
            )
            exploratory_reads = tuple(
                {
                    str(read.record["id"]): read
                    for read in [*exploratory_reads, *newly_read_context]
                }.values()
            )
            refreshed = _refresh_records(ctx, [*seed_records, *exploratory_records])
            if refreshed:
                frontier = _frontier_with_refreshed_records(
                    frontier,
                    refreshed,
                    sampled_count=len(sampled_records),
                    exploratory_reads=exploratory_reads,
                    retain_work_item=False,
                )
            else:
                frontier = _frontier_with_refreshed_records(
                    frontier,
                    [*seed_records, *exploratory_records],
                    sampled_count=len(sampled_records),
                    exploratory_reads=exploratory_reads,
                    retain_work_item=False,
                )
            for record in newly_read_records:
                memory_types[UUID(record.id)] = record.type
            next_result = await harness.run(frontier)
            cumulative_tool_calls += _authoritative_read_tool_calls(next_result)
            correction_turns += 1
            if (
                next_result.result.outcome is CurationRunOutcome.NO_OP
                and not next_result.result.receipts
            ):
                termination_reason = "no_useful_work"
                break
            result = next_result
        if correction_turns == CURATOR_MAX_FEEDBACK_ITERATIONS:
            termination_reason = (
                _feedback_termination_reason(result) or "iteration_limit"
            )
    except Exception:
        if _work_item_is_running(ctx, claimed_work_item):
            release_work_item(ctx, claimed_work_item.id)
        if session is not None:
            await session.close()
        raise
    if session is not None:
        await session.close()
    plan = None if result.validation is None else result.validation.plan
    decisions = _curator_decisions(result)
    campaign_result = _curator_campaign_result(result)
    no_op_reason = _curation_no_op_reason(result.result.outcome, plan, investigation)
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=None if plan is None else plan.rationale,
        execution_mode="curation_verified_campaign",
        claimed_work_item_count=1 if claimed_work_item is not None else 0,
        tool_calls_executed=cumulative_tool_calls,
        mutations=campaign_result["mutation_count"],
        curation_run_id=str(result.run.run_id),
        curation_plan_id=str(result.result.plan_id),
        curation_outcome=str(result.result.outcome),
        curation_work_item_action=str(result.work_item.action),
        curation_rejection_codes=result.result.rejection_codes,
        curation_planner_attempts=result.planner_attempts,
        curation_failure_details=result.result.failure_details,
        curation_plan=None if plan is None else plan.model_dump(mode="json"),
        curation_decisions=decisions,
        curation_campaign_result=campaign_result,
        curation_no_op_reason=no_op_reason,
        curation_investigation={
            "status": investigation.status,
            "rounds": investigation.rounds,
            "tool_calls": investigation.tool_calls,
            "record_ids": list(investigation.record_ids),
            "record_count": len(exploratory_reads),
            "reason": investigation.reason,
        },
        curation_iterations=correction_turns,
        curation_iteration_termination=termination_reason,
        curation_cumulative_budget={
            "proposed_actions": cumulative_proposed + _proposed_action_count(result),
            "accepted_mutations": cumulative_accepted
            + int(getattr(result.result, "verified_action_count", len(result.result.receipts))),
            "max_proposed_actions": configured_budget.max_proposed_actions,
            "max_accepted_mutations": configured_budget.max_accepted_mutations,
        },
    )


def _proposed_action_count(result: Any) -> int:
    plan = getattr(getattr(result, "validation", None), "plan", None)
    return 0 if plan is None else len(plan.actions)


def _authoritative_read_tool_calls(result: Any) -> int:
    budget_usage = getattr(getattr(result, "result", None), "budget_usage", None)
    return int(getattr(budget_usage, "read_tool_calls", 0))


def _remaining_mutation_budget(
    budget: CurationMutationBudget,
    *,
    cumulative_proposed: int,
    cumulative_accepted: int,
) -> CurationMutationBudget:
    return CurationMutationBudget(
        max_proposed_actions=max(budget.max_proposed_actions - cumulative_proposed, 0),
        max_accepted_mutations=max(budget.max_accepted_mutations - cumulative_accepted, 0),
    )


def _refresh_records(ctx: ApplicationContext, records: list[Any]) -> list[Any]:
    peek = getattr(getattr(ctx, "relational_search", None), "peek_memory", None)
    if not callable(peek):
        return []
    refreshed: list[Any] = []
    for record in records:
        current = peek(str(record.id))
        if current is not None:
            refreshed.append(cast(Any, current).record)
    return refreshed


def _frontier_with_refreshed_records(
    frontier: CurationFrontier,
    records: list[Any],
    *,
    sampled_count: int,
    exploratory_reads: tuple[AcceptedMaintenanceRead, ...] | None = None,
    retain_work_item: bool = True,
) -> CurationFrontier:
    sampled = records[:sampled_count]
    support = records[sampled_count:]
    args = {
        "family": frontier.family,
        "strategy": frontier.strategy,
        "seed_reads": tuple(_record_read(record) for record in sampled),
        "support_reads": tuple(_record_read(record) for record in support),
        "exploratory_reads": (
            frontier.exploratory_reads
            if exploratory_reads is None
            else exploratory_reads
        ),
        "task_id": frontier.task_id,
        "campaign_hypothesis": frontier.campaign_hypothesis,
    }
    return (
        CurationFrontier.claimed(frontier.work_item_id, **args)
        if retain_work_item and frontier.work_item_id is not None
        else CurationFrontier.direct(**args)
    )


def _curation_no_op_reason(
    outcome: CurationRunOutcome,
    plan: Any,
    investigation: CurationInvestigationResult,
) -> str | None:
    if outcome is CurationRunOutcome.NO_OP:
        if investigation.status != "completed":
            return "investigation_unavailable"
        return "planner_retained" if plan is not None and not plan.actions else "execution_no_op"
    if outcome in {
        CurationRunOutcome.INVALID_PLAN,
        CurationRunOutcome.STALE_PLAN,
        CurationRunOutcome.VERIFICATION_FAILED,
    }:
        return "validation_rejected"
    if outcome is CurationRunOutcome.PROVIDER_FAILED:
        return "provider_failed"
    return None


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


def _curation_plan_submission_tool(buffer: CurationPlanSubmissionBuffer) -> Any:
    from copilot.tools import Tool, ToolInvocation, ToolResult

    def submit(invocation: ToolInvocation) -> ToolResult:
        buffer.submit(invocation.arguments)
        return ToolResult(text_result_for_llm="Curation plan captured for validation.")

    return Tool(
        name="submit_curation_plan",
        description="Submit one complete typed curation plan for application validation.",
        parameters={
            "type": "object",
            "properties": {
                "plan": CurationPlan.model_json_schema(),
                "override_confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "override_reason": {"type": ["string", "null"]},
            },
            "required": ["plan"],
        },
        handler=submit,
        skip_permission=True,
        defer="never",
    )


def _incremental_curation_tools(state: IncrementalCurationPlanState) -> list[Any]:
    from copilot.tools import Tool, ToolInvocation, ToolResult

    action_adapter = TypeAdapter(CurationAction)

    def propose(invocation: ToolInvocation) -> ToolResult:
        arguments = invocation.arguments
        payload = arguments.get("action") if isinstance(arguments, dict) else None
        try:
            action = action_adapter.validate_python(payload)
            state.add_action(action)
        except ValidationError as error:
            return ToolResult(
                text_result_for_llm=f"Action rejected as structurally invalid: {error}"
            )
        except (RuntimeError, ValueError) as error:
            return ToolResult(text_result_for_llm=f"Action not added: {error}")
        return ToolResult(text_result_for_llm="Curation action added to the pending plan.")

    def replace(invocation: ToolInvocation) -> ToolResult:
        arguments = invocation.arguments
        payload = arguments.get("action") if isinstance(arguments, dict) else None
        try:
            action = action_adapter.validate_python(payload)
            state.replace_action(action)
        except ValidationError as error:
            return ToolResult(
                text_result_for_llm=f"Replacement rejected as structurally invalid: {error}"
            )
        except (RuntimeError, ValueError) as error:
            return ToolResult(text_result_for_llm=f"Replacement not applied: {error}")
        return ToolResult(text_result_for_llm="Curation action replaced in the pending plan.")

    def remove(invocation: ToolInvocation) -> ToolResult:
        arguments = invocation.arguments
        raw_action_id = arguments.get("action_id") if isinstance(arguments, dict) else None
        try:
            state.remove_action(UUID(str(raw_action_id)))
        except (RuntimeError, ValueError) as error:
            return ToolResult(text_result_for_llm=f"Action not removed: {error}")
        return ToolResult(text_result_for_llm="Curation action removed from the pending plan.")

    def retain(invocation: ToolInvocation) -> ToolResult:
        arguments = invocation.arguments
        payload = arguments.get("decision") if isinstance(arguments, dict) else None
        try:
            decision = RetentionDecision.model_validate(payload)
            state.add_retention(decision)
        except ValidationError as error:
            return ToolResult(
                text_result_for_llm=f"Retention decision rejected as invalid: {error}"
            )
        except (RuntimeError, ValueError) as error:
            return ToolResult(text_result_for_llm=f"Retention decision not added: {error}")
        return ToolResult(text_result_for_llm="Retention decision added to the pending plan.")

    action_schema = TypeAdapter(CurationAction).json_schema()
    return [
        Tool(
            name="propose_curation_action",
            description="Add one typed mutation action to the pending curation plan.",
            parameters={
                "type": "object",
                "properties": {"action": action_schema},
                "required": ["action"],
            },
            handler=propose,
            skip_permission=True,
            defer="never",
        ),
        Tool(
            name="replace_curation_action",
            description="Replace one previously proposed action using the same action_id.",
            parameters={
                "type": "object",
                "properties": {"action": action_schema},
                "required": ["action"],
            },
            handler=replace,
            skip_permission=True,
            defer="never",
        ),
        Tool(
            name="remove_curation_action",
            description="Remove one previously proposed action by action_id.",
            parameters={
                "type": "object",
                "properties": {"action_id": {"type": "string", "format": "uuid"}},
                "required": ["action_id"],
            },
            handler=remove,
            skip_permission=True,
            defer="never",
        ),
        Tool(
            name="retain_curation_memory",
            description="Record one seed memory as intentionally retained without mutation.",
            parameters={
                "type": "object",
                "properties": {"decision": RetentionDecision.model_json_schema()},
                "required": ["decision"],
            },
            handler=retain,
            skip_permission=True,
            defer="never",
        ),
    ]


def _disclosure_context(
    ctx: ApplicationContext,
    provider: Any,
    records: list[Any],
) -> tuple[
    ProviderTrust,
    dict[UUID, frozenset[ProtectionMode]] | None,
    dict[UUID, frozenset[str]] | None,
    bool,
]:
    """Resolve provider and policy facts before constructing provider context."""

    raw_trust = next(
        (
            getattr(provider, name, None)
            for name in ("provider_trust_class", "_provider_trust_class", "trust_class", "_trust_class")
            if getattr(provider, name, None) is not None
        ),
        None,
    )
    underlying = getattr(provider, "_provider", None)
    if raw_trust is None and underlying is not None:
        raw_trust = next(
            (
                getattr(underlying, name, None)
                for name in ("provider_trust_class", "trust_class")
                if getattr(underlying, name, None) is not None
            ),
            None,
        )
    try:
        trust_class = ProviderTrustClass(str(raw_trust))
    except (TypeError, ValueError):
        # An unannotated route is not safe to treat as local.  This also makes
        # missing provider metadata visible as a disclosure denial.
        return ProviderTrust(ProviderTrustClass.EXTERNAL, allowlisted=False), None, None, True

    raw_allowlisted = next(
        (
            getattr(provider, name, None)
            for name in ("provider_allowlisted", "_provider_allowlisted")
            if getattr(provider, name, None) is not None
        ),
        None,
    )
    if raw_allowlisted is None and underlying is not None:
        raw_allowlisted = getattr(underlying, "provider_allowlisted", None)
    allowlisted = (
        trust_class is ProviderTrustClass.LOCAL
        if raw_allowlisted is None
        else bool(raw_allowlisted)
    )
    provider_trust = ProviderTrust(trust_class, allowlisted=allowlisted)
    history = getattr(ctx, "mutation_history", None)
    if history is None:
        return provider_trust, None, None, True

    protections: dict[UUID, frozenset[ProtectionMode]] = {}
    sensitive_fields: dict[UUID, frozenset[str]] = {}
    now = datetime.now(UTC)
    try:
        for record in records:
            memory_id = UUID(str(record.id))
            active_modes = set()
            for protection in history.get_protections(memory_id):
                expires_at = getattr(protection, "expires_at", None)
                if is_protection_active(expires_at, now=now):
                    mode = getattr(protection, "mode", None)
                    if mode is not None:
                        active_modes.add(ProtectionMode(str(mode)))
            protections[memory_id] = frozenset(active_modes)
            metadata = getattr(record, "metadata", {})
            raw_fields = metadata.get("sensitive_fields", ()) if isinstance(metadata, dict) else ()
            sensitive_fields[memory_id] = frozenset(
                field for field in raw_fields if isinstance(field, str) and field.strip()
            ) if isinstance(raw_fields, (list, tuple, set, frozenset)) else frozenset()
    except (AttributeError, TypeError, ValueError):
        return provider_trust, None, None, True
    if trust_class is ProviderTrustClass.LOCAL:
        return provider_trust, protections, None, False
    return provider_trust, protections, sensitive_fields, True


def _record_read(
    record: Any,
    *,
    edges: tuple[dict[str, Any], ...] = (),
    selection_reason: str | None = None,
    selection_signals: dict[str, float] | None = None,
    selection_scores: dict[str, float] | None = None,
    quality_feedback: dict[str, Any] | None = None,
) -> AcceptedMaintenanceRead:
    selection = curator_seed_payload_item(
        record,
        selection_reason=selection_reason,
        selection_signals=selection_signals,
        selection_scores=selection_scores,
        quality_feedback=quality_feedback,
    )
    return AcceptedMaintenanceRead(
        record={
            "id": record.id,
            "title": record.title,
            "content": record.content,
            "summary": record.summary,
            "type": record.type,
            "status": record.status,
            "tags": list(record.tags),
            "workspace_ids": list(getattr(record, "workspace_ids", ())),
            "metadata": {
                **dict(getattr(record, "metadata", {})),
                "curation_read_source": "authoritative",
                "mutation_eligible": True,
            },
            **{
                key: selection[key]
                for key in (
                    "read_count",
                    "last_surfaced_at",
                    "content_size_chars",
                    "size_band",
                    "oversized_for_curator",
                    "retrieval_friction_flags",
                    "selection_reason",
                    "selection_signals",
                    "selection_scores",
                    "quality_feedback",
                )
            },
        },
        edges=edges,
    )


def _record_edges(ctx: ApplicationContext, record: Any) -> tuple[dict[str, Any], ...]:
    search = getattr(ctx, "relational_search", None)
    peek = getattr(search, "peek_memory", None)
    if not callable(peek):
        return ()
    result = peek(str(record.id))
    return () if result is None else _relationship_edges(result)


def _relationship_edges(result: Any) -> tuple[dict[str, Any], ...]:
    relationships = getattr(result, "relationships", {})
    edges: list[dict[str, Any]] = []
    if not isinstance(relationships, dict):
        return ()
    for direction, links in relationships.items():
        if not isinstance(links, (list, tuple)):
            continue
        for link in links:
            edges.append(
                {
                    "direction": direction,
                    "link": {
                        "source_id": link.source_id,
                        "target_id": link.target_id,
                        "type": link.link_type,
                        "context": link.context,
                    },
                }
            )
    return tuple(edges)


def _investigated_reads(
    ctx: ApplicationContext,
    investigation: CurationInvestigationResult,
    seed_records: list[Any],
) -> tuple[list[Any], tuple[AcceptedMaintenanceRead, ...]]:
    if investigation.status != "completed":
        return [], ()
    search = getattr(ctx, "relational_search", None)
    peek = getattr(search, "peek_memory", None)
    if not callable(peek):
        return [], ()
    seed_ids = {str(record.id) for record in seed_records}
    records: list[Any] = []
    reads: list[AcceptedMaintenanceRead] = []
    for raw_id in investigation.record_ids:
        memory_id = _resolve_memory_id(ctx, raw_id)
        if memory_id is None or memory_id in seed_ids:
            continue
        result = cast(Any, peek(memory_id))
        if result is None:
            continue
        record = result.record
        records.append(record)
        reads.append(_record_read(record, edges=_relationship_edges(result)))
    return records, tuple(reads)


def _resolve_memory_id(ctx: ApplicationContext, value: str) -> str | None:
    try:
        return str(UUID(value))
    except ValueError:
        resolver = getattr(getattr(ctx, "repository", None), "resolve_memory_id", None)
        resolved = resolver(value) if callable(resolver) else None
        return None if resolved is None else str(resolved)


def _strategy_signals(seed_batch: Any) -> dict[str, float]:
    snapshot = getattr(seed_batch, "selector_feature_snapshot", None)
    if not isinstance(snapshot, dict):
        return {}
    signals = snapshot.get("strategy_signals")
    return dict(signals) if isinstance(signals, dict) else {}


def _task_uuid(task_id: str) -> UUID | None:
    try:
        return UUID(task_id)
    except ValueError:
        return None


def _curator_decisions(result: Any) -> list[dict[str, Any]]:
    validation = getattr(result, "validation", None)
    if validation is None:
        return []
    decisions = [
        {
            "action_id": str(item.action.action_id),
            "operation": item.action.operation,
            "decision": "accepted",
            "primary_family": str(item.family),
        }
        for item in validation.accepted_actions
    ]
    decisions.extend(
        {
            "action_id": str(item.action.action_id),
            "operation": item.action.operation,
            "decision": "rejected",
            "reason_codes": [str(code) for code in item.reason_codes],
        }
        for item in validation.rejected_actions
    )
    decisions.extend(
        {
            "action_id": str(item.action.action_id),
            "operation": item.action.operation,
            "decision": "specialist_route",
            "primary_family": str(item.family),
            "reason_codes": [str(item.reason_code)],
        }
        for item in validation.specialist_routes
    )
    return decisions


def _curator_campaign_result(result: Any) -> dict[str, Any]:
    validation = getattr(result, "validation", None)
    plan = None if validation is None else validation.plan
    receipts = [receipt.model_dump(mode="json") for receipt in getattr(result.result, "receipts", ())]
    specialist_routes: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    if validation is not None:
        for route in validation.specialist_routes:
            family = str(route.family)
            family_counts[family] = family_counts.get(family, 0) + 1
            specialist_routes.append(
                {
                    "action_id": str(route.action.action_id),
                    "operation": route.action.operation,
                    "primary_family": family,
                    "reason_code": str(route.reason_code),
                }
            )
    return {
        **result.result.model_dump(mode="json"),
        "receipts": receipts,
        "mutation_count": int(getattr(result.result, "verified_action_count", 0)),
        "verification_failure_count": sum(
            1 for receipt in getattr(result.result, "receipts", ()) if str(receipt.status) == "verification_failed"
        ),
        "retention_count": 0 if plan is None else len(plan.retained),
        "no_op_count": 0 if plan is None or str(result.result.outcome) != "no_op" else len(plan.retained),
        "specialist_family_routing": {
            "route_count": len(specialist_routes),
            "family_counts": family_counts,
            "routes": specialist_routes,
            "work_item_ids": [str(item.id) for item in getattr(result, "specialist_work_items", ())],
        },
    }

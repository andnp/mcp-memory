"""Curator rollout routing through the verified curation campaign."""

from __future__ import annotations

from datetime import UTC, datetime
from collections.abc import Awaitable, Callable
from typing import Any, cast
from uuid import UUID

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_context import AcceptedMaintenanceRead
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier, CurationHarnessConfig
from mcp_memory.core.curation_investigation import (
    CurationInvestigationLimits,
    CurationInvestigationResult,
    READ_ONLY_CURATOR_INVESTIGATION_TOOLS,
    run_curator_investigation,
)
from mcp_memory.core.curation_planner import InstrumentedCurationPlanner, SessionCurationPlanner
from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.curation_models import CampaignHypothesis, CurationRunOutcome
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
    planner_provider = _curator_json_provider(ctx, task, provider)

    action_store = getattr(ctx, "curation_action_store", None)
    if action_store is None or ctx.curation is None or ctx.relational_search is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise RuntimeError("verified campaign storage is unavailable")

    investigator = _curator_agentic_provider(ctx, task, provider)
    session = None
    if investigator is not None:
        opener = getattr(investigator, "open_agent_session", None)
        if callable(opener):
            session = await cast(Callable[..., Awaitable[Any]], opener)(
                allowed_tool_names=READ_ONLY_CURATOR_INVESTIGATION_TOOLS
            )
    if session is None and (planner_provider is None or not _supports_json_planning(planner_provider)):
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
            reason="campaign_json_provider_not_configured",
        )
    investigation = (
        await run_curator_investigation(
            investigator,
            seed_records=seed_records,
            task_id=task.id,
            limits=investigation_limits,
            session=session,
        )
        if investigator is not None
        else CurationInvestigationResult("skipped", reason="agentic_provider_unavailable")
    )
    exploratory_records, exploratory_reads = _investigated_reads(
        ctx, investigation, seed_records
    )
    context_records = [*seed_records, *exploratory_records]

    if session is not None:
        planner = SessionCurationPlanner(
            session,
            provider_key=str(getattr(planner_provider, "_provider_key", "curation-session")),
            provider_name=str(getattr(planner_provider, "_provider_name", "curation-session")),
            model_name=str(getattr(planner_provider, "_model_name", "curation-session")),
        )
    else:
        planner = InstrumentedCurationPlanner(
            planner_provider,
            provider_key=str(getattr(planner_provider, "_provider_key", "curation-executor")),
            provider_name=str(getattr(planner_provider, "_provider_name", "curation-executor")),
            model_name=str(getattr(planner_provider, "_model_name", "curation-executor")),
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
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(ctx, planner_provider, context_records)
    memory_types = {UUID(record.id): record.type for record in context_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            mutation_budget=mutation_budget or CurationMutationBudget(),
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
        restore_action_store=action_store,
        mutation_history=getattr(ctx, "mutation_history", None),
    )
    try:
        result = await harness.run(frontier)
        correction_turns = 0
        while (
            session is not None
            and result.result.outcome is CurationRunOutcome.QUALITY_REJECTED
            and correction_turns < 2
        ):
            correction_turns += 1
            feedback = _quality_feedback_payload(result)
            cast(Any, planner).set_quality_feedback(feedback)
            refreshed = _refresh_records(ctx, [*seed_records, *exploratory_records])
            if refreshed:
                frontier = _frontier_with_refreshed_records(
                    frontier,
                    refreshed,
                    sampled_count=len(sampled_records),
                )
            result = await harness.run(frontier)
    except Exception:
        if claimed_work_item is not None:
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
        tool_calls_executed=0,
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
    )


def _quality_feedback_payload(result: Any) -> dict[str, object]:
    evidence = result.result.quality_evidence
    return {
        "latest_failed_live_run": {
            "diagnosis": "normalize+split campaign, retrieval utility delta -1.2, one top-k retrieval loss, no content gain, mixed-wave restore unsupported",
            "evidence": evidence,
        },
        "required_response": [
            "Challenge weak split evidence.",
            "Preserve exact search anchors and concrete entities.",
            "Avoid speculative multi-action waves.",
            "Use measured feedback to change strategy rather than repeat.",
        ],
    }


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
) -> CurationFrontier:
    sampled = records[:sampled_count]
    support = records[sampled_count:]
    args = {
        "family": frontier.family,
        "strategy": frontier.strategy,
        "seed_reads": tuple(_record_read(record) for record in sampled),
        "support_reads": tuple(_record_read(record) for record in support),
        "exploratory_reads": frontier.exploratory_reads,
        "task_id": frontier.task_id,
        "campaign_hypothesis": frontier.campaign_hypothesis,
    }
    return (
        CurationFrontier.claimed(frontier.work_item_id, **args)
        if frontier.work_item_id is not None
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


def _supports_json_planning(provider: Any) -> bool:
    if not callable(getattr(provider, "ask_json", None)):
        return False
    supports_agentic = getattr(provider, "supports_agentic", None)
    return not callable(supports_agentic) or not bool(supports_agentic())


def _supports_agentic_investigation(provider: Any) -> bool:
    return callable(getattr(provider, "run_agent", None))


def _curator_agentic_provider(ctx: ApplicationContext, task: TaskRecord, provider: Any) -> Any:
    candidate = provider if _supports_agentic_investigation(provider) else getattr(ctx, "ai_agent_provider", None)
    if not _supports_agentic_investigation(candidate):
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


def _curator_json_provider(ctx: ApplicationContext, task: TaskRecord, provider: Any) -> Any:
    """Prefer the task route and bind the registry fallback to this execution."""
    if _supports_json_planning(provider):
        return provider

    planner_provider = getattr(ctx, "ai_json_provider", None)
    if not _supports_json_planning(planner_provider):
        return planner_provider

    with_usage_context = getattr(planner_provider, "with_usage_context", None)
    if callable(with_usage_context):
        return with_usage_context(
            task_name=task.task_name,
            task_id=task.id,
            execution_epoch=task.execution_epoch,
            workspace_id=task.workspace_id,
        )
    return planner_provider


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
            "metadata": dict(getattr(record, "metadata", {})),
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

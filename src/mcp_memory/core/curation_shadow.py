"""Curator rollout routing through the planning-only curation harness."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_context import AcceptedMaintenanceRead
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier, CurationHarnessConfig
from mcp_memory.core.curation_planner import InstrumentedCurationPlanner
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import release_work_item
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.mutation_history import ProtectionMode, is_protection_active


def curator_shadow_mode_enabled(ctx: ApplicationContext) -> bool:
    """Return the explicit, opt-in curator shadow policy."""

    config = getattr(ctx, "config", None)
    curation = getattr(config, "curation", None)
    return bool(getattr(curation, "shadow_mode_enabled", False))


def curator_normalize_execution_enabled(ctx: ApplicationContext) -> bool:
    """Return the explicit production gate for normalize execution."""

    config = getattr(ctx, "config", None)
    curation = getattr(config, "curation", None)
    return bool(getattr(curation, "normalize_execution_enabled", False))


def curator_create_link_execution_enabled(ctx: ApplicationContext) -> bool:
    """Return the independent production gate for create-link execution."""

    config = getattr(ctx, "config", None)
    curation = getattr(config, "curation", None)
    return bool(getattr(curation, "create_link_execution_enabled", False))


async def run_curator_shadow_mode(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    provider: Any,
    seed_batch: Any,
    sampled_records: list[Any],
    seed_records: list[Any],
    claimed_work_item: Any,
    work_item_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Plan one curator frontier without invoking a mutation executor."""

    planner_provider = _curator_json_provider(ctx, task, provider)
    if planner_provider is None or not _supports_json_planning(planner_provider):
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            shadow_mode=True,
            execution_mode="curation_shadow_harness",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="shadow_json_provider_not_configured",
        )

    planner = InstrumentedCurationPlanner(
        planner_provider,
        provider_key=str(getattr(planner_provider, "_provider_key", "curation-shadow")),
        provider_name=str(getattr(planner_provider, "_provider_name", "curation-shadow")),
        model_name=str(getattr(planner_provider, "_model_name", "curation-shadow")),
    )
    support_records = seed_records[len(sampled_records) :] if len(seed_records) > len(sampled_records) else []
    frontier_args = {
        "family": "curator",
        "strategy": seed_batch.strategy_used,
        "seed_reads": (_record_read(record) for record in sampled_records),
        "support_reads": (_record_read(record) for record in support_records),
        "task_id": _task_uuid(task.id),
    }
    frontier = (
        CurationFrontier.claimed(claimed_work_item.id, **frontier_args)
        if claimed_work_item is not None
        else CurationFrontier.direct(**frontier_args)
    )
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(ctx, planner_provider, seed_records)
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            require_authoritative_disclosure_context=require_policy,
        ),
        protections_by_memory=protections,
        sensitive_fields_by_memory=sensitive_fields,
    )
    try:
        result = await harness.run(frontier)
    except Exception:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise
    plan = None if result.validation is None else result.validation.plan
    decisions = []
    if result.validation is not None:
        decisions = [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "accepted",
            }
            for item in result.validation.accepted_actions
        ] + [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "rejected",
                "reason_codes": [str(code) for code in item.reason_codes],
            }
            for item in result.validation.rejected_actions
        ]
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=None if plan is None else plan.rationale,
        shadow_mode=True,
        execution_mode="curation_shadow_harness",
        claimed_work_item_count=1 if claimed_work_item is not None else 0,
        tool_calls_executed=0,
        mutations=0,
        curation_run_id=str(result.run.run_id),
        curation_plan_id=str(result.result.plan_id),
        curation_outcome=str(result.result.outcome),
        curation_work_item_action=str(result.work_item.action),
        curation_rejection_codes=result.result.rejection_codes,
        curation_planner_attempts=result.planner_attempts,
        curation_plan=None if plan is None else plan.model_dump(mode="json"),
        curation_decisions=decisions,
    )


async def run_curator_verified_normalize_execution(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    provider: Any,
    seed_batch: Any,
    sampled_records: list[Any],
    seed_records: list[Any],
    claimed_work_item: Any,
    work_item_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Plan with the provider, then execute accepted normalizes locally."""
    planner_provider = _curator_json_provider(ctx, task, provider)
    if planner_provider is None or not _supports_json_planning(planner_provider):
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            shadow_mode=False,
            execution_mode="curation_verified_executor",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="normalize_json_provider_not_configured",
        )

    action_store = getattr(ctx, "curation_action_store", None)
    if action_store is None and ctx.db_manager is not None:
        if ctx.storage_backend == "postgres":
            from mcp_memory.storage.postgres_curation_action_store import PostgresCurationActionStore

            action_store = PostgresCurationActionStore(ctx.db_manager)
        else:
            from mcp_memory.curation_action_store import SQLiteCurationActionStore

            action_store = SQLiteCurationActionStore(ctx.db_manager)
    if action_store is None or ctx.curation is None or ctx.relational_search is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise RuntimeError("normalize execution storage is unavailable")

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
        "seed_reads": (_record_read(record) for record in sampled_records),
        "support_reads": (_record_read(record) for record in support_records),
        "task_id": _task_uuid(task.id),
    }
    frontier = (
        CurationFrontier.claimed(claimed_work_item.id, **frontier_args)
        if claimed_work_item is not None
        else CurationFrontier.direct(**frontier_args)
    )
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(ctx, planner_provider, seed_records)
    memory_types = {UUID(record.id): record.type for record in seed_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            execute_accepted_normalize_actions=True,
            require_authoritative_disclosure_context=require_policy,
        ),
        memory_types=memory_types,
        protections_by_memory=protections,
        sensitive_fields_by_memory=sensitive_fields,
        executor=CurationExecutor(action_store),
        verifier=CurationVerifier(ctx.curation, ctx.relational_search),
    )
    try:
        result = await harness.run(frontier)
    except Exception:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise
    plan = None if result.validation is None else result.validation.plan
    decisions = []
    if result.validation is not None:
        decisions = [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "accepted",
            }
            for item in result.validation.accepted_actions
        ] + [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "rejected",
                "reason_codes": [str(code) for code in item.reason_codes],
            }
            for item in result.validation.rejected_actions
        ]
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=None if plan is None else plan.rationale,
        shadow_mode=False,
        execution_mode="curation_verified_executor",
        claimed_work_item_count=1 if claimed_work_item is not None else 0,
        tool_calls_executed=0,
        mutations=(
            len(result.validation.accepted_actions)
            if result.validation is not None and result.outcome.value in {"applied", "partially_applied"}
            else 0
        ),
        curation_run_id=str(result.run.run_id),
        curation_plan_id=str(result.result.plan_id),
        curation_outcome=str(result.result.outcome),
        curation_work_item_action=str(result.work_item.action),
        curation_rejection_codes=result.result.rejection_codes,
        curation_planner_attempts=result.planner_attempts,
        curation_plan=None if plan is None else plan.model_dump(mode="json"),
        curation_decisions=decisions,
    )


async def run_curator_verified_create_link_execution(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    provider: Any,
    seed_batch: Any,
    sampled_records: list[Any],
    seed_records: list[Any],
    claimed_work_item: Any,
    work_item_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Plan with the provider, then execute accepted create-links locally."""
    planner_provider = _curator_json_provider(ctx, task, provider)
    if planner_provider is None or not _supports_json_planning(planner_provider):
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            shadow_mode=False,
            execution_mode="curation_verified_executor",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="create_link_json_provider_not_configured",
        )

    action_store = getattr(ctx, "curation_action_store", None)
    if action_store is None and ctx.db_manager is not None:
        if ctx.storage_backend == "postgres":
            from mcp_memory.storage.postgres_curation_action_store import PostgresCurationActionStore

            action_store = PostgresCurationActionStore(ctx.db_manager)
        else:
            from mcp_memory.curation_action_store import SQLiteCurationActionStore

            action_store = SQLiteCurationActionStore(ctx.db_manager)
    if action_store is None or ctx.curation is None or ctx.relational_search is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise RuntimeError("create-link execution storage is unavailable")

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
        "seed_reads": (_record_read(record) for record in sampled_records),
        "support_reads": (_record_read(record) for record in support_records),
        "task_id": _task_uuid(task.id),
    }
    frontier = (
        CurationFrontier.claimed(claimed_work_item.id, **frontier_args)
        if claimed_work_item is not None
        else CurationFrontier.direct(**frontier_args)
    )
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(ctx, planner_provider, seed_records)
    memory_types = {UUID(record.id): record.type for record in seed_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            execute_accepted_create_link_actions=True,
            require_authoritative_disclosure_context=require_policy,
        ),
        memory_types=memory_types,
        protections_by_memory=protections,
        sensitive_fields_by_memory=sensitive_fields,
        executor=CurationExecutor(action_store),
        verifier=CurationVerifier(ctx.curation, ctx.relational_search),
    )
    try:
        result = await harness.run(frontier)
    except Exception:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise
    plan = None if result.validation is None else result.validation.plan
    decisions = []
    create_link_actions = []
    if result.validation is not None:
        decisions = [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "accepted",
            }
            for item in result.validation.accepted_actions
            if item.action.operation == "create_link"
        ] + [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "specialist_route",
                "primary_family": str(item.family),
                "reason_codes": [str(item.reason_code)],
            }
            for item in result.validation.specialist_routes
            if item.action.operation == "create_link"
        ] + [
            {
                "action_id": str(item.action.action_id),
                "operation": item.action.operation,
                "decision": "rejected",
                "reason_codes": [str(code) for code in item.reason_codes],
            }
            for item in result.validation.rejected_actions
        ]
        create_link_actions = [
            item.action
            for item in result.validation.accepted_actions
            if item.action.operation == "create_link"
        ]
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=None if plan is None else plan.rationale,
        shadow_mode=False,
        execution_mode="curation_verified_executor",
        claimed_work_item_count=1 if claimed_work_item is not None else 0,
        tool_calls_executed=0,
        mutations=(
            len(create_link_actions)
            if result.outcome.value in {"applied", "partially_applied"}
            else 0
        ),
        curation_run_id=str(result.run.run_id),
        curation_plan_id=str(result.result.plan_id),
        curation_outcome=str(result.result.outcome),
        curation_work_item_action=str(result.work_item.action),
        curation_rejection_codes=result.result.rejection_codes,
        curation_planner_attempts=result.planner_attempts,
        curation_plan=None if plan is None else plan.model_dump(mode="json"),
        curation_decisions=decisions,
    )


async def run_curator_verified_campaign(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    provider: Any,
    seed_batch: Any,
    sampled_records: list[Any],
    seed_records: list[Any],
    claimed_work_item: Any,
    work_item_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Plan and execute the canonical verified curator campaign path."""
    planner_provider = _curator_json_provider(ctx, task, provider)
    if planner_provider is None or not _supports_json_planning(planner_provider):
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            shadow_mode=False,
            execution_mode="curation_verified_campaign",
            claimed_work_item_count=0,
            tool_calls_executed=0,
            mutations=0,
            reason="campaign_json_provider_not_configured",
        )

    action_store = getattr(ctx, "curation_action_store", None)
    if action_store is None and ctx.db_manager is not None:
        if ctx.storage_backend == "postgres":
            from mcp_memory.storage.postgres_curation_action_store import PostgresCurationActionStore

            action_store = PostgresCurationActionStore(ctx.db_manager)
        else:
            from mcp_memory.curation_action_store import SQLiteCurationActionStore

            action_store = SQLiteCurationActionStore(ctx.db_manager)
    if action_store is None or ctx.curation is None or ctx.relational_search is None:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise RuntimeError("verified campaign storage is unavailable")

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
        "seed_reads": (_record_read(record) for record in sampled_records),
        "support_reads": (_record_read(record) for record in support_records),
        "task_id": _task_uuid(task.id),
    }
    frontier = (
        CurationFrontier.claimed(claimed_work_item.id, **frontier_args)
        if claimed_work_item is not None
        else CurationFrontier.direct(**frontier_args)
    )
    provider_trust, protections, sensitive_fields, require_policy = _disclosure_context(ctx, planner_provider, seed_records)
    memory_types = {UUID(record.id): record.type for record in seed_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(
            provider=provider_trust,
            execute_accepted_actions=True,
            execute_accepted_normalize_actions=True,
            execute_accepted_create_link_actions=True,
            require_authoritative_disclosure_context=require_policy,
        ),
        memory_types=memory_types,
        protections_by_memory=protections,
        sensitive_fields_by_memory=sensitive_fields,
        executor=CurationExecutor(action_store),
        verifier=CurationVerifier(ctx.curation, ctx.relational_search),
    )
    try:
        result = await harness.run(frontier)
    except Exception:
        if claimed_work_item is not None:
            release_work_item(ctx, claimed_work_item.id)
        raise
    plan = None if result.validation is None else result.validation.plan
    decisions = _curator_decisions(result)
    campaign_result = _curator_campaign_result(result)
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=None if plan is None else plan.rationale,
        shadow_mode=False,
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
        curation_plan=None if plan is None else plan.model_dump(mode="json"),
        curation_decisions=decisions,
        curation_campaign_result=campaign_result,
    )


def _supports_json_planning(provider: Any) -> bool:
    if not callable(getattr(provider, "ask_json", None)):
        return False
    supports_agentic = getattr(provider, "supports_agentic", None)
    return not callable(supports_agentic) or not bool(supports_agentic())


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


def _record_read(record: Any) -> AcceptedMaintenanceRead:
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
        }
    )


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

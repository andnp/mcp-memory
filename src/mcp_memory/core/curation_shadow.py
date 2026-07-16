"""Curator rollout routing through the planning-only curation harness."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_context import AcceptedMaintenanceRead
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier, CurationHarnessConfig
from mcp_memory.core.curation_planner import InstrumentedCurationPlanner
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import release_work_item
from mcp_memory.core.tasks import TaskRecord


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

    planner_provider = getattr(ctx, "ai_json_provider", None)
    if planner_provider is None and _supports_json_planning(provider):
        planner_provider = provider
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
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=None,
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
    planner_provider = getattr(ctx, "ai_json_provider", None)
    if planner_provider is None and _supports_json_planning(provider):
        planner_provider = provider
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
    memory_types = {UUID(record.id): record.type for record in seed_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(execute_accepted_normalize_actions=True),
        memory_types=memory_types,
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
    planner_provider = getattr(ctx, "ai_json_provider", None)
    if planner_provider is None and _supports_json_planning(provider):
        planner_provider = provider
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
    memory_types = {UUID(record.id): record.type for record in seed_records}
    harness = CurationDryRunHarness(
        curation_store=ctx.curation,
        planner=planner,
        work_items=ctx.work_items,
        config=CurationHarnessConfig(execute_accepted_create_link_actions=True),
        memory_types=memory_types,
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
            for item in (*result.validation.accepted_actions, *result.validation.specialist_routes)
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
            for item in (*result.validation.accepted_actions, *result.validation.specialist_routes)
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


def _supports_json_planning(provider: Any) -> bool:
    if not callable(getattr(provider, "ask_json", None)):
        return False
    supports_agentic = getattr(provider, "supports_agentic", None)
    return not callable(supports_agentic) or not bool(supports_agentic())


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

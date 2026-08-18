from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_identity import canonical_token
from mcp_memory.core.curation_models import CampaignHypothesis
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import CurationSpecialistRoute
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.ports.work_items import (
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_CONFLICT_REVIEW,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
    WORK_FAMILY_MEMORY_TAGGING,
    WORK_FAMILY_OPERATOR_REVIEW,
    WorkItemRecordLike,
)
from mcp_memory.core.sampling import SamplingBatch
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.management.models import NerdQualityRemediationSignalPayload

SPECIALIST_ROUTE_WORK_FAMILIES: dict[MaintenanceFamily, str] = {
    MaintenanceFamily.CONFLICT_REVIEW: WORK_FAMILY_CONFLICT_REVIEW,
    MaintenanceFamily.DEDUPLICATOR: WORK_FAMILY_MEMORY_DEDUP_REVIEW,
    MaintenanceFamily.GRAPH_LINKER: WORK_FAMILY_GRAPH_LINK_REVIEW,
    MaintenanceFamily.TAXONOMIST: WORK_FAMILY_MEMORY_TAGGING,
}


def enqueue_specialist_routes(
    work_items: Any,
    routes: Iterable[CurationSpecialistRoute],
    *,
    context: Any | None = None,
    workspace_id: str | None = None,
    priority: int = 100,
) -> tuple[WorkItemRecordLike, ...]:
    """Persist accepted specialist routes without claiming or executing them."""

    records: list[WorkItemRecordLike] = []
    for route in routes:
        record, _ = enqueue_specialist_route(
            work_items,
            route,
            context=context,
            workspace_id=workspace_id,
            priority=priority,
        )
        records.append(record)
    return tuple(records)


def enqueue_specialist_route(
    work_items: Any,
    route: CurationSpecialistRoute,
    *,
    context: Any | None = None,
    workspace_id: str | None = None,
    priority: int = 100,
) -> tuple[WorkItemRecordLike, bool]:
    """Create one idempotent work item for a validated specialist route."""

    requested_family = _family_value(route.family)
    try:
        primary_family = MaintenanceFamily(requested_family)
    except ValueError:
        primary_family = None
    work_family = (
        SPECIALIST_ROUTE_WORK_FAMILIES[primary_family]
        if primary_family is not None and primary_family in SPECIALIST_ROUTE_WORK_FAMILIES
        else WORK_FAMILY_OPERATOR_REVIEW
    )
    reason_code = _enum_value(route.reason_code)
    target_ids = _route_target_ids(route)
    target_revision_token = _route_target_revision_token(route, target_ids=target_ids, context=context)
    action_payload = route.action.model_dump(mode="json")
    operation_payload = {
        key: value
        for key, value in action_payload.items()
        if key not in {"confidence", "rationale", "evidence"}
    }
    idempotency_key = "specialist-route:" + canonical_token(
        {
            "family": requested_family,
            "reason_code": reason_code,
            "target_revision_token": target_revision_token,
            "operation": operation_payload,
        }
    )
    payload: dict[str, Any] = {
        "route_kind": "specialist_route",
        "primary_family": requested_family,
        "work_family": work_family,
        "operation": _enum_value(route.action.operation),
        "action_id": str(route.action.action_id),
        "target_ids": target_ids,
        "target_revision_token": target_revision_token,
        "reason_code": reason_code,
        "action": action_payload,
    }
    if work_family == WORK_FAMILY_OPERATOR_REVIEW:
        payload["operator_review_required"] = True
        payload["operator_review_reason"] = "unsupported_specialist_family"
    record, created = work_items.enqueue_unique(
        family_key=work_family,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        priority=priority,
        idempotency_key=idempotency_key,
        payload=payload,
    )
    if not created and canonical_token(record.payload) != canonical_token(payload):
        raise RuntimeError("specialist_route_idempotency_conflict")
    return record, created


def enqueue_producer_remediation_signals(
    work_items: Any,
    signals: Iterable[NerdQualityRemediationSignalPayload],
    *,
    workspace_id: str | None = None,
    priority: int = 100,
) -> tuple[WorkItemRecordLike, ...]:
    """Persist producer-defect signals using the operator-review work family."""
    records: list[WorkItemRecordLike] = []
    for signal in signals:
        payload = signal.model_dump(mode="json")
        payload.update(
            {
                "work_item_kind": "producer_remediation",
                "operator_review_required": True,
                "operator_review_reason": "repeated_producer_defect",
            }
        )
        record, _ = work_items.enqueue_unique(
            family_key=WORK_FAMILY_OPERATOR_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=workspace_id,
            priority=priority,
            idempotency_key=signal.idempotency_key,
            payload=payload,
        )
        records.append(record)
    return tuple(records)


def _route_target_ids(route: CurationSpecialistRoute) -> list[str]:
    action = route.action
    values: list[Any] = []
    for name in ("target_id", "source_id", "canonical_id"):
        if hasattr(action, name):
            values.append(getattr(action, name))
    values.extend(getattr(action, "source_ids", ()))
    return sorted({str(value) for value in values})


def _route_target_revision_token(
    route: CurationSpecialistRoute,
    *,
    target_ids: list[str],
    context: Any | None,
) -> str:
    revision_tokens: list[str] = []
    for evidence in route.action.evidence:
        if evidence.revision_token:
            revision_tokens.append(evidence.revision_token)
    for memory_id, token in route.action.preconditions.record_tokens.items():
        if str(memory_id) in target_ids:
            revision_tokens.append(token)

    record_tokens = _context_value(context, "record_tokens")
    graph_tokens = _context_value(context, "graph_tokens")
    if isinstance(record_tokens, Mapping) or isinstance(graph_tokens, Mapping):
        for memory_id in target_ids:
            record_token = record_tokens.get(memory_id) if isinstance(record_tokens, Mapping) else None
            graph_token = graph_tokens.get(memory_id) if isinstance(graph_tokens, Mapping) else None
            if record_token is not None and graph_token is not None:
                revision_tokens.append(canonical_token({"record": record_token, "graph": graph_token}))
            elif record_token is not None:
                revision_tokens.append(str(record_token))
            elif graph_token is not None:
                revision_tokens.append(str(graph_token))

    return canonical_token(
        {
            "target_ids": target_ids,
            "revision_tokens": sorted(set(str(token) for token in revision_tokens)),
        }
    )


def _context_value(context: Any | None, name: str) -> Any:
    if isinstance(context, Mapping):
        return context.get(name)
    return getattr(context, name, None)


def _family_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def claim_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    execution_lane: str,
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    return work_items.claim_batch(
        family_key=family_key,
        execution_lane=execution_lane,
        lease_owner=task.id,
        limit=limit,
        workspace_id=task.workspace_id,
    )


def enqueue_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    execution_lane: str,
    workspace_id: str | None,
    idempotency_prefix: str,
    payload_memory_ids_key: str,
    memory_ids: list[str],
    strategy_used: str | None,
    candidate_count: int | None = None,
    extra_payload: dict[str, Any] | None = None,
    campaign_hypothesis: CampaignHypothesis | None = None,
) -> tuple[Any, bool]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        raise ValueError("work_items repository is not configured")
    sorted_memory_ids = sorted(memory_ids)
    payload: dict[str, Any] = {
        "workspace_id": workspace_id,
        payload_memory_ids_key: sorted_memory_ids,
        "strategy_used": strategy_used,
    }
    if candidate_count is not None:
        payload["candidate_count"] = candidate_count
    if campaign_hypothesis is not None:
        payload["campaign_hypothesis"] = campaign_hypothesis.model_dump(mode="json")
    if extra_payload:
        payload.update(extra_payload)
    return work_items.enqueue_unique(
        family_key=family_key,
        execution_lane=execution_lane,
        workspace_id=workspace_id,
        priority=task.priority,
        idempotency_key=f"{idempotency_prefix}:{'|'.join(sorted_memory_ids)}",
        payload=payload,
    )


def payload_memory_records(
    ctx: ApplicationContext,
    payload: dict[str, Any],
    *,
    payload_memory_ids_key: str,
    allowed_types: set[str] | None = None,
) -> list[Any]:
    if ctx.repository is None:
        return []
    memory_ids = payload.get(payload_memory_ids_key)
    if not isinstance(memory_ids, list):
        return []
    records: list[Any] = []
    for memory_id in memory_ids:
        if not isinstance(memory_id, str):
            continue
        record = ctx.repository.get_memory(memory_id)
        if record is None or record.status != "active":
            continue
        if allowed_types is not None and record.type not in allowed_types:
            continue
        records.append(record)
    return records


def work_item_result_metadata(
    *,
    family_key: str,
    execution_lane: str,
    seed_source: str,
    seed_records: list[Any],
    claimed_work_item: Any | None = None,
    created_work_item: Any | None = None,
    campaign_hypothesis: CampaignHypothesis | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "work_item_family": family_key,
        "work_item_execution_lane": execution_lane,
        "seed_source": seed_source,
        "seed_record_count": len(seed_records),
    }
    if claimed_work_item is not None:
        metadata["claimed_work_item_id"] = claimed_work_item.id
    if created_work_item is not None:
        metadata["created_work_item_id"] = created_work_item.id
    if campaign_hypothesis is not None:
        metadata["campaign_hypothesis"] = campaign_hypothesis.model_dump(mode="json")
    return metadata


def complete_work_item(ctx: ApplicationContext, work_item_id: str) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.complete_item(work_item_id)


def defer_work_item(
    ctx: ApplicationContext,
    work_item_id: str,
    *,
    error: str,
    retry_delay_seconds: float | None,
) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.defer_item(
        work_item_id,
        error=error,
        retry_delay_seconds=0.0 if retry_delay_seconds is None else float(retry_delay_seconds),
    )


def release_work_item(ctx: ApplicationContext, work_item_id: str) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.release_item(work_item_id)


async def run_claimed_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    load_candidates: Callable[[ApplicationContext, dict[str, Any]], list[Any]],
    propose_pairs: Callable[[list[Any]], Awaitable[Any]],
    apply_pairs: Callable[[Any], int],
) -> dict[str, Any] | None:
    claimed_review_items = claim_work_batch(
        ctx,
        task=task,
        family_key=family_key,
        execution_lane="agentic",
        limit=1,
    )
    if not claimed_review_items:
        return None

    review_item = claimed_review_items[0]
    try:
        candidates = load_candidates(ctx, review_item.payload)
        proposed_pairs = await propose_pairs(candidates)
        created = apply_pairs(proposed_pairs)
    except Exception:
        release_work_item(ctx, review_item.id)
        raise

    complete_work_item(ctx, review_item.id)
    result = {
        "created": created,
        "claimed_work_item_count": 1,
        "execution_mode": "agentic_review",
    }
    result.update(
        work_item_result_metadata(
            family_key=family_key,
            execution_lane="agentic",
            seed_source="claimed_review_work_item",
            seed_records=candidates,
            claimed_work_item=review_item,
        )
    )
    return result


async def run_sparse_frontier_review_task(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    sampled_batch: SamplingBatch,
    candidates: list[Any],
    family_key: str,
    propose_pairs: Callable[[list[Any]], Awaitable[Any]],
    apply_pairs: Callable[[Any], int],
    should_seed: Callable[[Any, list[Any]], bool],
    enqueue_review_work_item: Callable[[list[Any]], tuple[Any, bool]],
) -> dict[str, Any]:
    created_work_item: Any | None = None
    fallback_pairs = await propose_pairs(candidates)
    created = apply_pairs(fallback_pairs)
    seeded_work_item_count = 0
    if should_seed(fallback_pairs, candidates):
        created_work_item, created_item = enqueue_review_work_item(candidates)
        seeded_work_item_count = 1 if created_item else 0

    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        extra=work_item_result_metadata(
            family_key=family_key,
            execution_lane="agentic",
            seed_source="frontier_seed",
            seed_records=candidates,
            created_work_item=created_work_item if seeded_work_item_count else None,
        ),
        created=created,
        seeded_work_item_count=seeded_work_item_count,
    )

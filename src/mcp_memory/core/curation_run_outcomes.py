"""Persistence-free projection of curation run outcomes and results."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol
from uuid import UUID

from mcp_memory.core.curation_context import CurationContextPacket as ImmutableCurationContextPacket
from mcp_memory.core.curation_models import (
    CurationBudgetUsage,
    CurationPlan,
    CurationRunOutcome,
    CurationRunResult,
    MutationReceipt,
)
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerError,
    CurationPlannerSchemaError,
    PlannerExecutionEnvelope,
)
from mcp_memory.core.curation_validation import CurationValidationResult


class CurationReceipt(Protocol):
    @property
    def status(self) -> str:
        ...

    @property
    def affected_ids(self) -> list[UUID]:
        ...

    def model_dump(self, *, mode: str, exclude: set[str]) -> dict[str, Any]:
        ...


def classify_outcome(
    plan: CurationPlan | None,
    validation: CurationValidationResult | None,
    failure: CurationPlannerError | BaseException | None,
) -> tuple[CurationRunOutcome, str]:
    if isinstance(failure, CurationPlannerCancelledError) or isinstance(failure, asyncio.CancelledError):
        return CurationRunOutcome.CANCELLED, "provider_cancelled"
    if failure is not None:
        if isinstance(failure, CurationPlannerSchemaError):
            return CurationRunOutcome.INVALID_PLAN, "invalid_plan"
        return CurationRunOutcome.PROVIDER_FAILED, getattr(failure, "reason_code", "provider_failed")
    if validation is None or not validation.valid or plan is None:
        return CurationRunOutcome.INVALID_PLAN, "invalid_plan"
    if not plan.actions:
        return CurationRunOutcome.NO_OP, "valid_no_op"
    if validation.rejected_actions and not validation.accepted_actions and not validation.specialist_routes:
        return CurationRunOutcome.DEFERRED, "policy_rejected"
    return CurationRunOutcome.DEFERRED, "dry_run_requires_executor"


def rejection_codes(validation: CurationValidationResult | None) -> list[str]:
    if validation is None:
        return []
    codes = [str(issue.code) for issue in validation.issues]
    codes.extend(str(code) for item in validation.rejected_actions for code in item.reason_codes)
    codes.extend(str(route.reason_code) for route in validation.specialist_routes)
    return list(dict.fromkeys(codes))


def budget_usage(
    context: ImmutableCurationContextPacket,
    envelopes: list[PlannerExecutionEnvelope[Any]],
    validation: CurationValidationResult | None,
) -> CurationBudgetUsage:
    token_usage = [envelope.token_usage for envelope in envelopes if envelope.token_usage is not None]
    token_source = next(
        (envelope.token_usage_source for envelope in reversed(envelopes) if envelope.token_usage_source is not None),
        None,
    )
    return CurationBudgetUsage(
        seed_records=context.usage.seed_records,
        support_records=context.usage.support_records,
        context_characters=context.usage.context_characters,
        read_tool_calls=context.usage.read_tool_calls,
        records_returned=context.usage.records_returned,
        proposed_actions=0 if validation is None or validation.plan is None else len(validation.plan.actions),
        accepted_mutations=0 if validation is None else len(validation.accepted_actions),
        planner_attempts=len(envelopes),
        premium_requests=sum(1 for envelope in envelopes if envelope.premium_request),
        token_usage=sum(token_usage) if token_usage else None,
        token_usage_source=token_source,
    )


def project_receipts(receipts: tuple[CurationReceipt, ...]) -> list[MutationReceipt]:
    return [
        MutationReceipt.model_validate(
            receipt.model_dump(mode="json", exclude={"mutation_event_id", "intent_hash"})
        )
        for receipt in receipts
    ]


def count_verified_receipts(receipts: tuple[CurationReceipt, ...]) -> int:
    return sum(1 for receipt in receipts if receipt.status in {"verified", "CurationReceiptState.VERIFIED"})


def count_affected_memory_ids(receipts: tuple[CurationReceipt, ...]) -> int:
    return len({str(memory_id) for receipt in receipts for memory_id in receipt.affected_ids})


def build_run_result(
    *,
    run_id: UUID,
    outcome: CurationRunOutcome,
    plan_id: UUID | None,
    receipts: tuple[CurationReceipt, ...],
    rejection_codes: list[str],
    retry_reason: str | None,
    budget_usage: CurationBudgetUsage,
    context_record_counts: dict[str, int],
) -> CurationRunResult:
    return CurationRunResult(
        run_id=run_id,
        outcome=outcome,
        plan_id=plan_id,
        receipts=project_receipts(receipts),
        rejection_codes=rejection_codes,
        retry_reason=retry_reason,
        budget_usage=budget_usage,
        context_record_counts=context_record_counts,
        verified_action_count=count_verified_receipts(receipts),
        affected_memory_count=count_affected_memory_ids(receipts),
    )

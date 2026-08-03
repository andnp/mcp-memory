"""Application-owned planning and provider-neutral validation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, cast
from uuid import UUID

from mcp_memory.core.curation_context import CurationContextPacket as ImmutableCurationContextPacket
from mcp_memory.core.curation_models import CurationPlan, CurationPlanningRequest
from mcp_memory.core.curation_planner import (
    CurationPlanner,
    CurationPlannerCancelledError,
    CurationPlannerError,
    CurationPlannerSchemaError,
    PlannerExecutionEnvelope,
)
from mcp_memory.core.curation_validation import (
    CurationMutationBudget,
    CurationRetryFeedback,
    CurationValidationResult,
    validate_curation_plan,
)
from mcp_memory.mutation_history import ProtectionMode


@dataclass(frozen=True, slots=True)
class CurationPlannerTools:
    context: ImmutableCurationContextPacket
    retry_feedback: CurationRetryFeedback | None = None
    quality_feedback: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CurationPlanningInput:
    request: CurationPlanningRequest
    context: ImmutableCurationContextPacket
    mutation_budget: CurationMutationBudget
    memory_types: Mapping[UUID, str] | None = None
    contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset()
    protections_by_memory: Mapping[UUID, set[ProtectionMode] | frozenset[ProtectionMode]] | None = None
    quality_feedback: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class CurationPlanningOutput:
    plan: CurationPlan | None
    validation: CurationValidationResult | None
    envelopes: list[PlannerExecutionEnvelope[Any]]
    retry_reason: str | None
    failure: CurationPlannerError | BaseException | None


async def plan_and_validate(
    planner: CurationPlanner, input: CurationPlanningInput
) -> CurationPlanningOutput:
    envelopes: list[PlannerExecutionEnvelope[Any]] = []
    feedback: CurationRetryFeedback | None = None
    retry_reason: str | None = None
    validation: CurationValidationResult | None = None
    failure: CurationPlannerError | BaseException | None = None
    for attempt in range(2):
        try:
            envelope = await planner.create_plan(
                input.request,
                cast(
                    Any,
                    CurationPlannerTools(
                        context=input.context,
                        retry_feedback=feedback,
                        quality_feedback=input.quality_feedback,
                    ),
                ),
            )
            envelopes.append(cast(PlannerExecutionEnvelope[Any], envelope))
        except CurationPlannerSchemaError as error:
            failure = error
            if error.envelope is not None:
                envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
            if attempt == 0:
                retry_reason = "schema_invalid"
                feedback = CurationRetryFeedback(
                    reason_code="schema_invalid",
                    message=_schema_retry_message(error),
                    fields=tuple(
                        issue.field for issue in error.validation_issues if issue.field is not None
                    ),
                    issue_codes=tuple(issue.code for issue in error.validation_issues),
                    expected_fields=error.expected_fields,
                    received_fields=error.received_fields,
                )
                continue
            return CurationPlanningOutput(None, None, envelopes, retry_reason, failure)
        except CurationPlannerCancelledError as error:
            failure = error
            if error.envelope is not None:
                envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
            return CurationPlanningOutput(None, None, envelopes, retry_reason, failure)
        except asyncio.CancelledError as error:
            return CurationPlanningOutput(None, None, envelopes, retry_reason, error)
        except CurationPlannerError as error:
            failure = error
            if error.envelope is not None:
                envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
            return CurationPlanningOutput(None, None, envelopes, retry_reason, failure)

        validation = validate_curation_plan(
            cast(Any, envelope.plan),
            request=input.request,
            context=input.context,
            mutation_budget=input.mutation_budget,
            memory_types=input.memory_types,
            contradictory_memory_ids=input.contradictory_memory_ids,
            protections_by_memory=input.protections_by_memory,
        )
        if validation.valid:
            return CurationPlanningOutput(cast(CurationPlan, validation.plan), validation, envelopes, retry_reason, None)
        if validation.retry_feedback is not None and attempt == 0:
            retry_reason = validation.retry_feedback.reason_code
            feedback = validation.retry_feedback
            continue
        return CurationPlanningOutput(None, validation, envelopes, retry_reason, None)
    return CurationPlanningOutput(None, validation, envelopes, retry_reason, failure)


def _schema_retry_message(error: CurationPlannerSchemaError) -> str:
    details = "; ".join(
        f"{issue.code} at {issue.field}: {issue.message}"
        if issue.field is not None
        else f"{issue.code}: {issue.message}"
        for issue in error.validation_issues
    )
    fields = (
        f"expected fields={list(error.expected_fields)}; "
        f"received fields={list(error.received_fields)}"
    )
    return f"{str(error)}; {details}; {fields}"[:1000]

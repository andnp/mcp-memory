import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_models import CurationPlan, CurationPlanningRequest, RetentionDecision, RetentionReason
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerProviderError,
    CurationPlannerSchemaError,
    FakeCurationPlanner,
    FakePlannerScenario,
    PlannerExecutionStatus,
)


def _request(seed_id: UUID) -> CurationPlanningRequest:
    return CurationPlanningRequest(
        run_id=uuid4(), plan_id=uuid4(), frontier_key="frontier", context_fingerprint="context"
    )


def _plan(request: CurationPlanningRequest, seed_id: UUID) -> CurationPlan:
    return CurationPlan(
        plan_id=request.plan_id,
        run_id=request.run_id,
        frontier_key=request.frontier_key,
        context_fingerprint=request.context_fingerprint,
        seed_memory_ids=[seed_id],
        retained=[RetentionDecision(memory_id=seed_id, reason=RetentionReason.ALREADY_FOCUSED, rationale="still useful")],
        rationale="no change",
    )


@pytest.mark.asyncio
async def test_fake_plan_is_deterministic_and_preserves_envelope() -> None:
    seed_id = uuid4()
    request = _request(seed_id)
    planner = FakeCurationPlanner(
        [FakePlannerScenario(plan=_plan(request, seed_id), request_id="request-7", token_usage=12)],
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    envelope = await planner.create_plan(request, object())

    assert envelope.plan == _plan(request, seed_id)
    assert envelope.result == envelope.plan
    assert envelope.request_id == "request-7"
    assert envelope.token_usage == 12
    assert envelope.started_at == envelope.completed_at


@pytest.mark.asyncio
async def test_fake_schema_failure_is_typed_and_enveloped() -> None:
    planner = FakeCurationPlanner([FakePlannerScenario(plan={"schema_version": 1})])

    with pytest.raises(CurationPlannerSchemaError) as raised:
        await planner.create_plan(_request(uuid4()), object())

    assert raised.value.envelope is not None
    assert raised.value.envelope.status == PlannerExecutionStatus.SCHEMA_FAILED
    assert raised.value.envelope.reason_code == "schema_invalid"


@pytest.mark.asyncio
async def test_fake_provider_failure_and_cancellation_are_scriptable() -> None:
    provider = CurationPlannerProviderError("unavailable", reason_code="rate_limited")
    planner = FakeCurationPlanner([
        FakePlannerScenario(failure=provider),
        FakePlannerScenario(failure=CurationPlannerCancelledError()),
    ])

    with pytest.raises(CurationPlannerProviderError) as provider_raised:
        await planner.create_plan(_request(uuid4()), object())
    assert provider_raised.value.envelope is not None
    assert provider_raised.value.envelope.reason_code == "rate_limited"
    assert provider_raised.value.envelope.status == PlannerExecutionStatus.PROVIDER_FAILED

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await planner.create_plan(_request(uuid4()), object())
    assert isinstance(cancelled.value, CurationPlannerCancelledError)
    assert cancelled.value.envelope is not None
    assert cancelled.value.envelope.status == PlannerExecutionStatus.CANCELLED

import asyncio
import json
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
    InstrumentedCurationPlanner,
    PlannerExecutionStatus,
)
from mcp_memory.core.providers.interfaces import ProviderJSONCall


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


@pytest.mark.asyncio
async def test_instrumented_adapter_translates_recorded_json_without_action_counts() -> None:
    seed_id = uuid4()
    request = _request(seed_id)
    response = _plan(request, seed_id).model_dump(mode="json")
    response["claimed_action_count"] = 999
    recorded = ProviderJSONCall(
        response=response,
        provider_key="copilot-strong",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
        request_id="request-55",
        attempt=2,
        started_at=10.0,
        completed_at=12.0,
        status="success",
        raw_text=json.dumps(response),
        parsed=response,
        admission_status="admitted",
        premium_request=True,
        token_usage=41,
        token_usage_source="recorded",
    )

    class _RecordedProvider:
        def __init__(self) -> None:
            self.prompt = ""

        async def ask_json_with_telemetry(self, prompt: str) -> ProviderJSONCall:
            self.prompt = prompt
            return recorded

    provider = _RecordedProvider()
    planner = InstrumentedCurationPlanner(provider)

    with pytest.raises(CurationPlannerSchemaError) as raised:
        await planner.create_plan(request, object())

    assert raised.value.envelope is not None
    assert raised.value.envelope.status == PlannerExecutionStatus.SCHEMA_FAILED
    assert raised.value.envelope.request_id == "request-55"
    assert raised.value.envelope.transcript_ref == "request-55"
    assert raised.value.envelope.premium_request is True
    assert raised.value.envelope.token_usage == 41
    assert json.loads(provider.prompt.split("\n", 1)[1])["available_tools"] == []


@pytest.mark.asyncio
async def test_instrumented_adapter_parses_recorded_valid_json_into_common_envelope() -> None:
    seed_id = uuid4()
    request = _request(seed_id)
    response = _plan(request, seed_id).model_dump(mode="json")

    class _RecordedProvider:
        async def ask_json_with_telemetry(self, prompt: str) -> ProviderJSONCall:
            del prompt
            return ProviderJSONCall(
                response=response,
                provider_key="copilot-strong",
                provider_name="Copilot SDK",
                model_name="gpt-5.4-mini",
                request_id="request-56",
                attempt=1,
                started_at=20.0,
                completed_at=21.5,
                status="success",
                raw_text=json.dumps(response),
                parsed=response,
                admission_status="admitted",
                premium_request=True,
            )

    envelope = await InstrumentedCurationPlanner(_RecordedProvider()).create_plan(request, object())

    assert envelope.plan == _plan(request, seed_id)
    assert envelope.request_id == "request-56"
    assert envelope.attempt == 1
    assert envelope.premium_request is True
    assert envelope.started_at == datetime.fromtimestamp(20.0, tz=UTC)


@pytest.mark.asyncio
async def test_instrumented_adapter_classifies_recorded_invalid_json_as_schema_failure() -> None:
    class _RecordedProvider:
        async def ask_json_with_telemetry(self, prompt: str) -> ProviderJSONCall:
            del prompt
            return ProviderJSONCall(
                response=None,
                provider_key="copilot-strong",
                provider_name="Copilot SDK",
                model_name="gpt-5.4-mini",
                request_id="request-57",
                attempt=1,
                started_at=30.0,
                completed_at=30.2,
                status="parse_error",
                error_text="Assistant message did not contain a JSON object",
                raw_text="not json",
                admission_status="admitted",
                premium_request=True,
            )

    with pytest.raises(CurationPlannerSchemaError) as raised:
        await InstrumentedCurationPlanner(_RecordedProvider()).create_plan(_request(uuid4()), object())

    assert raised.value.envelope is not None
    assert raised.value.envelope.status == PlannerExecutionStatus.SCHEMA_FAILED
    assert raised.value.envelope.reason_code == "schema_invalid"

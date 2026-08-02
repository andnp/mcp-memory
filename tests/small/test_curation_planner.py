import asyncio
import json
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_models import (
    CurationPlan,
    CurationPlanningRequest,
    RetentionDecision,
    RetentionReason,
)
from mcp_memory.core.curation_validation import CurationRetryFeedback
from mcp_memory.core.curation_harness import CurationPlannerTools
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerProviderError,
    CurationPlannerSchemaError,
    FakeCurationPlanner,
    FakePlannerScenario,
    InstrumentedCurationPlanner,
    PlannerExecutionStatus,
    _build_planner_prompt,
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


def test_planner_prompt_contains_exact_schema_and_no_mutation_tools() -> None:
    request = _request(uuid4())
    payload = json.loads(_build_planner_prompt(request, CurationPlannerTools(context=cast(Any, {}))).split("\n", 1)[1])
    schema = payload["schema"]

    assert payload["available_tools"] == []
    assert set(schema["required"]) >= {
        "run_id",
        "plan_id",
        "frontier_key",
        "context_fingerprint",
    }
    assert {"actions", "retained", "seed_memory_ids"} <= schema["properties"].keys()
    action_schema = schema["properties"]["actions"]["items"]
    assert set(action_schema["discriminator"]["mapping"]) == {
        "archive_memory",
        "create_link",
        "merge_memories",
        "normalize_memory",
        "remove_link",
        "rewrite_memory",
        "split_memory",
    }
    assert "delete" not in action_schema["discriminator"]["mapping"]
    assert "RetentionDecision" in schema["$defs"]
    assert set(schema["$defs"]["RetentionDecision"]["required"]) >= {"memory_id", "reason", "rationale"}
    for action_name in ("CreateLinkAction", "RemoveLinkAction"):
        assert schema["$defs"][action_name]["properties"]["link_type"]["pattern"] == r"^[A-Z][A-Z0-9_]*$"

    contract = " ".join(payload["planner_contract"])
    assert "visible in context" in contract
    assert "merging never creates a record" in contract
    assert "both endpoints to be visible" in contract
    assert "Only split_memory may introduce child records" in contract
    assert "retention decision instead of inventing an ID" in contract
    assert "fail-closed" in contract
    assert "existing visible memory ID" in schema["$defs"]["CreateLinkAction"]["properties"]["source_id"]["description"]
    assert "never creates a record" in schema["$defs"]["MergeMemoriesAction"]["properties"]["canonical_id"]["description"]
    assert "Typed child content only" in schema["$defs"]["SplitMemoryAction"]["properties"]["children"]["description"]

    prompt = _build_planner_prompt(request, CurationPlannerTools(context=cast(Any, {})))
    assert "Every memory ID in an action or retention decision must be copied" in prompt
    assert "If no visible canonical is appropriate, retain the memory" in prompt


def test_planner_prompt_includes_only_bounded_retry_feedback_when_present() -> None:
    request = _request(uuid4())
    context = cast(Any, {})
    initial = json.loads(_build_planner_prompt(request, CurationPlannerTools(context=context)).split("\n", 1)[1])
    feedback = CurationRetryFeedback(reason_code="formatting_only", message="fix the JSON" + "!" * 2000)
    retry = json.loads(
        _build_planner_prompt(request, CurationPlannerTools(context=context, retry_feedback=feedback)).split("\n", 1)[1]
    )

    assert "retry_feedback" not in initial
    assert retry["retry_feedback"]["reason_code"] == "formatting_only"
    assert len(retry["retry_feedback"]["message"]) == 1000
    assert retry["retry_feedback"]["message"].startswith("fix the JSON")


def test_planner_prompt_preserves_schema_retry_diagnostics() -> None:
    request = _request(uuid4())
    feedback = CurationRetryFeedback(
        reason_code="schema_invalid",
        message="missing required fields",
        fields=("run_id",),
        issue_codes=("missing",),
        expected_fields=("run_id", "plan_id"),
        received_fields=("schema_version",),
    )

    payload = json.loads(
        _build_planner_prompt(
            request,
            CurationPlannerTools(context=cast(Any, {}), retry_feedback=feedback),
        ).split("\n", 1)[1]
    )

    assert payload["retry_feedback"] == {
        "reason_code": "schema_invalid",
        "message": "missing required fields",
        "fields": ["run_id"],
        "issue_codes": ["missing"],
        "expected_fields": ["run_id", "plan_id"],
        "received_fields": ["schema_version"],
    }


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
    assert raised.value.expected_fields
    assert raised.value.received_fields == ("schema_version",)
    assert raised.value.validation_issues
    assert raised.value.envelope.metadata["expected_fields"] == list(raised.value.expected_fields)
    assert raised.value.envelope.metadata["received_fields"] == ["schema_version"]
    assert raised.value.envelope.metadata["validation_issues"]


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

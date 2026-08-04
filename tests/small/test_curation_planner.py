import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    build_context_packet,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_harness import CurationPlannerTools
from mcp_memory.core.curation_planning_service import (
    CurationPlanningInput,
    plan_and_validate,
)
from mcp_memory.core.curation_models import (
    CampaignHypothesis,
    CampaignRetrievalProblem,
    CurationPlan,
    CurationPlanningRequest,
    NormalizeMemoryAction,
    RetentionDecision,
    RetentionReason,
)
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerProviderError,
    CurationPlannerSchemaError,
    CurationPlanSubmissionBuffer,
    FakeCurationPlanner,
    FakePlannerScenario,
    IncrementalCurationPlanState,
    InstrumentedCurationPlanner,
    PlannerExecutionStatus,
    SessionCurationPlanner,
    _build_planner_prompt,
    _request_for_packet,
)
from mcp_memory.core.curation_validation import CurationMutationBudget, CurationRetryFeedback
from mcp_memory.core.providers.interfaces import AgenticRunResult
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
        "seed_memory_ids",
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
    assert "primary objective is to improve durable memory quality" in contract
    assert "Any record visible in context may be modified" in contract
    assert "coherent set of mutations" in contract
    assert "Review the full visible context before stopping" in contract
    assert "rather than stopping after the first valid mutation" in contract
    assert "Retain a seed only when it is already focused and durable" in contract
    assert "visible in context" in contract
    assert "seed_memory_ids is required" in contract
    assert "merging never creates a record" in contract
    assert "both endpoints to be visible" in contract
    assert "exact context.record_tokens value" in contract
    assert "descriptive relationship context" in contract
    assert "labels and memory excerpts alone are not link evidence" in contract
    assert "exact absent-link precondition" in contract
    assert "containing only source_id, target_id, and link_type" in contract
    assert "Only split_memory may introduce child records" in contract
    assert "content is the final persisted durable memory body" in contract
    assert "canonical_id is the retained base" in contract
    assert "Do not write mutation-status prose" in contract
    assert "exactly one disposition" in contract
    assert "action source, target, canonical, and child-source IDs all count as affected" in contract
    assert "target 1600 characters or less" in contract
    assert "split content above 3000 characters" in contract
    assert "quality_feedback field means a previous curator mutation" in contract
    assert "Action failure feedback is execution evidence" in contract
    assert "retention decision instead of inventing an ID" in contract
    assert "fail-closed" in contract
    assert "existing visible memory ID" in schema["$defs"]["CreateLinkAction"]["properties"]["source_id"]["description"]
    assert "never creates a record" in schema["$defs"]["MergeMemoriesAction"]["properties"]["canonical_id"]["description"]
    assert "Typed child content only" in schema["$defs"]["SplitMemoryAction"]["properties"]["children"]["description"]

    prompt = _build_planner_prompt(request, CurationPlannerTools(context=cast(Any, {})))
    assert "Every memory ID in an action or retention decision must be copied" in prompt
    assert "Do not optimize for safe no-ops" in prompt
    assert "any visible seed, support, or exploratory memory may be changed" in prompt
    assert "Include seed_memory_ids exactly as the context seed IDs" in prompt
    assert "content is the final persisted durable memory body" in prompt
    assert "do not write status prose" in prompt
    assert "evidence.link matching the exact endpoints" in prompt
    assert "absent-link precondition containing only source_id, target_id, and link_type" in prompt
    assert "Give every seed exactly one disposition" in prompt
    assert "If no visible canonical is appropriate, retain the memory" in prompt
    assert "use the available action budget" in prompt
    assert "Do not manufacture work for focused records" in prompt


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


def test_incremental_plan_state_supports_action_revisions() -> None:
    seed_id = uuid4()
    other_seed_id = uuid4()
    request = _request(seed_id)
    state = IncrementalCurationPlanState(max_proposed_actions=2)
    state.begin(request, seed_memory_ids=(seed_id, other_seed_id))
    action = NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=seed_id,
        summary="A focused summary.",
        confidence=0.9,
        rationale="make the target specific",
    )
    state.add_action(action)
    with pytest.raises(ValueError, match="already been proposed"):
        state.add_action(action)

    replacement = action.model_copy(update={"summary": "A more specific summary."})
    state.replace_action(replacement)
    state.remove_action(replacement.action_id)
    state.add_retention(
        RetentionDecision(
            memory_id=seed_id,
            reason=RetentionReason.ALREADY_FOCUSED,
            rationale="retain the focused target",
        )
    )
    state.add_retention(
        RetentionDecision(
            memory_id=other_seed_id,
            reason=RetentionReason.ALREADY_FOCUSED,
            rationale="retain the supporting target",
        )
    )
    plan = state.build(rationale="incremental test plan")

    assert plan.actions == []
    assert [item.memory_id for item in plan.retained] == [seed_id, other_seed_id]


@pytest.mark.asyncio
async def test_session_planner_assembles_incremental_tool_plan() -> None:
    seed_id = uuid4()
    other_seed_id = uuid4()
    request = _request(seed_id)
    state = IncrementalCurationPlanState(max_proposed_actions=4)

    class _Session:
        async def run_agent(self, prompt: str) -> AgenticRunResult:
            assert "propose_curation_action" in prompt
            state.add_action(
                NormalizeMemoryAction(
                    action_id=uuid4(),
                    target_id=seed_id,
                    summary="A focused summary.",
                    confidence=0.9,
                    rationale="make the target specific",
                )
            )
            state.add_retention(
                RetentionDecision(
                    memory_id=other_seed_id,
                    reason=RetentionReason.ALREADY_FOCUSED,
                    rationale="retain the supporting target",
                )
            )
            return AgenticRunResult(status="success", raw_text="incremental rationale")

    planner = SessionCurationPlanner(cast(Any, _Session()), plan_state=state)
    result = await planner.create_plan(
        request,
        CurationPlannerTools(
            context=cast(Any, SimpleNamespace(seed_memory_ids=(seed_id, other_seed_id)))
        ),
    )

    assert result.plan is not None
    assert len(result.plan.actions) == 1
    assert result.plan.retained[0].memory_id == other_seed_id
    assert result.plan.rationale == "incremental rationale"


@pytest.mark.asyncio
async def test_session_planner_reuses_conversation_for_quality_feedback() -> None:
    seed_id = uuid4()
    request = _request(seed_id)

    class _Session:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(status="success")

    session = _Session()
    submission_buffer = CurationPlanSubmissionBuffer()
    planner = SessionCurationPlanner(cast(Any, session), submission_buffer=submission_buffer)
    submission_buffer.submit({"plan": _plan(request, seed_id).model_dump(mode="json")})
    first = await planner.create_plan(request, CurationPlannerTools(context=cast(Any, {})))
    planner.set_quality_feedback(
        {
            "latest_failed_live_run": {
                "diagnosis": "retrieval utility delta -1.2",
            }
        }
    )
    submission_buffer.submit({"plan": _plan(request, seed_id).model_dump(mode="json")})
    second = await planner.create_plan(request, CurationPlannerTools(context=cast(Any, {})))

    assert first.plan is not None
    assert second.plan is not None
    assert len(session.prompts) == 2
    assert "retrieval utility delta -1.2" in session.prompts[1]
    assert "Challenge weak split evidence" in session.prompts[1]
    assert "preserve exact search anchors" in session.prompts[1]
    assert "avoid speculative multi-action waves" in session.prompts[1]
    assert "change strategy rather than repeat" in session.prompts[1]
    assert "rejected_actions feedback as execution evidence" in session.prompts[1]


@pytest.mark.asyncio
async def test_session_planner_records_final_override_signal() -> None:
    seed_id = uuid4()
    request = _request(seed_id)

    class _Session:
        async def run_agent(self, prompt: str) -> AgenticRunResult:
            del prompt
            return AgenticRunResult(status="success")

    submission_buffer = CurationPlanSubmissionBuffer()
    planner = SessionCurationPlanner(cast(Any, _Session()), submission_buffer=submission_buffer)
    submission_buffer.submit(
        {
            "plan": _plan(request, seed_id).model_dump(mode="json"),
            "override_confidence": 0.95,
            "override_reason": "Measured utility remains positive.",
        }
    )
    result = await planner.create_plan(request, CurationPlannerTools(context=cast(Any, {})))

    assert result.plan is not None
    assert planner.override_confidence == pytest.approx(0.95)
    assert planner.override_reason == "Measured utility remains positive."


def test_planner_prompt_includes_bounded_curator_selection_metadata() -> None:
    memory_id = uuid4()
    context = build_context_packet(
        family="curator",
        strategy="quality-signal",
        seed_reads=[
            AcceptedMaintenanceRead(
                {
                    "id": memory_id,
                    "title": "A focused memory",
                    "content": "Authoritative content",
                    "summary": "Added a focused memory.",
                    "type": "observation",
                    "status": "active",
                    "tags": [],
                    "workspace_ids": [],
                    "read_count": 2,
                    "last_surfaced_at": "2026-01-01T00:00:00+00:00",
                    "content_size_chars": 22,
                    "size_band": "target",
                    "retrieval_friction_flags": [
                        "generic_summary",
                        "untagged_observation",
                        "raw_ingress",
                    ],
                    "selection_reason": "selected=quality-signal",
                    "selection_signals": {"quality_signal_share": 0.5},
                    "selection_scores": {"quality-signal": 0.7},
                    "quality_feedback": {
                        "reason": "retrieval_regression",
                        "escalation_count": 2,
                        "last_strategy": "semantic",
                    },
                }
            )
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )

    payload = json.loads(
        _build_planner_prompt(
            _request(memory_id),
            CurationPlannerTools(context=context),
        ).split("\n", 1)[1]
    )
    seed = payload["context"]["seeds"][0]

    assert seed["retrieval_friction_flags"] == [
        "generic_summary",
        "untagged_observation",
        "raw_ingress",
    ]
    assert seed["read_count"] == 2
    assert seed["last_surfaced_at"] == "2026-01-01T00:00:00+00:00"
    assert seed["content_size_chars"] == 22
    assert seed["size_band"] == "target"
    assert seed["selection_reason"] == "selected=quality-signal"
    assert seed["selection_signals"] == {"quality_signal_share": 0.5}
    assert seed["selection_scores"] == {"quality-signal": 0.7}
    assert seed["quality_feedback"]["reason"] == "retrieval_regression"
    assert seed["quality_feedback"]["escalation_count"] == 2
    assert seed["content"] == "Authoritative content"


def test_campaign_hypothesis_reaches_planner_request_and_context_payload() -> None:
    seed_id = uuid4()
    hypothesis = CampaignHypothesis(
        query="find the authentication goal",
        retrieval_problem=CampaignRetrievalProblem.RETRIEVAL_QUALITY,
        expected_memory_ids=[seed_id],
        minimum_improvement=0.5,
    )
    context = build_context_packet(
        family="curator",
        strategy="quality-signal",
        seed_reads=[
            AcceptedMaintenanceRead(
                {
                    "id": seed_id,
                    "title": "Authentication goal",
                    "content": "The durable authentication conclusion.",
                    "summary": "Authentication conclusion.",
                    "type": "fact",
                    "status": "active",
                    "tags": [],
                    "workspace_ids": [],
                }
            )
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
        campaign_hypothesis=hypothesis,
    )

    request = _request_for_packet(context, run_id=uuid4())
    payload = json.loads(
        _build_planner_prompt(request, CurationPlannerTools(context=context)).split("\n", 1)[1]
    )
    expected = hypothesis.model_dump(mode="json")

    assert request.campaign_hypothesis == hypothesis
    assert payload["request"]["campaign_hypothesis"] == expected
    assert payload["context"]["campaign_hypothesis"] == expected


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


def test_planner_prompt_preserves_contract_retry_diagnostics() -> None:
    request = _request(uuid4())
    feedback = CurationRetryFeedback(
        reason_code="contract_invalid",
        message="target_not_visible: target is absent from context",
        issue_codes=("target_not_visible",),
    )

    payload = json.loads(
        _build_planner_prompt(
            request,
            CurationPlannerTools(context=cast(Any, {}), retry_feedback=feedback),
        ).split("\n", 1)[1]
    )

    assert payload["retry_feedback"]["reason_code"] == "contract_invalid"
    assert payload["retry_feedback"]["issue_codes"] == ["target_not_visible"]


@pytest.mark.asyncio
async def test_planning_retries_provider_failure_with_bounded_error_feedback() -> None:
    seed_id = uuid4()
    context = build_context_packet(
        family="curator",
        strategy="quality-signal",
        seed_reads=[
            AcceptedMaintenanceRead(
                {
                    "id": seed_id,
                    "title": "A focused memory",
                    "content": "Authoritative content",
                    "summary": "A focused summary.",
                    "type": "observation",
                    "status": "active",
                    "tags": [],
                    "workspace_ids": [],
                }
            )
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    request = _request_for_packet(context, run_id=uuid4())

    class _RetryingPlanner:
        def __init__(self) -> None:
            self.feedback: list[Any] = []
            self.inner = FakeCurationPlanner(
                [
                    FakePlannerScenario(
                        failure=CurationPlannerProviderError(
                            "assistant completion was not usable",
                            reason_code="provider_failed",
                        )
                    ),
                    FakePlannerScenario(plan=_plan(request, seed_id)),
                ]
            )

        async def create_plan(self, request, tools):
            self.feedback.append(tools.retry_feedback)
            return await self.inner.create_plan(request, tools)

    planner = _RetryingPlanner()
    result = await plan_and_validate(
        planner,
        CurationPlanningInput(
            request=request,
            context=context,
            mutation_budget=CurationMutationBudget(),
        ),
    )

    assert result.plan is not None
    assert planner.inner.calls == 2
    assert result.retry_reason == "provider_failed"
    assert planner.feedback[0] is None
    assert planner.feedback[1].reason_code == "provider_failed"
    assert "assistant completion was not usable" in planner.feedback[1].message


@pytest.mark.asyncio
async def test_planning_retries_context_contract_failure() -> None:
    seed_id = uuid4()
    hidden_id = uuid4()
    context = build_context_packet(
        family="curator",
        strategy="quality-signal",
        seed_reads=[
            AcceptedMaintenanceRead(
                {
                    "id": seed_id,
                    "title": "A focused memory",
                    "content": "Authoritative content",
                    "summary": "A focused summary.",
                    "type": "observation",
                    "status": "active",
                    "tags": [],
                    "workspace_ids": [],
                }
            )
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    request = _request_for_packet(context, run_id=uuid4())
    invalid_plan = CurationPlan(
        plan_id=request.plan_id,
        run_id=request.run_id,
        frontier_key=request.frontier_key,
        context_fingerprint=request.context_fingerprint,
        seed_memory_ids=[seed_id],
        retained=[
            RetentionDecision(
                memory_id=seed_id,
                reason=RetentionReason.ALREADY_FOCUSED,
                rationale="still useful",
            )
        ],
        actions=[
            NormalizeMemoryAction(
                action_id=uuid4(),
                target_id=hidden_id,
                confidence=1,
                rationale="improve metadata",
                summary="A clearer summary.",
            )
        ],
        rationale="improve the visible neighborhood",
    )

    class _RetryingPlanner:
        def __init__(self) -> None:
            self.feedback: list[Any] = []
            self.inner = FakeCurationPlanner(
                [
                    FakePlannerScenario(plan=invalid_plan),
                    FakePlannerScenario(plan=_plan(request, seed_id)),
                ]
            )

        async def create_plan(self, request, tools):
            self.feedback.append(tools.retry_feedback)
            return await self.inner.create_plan(request, tools)

    planner = _RetryingPlanner()
    result = await plan_and_validate(
        planner,
        CurationPlanningInput(
            request=request,
            context=context,
            mutation_budget=CurationMutationBudget(),
        ),
    )

    assert result.plan is not None
    assert planner.inner.calls == 2
    assert result.retry_reason == "contract_invalid"
    assert planner.feedback[1].reason_code == "contract_invalid"
    assert "target_not_visible" in planner.feedback[1].message


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


@pytest.mark.asyncio
async def test_instrumented_adapter_classifies_provider_route_failure() -> None:
    class _FailingProvider:
        async def ask_json(self, prompt: str) -> dict[str, object]:
            del prompt
            raise RuntimeError("route unavailable")

    planner = InstrumentedCurationPlanner(
        _FailingProvider(),
        provider_key="route-a",
        provider_profile="profile-a",
        model_name="model-a",
        route_available=True,
    )

    with pytest.raises(CurationPlannerProviderError) as raised:
        await planner.create_plan(_request(uuid4()), object())

    error = raised.value
    assert error.reason_code == "provider_execution_error"
    assert error.reason_category == "execution"
    assert error.route_available is True
    assert error.envelope is not None
    assert error.envelope.metadata["provider_profile"] == "profile-a"
    assert error.envelope.metadata["route_available"] is True

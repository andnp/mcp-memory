"""Provider-neutral planner contracts and a deterministic test planner."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from mcp_memory.core.curation_context import (
    CurationContextPacket as ImmutableCurationContextPacket,
)
from mcp_memory.core.curation_models import (
    CurationAction,
    CurationContextPacket,
    CurationPlan,
    CurationPlanningRequest,
    RetentionDecision,
)
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.providers.interfaces import AgenticSession, ProviderJSONCall


class CurationReadTools(Protocol):
    """Read-only tools supplied by the harness to a planner."""

    ...


T = TypeVar("T")


class PlannerExecutionStatus(str):
    COMPLETED = "completed"
    SCHEMA_FAILED = "schema_failed"
    PROVIDER_FAILED = "provider_failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PlannerExecutionEnvelope(Generic[T]):
    """The provider lifecycle and telemetry accompanying a planner result."""

    plan: T | None
    provider_key: str
    model_name: str
    request_id: str | None
    attempt: int
    started_at: datetime
    completed_at: datetime
    status: str = PlannerExecutionStatus.COMPLETED
    reason_code: str | None = None
    cancellation_requested: bool = False
    admission_status: str | None = None
    retry_delay_seconds: float | None = None
    transcript_ref: str | None = None
    provider_call: bool | None = None
    token_usage: int | None = None
    token_usage_source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def result(self) -> T | None:
        """Alias used by adapters that call the typed result a result."""
        return self.plan


@dataclass(frozen=True, slots=True)
class CurationPlannerSchemaIssue:
    code: str
    message: str
    field: str | None = None


class CurationPlannerError(Exception):
    """Base class for failures returned by a planner boundary."""

    def __init__(self, message: str, *, envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        super().__init__(message)
        self.envelope = envelope


class CurationPlannerSchemaError(CurationPlannerError):
    """The provider response was not a valid typed curation plan."""

    reason_code = "schema_invalid"

    def __init__(
        self,
        message: str,
        *,
        validation_error: ValidationError | None = None,
        validation_issues: tuple[CurationPlannerSchemaIssue, ...] = (),
        expected_fields: tuple[str, ...] = (),
        received_fields: tuple[str, ...] = (),
        envelope: PlannerExecutionEnvelope[None] | None = None,
    ) -> None:
        super().__init__(message, envelope=envelope)
        self.validation_error = validation_error
        self.validation_issues = validation_issues
        self.expected_fields = expected_fields
        self.received_fields = received_fields


class CurationPlannerProviderError(CurationPlannerError):
    """The provider failed before producing a usable plan."""

    reason_code = "provider_failed"

    def __init__(self, message: str, *, reason_code: str = "provider_failed",
                 reason_category: str | None = None,
                 retry_delay_seconds: float | None = None,
                 route_available: bool | None = None,
                 envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        super().__init__(message, envelope=envelope)
        self.reason_code = reason_code
        self.reason_category = reason_category
        self.retry_delay_seconds = retry_delay_seconds
        self.route_available = route_available


class CurationPlannerCancelledError(asyncio.CancelledError, CurationPlannerError):
    """A planner attempt was cancelled, retaining its execution envelope."""

    reason_code = "provider_cancelled"

    def __init__(self, message: str = "curation planner was cancelled",
                 *, envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        CurationPlannerError.__init__(self, message, envelope=envelope)


class CurationPlanner(Protocol):
    async def create_plan(
        self, request: CurationPlanningRequest, tools: CurationReadTools
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        """Create a typed plan without owning execution or lifecycle state."""

        ...


@dataclass(slots=True)
class CurationPlanSubmissionBuffer:
    payload: Mapping[str, Any] | None = None

    def submit(self, payload: Any) -> None:
        self.payload = payload if isinstance(payload, MappingABC) else None

    def consume(self) -> Mapping[str, Any] | None:
        payload = self.payload
        self.payload = None
        return payload


@dataclass(slots=True)
class IncrementalCurationPlanBuilder:
    """Accumulate a typed plan without mutating memory storage."""

    request: CurationPlanningRequest
    seed_memory_ids: tuple[UUID, ...]
    _actions: list[CurationAction] = field(default_factory=list)
    _retained: list[RetentionDecision] = field(default_factory=list)

    @property
    def action_count(self) -> int:
        return len(self._actions)

    def add_action(self, action: CurationAction) -> None:
        if any(existing.action_id == action.action_id for existing in self._actions):
            raise ValueError(f"action {action.action_id} has already been proposed")
        self._actions.append(action)

    def replace_action(self, action: CurationAction) -> None:
        for index, existing in enumerate(self._actions):
            if existing.action_id == action.action_id:
                self._actions[index] = action
                return
        raise ValueError(f"action {action.action_id} has not been proposed")

    def remove_action(self, action_id: UUID) -> None:
        for index, action in enumerate(self._actions):
            if action.action_id == action_id:
                del self._actions[index]
                return
        raise ValueError(f"action {action_id} has not been proposed")

    def add_retention(self, decision: RetentionDecision) -> None:
        if any(existing.memory_id == decision.memory_id for existing in self._retained):
            raise ValueError(f"memory {decision.memory_id} already has a retention decision")
        self._retained.append(decision)

    def build(self, *, rationale: str) -> CurationPlan:
        return CurationPlan(
            plan_id=self.request.plan_id,
            run_id=self.request.run_id,
            frontier_key=self.request.frontier_key,
            context_fingerprint=self.request.context_fingerprint,
            actions=list(self._actions),
            retained=list(self._retained),
            rationale=rationale,
            seed_memory_ids=list(self.seed_memory_ids),
        )


@dataclass(slots=True)
class IncrementalCurationPlanState:
    """Session-owned state shared by incremental planning tools."""

    max_proposed_actions: int | None = None
    builder: IncrementalCurationPlanBuilder | None = None

    def begin(
        self,
        request: CurationPlanningRequest,
        *,
        seed_memory_ids: tuple[UUID, ...],
    ) -> None:
        self.builder = IncrementalCurationPlanBuilder(request, seed_memory_ids)

    def require_builder(self) -> IncrementalCurationPlanBuilder:
        if self.builder is None:
            raise RuntimeError("incremental curation plan has not started")
        return self.builder

    def add_action(self, action: CurationAction) -> None:
        builder = self.require_builder()
        if (
            self.max_proposed_actions is not None
            and builder.action_count >= self.max_proposed_actions
        ):
            raise ValueError("proposed action budget is exhausted")
        builder.add_action(action)

    def replace_action(self, action: CurationAction) -> None:
        self.require_builder().replace_action(action)

    def remove_action(self, action_id: UUID) -> None:
        self.require_builder().remove_action(action_id)

    def add_retention(self, decision: RetentionDecision) -> None:
        self.require_builder().add_retention(decision)

    def build(self, *, rationale: str) -> CurationPlan:
        return self.require_builder().build(rationale=rationale)


class CancellationScope(Protocol):
    """Minimal cancellation surface accepted by the provider adapter."""

    def is_cancelled(self) -> bool:
        ...


@dataclass(frozen=True, slots=True)
class _AdapterPlannerTools:
    context: Any


class InstrumentedCurationPlanner:
    """Adapt an instrumented JSON provider to the typed curation planner."""

    def __init__(
        self,
        provider,
        *,
        provider_key: str = "provider",
        provider_name: str = "provider",
        model_name: str = "model",
        provider_profile: str | None = None,
        route_available: bool | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        supports_agentic = getattr(provider, "supports_agentic", None)
        agentic = supports_agentic() if callable(supports_agentic) else callable(getattr(provider, "run_agent", None))
        if agentic:
            raise ValueError("curation planner requires a JSON-only provider")
        self._provider = provider
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._model_name = model_name
        self._provider_profile = provider_profile or _provider_profile(provider, provider_key)
        self._route_available = (
            route_available if route_available is not None else _route_available(provider)
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    async def create_plan(
        self, request: CurationPlanningRequest, tools: CurationReadTools
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        prompt = _build_planner_prompt(request, tools)
        started_at = self._clock()
        try:
            call = await self._call_json(prompt, started_at=started_at)
        except asyncio.CancelledError as exc:
            recorded_call = getattr(exc, "provider_json_call", None)
            envelope = (
                PlannerExecutionEnvelope[None](
                    plan=None,
                    status=PlannerExecutionStatus.CANCELLED,
                    reason_code="provider_cancelled",
                    cancellation_requested=True,
                    **self._envelope_base(cast(ProviderJSONCall, recorded_call)),
                )
                if isinstance(recorded_call, ProviderJSONCall)
                else self._local_cancellation_envelope(request, started_at)
            )
            raise CurationPlannerCancelledError(envelope=envelope) from exc
        return self._translate_call(request, call)

    async def plan(
        self,
        context_packet: ImmutableCurationContextPacket | CurationContextPacket,
        *,
        run_id: UUID,
        cancellation: CancellationScope | asyncio.Event | None = None,
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        """Plan from an already bounded packet without exposing provider tools."""

        if _cancellation_requested(cancellation):
            now = self._clock()
            request = _request_for_packet(context_packet, run_id=run_id)
            envelope = self._local_cancellation_envelope(request, now)
            raise CurationPlannerCancelledError(envelope=envelope)
        request = _request_for_packet(context_packet, run_id=run_id)
        result = await self.create_plan(request, _AdapterPlannerTools(context_packet))
        if _cancellation_requested(cancellation):
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                provider_key=result.provider_key,
                model_name=result.model_name,
                request_id=result.request_id,
                attempt=result.attempt,
                started_at=result.started_at,
                completed_at=result.completed_at,
                status=PlannerExecutionStatus.CANCELLED,
                reason_code="provider_cancelled",
                cancellation_requested=True,
                admission_status=result.admission_status,
                retry_delay_seconds=result.retry_delay_seconds,
                transcript_ref=result.transcript_ref,
                provider_call=result.provider_call,
                token_usage=result.token_usage,
                token_usage_source=result.token_usage_source,
                metadata=result.metadata,
            )
            raise CurationPlannerCancelledError(envelope=envelope)
        return result

    async def _call_json(self, prompt: str, *, started_at: datetime) -> ProviderJSONCall:
        instrumented_call = getattr(self._provider, "ask_json_with_telemetry", None)
        if callable(instrumented_call):
            return await cast(Callable[[str], Any], instrumented_call)(prompt)
        ask_json = getattr(self._provider, "ask_json", None)
        if not callable(ask_json):
            completed_at = self._clock()
            return ProviderJSONCall(
                response=None,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                request_id="",
                attempt=0,
                started_at=started_at.timestamp(),
                completed_at=completed_at.timestamp(),
                status="error",
                error_text="provider does not support JSON planning",
                reason_code="provider_capability_missing",
                reason_category="routing",
                admission_status="not_started",
                provider_call=False,
            )
        try:
            response = await cast(Callable[[str], Any], ask_json)(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            completed_at = self._clock()
            classification = classify_provider_failure(exc)
            return ProviderJSONCall(
                response=None,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                request_id="",
                attempt=1,
                started_at=started_at.timestamp(),
                completed_at=completed_at.timestamp(),
                status="error",
                error_text=str(exc),
                reason_code=classification.reason_code,
                reason_category=classification.reason_category,
                retry_delay_seconds=classification.retry_delay_seconds,
                admission_status="admitted",
                provider_call=True,
            )
        completed_at = self._clock()
        parsed_response: dict[str, Any] | None = response if isinstance(response, dict) else None
        response_status = "success"
        response_error: str | None = None
        if isinstance(response, str):
            try:
                parsed = json.loads(response)
            except json.JSONDecodeError as exc:
                response_status = "parse_error"
                response_error = str(exc)
            else:
                if isinstance(parsed, dict):
                    parsed_response = parsed
                else:
                    response_status = "parse_error"
                    response_error = "provider JSON response was not an object"
        elif not isinstance(response, dict):
            response_status = "parse_error"
            response_error = "provider JSON response was not an object"
        return ProviderJSONCall(
            response=parsed_response,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            request_id="",
            attempt=1,
            started_at=started_at.timestamp(),
            completed_at=completed_at.timestamp(),
            status=response_status,
            error_text=response_error,
            raw_text=json.dumps(response, sort_keys=True) if isinstance(response, dict) else str(response),
            parsed=parsed_response,
            admission_status="admitted",
            provider_call=True,
        )

    def _translate_call(
        self,
        request: CurationPlanningRequest,
        call: ProviderJSONCall,
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        envelope_base = self._envelope_base(call)
        if call.status == "cancelled" or call.cancellation_requested:
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=PlannerExecutionStatus.CANCELLED,
                reason_code="provider_cancelled",
                cancellation_requested=True,
                **envelope_base,
            )
            raise CurationPlannerCancelledError(envelope=envelope)
        if call.response is None:
            if call.status == "parse_error":
                schema_error = _schema_error(
                    "provider returned invalid JSON for curation planning",
                    expected_fields=_expected_plan_fields(),
                    received_fields=(),
                    envelope_base=envelope_base,
                )
                raise schema_error
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=PlannerExecutionStatus.PROVIDER_FAILED,
                reason_code=call.reason_code or "provider_failed",
                **envelope_base,
            )
            raise CurationPlannerProviderError(
                call.error_text or "provider failed during curation planning",
                reason_code=call.reason_code or "provider_failed",
                reason_category=call.reason_category,
                retry_delay_seconds=call.retry_delay_seconds,
                route_available=self._route_available,
                envelope=envelope,
            )
        try:
            plan = CurationPlan.model_validate(call.response)
        except ValidationError as exc:
            raise _schema_error(
                "provider returned JSON that is not a valid curation plan",
                validation_error=exc,
                expected_fields=_expected_plan_fields(),
                received_fields=tuple(sorted(call.response)),
                envelope_base=envelope_base,
            ) from exc
        return PlannerExecutionEnvelope(plan=plan, **envelope_base)

    def _local_cancellation_envelope(
        self, request: CurationPlanningRequest, started_at: datetime
    ) -> PlannerExecutionEnvelope[None]:
        return PlannerExecutionEnvelope(
            plan=None,
            provider_key=self._provider_key,
            model_name=self._model_name,
            request_id=None,
            attempt=0,
            started_at=started_at,
            completed_at=self._clock(),
            status=PlannerExecutionStatus.CANCELLED,
            reason_code="provider_cancelled",
            cancellation_requested=True,
            admission_status="not_started",
            provider_call=False,
        )

    def _envelope_base(self, call: ProviderJSONCall) -> dict[str, Any]:
        return _envelope_base(
            call,
            provider_profile=self._provider_profile,
            route_available=self._route_available,
        )


class SessionCurationPlanner(InstrumentedCurationPlanner):
    """Use one persistent agent conversation for typed planning turns."""

    def __init__(
        self,
        session: AgenticSession,
        *,
        submission_buffer: CurationPlanSubmissionBuffer | None = None,
        plan_state: IncrementalCurationPlanState | None = None,
        provider_key: str = "agentic-session",
        provider_name: str = "agentic-session",
        model_name: str = "agentic-session",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._session = session
        self._submission_buffer = submission_buffer
        self._plan_state = plan_state
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._model_name = model_name
        self._provider_profile = provider_key
        self._route_available = True
        self._clock = clock or (lambda: datetime.now(UTC))
        self._quality_feedback: Mapping[str, object] | None = None
        self.override_confidence: float | None = None
        self.override_reason: str | None = None
        self.override_eligible = False

    def set_quality_feedback(self, feedback: Mapping[str, object] | None) -> None:
        self._quality_feedback = None if feedback is None else dict(feedback)
        self.override_eligible = feedback is not None

    async def create_plan(
        self, request: CurationPlanningRequest, tools: CurationReadTools
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        if self._plan_state is not None:
            context = getattr(tools, "context", None)
            seed_memory_ids = tuple(getattr(context, "seed_memory_ids", ()))
            self._plan_state.begin(request, seed_memory_ids=seed_memory_ids)
            prompt = _build_planner_prompt(
                request,
                tools,
                incremental_tool_names=(
                    "propose_curation_action",
                    "replace_curation_action",
                    "remove_curation_action",
                    "retain_curation_memory",
                ),
            )
        else:
            prompt = _build_planner_prompt(
                request,
                tools,
                submission_tool_name="submit_curation_plan",
            )
        if self._quality_feedback is not None:
            prompt += "\nMeasured quality feedback:\n" + json.dumps(
                self._quality_feedback, sort_keys=True, default=str
            )
        started_at = self._clock()
        try:
            result = await self._session.run_agent(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            completed_at = self._clock()
            call = ProviderJSONCall(
                response=None,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                request_id=str(uuid4()),
                attempt=1,
                started_at=started_at.timestamp(),
                completed_at=completed_at.timestamp(),
                status="error",
                error_text=str(exc),
                reason_code="provider_failed",
                reason_category="execution",
                admission_status="admitted",
                provider_call=True,
            )
            return self._translate_call(request, call)
        if self._plan_state is not None:
            raw_text = getattr(result, "raw_text", None)
            rationale = (
                raw_text.strip()[:2000]
                if isinstance(raw_text, str) and raw_text.strip()
                else "Incremental plan assembled from typed curation actions."
            )
            try:
                plan = self._plan_state.build(rationale=rationale)
            except ValueError as exc:
                envelope = PlannerExecutionEnvelope[None](
                    plan=None,
                    provider_key=self._provider_key,
                    model_name=self._model_name,
                    request_id=str(uuid4()),
                    attempt=1,
                    started_at=started_at,
                    completed_at=self._clock(),
                    status=PlannerExecutionStatus.SCHEMA_FAILED,
                    reason_code="schema_invalid",
                    provider_call=True,
                    metadata={"error": str(exc)},
                )
                raise CurationPlannerSchemaError(
                    "incremental curation plan is incomplete",
                    validation_issues=(
                        CurationPlannerSchemaIssue(
                            code="invalid_incremental_plan",
                            message=str(exc),
                        ),
                    ),
                    expected_fields=_expected_plan_fields(),
                    envelope=envelope,
                ) from exc
            return PlannerExecutionEnvelope(
                plan=plan,
                provider_key=self._provider_key,
                model_name=self._model_name,
                request_id=str(uuid4()),
                attempt=1,
                started_at=started_at,
                completed_at=self._clock(),
                provider_call=True,
            )
        submission_buffer = self._submission_buffer
        if submission_buffer is None:
            raise RuntimeError("legacy curation submission buffer is unavailable")
        submission = submission_buffer.consume()
        parsed = None if submission is None else dict(submission)
        self.override_confidence = _bounded_override_confidence(
            None if parsed is None else parsed.get("override_confidence")
        )
        self.override_reason = (
            None
            if parsed is None or not isinstance(parsed.get("override_reason"), str)
            else parsed["override_reason"].strip() or None
        )
        if isinstance(parsed, dict) and isinstance(parsed.get("plan"), dict):
            parsed = cast(dict[str, Any], parsed["plan"])
        call = ProviderJSONCall(
            response=parsed,
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            request_id=str(uuid4()),
            attempt=1,
            started_at=started_at.timestamp(),
            completed_at=self._clock().timestamp(),
            status="success" if parsed is not None else "parse_error",
            error_text=None if parsed is not None else "agentic session did not submit a curation plan",
            raw_text=result.raw_text,
            parsed=parsed,
            admission_status="admitted",
            provider_call=True,
        )
        return self._translate_call(request, call)

# Descriptive alias for callers that name the boundary after the provider.
ProviderCurationPlanner = InstrumentedCurationPlanner


@dataclass(frozen=True, slots=True)
class FakePlannerScenario:
    """One scripted fake result; scenarios are consumed in call order."""

    plan: CurationPlan | Mapping[str, Any] | None = None
    failure: CurationPlannerError | None = None
    provider_key: str = "fake"
    model_name: str = "fake-model"
    request_id: str | None = None
    attempt: int = 1
    transcript_ref: str | None = None
    provider_call: bool | None = None
    token_usage: int | None = None
    token_usage_source: str | None = None
    admission_status: str | None = "admitted"
    retry_delay_seconds: float | None = None


class FakeCurationPlanner:
    """A deterministic, storage-free planner whose scenarios are scriptable."""

    def __init__(
        self,
        scenarios: list[FakePlannerScenario] | tuple[FakePlannerScenario, ...],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not scenarios:
            raise ValueError("at least one fake planner scenario is required")
        self._scenarios = tuple(scenarios)
        self._clock = clock or (lambda: datetime(1970, 1, 1, tzinfo=UTC))
        self.calls = 0

    async def create_plan(
        self, request: CurationPlanningRequest, tools: CurationReadTools
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        del tools
        index = min(self.calls, len(self._scenarios) - 1)
        scenario = self._scenarios[index]
        self.calls += 1
        started_at = self._clock()
        completed_at = self._clock()
        request_id = scenario.request_id or f"fake-request-{self.calls}"
        base = dict(
            provider_key=scenario.provider_key,
            model_name=scenario.model_name,
            request_id=request_id,
            attempt=scenario.attempt,
            started_at=started_at,
            completed_at=completed_at,
            admission_status=scenario.admission_status,
            retry_delay_seconds=scenario.retry_delay_seconds,
            transcript_ref=scenario.transcript_ref,
            provider_call=scenario.provider_call,
            token_usage=scenario.token_usage,
            token_usage_source=scenario.token_usage_source,
        )
        if scenario.failure is not None:
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=_failure_status(scenario.failure),
                reason_code=getattr(scenario.failure, "reason_code", None),
                cancellation_requested=isinstance(scenario.failure, asyncio.CancelledError),
                **cast(Any, base),
            )
            scenario.failure.envelope = envelope
            raise scenario.failure
        try:
            plan = scenario.plan if isinstance(scenario.plan, CurationPlan) else CurationPlan.model_validate(scenario.plan)
        except ValidationError as exc:
            raise _schema_error(
                "fake planner returned an invalid curation plan",
                validation_error=exc,
                expected_fields=_expected_plan_fields(),
                received_fields=tuple(sorted(scenario.plan)) if isinstance(scenario.plan, Mapping) else (),
                envelope_base=cast(Any, base),
            ) from exc
        return PlannerExecutionEnvelope(plan=plan, **cast(Any, base))


def _failure_status(failure: CurationPlannerError) -> str:
    if isinstance(failure, asyncio.CancelledError):
        return PlannerExecutionStatus.CANCELLED
    if isinstance(failure, CurationPlannerSchemaError):
        return PlannerExecutionStatus.SCHEMA_FAILED
    return PlannerExecutionStatus.PROVIDER_FAILED


def _request_for_packet(
    context_packet: ImmutableCurationContextPacket | CurationContextPacket,
    *,
    run_id: UUID,
) -> CurationPlanningRequest:
    if isinstance(context_packet, CurationContextPacket):
        return CurationPlanningRequest(
            run_id=run_id,
            plan_id=uuid4(),
            frontier_key=context_packet.context_fingerprint,
            context_fingerprint=context_packet.context_fingerprint,
            campaign_hypothesis=context_packet.campaign_hypothesis,
            context=context_packet,
        )
    return CurationPlanningRequest(
        run_id=run_id,
        plan_id=uuid4(),
        frontier_key=context_packet.frontier_fingerprint,
        context_fingerprint=context_packet.context_fingerprint,
        campaign_hypothesis=context_packet.campaign_hypothesis,
        context=CurationContextPacket.from_visible_ids(
            seed_memory_ids=context_packet.seed_memory_ids,
            support_memory_ids=context_packet.support_memory_ids,
            exploratory_memory_ids=context_packet.exploratory_memory_ids,
            context_fingerprint=context_packet.context_fingerprint,
            campaign_hypothesis=context_packet.campaign_hypothesis,
        ),
    )


def _build_planner_prompt(
    request: CurationPlanningRequest,
    tools: CurationReadTools,
    *,
    submission_tool_name: str | None = None,
    incremental_tool_names: tuple[str, ...] | None = None,
) -> str:
    context = getattr(tools, "context", None)
    if context is None:
        context_payload: Any = None
    elif callable(getattr(context, "as_dict", None)):
        context_payload = context.as_dict()
    elif callable(getattr(context, "model_dump", None)):
        context_payload = context.model_dump(mode="json")
    elif isinstance(context, MappingABC):
        context_payload = dict(context)
    else:
        context_payload = str(context)
    payload = {
        "request": request.model_dump(mode="json"),
        "context": context_payload,
        "available_tools": list(incremental_tool_names or ()),
        "schema": CurationPlan.model_json_schema(),
        "planner_contract": [
            "The primary objective is to improve durable memory quality, not to minimize action count or maximize retention.",
            "Diagnose each seed for focused scope, conclusion-first wording, concrete evidence, title and summary specificity, stale status prose, duplication, and missing relationships before deciding.",
            "Treat raw_ingress as a likely system1-created record: normalize weak metadata, and use rewrite_memory when the body itself is raw, status-shaped, or combines unrelated takeaways.",
            "Use support and exploratory records as active repair candidates. Any record visible in context may be modified; only initial seeds require exactly one disposition.",
            "Prefer a coherent set of mutations when several changes together improve a local memory cluster, even if one isolated change would be incomplete.",
            "Review the full visible context before stopping; when several independent, evidence-backed improvements exist, propose them in the same turn and use the available action budget rather than stopping after the first valid mutation.",
            "Retain a seed only when it is already focused and durable, or when the available evidence does not support a specific improvement hypothesis.",
            "Every target_id, source_id, canonical_id, source_ids entry, and retained memory_id must refer to a memory visible in context.",
            "seed_memory_ids is required and must exactly equal the initial context seed IDs; every seed must receive exactly one disposition.",
            "Exploratory records are separately audited visible context. Actions may use their IDs, but exploratory IDs must never be silently counted as initial seeds.",
            "merge_memories canonical_id must be an existing visible record; merging never creates a record.",
            "create_link and remove_link require both endpoints to be visible in context.",
            "Every action affecting a visible memory must include its exact context.record_tokens value in preconditions.record_tokens.",
            "create_link requires descriptive relationship context with at least two words, exact endpoint/type/context link evidence, and an exact absent-link precondition containing only source_id, target_id, and link_type; labels and memory excerpts alone are not link evidence.",
            "Only split_memory may introduce child records, and only through its typed children surface; never invent child IDs.",
            "For rewrite_memory and merge_memories, content is the final persisted durable memory body, not a description of the mutation.",
            "For merge_memories, canonical_id is the retained base; preserve its supported claims, integrate justified source claims, and make content stand alone after sources are archived.",
            "Use claim_manifest to map preserved or transformed claims to source records; do not use it as a substitute for writing those claims into the final content.",
            "Do not write mutation-status prose such as 'merged ... into the canonical' or 'added ... to the record' as memory content.",
            "Each seed memory must have exactly one disposition: it appears in an action's affected IDs or in retained, never both; action source, target, canonical, and child-source IDs all count as affected.",
            "Prefer one conclusion-first takeaway with concrete evidence; target 1600 characters or less, and split content above 3000 characters when it contains multiple takeaways.",
            "Treat selection signals and retrieval-friction flags as review clues, not proof; inspect the record content and relationships before mutating.",
            "A quality_feedback field means a previous curator mutation caused a measured retrieval regression; do not repeat that mutation blindly, but investigate and propose a different targeted repair when the evidence supports one.",
            "Action failure feedback is execution evidence: repair listed contract, stale, or transient failures instead of repeating the same invalid action shape; keep terminal safety failures out of the next plan.",
            "For quality-feedback records, preserve exact search anchors, concrete entities, and the current durable meaning. Retain only when no evidence-backed repair hypothesis survives investigation.",
            "Challenge weak split evidence, preserve exact search anchors, avoid speculative multi-action waves, and use measured feedback to change strategy rather than repeat.",
            "When no visible canonical is appropriate, make a retention decision instead of inventing an ID or proposing a merge, link, or normalize action against one.",
            "These constraints are fail-closed: an action with an ID absent from context is invalid and must not be executed.",
        ],
    }
    retry_feedback = getattr(tools, "retry_feedback", None)
    if retry_feedback is not None:
        payload["retry_feedback"] = {
            "reason_code": str(retry_feedback.reason_code),
            "message": str(retry_feedback.message)[:1000],
            "fields": list(retry_feedback.fields),
            "issue_codes": list(retry_feedback.issue_codes),
            "expected_fields": list(retry_feedback.expected_fields),
            "received_fields": list(retry_feedback.received_fields),
        }
    quality_feedback = getattr(tools, "quality_feedback", None)
    if quality_feedback is not None:
        payload["quality_feedback"] = dict(quality_feedback)
    instruction = (
        "You are a curation planner whose job is to improve the durable quality of "
        "the memory base. Do not optimize for safe no-ops: inspect the visible "
        "neighborhood, find the highest-confidence quality gains, and make a coherent "
        "set of changes when the evidence supports them. Return exactly one JSON "
        "object matching the provided CurationPlan JSON schema. Include "
        "seed_memory_ids exactly as the "
        "context seed IDs (the initial context seed IDs); exploratory records are visible but separately "
        "audited. Planning only: do not execute mutations, "
        "call tools, or report a claimed action count; actions are counted only after "
        "validation. Every memory ID in an action or retention decision must be copied "
        "from a memory visible in the provided context; any visible seed, support, or "
        "exploratory memory may be changed. A merge canonical_id must be "
        "an existing visible record and merge never creates records; both create-link "
        "endpoints must be visible. Copy exact record tokens from context.record_tokens "
        "into preconditions.record_tokens for every affected visible memory. A "
        "create_link requires descriptive relationship context with at least two "
        "words, an evidence.link matching the exact endpoints, link type, and "
        "context, plus an absent-link precondition containing only source_id, "
        "target_id, and link_type; labels and memory excerpts alone are not link "
        "evidence. "
        "Only split may introduce child "
        "records through its typed children surface. For rewrite and merge actions, "
        "content is the final persisted durable memory body, not a description of the "
        "mutation: preserve the canonical's supported claims, integrate justified "
        "source claims, and do not write status prose such as 'merged into the "
        "canonical'. Give every seed exactly one disposition: put it in an action "
        "or in retained, never both; any action source, target, canonical, or "
        "child-source ID counts as acted on. Use one conclusion-first takeaway with concrete evidence, target "
        "1600 characters or less, and split multi-takeaway content above 3000 "
        "characters. A quality_feedback field marks a previous curator mutation as a "
        "measured retrieval regression: do not repeat that mutation blindly; investigate "
        "and propose a different targeted repair when evidence supports one. Treat "
        "rejected_actions feedback as execution evidence: repair listed contract, stale, "
        "or transient failures instead of repeating the same invalid action shape. "
        "raw_ingress as a likely system1-created record: normalize weak metadata, and "
        "use rewrite_memory when the body itself is raw, status-shaped, or combines "
        "unrelated takeaways. Preserve "
        "exact search anchors and concrete entities; retain only when no evidence-backed "
        "repair hypothesis survives investigation. If no visible canonical is appropriate, retain the memory instead "
        "of inventing an ID. Review the full visible context before stopping; when "
        "several independent, evidence-backed improvements exist, propose them in "
        "the same turn and use the available action budget rather than stopping "
        "after the first valid mutation. Do not manufacture work for focused records."
    )
    if submission_tool_name is not None:
        instruction += (
            f" Call the {submission_tool_name} tool exactly once with the complete plan; "
            "the tool submission is authoritative and assistant text is not parsed as a plan."
        )
    elif incremental_tool_names is not None:
        instruction += (
            " Build the plan incrementally using the typed MCP tools. Call "
            "propose_curation_action once per mutation, retain_curation_memory once "
            "per retained seed, and use replace_curation_action or "
            "remove_curation_action when correcting an earlier proposal. Do not "
            "return a complete JSON plan; the tool calls are authoritative."
        )
    return instruction + "\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _envelope_base(
    call: ProviderJSONCall,
    *,
    provider_profile: str | None = None,
    route_available: bool | None = None,
) -> dict[str, Any]:
    token_usage = call.token_usage
    token_usage_source = call.token_usage_source
    if call.token_usage_details is not None:
        if token_usage is None:
            token_usage = call.token_usage_details.total_tokens
        if token_usage_source is None:
            token_usage_source = call.token_usage_details.source
    return {
        "provider_key": call.provider_key,
        "model_name": call.model_name,
        "request_id": call.request_id or None,
        "attempt": call.attempt,
        "started_at": datetime.fromtimestamp(call.started_at, tz=UTC),
        "completed_at": datetime.fromtimestamp(call.completed_at, tz=UTC),
        "admission_status": call.admission_status,
        "retry_delay_seconds": call.retry_delay_seconds,
        "transcript_ref": call.request_id or None,
        "provider_call": call.provider_call,
        "token_usage": token_usage,
        "token_usage_source": token_usage_source,
        "metadata": {
            "provider_status": call.status,
            "raw_response_recorded": call.raw_text is not None,
            "provider_profile": provider_profile or call.provider_key,
            "route_available": route_available,
            "reason_category": getattr(call, "reason_category", None),
            "error_text": call.error_text,
            "retry_delay_seconds": call.retry_delay_seconds,
        },
    }


def _provider_profile(provider: Any, provider_key: str) -> str:
    for name in ("provider_profile", "_provider_profile", "profile_name", "_profile_name"):
        value = getattr(provider, name, None)
        if value is not None:
            return str(value)
    return provider_key


def _bounded_override_confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if 0.0 <= float(value) <= 1.0:
        return float(value)
    return None


def _route_available(provider: Any) -> bool | None:
    for name in ("route_available", "_route_available", "alternative_route_available"):
        value = getattr(provider, name, None)
        if value is not None:
            return bool(value)
    return None


def _expected_plan_fields() -> tuple[str, ...]:
    return tuple(sorted(CurationPlan.model_fields))


def _schema_error(
    message: str,
    *,
    validation_error: ValidationError | None = None,
    expected_fields: tuple[str, ...],
    received_fields: tuple[str, ...],
    envelope_base: Mapping[str, Any],
) -> CurationPlannerSchemaError:
    issues = _schema_issues(validation_error)
    metadata = dict(envelope_base.get("metadata", {}))
    metadata.update(
        {
            "validation_issues": [
                {"code": issue.code, "message": issue.message, "field": issue.field}
                for issue in issues
            ],
            "expected_fields": list(expected_fields),
            "received_fields": list(received_fields),
        }
    )
    envelope = PlannerExecutionEnvelope[None](
        plan=None,
        status=PlannerExecutionStatus.SCHEMA_FAILED,
        reason_code=CurationPlannerSchemaError.reason_code,
        **{**dict(envelope_base), "metadata": metadata},
    )
    return CurationPlannerSchemaError(
        message,
        validation_error=validation_error,
        validation_issues=issues,
        expected_fields=expected_fields,
        received_fields=received_fields,
        envelope=envelope,
    )


def _schema_issues(error: ValidationError | None) -> tuple[CurationPlannerSchemaIssue, ...]:
    if error is None:
        return (CurationPlannerSchemaIssue("invalid_json", "provider response was not valid JSON"),)
    return tuple(
        CurationPlannerSchemaIssue(
            code=str(item.get("type", "validation_error")),
            message=str(item.get("msg", "schema validation failed")),
            field=".".join(str(part) for part in item.get("loc", ())) or None,
        )
        for item in error.errors()
    )


def _cancellation_requested(scope: CancellationScope | asyncio.Event | None) -> bool:
    if scope is None:
        return False
    is_set = getattr(scope, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    is_cancelled = getattr(scope, "is_cancelled", None)
    return callable(is_cancelled) and bool(is_cancelled())

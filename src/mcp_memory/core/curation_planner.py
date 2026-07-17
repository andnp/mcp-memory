"""Provider-neutral planner contracts and a deterministic test planner."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from collections.abc import Mapping as MappingABC
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, cast
from uuid import UUID, uuid4

from pydantic import ValidationError

from mcp_memory.core.curation_models import CurationContextPacket
from mcp_memory.core.curation_models import CurationPlan, CurationPlanningRequest
from mcp_memory.core.curation_context import CurationContextPacket as ImmutableCurationContextPacket
from mcp_memory.core.providers.interfaces import ProviderJSONCall


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
    premium_request: bool | None = None
    token_usage: int | None = None
    token_usage_source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def result(self) -> T | None:
        """Alias used by adapters that call the typed result a result."""
        return self.plan


class CurationPlannerError(Exception):
    """Base class for failures returned by a planner boundary."""

    def __init__(self, message: str, *, envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        super().__init__(message)
        self.envelope = envelope


class CurationPlannerSchemaError(CurationPlannerError):
    """The provider response was not a valid typed curation plan."""

    reason_code = "schema_invalid"

    def __init__(self, message: str, *, validation_error: ValidationError | None = None,
                 envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        super().__init__(message, envelope=envelope)
        self.validation_error = validation_error


class CurationPlannerProviderError(CurationPlannerError):
    """The provider failed before producing a usable plan."""

    reason_code = "provider_failed"

    def __init__(self, message: str, *, reason_code: str = "provider_failed",
                 envelope: PlannerExecutionEnvelope[None] | None = None) -> None:
        super().__init__(message, envelope=envelope)
        self.reason_code = reason_code


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
                    **_envelope_base(cast(ProviderJSONCall, recorded_call)),
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
                premium_request=result.premium_request,
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
                admission_status="not_started",
                premium_request=False,
            )
        try:
            response = await cast(Callable[[str], Any], ask_json)(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            completed_at = self._clock()
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
                admission_status="admitted",
                premium_request=True,
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
            premium_request=True,
        )

    def _translate_call(
        self,
        request: CurationPlanningRequest,
        call: ProviderJSONCall,
    ) -> PlannerExecutionEnvelope[CurationPlan]:
        envelope_base = _envelope_base(call)
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
                envelope = PlannerExecutionEnvelope[None](
                    plan=None,
                    status=PlannerExecutionStatus.SCHEMA_FAILED,
                    reason_code=CurationPlannerSchemaError.reason_code,
                    **envelope_base,
                )
                raise CurationPlannerSchemaError(
                    "provider returned invalid JSON for curation planning", envelope=envelope
                )
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=PlannerExecutionStatus.PROVIDER_FAILED,
                reason_code=call.reason_code or "provider_failed",
                **envelope_base,
            )
            raise CurationPlannerProviderError(
                call.error_text or "provider failed during curation planning",
                reason_code=call.reason_code or "provider_failed",
                envelope=envelope,
            )
        try:
            plan = CurationPlan.model_validate(call.response)
        except ValidationError as exc:
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=PlannerExecutionStatus.SCHEMA_FAILED,
                reason_code=CurationPlannerSchemaError.reason_code,
                **envelope_base,
            )
            raise CurationPlannerSchemaError(
                "provider returned JSON that is not a valid curation plan",
                validation_error=exc,
                envelope=envelope,
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
            premium_request=False,
        )


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
    premium_request: bool | None = None
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
            premium_request=scenario.premium_request,
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
            envelope = PlannerExecutionEnvelope[None](
                plan=None,
                status=PlannerExecutionStatus.SCHEMA_FAILED,
                reason_code=CurationPlannerSchemaError.reason_code,
                **cast(Any, base),
            )
            raise CurationPlannerSchemaError("fake planner returned an invalid curation plan",
                                            validation_error=exc, envelope=envelope) from exc
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
            context=context_packet,
        )
    return CurationPlanningRequest(
        run_id=run_id,
        plan_id=uuid4(),
        frontier_key=context_packet.frontier_fingerprint,
        context_fingerprint=context_packet.context_fingerprint,
        context=CurationContextPacket(
            seed_memory_ids=[UUID(value) for value in context_packet.seed_memory_ids],
            support_memory_ids=[UUID(value) for value in context_packet.support_memory_ids],
            context_fingerprint=context_packet.context_fingerprint,
        ),
    )


def _build_planner_prompt(request: CurationPlanningRequest, tools: CurationReadTools) -> str:
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
        "available_tools": [],
        "schema": CurationPlan.model_json_schema(),
    }
    retry_feedback = getattr(tools, "retry_feedback", None)
    if retry_feedback is not None:
        payload["retry_feedback"] = {
            "reason_code": str(retry_feedback.reason_code),
            "message": str(retry_feedback.message)[:1000],
        }
    return (
        "You are a curation planner. Return exactly one JSON object matching the "
        "provided CurationPlan JSON schema. Planning only: do not execute mutations, "
        "call tools, or report a claimed action count; actions are counted only after "
        "validation.\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )


def _envelope_base(call: ProviderJSONCall) -> dict[str, Any]:
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
        "premium_request": call.premium_request,
        "token_usage": call.token_usage,
        "token_usage_source": call.token_usage_source,
        "metadata": {
            "provider_status": call.status,
            "raw_response_recorded": call.raw_text is not None,
        },
    }


def _cancellation_requested(scope: CancellationScope | asyncio.Event | None) -> bool:
    if scope is None:
        return False
    is_set = getattr(scope, "is_set", None)
    if callable(is_set):
        return bool(is_set())
    is_cancelled = getattr(scope, "is_cancelled", None)
    return callable(is_cancelled) and bool(is_cancelled())

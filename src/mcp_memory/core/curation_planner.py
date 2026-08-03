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
    CurationContextPacket,
    CurationPlan,
    CurationPlanningRequest,
)
from mcp_memory.core.provider_admission import classify_provider_failure
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
                reason_code="provider_capability_missing",
                reason_category="routing",
                admission_status="not_started",
                premium_request=False,
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
            premium_request=False,
        )

    def _envelope_base(self, call: ProviderJSONCall) -> dict[str, Any]:
        return _envelope_base(
            call,
            provider_profile=self._provider_profile,
            route_available=self._route_available,
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
            context_fingerprint=context_packet.context_fingerprint,
            campaign_hypothesis=context_packet.campaign_hypothesis,
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
        "planner_contract": [
            "Every target_id, source_id, canonical_id, source_ids entry, and retained memory_id must refer to a memory visible in context.",
            "seed_memory_ids is required and must exactly equal the context seed IDs; every seed must receive exactly one disposition.",
            "merge_memories canonical_id must be an existing visible record; merging never creates a record.",
            "create_link and remove_link require both endpoints to be visible in context.",
            "Every action affecting a visible memory must include its exact context.record_tokens value in preconditions.record_tokens.",
            "create_link requires descriptive relationship context with at least two words, exact endpoint/type/context link evidence, and an exact absent-link precondition; labels and memory excerpts alone are not link evidence.",
            "Only split_memory may introduce child records, and only through its typed children surface; never invent child IDs.",
            "For rewrite_memory and merge_memories, content is the final persisted durable memory body, not a description of the mutation.",
            "For merge_memories, canonical_id is the retained base; preserve its supported claims, integrate justified source claims, and make content stand alone after sources are archived.",
            "Use claim_manifest to map preserved or transformed claims to source records; do not use it as a substitute for writing those claims into the final content.",
            "Do not write mutation-status prose such as 'merged ... into the canonical' or 'added ... to the record' as memory content.",
            "Each seed memory must have exactly one disposition: it appears in an action's affected IDs or in retained, never both; action source, target, canonical, and child-source IDs all count as affected.",
            "Prefer one conclusion-first takeaway with concrete evidence; target 1600 characters or less, and split content above 3000 characters when it contains multiple takeaways.",
            "Treat selection signals and retrieval-friction flags as review clues, not proof; inspect the record content and relationships before mutating.",
            "A quality_feedback field means a previous curator mutation caused a measured retrieval regression; treat the record as a failed repair, retain it by default, and do not repeat a broad split, normalize, rewrite, or merge without a specific evidence-backed repair.",
            "For quality-feedback records, preserve exact search anchors, concrete entities, and the current durable meaning; if no targeted repair is clearly justified, put the seed in retained.",
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
    return (
        "You are a curation planner. Return exactly one JSON object matching the "
        "provided CurationPlan JSON schema. Include seed_memory_ids exactly as the "
        "context seed IDs. Planning only: do not execute mutations, "
        "call tools, or report a claimed action count; actions are counted only after "
        "validation. Every memory ID in an action or retention decision must be copied "
        "from a memory visible in the provided context. A merge canonical_id must be "
        "an existing visible record and merge never creates records; both create-link "
        "endpoints must be visible. Copy exact record tokens from context.record_tokens "
        "into preconditions.record_tokens for every affected visible memory. A "
        "create_link requires descriptive relationship context with at least two "
        "words, an evidence.link "
        "matching the exact endpoints, link type, and context, plus an exact "
        "absent-link precondition; labels and memory excerpts alone are not link "
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
        "measured retrieval regression: retain that record by default and do not repeat "
        "a broad split, normalize, rewrite, or merge without a specific evidence-backed "
        "repair. Preserve exact search anchors and concrete entities; if no targeted "
        "repair is clearly justified, put the seed in retained. If no visible canonical is appropriate, retain the memory instead "
        "of inventing an ID.\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    )


def _envelope_base(
    call: ProviderJSONCall,
    *,
    provider_profile: str | None = None,
    route_available: bool | None = None,
) -> dict[str, Any]:
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

"""Provider-neutral planner contracts and a deterministic test planner."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable, Generic, Mapping, Protocol, TypeVar, cast

from pydantic import ValidationError

from mcp_memory.core.curation_models import CurationPlan, CurationPlanningRequest


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

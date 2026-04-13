from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Protocol, TypeAlias


@dataclass(slots=True)
class AgenticRunResult:
    status: str
    summary: str | None = None
    raw_text: str | None = None
    parsed: dict[str, Any] | None = None
    subprocess_pid: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderObserverEvent:
    attempt: int
    prompt: str | None
    subprocess_pid: int | None


@dataclass(frozen=True, slots=True)
class ProviderAttemptStartedEvent(ProviderObserverEvent):
    event: ClassVar[str] = "started"

    started_at: float


@dataclass(frozen=True, slots=True)
class ProviderAttemptHeartbeatEvent(ProviderObserverEvent):
    event: ClassVar[str] = "heartbeat"

    started_at: float
    heartbeat_at: float
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ProviderAttemptFinishedEvent(ProviderObserverEvent):
    event: ClassVar[str] = "finished"

    started_at: float
    completed_at: float
    duration_seconds: float
    status: str | None = None
    returncode: int | None = None
    raw_text: str | None = None
    parsed: dict[str, Any] | None = None
    error_text: str | None = None
    reason_category: str | None = None
    reason_code: str | None = None
    retry_delay_seconds: float | None = None

    @property
    def error(self) -> str | None:
        return self.error_text


# Deprecated import-compat aliases. Provider observers accept typed dataclass events only.
LegacyProviderObserverPayload: TypeAlias = Mapping[str, Any]
ProviderObserverInput: TypeAlias = ProviderObserverEvent
ProviderObserver = Callable[[ProviderObserverEvent], None]


class ProviderBudgetExceeded(RuntimeError):
    def __init__(self, provider_key: str, *, calls_last_day: int, daily_call_limit: int) -> None:
        super().__init__(
            f"provider_budget_exceeded: provider_key={provider_key} calls_last_day={calls_last_day} daily_call_limit={daily_call_limit}"
        )
        self.provider_key = provider_key
        self.calls_last_day = calls_last_day
        self.daily_call_limit = daily_call_limit


class ProviderRateLimitExceeded(RuntimeError):
    def __init__(
        self,
        model_name: str,
        *,
        calls_in_window: int,
        burst_call_limit: int,
        burst_window_seconds: float,
        retry_delay_seconds: float | None = None,
    ) -> None:
        super().__init__(
            "provider_rate_limit_exceeded: "
            f"model_name={model_name} calls_in_window={calls_in_window} "
            f"burst_call_limit={burst_call_limit} burst_window_seconds={burst_window_seconds}"
        )
        self.model_name = model_name
        self.calls_in_window = calls_in_window
        self.burst_call_limit = burst_call_limit
        self.burst_window_seconds = burst_window_seconds
        self.retry_delay_seconds = retry_delay_seconds
        self.same_run_failover_eligible = True


class ProviderAdmissionDeferred(RuntimeError):
    def __init__(
        self,
        provider_key: str,
        model_name: str,
        *,
        reason_code: str,
        reason_category: str,
        retry_delay_seconds: float | None = None,
        error_text: str | None = None,
    ) -> None:
        message = (
            "provider_admission_deferred: "
            f"provider_key={provider_key} model_name={model_name} "
            f"reason_code={reason_code} reason_category={reason_category}"
        )
        if error_text:
            message += f" detail={error_text}"
        super().__init__(message)
        self.provider_key = provider_key
        self.model_name = model_name
        self.reason_code = reason_code
        self.reason_category = reason_category
        self.retry_delay_seconds = retry_delay_seconds
        self.error_text = error_text
        self.same_run_failover_eligible = True


class ProviderAuthenticationRequired(RuntimeError):
    def __init__(self, provider_name: str, *, error_text: str | None = None) -> None:
        message = f"provider_authentication_required: provider_name={provider_name}"
        if error_text:
            message += f" detail={error_text}"
        super().__init__(message)
        self.provider_name = provider_name
        self.error_text = error_text
        self.same_run_failover_eligible = True


class JSONTaskProvider(Protocol):
    async def ask_json(self, prompt: str) -> dict[str, Any]:
        ...


class AgenticTaskProvider(Protocol):
    async def run_agent(self, prompt: str) -> AgenticRunResult:
        ...

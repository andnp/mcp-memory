from __future__ import annotations

from dataclasses import dataclass

from mcp_memory.core.ports.providers import ProviderUsagePort
from mcp_memory.core.providers._json_cli import ProviderBackoffError, looks_like_interactive_auth_prompt
from mcp_memory.core.providers.interfaces import (
    ProviderAdmissionDeferred,
    ProviderAuthenticationRequired,
    ProviderBudgetExceeded,
    ProviderRateLimitExceeded,
)


@dataclass(frozen=True)
class ProviderAdmissionDecision:
    allowed: bool
    reason: str | None = None
    reason_category: str | None = None
    retry_delay_seconds: float | None = None
    error_text: str | None = None
    calls_last_day: int | None = None
    daily_call_limit: int | None = None
    calls_in_window: int | None = None
    burst_call_limit: int | None = None
    burst_window_seconds: float | None = None


@dataclass(frozen=True)
class ProviderOutcomeReason:
    reason_code: str
    reason_category: str
    retry_delay_seconds: float | None = None
    error_text: str | None = None


def evaluate_provider_admission(
    usage_repository: ProviderUsagePort,
    *,
    provider_key: str,
    budget_key: str,
    model_name: str,
    daily_call_limit: int | None,
    model_burst_call_limit: int | None,
    model_burst_window_seconds: float | None,
    now: float | None = None,
) -> ProviderAdmissionDecision:
    active_state = usage_repository.get_active_admission_state(
        provider_key=provider_key,
        model_name=model_name,
        now=now,
    )
    if active_state is not None:
        retry_delay_seconds = active_state.retry_delay_seconds
        if now is not None:
            retry_delay_seconds = max(active_state.active_until - now, 0.0)
        return ProviderAdmissionDecision(
            allowed=False,
            reason=active_state.reason_code,
            reason_category=active_state.reason_category,
            retry_delay_seconds=retry_delay_seconds,
            error_text=active_state.error_text,
        )

    calls_last_day = None
    if daily_call_limit is not None:
        calls_last_day = usage_repository.count_recent_calls(
            provider_keys=[budget_key, f"{budget_key}:agentic"],
            now=now,
        )
        if calls_last_day >= daily_call_limit:
            return ProviderAdmissionDecision(
                allowed=False,
                reason="profile_daily_budget_exceeded",
                reason_category="admission",
                calls_last_day=calls_last_day,
                daily_call_limit=daily_call_limit,
            )

    if model_burst_call_limit is None or model_burst_window_seconds is None:
        return ProviderAdmissionDecision(allowed=True)

    calls_in_window = usage_repository.count_recent_model_calls(
        model_names=[model_name],
        now=now,
        window_seconds=model_burst_window_seconds,
    )
    if calls_in_window >= model_burst_call_limit:
        return ProviderAdmissionDecision(
            allowed=False,
            reason="model_burst_limit_exceeded",
            reason_category="admission",
            retry_delay_seconds=model_burst_window_seconds,
            calls_in_window=calls_in_window,
            burst_call_limit=model_burst_call_limit,
            burst_window_seconds=model_burst_window_seconds,
        )

    return ProviderAdmissionDecision(allowed=True)


def build_provider_admission_exception(
    decision: ProviderAdmissionDecision,
    *,
    provider_key: str,
    model_name: str,
):
    if decision.reason == "profile_daily_budget_exceeded":
        assert decision.calls_last_day is not None
        assert decision.daily_call_limit is not None
        return ProviderBudgetExceeded(
            provider_key,
            calls_last_day=decision.calls_last_day,
            daily_call_limit=decision.daily_call_limit,
        )
    if decision.reason == "model_burst_limit_exceeded":
        assert decision.calls_in_window is not None
        assert decision.burst_call_limit is not None
        assert decision.burst_window_seconds is not None
        return ProviderRateLimitExceeded(
            model_name,
            calls_in_window=decision.calls_in_window,
            burst_call_limit=decision.burst_call_limit,
            burst_window_seconds=decision.burst_window_seconds,
            retry_delay_seconds=decision.retry_delay_seconds,
        )
    if decision.reason is not None and decision.reason_category is not None:
        return ProviderAdmissionDeferred(
            provider_key,
            model_name,
            reason_code=decision.reason,
            reason_category=decision.reason_category,
            retry_delay_seconds=decision.retry_delay_seconds,
            error_text=decision.error_text,
        )
    return RuntimeError(f"provider_admission_denied: provider_key={provider_key} model_name={model_name}")


def classify_provider_failure(
    exc: Exception,
    *,
    event_status: str | None = None,
    raw_text: str | None = None,
) -> ProviderOutcomeReason:
    if isinstance(exc, ProviderAdmissionDeferred):
        return ProviderOutcomeReason(
            reason_code=exc.reason_code,
            reason_category=exc.reason_category,
            retry_delay_seconds=exc.retry_delay_seconds,
            error_text=exc.error_text or str(exc),
        )
    if isinstance(exc, ProviderBudgetExceeded):
        return ProviderOutcomeReason(
            reason_code="profile_daily_budget_exceeded",
            reason_category="admission",
            error_text=str(exc),
        )
    if isinstance(exc, ProviderRateLimitExceeded):
        return ProviderOutcomeReason(
            reason_code="model_burst_limit_exceeded",
            reason_category="admission",
            retry_delay_seconds=exc.retry_delay_seconds,
            error_text=str(exc),
        )
    if isinstance(exc, ProviderAuthenticationRequired):
        return ProviderOutcomeReason(
            reason_code="interactive_auth_required",
            reason_category="auth",
            error_text=exc.error_text or str(exc),
        )

    error_text = str(exc)
    retry_delay_seconds = getattr(exc, "retry_delay_seconds", None)
    if isinstance(exc, ProviderBackoffError) or isinstance(retry_delay_seconds, (int, float)):
        lowered = error_text.lower()
        if "quota" in lowered or "quota_exhausted" in lowered:
            return ProviderOutcomeReason(
                reason_code="provider_quota_exhausted",
                reason_category="upstream",
                retry_delay_seconds=None if retry_delay_seconds is None else float(retry_delay_seconds),
                error_text=error_text,
            )
        return ProviderOutcomeReason(
            reason_code="provider_backoff_requested",
            reason_category="upstream",
            retry_delay_seconds=None if retry_delay_seconds is None else float(retry_delay_seconds),
            error_text=error_text,
        )

    normalized_status = (event_status or "").strip().lower()
    raw_output = raw_text or ""
    combined_text = f"{error_text}\n{raw_output}".strip().lower()
    if normalized_status == "timeout":
        return ProviderOutcomeReason("provider_timeout", "transport", error_text=error_text)
    if normalized_status == "parse_error":
        if looks_like_interactive_auth_prompt(combined_text):
            return ProviderOutcomeReason("interactive_auth_required", "auth", error_text=error_text)
        return ProviderOutcomeReason("invalid_json_response", "response", error_text=error_text)
    if "command not found" in combined_text:
        return ProviderOutcomeReason("provider_command_not_found", "environment", error_text=error_text)
    if "command cancelled" in combined_text:
        return ProviderOutcomeReason("provider_cancelled", "cancellation", error_text=error_text)
    if error_text.startswith("Exit code ") or error_text.startswith("Terminated by signal"):
        return ProviderOutcomeReason("provider_subprocess_exit", "subprocess", error_text=error_text)
    return ProviderOutcomeReason("provider_execution_error", "execution", error_text=error_text)


def should_persist_admission_backoff(reason: ProviderOutcomeReason) -> bool:
    return (
        reason.reason_category == "upstream"
        and isinstance(reason.retry_delay_seconds, (int, float))
        and float(reason.retry_delay_seconds) > 0.0
    )
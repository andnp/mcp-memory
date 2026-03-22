import time

import pytest

from mcp_memory.core.provider_admission import build_provider_admission_exception
from mcp_memory.core.provider_admission import evaluate_provider_admission
from mcp_memory.core.providers.interfaces import ProviderBudgetExceeded
from mcp_memory.core.providers.interfaces import ProviderAdmissionDeferred
from mcp_memory.core.providers.interfaces import ProviderRateLimitExceeded
from mcp_memory.provider_usage_store import ProviderUsageRepository


pytestmark = pytest.mark.small


def test_provider_admission_allows_provider_under_limits(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")

    decision = evaluate_provider_admission(
        repository,
        provider_key="copilot-mini",
        budget_key="copilot-mini",
        model_name="gpt-5-mini",
        daily_call_limit=50,
        model_burst_call_limit=1,
        model_burst_window_seconds=600.0,
        now=1000.0,
    )

    assert decision.allowed is True
    assert decision.reason is None


def test_provider_admission_blocks_daily_profile_budget(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="ingest-system1",
        task_id="task-a",
        request_id="req-a",
        subprocess_pid=None,
        provider_key="copilot-mini:agentic",
        provider_name="Copilot CLI Agentic",
        model_name="gpt-5-mini",
        status="success",
        duration_seconds=1.0,
        created_at=time.time(),
        error_text=None,
    )

    decision = evaluate_provider_admission(
        repository,
        provider_key="copilot-mini",
        budget_key="copilot-mini",
        model_name="gpt-5-mini",
        daily_call_limit=1,
        model_burst_call_limit=2,
        model_burst_window_seconds=600.0,
    )

    assert decision.allowed is False
    assert decision.reason == "profile_daily_budget_exceeded"
    exc = build_provider_admission_exception(decision, provider_key="copilot-mini", model_name="gpt-5-mini")
    assert isinstance(exc, ProviderBudgetExceeded)


def test_provider_admission_blocks_model_burst_limit(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="graph-linker",
        task_id="task-a",
        request_id="req-a",
        subprocess_pid=None,
        provider_key="copilot-mini",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
        status="success",
        duration_seconds=0.5,
        created_at=time.time(),
        error_text=None,
    )

    decision = evaluate_provider_admission(
        repository,
        provider_key="copilot-strong",
        budget_key="copilot-strong",
        model_name="gpt-5-mini",
        daily_call_limit=20,
        model_burst_call_limit=1,
        model_burst_window_seconds=600.0,
    )

    assert decision.allowed is False
    assert decision.reason == "model_burst_limit_exceeded"
    exc = build_provider_admission_exception(decision, provider_key="copilot-strong", model_name="gpt-5-mini")
    assert isinstance(exc, ProviderRateLimitExceeded)


def test_provider_admission_blocks_active_upstream_backoff_state(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.upsert_admission_state(
        provider_key="gemini-cli",
        model_name="gemini-3-flash-preview",
        reason_category="upstream",
        reason_code="provider_quota_exhausted",
        error_text="quota reset pending",
        retry_delay_seconds=90.0,
        active_until=1_090.0,
        updated_at=1_000.0,
    )

    decision = evaluate_provider_admission(
        repository,
        provider_key="gemini-cli",
        budget_key="gemini-cheap",
        model_name="gemini-3-flash-preview",
        daily_call_limit=50,
        model_burst_call_limit=2,
        model_burst_window_seconds=600.0,
        now=1_030.0,
    )

    assert decision.allowed is False
    assert decision.reason == "provider_quota_exhausted"
    assert decision.reason_category == "upstream"
    assert decision.retry_delay_seconds == pytest.approx(60.0)
    exc = build_provider_admission_exception(decision, provider_key="gemini-cli", model_name="gemini-3-flash-preview")
    assert isinstance(exc, ProviderAdmissionDeferred)
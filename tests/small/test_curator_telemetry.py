from __future__ import annotations

import pytest

from mcp_memory.core.curator_telemetry import aggregate_provider_attempts
from mcp_memory.operational_store_rows import ProviderUsageSample


pytestmark = pytest.mark.small


def _sample(identity: str | None, *, status: str, created_at: float, attempt: int | None = 1) -> ProviderUsageSample:
    return ProviderUsageSample(
        task_name="memory-curator",
        task_id="task-1" if identity else None,
        execution_epoch=3 if identity else None,
        request_id="request-1" if identity else None,
        attempt=attempt,
        attempt_identity=identity,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        status=status,
        duration_seconds=1.0,
        created_at=created_at,
        reason_code=None,
        total_tokens=100,
    )


def test_provider_attempt_aggregation_deduplicates_restart_rows() -> None:
    result = aggregate_provider_attempts(
        [
            _sample("task-1:3:req-1:1", status="running", created_at=1.0),
            _sample("task-1:3:req-1:1", status="success", created_at=2.0),
            _sample("task-1:3:req-2:2", status="error", created_at=3.0, attempt=2),
        ]
    )

    assert result.provider_attempts == 2
    assert result.failures == 1
    assert result.retries == 1
    assert result.token_totals.total_tokens == 200


def test_provider_attempt_aggregation_surfaces_missing_identity() -> None:
    result = aggregate_provider_attempts([_sample(None, status="success", created_at=1.0)])

    assert result.provider_attempts == 0
    assert result.unattributed_calls == 1

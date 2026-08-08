from __future__ import annotations

import pytest

from mcp_memory.core.curator_telemetry import aggregate_provider_attempts
from mcp_memory.management.reporting_rows import build_task_result_view
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


def test_task_result_projection_uses_persisted_evidence_over_provider_claims() -> None:
    view = build_task_result_view(
        {
            "tool_calls_executed": 3,
            "mutations": 2,
            "verified_mutation_count": 2,
            "curation_campaign_result": {
                "tool_calls_executed": 3,
                "productive_mutation_count": 2,
                "verified_mutation_count": 2,
                "provider_calls_used": 1,
                "provider_reported_tool_calls": 99,
                "provider_reported_mutations": 88,
                "quality_evidence": [{"useful_work": True}],
            },
        }
    )

    assert view.curator_telemetry is not None
    assert view.curator_telemetry.internal_tool_calls == 3
    assert view.curator_telemetry.productive_mutations == 2
    assert view.curator_telemetry.verified_mutations == 2
    assert view.curator_telemetry.quality_evidence == 1
    assert view.curator_telemetry.discrepancies.reported_mutations_delta == 86
    assert view.metadata.mutations == 2

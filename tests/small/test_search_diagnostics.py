from types import SimpleNamespace

import pytest

from mcp_memory.management.health_reporting import build_search_health
from mcp_memory.mcp.telemetry import (
    SearchDiagnosticsSampler,
    get_search_diagnostics_sampler,
    record_search_diagnostics,
    search_diagnostics_snapshot,
)

pytestmark = pytest.mark.small


def test_search_diagnostics_sampling_boundaries_are_deterministic() -> None:
    """Honor disabled and full sampling boundaries without randomness.

    The same invocation identity must always make the same sampling decision.
    """
    disabled = SearchDiagnosticsSampler(sample_rate=0.0)
    enabled = SearchDiagnosticsSampler(sample_rate=1.0)

    assert disabled.should_sample("request-1") is False
    assert enabled.should_sample("request-1") is True

    first = SearchDiagnosticsSampler(sample_rate=0.5)
    second = SearchDiagnosticsSampler(sample_rate=0.5)
    assert first.should_sample("request-2") == second.should_sample("request-2")


def test_search_diagnostics_aggregate_counts_supported_fields_without_private_data() -> None:
    """Aggregate degradation, planner, cache, and failure signals safely.

    Query text, content, and identifiers must never appear in the snapshot.
    """
    sampler = SearchDiagnosticsSampler(sample_rate=1.0)
    sampler.observe(
        "request-1",
        {
            "query": "private query text",
            "content": "private record content",
            "degraded": True,
            "lane_decisions": {
                "enabled": ["keyword", "vector"],
                "skipped": ["graph:awaiting_seed_confidence"],
            },
            "cache_diagnostics": ["fresh_exact_hit", "projection_fallback"],
            "failures": [{"stage": "vector_search", "message": "private detail"}],
        },
    )

    snapshot = sampler.snapshot()

    assert snapshot["observed_searches"] == 1
    assert snapshot["sampled_searches"] == 1
    assert snapshot["serialized_diagnostics"] == 1
    assert snapshot["degraded_count"] == 1
    assert snapshot["degraded_rate"] == 1.0
    assert snapshot["planner_decisions"] == {
        "enabled:keyword": 1,
        "enabled:vector": 1,
        "skipped:graph:awaiting_seed_confidence": 1,
    }
    assert snapshot["cache_status"] == {"fresh": 1, "projection": 1}
    assert snapshot["failure_stages"] == {"vector_search": 1}
    assert "private query text" not in repr(snapshot)
    assert "private record content" not in repr(snapshot)
    assert "private detail" not in repr(snapshot)


def test_search_diagnostics_empty_input_reports_unavailable_nested_metrics() -> None:
    """Keep unsupported diagnostic categories distinct from observed zeroes."""
    sampler = SearchDiagnosticsSampler(sample_rate=1.0)

    snapshot = sampler.snapshot()

    assert snapshot["available"] is True
    assert snapshot["observed_searches"] == 0
    assert snapshot["sampled_rate"] is None
    assert snapshot["degraded_count"] is None
    assert snapshot["degraded_rate"] is None
    assert snapshot["planner_decisions"] is None
    assert snapshot["cache_status"] is None
    assert snapshot["failure_stages"] is None


def test_search_diagnostics_reports_serialization_failures_without_blocking() -> None:
    """Report diagnostic serialization failures while preserving search telemetry."""

    class BrokenDiagnostics:
        def to_payload(self) -> dict[str, object]:
            raise ValueError("private serialization detail")

    sampler = SearchDiagnosticsSampler(sample_rate=1.0)
    sampler.observe("request-1", BrokenDiagnostics())

    snapshot = sampler.snapshot()

    assert snapshot["sampled_searches"] == 1
    assert snapshot["serialized_diagnostics"] == 0
    assert snapshot["diagnostic_serialization_failures"] == 1
    assert "private serialization detail" not in repr(snapshot)


def test_management_search_health_preserves_legacy_fields_with_aggregate() -> None:
    """Add aggregate diagnostics without removing the existing health contract."""
    telemetry = SimpleNamespace()
    get_search_diagnostics_sampler(telemetry, sample_rate=1.0)
    record_search_diagnostics(
        telemetry,
        invocation_id="request-1",
        diagnostics={"degraded": False},
    )

    payload = build_search_health(
        None,
        search_diagnostics=search_diagnostics_snapshot(telemetry),
    )

    assert payload.available is False
    assert payload.degraded is False
    assert payload.search_diagnostics.available is True
    assert payload.search_diagnostics.degraded_count == 0
    assert payload.model_dump()["degraded"] is False

from __future__ import annotations

from types import SimpleNamespace

from mcp_memory.core.sampling import SamplingBatch
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload


def test_sampling_payload_merges_strategy_metadata_and_result_metrics() -> None:
    batch = SamplingBatch(
        requested_strategy="semantic",
        strategy_used="semantic",
        strategy_fallback_reason=None,
        candidate_count=4,
        records=[],
        strategy_selection_mode="deterministic_scores",
        strategy_selection_reason="selected=semantic",
        strategy_selection_scores={"semantic": 0.75, "anomaly": 0.2},
        sampler_priority_score=0.75,
        sampler_priority_explanation="quality_pass_rate=0.800",
        selector_feature_snapshot={
            "strategy_signals": {"semantic_overlap_share": 0.8},
            "candidate_population": {
                "count": 4,
                "metrics": {"content_chars": {"count": 4, "min": 10.0, "p50": 20.0, "p90": 30.0, "max": 40.0, "mean": 25.0}},
                "shares": {"never_surfaced_share": 0.25},
            },
            "selected_population": {
                "count": 2,
                "metrics": {"content_chars": {"count": 2, "min": 20.0, "p50": 30.0, "p90": 30.0, "max": 30.0, "mean": 25.0}},
                "shares": {"never_surfaced_share": 0.5},
            },
        },
    )
    sampled_records = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    seed_records = [SimpleNamespace(id="seed-a")]

    payload = sampling_payload(
        batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        created=2,
        archived=1,
    )

    assert payload == {
        "requested_strategy": "semantic",
        "strategy_used": "semantic",
        "strategy_fallback_reason": None,
        "candidate_count": 4,
        "strategy_selection_mode": "deterministic_scores",
        "strategy_selection_reason": "selected=semantic",
        "strategy_selection_scores": {"semantic": 0.75, "anomaly": 0.2},
        "sampler_priority_score": 0.75,
        "sampler_priority_explanation": "quality_pass_rate=0.800",
        "selector_feature_snapshot": {
            "strategy_signals": {"semantic_overlap_share": 0.8},
            "candidate_population": {
                "count": 4,
                "metrics": {"content_chars": {"count": 4, "min": 10.0, "p50": 20.0, "p90": 30.0, "max": 40.0, "mean": 25.0}},
                "shares": {"never_surfaced_share": 0.25},
            },
            "selected_population": {
                "count": 2,
                "metrics": {"content_chars": {"count": 2, "min": 20.0, "p50": 30.0, "p90": 30.0, "max": 30.0, "mean": 25.0}},
                "shares": {"never_surfaced_share": 0.5},
            },
        },
        "sampled_memory_ids": ["a", "b"],
        "seed_memory_ids": ["seed-a"],
        "created": 2,
        "archived": 1,
    }
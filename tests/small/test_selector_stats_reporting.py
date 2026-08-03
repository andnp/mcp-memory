from __future__ import annotations

import pytest

from mcp_memory.management.models import SelectorFeatureSnapshotPayload
from mcp_memory.management.models import AgentRunHistoryPayload, RunResultMetadataPayload
from mcp_memory.management.selector_stats_reporting import (
    FRESH_SELECTOR,
    SEEDED_CLAIMED,
    UNKNOWN_CLASSIFICATION,
    build_selector_stats_payload,
    classify_selector_run,
)


pytestmark = pytest.mark.small


def _run(
    *,
    task_id: str,
    task_name: str,
    completed_at: float,
    claimed_work_item_count: int | None,
    strategy_used: str | None,
    strategy_selection_mode: str | None,
    strategy_selection_reason: str | None,
    strategy_fallback_reason: str | None = None,
    candidate_count: int | None = None,
    mutations: int | None = None,
    tool_calls_executed: int | None = None,
    strategy_selection_scores: dict[str, float] | None = None,
    selector_feature_snapshot: dict[str, object] | None = None,
) -> AgentRunHistoryPayload:
    return AgentRunHistoryPayload(
        task_id=task_id,
        task_name=task_name,
        status="completed",
        started_at=completed_at - 2.0,
        completed_at=completed_at,
        duration_seconds=2.0,
        result_summary=f"run={task_id}",
        result_metadata=RunResultMetadataPayload(
            requested_strategy=strategy_used,
            strategy_used=strategy_used,
            strategy_selection_mode=strategy_selection_mode,
            strategy_selection_reason=strategy_selection_reason,
            strategy_fallback_reason=strategy_fallback_reason,
            strategy_selection_scores=strategy_selection_scores or {},
            candidate_count=candidate_count,
            claimed_work_item_count=claimed_work_item_count,
            mutations=mutations,
            tool_calls_executed=tool_calls_executed,
            selector_feature_snapshot=(
                SelectorFeatureSnapshotPayload.model_validate(selector_feature_snapshot)
                if selector_feature_snapshot is not None
                else SelectorFeatureSnapshotPayload()
            ),
        ),
    )


def test_classify_selector_run_prefers_unknown_over_fake_certainty() -> None:
    assert classify_selector_run(RunResultMetadataPayload(claimed_work_item_count=0))[0] == FRESH_SELECTOR
    assert classify_selector_run(RunResultMetadataPayload(claimed_work_item_count=2))[0] == SEEDED_CLAIMED
    assert classify_selector_run(RunResultMetadataPayload(claimed_work_item_count=None))[0] == UNKNOWN_CLASSIFICATION


def test_classify_selector_run_reports_priority_score_selection() -> None:
    payload = build_selector_stats_payload(
        [
            _run(
                task_id="priority-1",
                task_name="memory-curator",
                completed_at=90.0,
                claimed_work_item_count=0,
                strategy_used="semantic",
                strategy_selection_mode="priority_scores_with_exploration",
                strategy_selection_reason="selected=semantic; priority_score=0.0200; exploration=bounded",
            )
        ],
        window_hours=1,
        run_limit=10,
        now=100.0,
    )

    assert payload.outcome_rows[0].reason_family == "priority_scores"


def test_build_selector_stats_payload_rolls_up_classifications_and_outcomes() -> None:
    payload = build_selector_stats_payload(
        [
            _run(
                task_id="fresh-1",
                task_name="memory-curator",
                completed_at=90.0,
                claimed_work_item_count=0,
                strategy_used="semantic",
                strategy_selection_mode="deterministic_scores",
                strategy_selection_reason="selected=semantic from deterministic scores",
                candidate_count=7,
                mutations=2,
                tool_calls_executed=4,
                strategy_selection_scores={"semantic": 0.91, "lexical": 0.22},
                selector_feature_snapshot={
                    "strategy_signals": {"semantic_overlap_share": 0.75},
                    "candidate_population": {
                        "count": 7,
                        "metrics": {
                            "content_chars": {"count": 7, "min": 10.0, "p50": 20.0, "p90": 35.0, "max": 40.0, "mean": 22.0},
                        },
                        "shares": {"never_surfaced_share": 0.5},
                    },
                    "selected_population": {
                        "count": 3,
                        "metrics": {
                            "content_chars": {"count": 3, "min": 15.0, "p50": 25.0, "p90": 35.0, "max": 35.0, "mean": 25.0},
                        },
                        "shares": {"never_surfaced_share": 0.3333},
                    },
                },
            ),
            _run(
                task_id="fresh-2",
                task_name="memory-curator",
                completed_at=85.0,
                claimed_work_item_count=0,
                strategy_used="semantic",
                strategy_selection_mode="deterministic_scores",
                strategy_selection_reason="selected=semantic from deterministic scores",
                candidate_count=8,
                mutations=1,
                tool_calls_executed=3,
                strategy_selection_scores={"semantic": 0.88, "lexical": 0.35},
                selector_feature_snapshot={
                    "strategy_signals": {
                        "semantic_overlap_share": 0.55,
                        "recency_share": 0.15,
                    },
                    "candidate_population": {
                        "count": 8,
                        "metrics": {
                            "content_chars": {"count": 8, "min": 12.0, "p50": 32.0, "p90": 52.0, "max": 60.0, "mean": 42.0},
                            "updated_age_seconds": {"count": 8, "min": 1200.0, "p50": 3600.0, "p90": 10800.0, "max": 14400.0, "mean": 7200.0},
                            "read_count": {"count": 8, "min": 0.0, "p50": 2.0, "p90": 5.0, "max": 6.0, "mean": 3.5},
                            "support_count": {"count": 8, "min": 0.0, "p50": 1.0, "p90": 2.0, "max": 2.0, "mean": 1.0},
                        },
                        "shares": {
                            "never_surfaced_share": 0.25,
                            "never_accessed_share": 0.125,
                            "cooldown_share": 0.5,
                            "low_support_share": 0.625,
                        },
                    },
                    "selected_population": {
                        "count": 4,
                        "metrics": {
                            "content_chars": {"count": 4, "min": 18.0, "p50": 40.0, "p90": 58.0, "max": 60.0, "mean": 45.0},
                            "updated_age_seconds": {"count": 4, "min": 900.0, "p50": 1800.0, "p90": 5400.0, "max": 7200.0, "mean": 2700.0},
                            "read_count": {"count": 4, "min": 1.0, "p50": 2.0, "p90": 4.0, "max": 4.0, "mean": 2.5},
                            "support_count": {"count": 4, "min": 1.0, "p50": 1.0, "p90": 2.0, "max": 2.0, "mean": 1.5},
                        },
                        "shares": {
                            "never_surfaced_share": 0.125,
                            "never_accessed_share": 0.0,
                            "cooldown_share": 0.25,
                            "low_support_share": 0.5,
                        },
                    },
                },
            ),
            _run(
                task_id="seeded-1",
                task_name="memory-curator",
                completed_at=80.0,
                claimed_work_item_count=3,
                strategy_used="semantic",
                strategy_selection_mode="utility_priors",
                strategy_selection_reason="utility prior tie-break",
                strategy_fallback_reason="insufficient_candidates",
                candidate_count=5,
                mutations=0,
                tool_calls_executed=2,
                strategy_selection_scores={"semantic": 0.77, "lexical": 0.63},
            ),
            _run(
                task_id="unknown-1",
                task_name="graph-linker",
                completed_at=70.0,
                claimed_work_item_count=None,
                strategy_used="lexical",
                strategy_selection_mode=None,
                strategy_selection_reason=None,
                candidate_count=4,
                mutations=1,
                tool_calls_executed=1,
            ),
        ],
        window_hours=1,
        run_limit=50,
        now=100.0,
    )

    assert payload.summary.total_runs == 4
    assert payload.summary.selector_signal_runs == 4
    assert payload.summary.fresh_selector_runs == 2
    assert payload.summary.seeded_claimed_runs == 1
    assert payload.summary.unknown_runs == 1
    assert payload.summary.fallback_runs == 1
    assert payload.summary.mutation_runs == 3
    assert payload.summary.no_op_runs == 1
    assert payload.summary.total_mutations == 4
    assert payload.summary.total_tool_calls == 10
    assert payload.summary.average_candidate_count == 6.0

    assert [(row.key, row.runs) for row in payload.classification_breakdown] == [
        (FRESH_SELECTOR, 2),
        (SEEDED_CLAIMED, 1),
        (UNKNOWN_CLASSIFICATION, 1),
    ]
    assert [
        (
            row.task_name,
            row.run_classification,
            row.strategy_used,
            row.strategy_selection_mode,
            row.reason_family,
            row.runs,
            row.fallback_count,
            row.total_mutations,
            row.total_tool_calls,
            row.average_candidate_count,
            row.no_op_rate,
        )
        for row in payload.outcome_rows
    ] == [
        ("graph-linker", UNKNOWN_CLASSIFICATION, "lexical", "unspecified", "unknown", 1, 0, 1, 1, 4.0, 0.0),
        ("memory-curator", FRESH_SELECTOR, "semantic", "deterministic_scores", "deterministic_signals", 2, 0, 3, 7, 7.5, 0.0),
        ("memory-curator", SEEDED_CLAIMED, "semantic", "utility_priors", "fallback", 1, 1, 0, 2, 5.0, 1.0),
    ]

    assert [
        (
            row.task_name,
            row.run_classification,
            row.runs,
            row.snapshot_runs,
            row.candidate_metric_means,
            row.selected_metric_means,
            row.candidate_share_means,
            row.selected_share_means,
            row.strategy_signal_means,
        )
        for row in payload.feature_rollup_rows
    ] == [
        (
            "memory-curator",
            FRESH_SELECTOR,
            2,
            2,
            {
                "content_chars": 32.0,
                "read_count": 3.5,
                "support_count": 1.0,
                "updated_age_seconds": 7200.0,
            },
            {
                "content_chars": 35.0,
                "read_count": 2.5,
                "support_count": 1.5,
                "updated_age_seconds": 2700.0,
            },
            {
                "cooldown_share": 0.5,
                "low_support_share": 0.625,
                "never_accessed_share": 0.125,
                "never_surfaced_share": 0.375,
            },
            {
                "cooldown_share": 0.25,
                "low_support_share": 0.5,
                "never_accessed_share": 0.0,
                "never_surfaced_share": 0.2291,
            },
            {
                "semantic_overlap_share": 0.65,
                "recency_share": 0.15,
            },
        ),
    ]

    assert payload.recent_runs[0].task_id == "fresh-1"
    assert payload.recent_runs[0].run_classification == FRESH_SELECTOR
    assert payload.recent_runs[0].selector_feature_snapshot.strategy_signals == {"semantic_overlap_share": 0.75}
    assert payload.recent_runs[0].selector_feature_snapshot.selected_population.count == 3
    assert payload.recent_runs[1].task_id == "fresh-2"
    assert payload.recent_runs[1].run_classification == FRESH_SELECTOR
    assert payload.recent_runs[2].task_id == "seeded-1"
    assert payload.recent_runs[2].run_classification == SEEDED_CLAIMED
    assert payload.recent_runs[3].task_id == "unknown-1"
    assert payload.recent_runs[3].run_classification == UNKNOWN_CLASSIFICATION
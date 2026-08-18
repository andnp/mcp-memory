from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    RunResultMetadataPayload,
    SelectorClassificationBreakdownPayload,
    SelectorFeatureRollupRowPayload,
    SelectorFeatureSnapshotPayload,
    SelectorOutcomeRowPayload,
    SelectorPopulationSnapshotPayload,
    SelectorRecentDiagnosticPayload,
    SelectorStatsPayload,
    SelectorStatsSummaryPayload,
)

UNKNOWN_SELECTOR_MODE = "unspecified"
UNKNOWN_SELECTOR_STRATEGY = "unknown"
FRESH_SELECTOR = "fresh_selector"
SEEDED_CLAIMED = "seeded_claimed"
UNKNOWN_CLASSIFICATION = "unknown"
_CLASSIFICATION_LABELS = {
    FRESH_SELECTOR: "Fresh selector",
    SEEDED_CLAIMED: "Seeded/claimed",
    UNKNOWN_CLASSIFICATION: "Unknown",
}
_ROLLUP_METRIC_KEYS = (
    "content_chars",
    "updated_age_seconds",
    "read_count",
    "support_count",
)
_ROLLUP_SHARE_KEYS = (
    "never_surfaced_share",
    "never_accessed_share",
    "cooldown_share",
    "low_support_share",
)
_ROLLUP_SIGNAL_LIMIT = 3


@dataclass
class _OutcomeAccumulator:
    task_name: str
    strategy_used: str
    strategy_selection_mode: str
    reason_family: str
    run_classification: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    candidate_count_total: int = 0
    candidate_count_observations: int = 0
    no_op_runs: int = 0


@dataclass
class _FeatureRollupAccumulator:
    task_name: str
    run_classification: str
    runs: int = 0
    snapshot_runs: int = 0
    candidate_metric_totals: dict[str, float] = field(default_factory=dict)
    candidate_metric_counts: dict[str, int] = field(default_factory=dict)
    selected_metric_totals: dict[str, float] = field(default_factory=dict)
    selected_metric_counts: dict[str, int] = field(default_factory=dict)
    candidate_share_totals: dict[str, float] = field(default_factory=dict)
    candidate_share_counts: dict[str, int] = field(default_factory=dict)
    selected_share_totals: dict[str, float] = field(default_factory=dict)
    selected_share_counts: dict[str, int] = field(default_factory=dict)
    strategy_signal_totals: dict[str, float] = field(default_factory=dict)
    strategy_signal_counts: dict[str, int] = field(default_factory=dict)


def build_selector_stats_payload(
    runs: Iterable[AgentRunHistoryPayload],
    *,
    window_hours: int,
    run_limit: int,
    now: float | None = None,
) -> SelectorStatsPayload:
    generated_at = time.time() if now is None else now
    cutoff = generated_at - max(window_hours, 0) * 3600
    filtered_runs = [run for run in runs if run.completed_at >= cutoff]
    outcome_rows: dict[tuple[str, str, str, str, str], _OutcomeAccumulator] = {}
    feature_rollup_rows: dict[tuple[str, str], _FeatureRollupAccumulator] = {}
    classification_counts = {
        FRESH_SELECTOR: 0,
        SEEDED_CLAIMED: 0,
        UNKNOWN_CLASSIFICATION: 0,
    }

    selector_signal_runs = 0
    fallback_runs = 0
    mutation_runs = 0
    no_op_runs = 0
    total_mutations = 0
    total_tool_calls = 0
    candidate_count_total = 0
    candidate_count_observations = 0
    recent_runs: list[SelectorRecentDiagnosticPayload] = []

    for run in filtered_runs:
        metadata = run.result_metadata
        run_classification, classification_reason = classify_selector_run(metadata)
        classification_counts[run_classification] += 1

        feature_rollup_key = (run.task_name, run_classification)
        feature_rollup = feature_rollup_rows.setdefault(
            feature_rollup_key,
            _FeatureRollupAccumulator(
                task_name=run.task_name,
                run_classification=run_classification,
            ),
        )
        feature_rollup.runs += 1

        if _has_feature_snapshot(metadata.selector_feature_snapshot):
            feature_rollup.snapshot_runs += 1
            _accumulate_feature_rollup(feature_rollup, metadata.selector_feature_snapshot)

        if _has_selector_signal(metadata):
            selector_signal_runs += 1
            if metadata.strategy_fallback_reason is not None:
                fallback_runs += 1

            mutation_count = metadata.mutations or 0
            tool_call_count = metadata.tool_calls_executed or 0
            total_mutations += mutation_count
            total_tool_calls += tool_call_count
            if mutation_count > 0:
                mutation_runs += 1
            else:
                no_op_runs += 1

            if metadata.candidate_count is not None:
                candidate_count_total += metadata.candidate_count
                candidate_count_observations += 1

            outcome_key = (
                run.task_name,
                metadata.strategy_used or UNKNOWN_SELECTOR_STRATEGY,
                metadata.strategy_selection_mode or UNKNOWN_SELECTOR_MODE,
                _derive_selector_reason_family(metadata),
                run_classification,
            )
            outcome_row = outcome_rows.setdefault(
                outcome_key,
                _OutcomeAccumulator(
                    task_name=run.task_name,
                    strategy_used=outcome_key[1],
                    strategy_selection_mode=outcome_key[2],
                    reason_family=outcome_key[3],
                    run_classification=outcome_key[4],
                ),
            )
            outcome_row.runs += 1
            if metadata.strategy_fallback_reason is not None:
                outcome_row.fallback_count += 1
            outcome_row.total_mutations += mutation_count
            outcome_row.total_tool_calls += tool_call_count
            if mutation_count > 0:
                outcome_row.mutation_runs += 1
            else:
                outcome_row.no_op_runs += 1
            if metadata.candidate_count is not None:
                outcome_row.candidate_count_total += metadata.candidate_count
                outcome_row.candidate_count_observations += 1

        recent_runs.append(
            SelectorRecentDiagnosticPayload(
                task_id=run.task_id,
                task_name=run.task_name,
                status=run.status,
                completed_at=run.completed_at,
                duration_seconds=run.duration_seconds,
                run_classification=run_classification,
                classification_reason=classification_reason,
                requested_strategy=metadata.requested_strategy,
                strategy_used=metadata.strategy_used,
                strategy_selection_mode=metadata.strategy_selection_mode,
                strategy_selection_reason=metadata.strategy_selection_reason,
                strategy_fallback_reason=metadata.strategy_fallback_reason,
                candidate_count=metadata.candidate_count,
                claimed_work_item_count=metadata.claimed_work_item_count,
                mutations=metadata.mutations,
                tool_calls_executed=metadata.tool_calls_executed,
                strategy_selection_scores=metadata.strategy_selection_scores,
                selector_feature_snapshot=metadata.selector_feature_snapshot,
                result_summary=run.result_summary,
            )
        )

    return SelectorStatsPayload(
        generated_at=generated_at,
        window_hours=window_hours,
        run_limit=run_limit,
        summary=SelectorStatsSummaryPayload(
            total_runs=len(filtered_runs),
            selector_signal_runs=selector_signal_runs,
            fresh_selector_runs=classification_counts[FRESH_SELECTOR],
            seeded_claimed_runs=classification_counts[SEEDED_CLAIMED],
            unknown_runs=classification_counts[UNKNOWN_CLASSIFICATION],
            fallback_runs=fallback_runs,
            mutation_runs=mutation_runs,
            no_op_runs=no_op_runs,
            total_mutations=total_mutations,
            total_tool_calls=total_tool_calls,
            average_candidate_count=_safe_ratio(candidate_count_total, candidate_count_observations),
        ),
        classification_breakdown=[
            SelectorClassificationBreakdownPayload(
                key=key,
                label=_CLASSIFICATION_LABELS[key],
                runs=classification_counts[key],
            )
            for key in (FRESH_SELECTOR, SEEDED_CLAIMED, UNKNOWN_CLASSIFICATION)
        ],
        outcome_rows=[
            SelectorOutcomeRowPayload(
                task_name=row.task_name,
                strategy_used=row.strategy_used,
                strategy_selection_mode=row.strategy_selection_mode,
                reason_family=row.reason_family,
                run_classification=row.run_classification,
                runs=row.runs,
                fallback_count=row.fallback_count,
                mutation_runs=row.mutation_runs,
                total_mutations=row.total_mutations,
                total_tool_calls=row.total_tool_calls,
                average_candidate_count=_safe_ratio(row.candidate_count_total, row.candidate_count_observations),
                no_op_runs=row.no_op_runs,
                no_op_rate=_safe_ratio(row.no_op_runs, row.runs) or 0.0,
            )
            for row in sorted(
                outcome_rows.values(),
                key=lambda item: (
                    item.task_name,
                    item.run_classification,
                    -item.runs,
                    item.strategy_used,
                    item.strategy_selection_mode,
                    item.reason_family,
                ),
            )
        ],
        feature_rollup_rows=[
            SelectorFeatureRollupRowPayload(
                task_name=row.task_name,
                run_classification=row.run_classification,
                runs=row.runs,
                snapshot_runs=row.snapshot_runs,
                candidate_metric_means=_build_rollup_means(row.candidate_metric_totals, row.candidate_metric_counts),
                selected_metric_means=_build_rollup_means(row.selected_metric_totals, row.selected_metric_counts),
                candidate_share_means=_build_rollup_means(row.candidate_share_totals, row.candidate_share_counts),
                selected_share_means=_build_rollup_means(row.selected_share_totals, row.selected_share_counts),
                strategy_signal_means=_build_top_rollup_signals(row),
            )
            for row in sorted(
                (item for item in feature_rollup_rows.values() if item.snapshot_runs > 0),
                key=lambda item: (
                    item.task_name,
                    item.run_classification,
                ),
            )
        ],
        recent_runs=recent_runs,
    )


def classify_selector_run(metadata: RunResultMetadataPayload) -> tuple[str, str]:
    claimed_work_item_count = metadata.claimed_work_item_count
    if claimed_work_item_count is not None:
        if claimed_work_item_count > 0:
            return SEEDED_CLAIMED, f"claimed_work_item_count={claimed_work_item_count}"
        return FRESH_SELECTOR, "claimed_work_item_count=0"
    return UNKNOWN_CLASSIFICATION, "claimed_work_item_count unavailable"


def _has_selector_signal(metadata: RunResultMetadataPayload) -> bool:
    return any(
        (
            metadata.strategy_used is not None,
            metadata.requested_strategy is not None,
            metadata.strategy_selection_mode is not None,
            metadata.strategy_selection_reason is not None,
            metadata.strategy_fallback_reason is not None,
            bool(metadata.strategy_selection_scores),
            metadata.candidate_count is not None,
            metadata.claimed_work_item_count is not None,
        )
    )


def _derive_selector_reason_family(metadata: RunResultMetadataPayload) -> str:
    if metadata.strategy_fallback_reason is not None:
        return "fallback"

    mode = (metadata.strategy_selection_mode or "").strip().lower()
    reason = (metadata.strategy_selection_reason or "").strip().lower()

    if mode == "requested_strategy" or "requested=" in reason or "requested strategy" in reason:
        return "requested"
    if any(token in mode for token in ("priority", "exploration")) or any(
        token in reason for token in ("priority", "exploration")
    ):
        return "priority_scores"
    if any(token in mode for token in ("seed", "random", "roulette")) or any(
        token in reason for token in ("seed", "random", "roulette")
    ):
        return "seeded_random"
    if any(token in mode for token in ("utility", "prior")) or any(token in reason for token in ("utility", "prior")):
        return "utility_priors"
    if any(token in mode for token in ("deterministic", "score", "signal")) or any(
        token in reason for token in ("deterministic", "score", "signal")
    ):
        return "deterministic_signals"
    if "fallback" in reason:
        return "fallback"
    return "unknown"


def _has_feature_snapshot(snapshot: SelectorFeatureSnapshotPayload) -> bool:
    return any(
        (
            bool(snapshot.strategy_signals),
            _population_has_snapshot(snapshot.candidate_population),
            _population_has_snapshot(snapshot.selected_population),
        )
    )


def _population_has_snapshot(population: SelectorPopulationSnapshotPayload) -> bool:
    return bool(population.count or population.metrics or population.shares)


def _accumulate_feature_rollup(
    accumulator: _FeatureRollupAccumulator,
    snapshot: SelectorFeatureSnapshotPayload,
) -> None:
    _accumulate_population_rollup(
        metric_totals=accumulator.candidate_metric_totals,
        metric_counts=accumulator.candidate_metric_counts,
        share_totals=accumulator.candidate_share_totals,
        share_counts=accumulator.candidate_share_counts,
        population=snapshot.candidate_population,
    )
    _accumulate_population_rollup(
        metric_totals=accumulator.selected_metric_totals,
        metric_counts=accumulator.selected_metric_counts,
        share_totals=accumulator.selected_share_totals,
        share_counts=accumulator.selected_share_counts,
        population=snapshot.selected_population,
    )

    for key, value in snapshot.strategy_signals.items():
        _accumulate_mean(accumulator.strategy_signal_totals, accumulator.strategy_signal_counts, key, value)


def _accumulate_population_rollup(
    *,
    metric_totals: dict[str, float],
    metric_counts: dict[str, int],
    share_totals: dict[str, float],
    share_counts: dict[str, int],
    population: SelectorPopulationSnapshotPayload,
) -> None:
    for key in _ROLLUP_METRIC_KEYS:
        metric = population.metrics.get(key)
        if metric is None or metric.count <= 0:
            continue
        if metric.mean is None:
            continue
        _accumulate_mean(metric_totals, metric_counts, key, metric.mean)

    for key in _ROLLUP_SHARE_KEYS:
        value = population.shares.get(key)
        if value is None:
            continue
        _accumulate_mean(share_totals, share_counts, key, value)


def _accumulate_mean(
    totals: dict[str, float],
    counts: dict[str, int],
    key: str,
    value: float,
) -> None:
    totals[key] = totals.get(key, 0.0) + value
    counts[key] = counts.get(key, 0) + 1


def _build_rollup_means(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        key: round(totals[key] / counts[key], 4)
        for key in sorted(totals)
        if counts.get(key, 0) > 0
    }


def _build_top_rollup_signals(accumulator: _FeatureRollupAccumulator) -> dict[str, float]:
    averaged = _build_rollup_means(accumulator.strategy_signal_totals, accumulator.strategy_signal_counts)
    if not averaged:
        return {}
    ranked_keys = sorted(
        averaged,
        key=lambda key: (-abs(averaged[key]), key),
    )[:_ROLLUP_SIGNAL_LIMIT]
    return {key: averaged[key] for key in ranked_keys}


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)
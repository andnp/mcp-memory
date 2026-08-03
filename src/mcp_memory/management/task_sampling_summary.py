from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    SamplingSummaryRowPayload,
    SelectorBehaviorSummaryPayload,
    SelectionStrategyUtilityPayload,
    TaskSamplingSummaryPayload,
)


UTILITY_PRIOR_MIN_RUNS = 3
UTILITY_PRIOR_MUTATION_RATE_WEIGHT = 0.45
UTILITY_PRIOR_MUTATIONS_PER_RUN_WEIGHT = 0.30
UTILITY_PRIOR_MUTATIONS_PER_TOOL_CALL_WEIGHT = 0.15
UTILITY_PRIOR_NO_OP_PENALTY_WEIGHT = 0.20
UTILITY_PRIOR_MUTATIONS_PER_RUN_SCALE = 2.0
UTILITY_PRIOR_QUALITY_WEIGHT = 0.85
UTILITY_PRIOR_QUALITY_REGRESSION_SCALE = 2.0
UNKNOWN_SELECTOR_MODE = "unspecified"
UNKNOWN_SELECTOR_STRATEGY = "unknown"


@dataclass
class _UsageAccumulator:
    name: str
    runs: int = 0
    fallbacks: int = 0
    tasks: set[str] = field(default_factory=set)


@dataclass
class _UtilityAccumulator:
    task_name: str
    strategy_used: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    candidate_count_total: int = 0
    candidate_count_observations: int = 0
    no_op_runs: int = 0
    quality_evidence_runs: int = 0
    useful_work_count: int = 0
    retrieval_regression_count: int = 0
    zero_result_change: int = 0


@dataclass
class _SelectorBehaviorAccumulator:
    task_name: str
    strategy_selection_mode: str
    strategy_used: str
    reason_family: str
    runs: int = 0
    fallback_count: int = 0
    mutation_runs: int = 0
    total_mutations: int = 0
    total_tool_calls: int = 0
    candidate_count_total: int = 0
    candidate_count_observations: int = 0
    no_op_runs: int = 0


def build_task_sampling_summary(runs: Iterable[AgentRunHistoryPayload]) -> TaskSamplingSummaryPayload:
    selection_rows: dict[str, _UsageAccumulator] = {}
    grouping_rows: dict[str, _UsageAccumulator] = {}
    utility_rows: dict[tuple[str, str], _UtilityAccumulator] = {}
    selector_behavior_rows: dict[tuple[str, str, str, str], _SelectorBehaviorAccumulator] = {}

    for run in runs:
        metadata = run.result_metadata

        if metadata.strategy_used is not None:
            selection_row = selection_rows.setdefault(metadata.strategy_used, _UsageAccumulator(name=metadata.strategy_used))
            selection_row.runs += 1
            if metadata.strategy_fallback_reason is not None:
                selection_row.fallbacks += 1
            selection_row.tasks.add(run.task_name)

            utility_key = (run.task_name, metadata.strategy_used)
            utility_row = utility_rows.setdefault(
                utility_key,
                _UtilityAccumulator(task_name=run.task_name, strategy_used=metadata.strategy_used),
            )
            utility_row.runs += 1
            if metadata.strategy_fallback_reason is not None:
                utility_row.fallback_count += 1

            mutation_count = metadata.mutations or 0
            tool_call_count = metadata.tool_calls_executed or 0
            utility_row.total_mutations += mutation_count
            utility_row.total_tool_calls += tool_call_count
            if mutation_count > 0:
                utility_row.mutation_runs += 1
            else:
                utility_row.no_op_runs += 1

            if metadata.candidate_count is not None:
                utility_row.candidate_count_total += metadata.candidate_count
                utility_row.candidate_count_observations += 1
            utility_row.quality_evidence_runs += metadata.quality_evidence_runs
            utility_row.useful_work_count += metadata.useful_work_count
            utility_row.retrieval_regression_count += metadata.retrieval_regression_count
            utility_row.zero_result_change += metadata.zero_result_change

        if _has_selector_behavior_signal(metadata):
            selector_mode = metadata.strategy_selection_mode or UNKNOWN_SELECTOR_MODE
            strategy_used = metadata.strategy_used or UNKNOWN_SELECTOR_STRATEGY
            selector_key = (
                run.task_name,
                selector_mode,
                strategy_used,
                _derive_selector_reason_family(metadata),
            )
            selector_row = selector_behavior_rows.setdefault(
                selector_key,
                _SelectorBehaviorAccumulator(
                    task_name=run.task_name,
                    strategy_selection_mode=selector_mode,
                    strategy_used=strategy_used,
                    reason_family=selector_key[3],
                ),
            )
            selector_row.runs += 1
            if metadata.strategy_fallback_reason is not None:
                selector_row.fallback_count += 1

            mutation_count = metadata.mutations or 0
            tool_call_count = metadata.tool_calls_executed or 0
            selector_row.total_mutations += mutation_count
            selector_row.total_tool_calls += tool_call_count
            if mutation_count > 0:
                selector_row.mutation_runs += 1
            else:
                selector_row.no_op_runs += 1

            if metadata.candidate_count is not None:
                selector_row.candidate_count_total += metadata.candidate_count
                selector_row.candidate_count_observations += 1

        if metadata.grouping_strategy_used is not None:
            grouping_row = grouping_rows.setdefault(
                metadata.grouping_strategy_used,
                _UsageAccumulator(name=metadata.grouping_strategy_used),
            )
            grouping_row.runs += 1
            if metadata.grouping_fallback_reason is not None:
                grouping_row.fallbacks += 1
            grouping_row.tasks.add(run.task_name)

    return TaskSamplingSummaryPayload(
        selection=[
            SamplingSummaryRowPayload(
                name=row.name,
                runs=row.runs,
                fallbacks=row.fallbacks,
                tasks=sorted(row.tasks),
            )
            for row in sorted(selection_rows.values(), key=lambda item: (-item.runs, item.name))
        ],
        grouping=[
            SamplingSummaryRowPayload(
                name=row.name,
                runs=row.runs,
                fallbacks=row.fallbacks,
                tasks=sorted(row.tasks),
            )
            for row in sorted(grouping_rows.values(), key=lambda item: (-item.runs, item.name))
        ],
        selection_utility=[
            SelectionStrategyUtilityPayload(
                task_name=row.task_name,
                strategy_used=row.strategy_used,
                runs=row.runs,
                fallback_count=row.fallback_count,
                mutation_runs=row.mutation_runs,
                total_mutations=row.total_mutations,
                total_tool_calls=row.total_tool_calls,
                average_candidate_count=_safe_ratio(row.candidate_count_total, row.candidate_count_observations),
                mutation_rate=_safe_ratio(row.mutation_runs, row.runs) or 0.0,
                mutations_per_run=_safe_ratio(row.total_mutations, row.runs) or 0.0,
                mutations_per_tool_call=_safe_ratio(row.total_mutations, row.total_tool_calls),
                no_op_runs=row.no_op_runs,
                no_op_rate=_safe_ratio(row.no_op_runs, row.runs) or 0.0,
                quality_evidence_runs=row.quality_evidence_runs,
                useful_work_count=row.useful_work_count,
                retrieval_regression_count=row.retrieval_regression_count,
                zero_result_change=row.zero_result_change,
            )
            for row in sorted(utility_rows.values(), key=lambda item: (item.task_name, -item.runs, item.strategy_used))
        ],
        selector_behavior=[
            SelectorBehaviorSummaryPayload(
                task_name=row.task_name,
                strategy_selection_mode=row.strategy_selection_mode,
                strategy_used=row.strategy_used,
                reason_family=row.reason_family,
                runs=row.runs,
                fallback_count=row.fallback_count,
                mutation_runs=row.mutation_runs,
                total_mutations=row.total_mutations,
                total_tool_calls=row.total_tool_calls,
                average_candidate_count=_safe_ratio(row.candidate_count_total, row.candidate_count_observations),
                mutation_rate=_safe_ratio(row.mutation_runs, row.runs) or 0.0,
                mutations_per_run=_safe_ratio(row.total_mutations, row.runs) or 0.0,
                mutations_per_tool_call=_safe_ratio(row.total_mutations, row.total_tool_calls),
                no_op_runs=row.no_op_runs,
                no_op_rate=_safe_ratio(row.no_op_runs, row.runs) or 0.0,
            )
            for row in sorted(
                selector_behavior_rows.values(),
                key=lambda item: (
                    item.task_name,
                    -item.runs,
                    item.strategy_selection_mode,
                    item.strategy_used,
                    item.reason_family,
                ),
            )
        ],
    )


def build_selection_strategy_utility_priors(
    runs: Iterable[AgentRunHistoryPayload],
    *,
    task_name: str,
    allowed_strategies: Iterable[str] | None = None,
    min_runs: int = UTILITY_PRIOR_MIN_RUNS,
) -> dict[str, float]:
    summary = build_task_sampling_summary(runs)
    return selection_utility_prior_scores(
        summary.selection_utility,
        task_name=task_name,
        allowed_strategies=allowed_strategies,
        min_runs=min_runs,
    )


def selection_utility_prior_scores(
    rows: Iterable[SelectionStrategyUtilityPayload],
    *,
    task_name: str,
    allowed_strategies: Iterable[str] | None = None,
    min_runs: int = UTILITY_PRIOR_MIN_RUNS,
) -> dict[str, float]:
    allowed = set(allowed_strategies or [])
    priors: dict[str, float] = {}
    for row in rows:
        if row.task_name != task_name or row.runs < max(min_runs, 1):
            continue
        if allowed and row.strategy_used not in allowed:
            continue
        priors[row.strategy_used] = _utility_prior_score(row)
    return priors


def _utility_prior_score(row: SelectionStrategyUtilityPayload) -> float:
    if row.quality_evidence_runs > 0:
        useful_rate = _clamp01(row.useful_work_count / row.quality_evidence_runs)
        regression_penalty = _clamp01(row.retrieval_regression_count / row.quality_evidence_runs / UTILITY_PRIOR_QUALITY_REGRESSION_SCALE)
        zero_result_penalty = _clamp01(max(row.zero_result_change, 0) / row.quality_evidence_runs / UTILITY_PRIOR_QUALITY_REGRESSION_SCALE)
        quality_score = _clamp01(useful_rate - regression_penalty - zero_result_penalty)
        return round(_clamp01(UTILITY_PRIOR_QUALITY_WEIGHT * quality_score - UTILITY_PRIOR_NO_OP_PENALTY_WEIGHT * _clamp01(row.no_op_rate)), 4)
    mutation_rate = _clamp01(row.mutation_rate)
    mutations_per_run = _clamp01(row.mutations_per_run / UTILITY_PRIOR_MUTATIONS_PER_RUN_SCALE)
    mutations_per_tool_call = _clamp01(row.mutations_per_tool_call or 0.0)
    no_op_rate = _clamp01(row.no_op_rate)
    score = (
        UTILITY_PRIOR_MUTATION_RATE_WEIGHT * mutation_rate
        + UTILITY_PRIOR_MUTATIONS_PER_RUN_WEIGHT * mutations_per_run
        + UTILITY_PRIOR_MUTATIONS_PER_TOOL_CALL_WEIGHT * mutations_per_tool_call
        - UTILITY_PRIOR_NO_OP_PENALTY_WEIGHT * no_op_rate
    )
    return round(_clamp01(score), 4)


def _clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _has_selector_behavior_signal(metadata) -> bool:
    return any(
        (
            metadata.strategy_used is not None,
            metadata.requested_strategy is not None,
            metadata.strategy_selection_mode is not None,
            metadata.strategy_selection_reason is not None,
            metadata.strategy_fallback_reason is not None,
        )
    )


def _derive_selector_reason_family(metadata) -> str:
    if metadata.strategy_fallback_reason is not None:
        return "fallback"

    mode = (metadata.strategy_selection_mode or "").strip().lower()
    reason = (metadata.strategy_selection_reason or "").strip().lower()

    if mode == "requested_strategy" or "requested=" in reason or "requested strategy" in reason:
        return "requested"
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


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)

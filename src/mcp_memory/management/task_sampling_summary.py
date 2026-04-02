from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    SamplingSummaryRowPayload,
    SelectionStrategyUtilityPayload,
    TaskSamplingSummaryPayload,
)


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


def build_task_sampling_summary(runs: Iterable[AgentRunHistoryPayload]) -> TaskSamplingSummaryPayload:
    selection_rows: dict[str, _UsageAccumulator] = {}
    grouping_rows: dict[str, _UsageAccumulator] = {}
    utility_rows: dict[tuple[str, str], _UtilityAccumulator] = {}

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
            )
            for row in sorted(utility_rows.values(), key=lambda item: (item.task_name, -item.runs, item.strategy_used))
        ],
    )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)
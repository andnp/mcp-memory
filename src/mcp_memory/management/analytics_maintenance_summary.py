from __future__ import annotations

from dataclasses import dataclass, field

from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SWEEPER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
)
from mcp_memory.management.models import (
    NerdMaintenanceAgentYieldPayload,
    NerdMaintenanceDeltaBucketPayload,
    NerdMaintenanceDeltaSeriesPayload,
    NerdMaintenanceSummaryPayload,
    NerdMaintenanceSummaryRowPayload,
)
from mcp_memory.management.reporting_rows import MaintenanceTaskRunRow
from mcp_memory.management.result_views import TaskResultView


_RUN_REPORTED_DELTA_KEYS: tuple[tuple[str, str], ...] = (
    ("created", "created_count"),
    ("merged", "merged_count"),
    ("updated", "updated_count"),
    ("archived", "archived_count"),
    ("degraded", "degraded_count"),
    ("restored", "restored_count"),
    ("meaningful_actions", "meaningful_actions"),
    ("lines_compressed", "lines_compressed"),
)
_MAINTENANCE_FAMILY_DEFS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "organization",
        "Organization & taxonomy",
        (PROJECT_MANAGER_TASK_NAME, TAXONOMIST_TASK_NAME),
    ),
    (
        "verification",
        "Verification & linkage",
        (FACT_CHECKER_TASK_NAME, GRAPH_LINKER_TASK_NAME, CONFLICT_DETECTOR_TASK_NAME),
    ),
    (
        "compaction",
        "Compaction & curation",
        (DEFRAGMENTER_TASK_NAME, DEDUPLICATOR_TASK_NAME, CURATOR_TASK_NAME),
    ),
    (
        "retention",
        "Retention",
        (SWEEPER_TASK_NAME,),
    ),
)
_MAINTENANCE_FAMILY_BY_TASK = {
    task_name: (family_key, family_label)
    for family_key, family_label, task_names in _MAINTENANCE_FAMILY_DEFS
    for task_name in task_names
}


@dataclass
class _MaintenanceSummaryAccumulator:
    task_names: set[str] = field(default_factory=set)
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    retry_runs: int = 0
    created_count: int = 0
    merged_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    degraded_count: int = 0
    restored_count: int = 0
    meaningful_actions: int = 0
    lines_compressed: int = 0

    @property
    def delta_total(self) -> int:
        return (
            self.created_count
            + self.merged_count
            + self.updated_count
            + self.archived_count
            + self.degraded_count
            + self.restored_count
        )


def build_maintenance_summary(
    maintenance_rows: list[MaintenanceTaskRunRow],
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdMaintenanceSummaryPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts or not maintenance_rows:
        return NerdMaintenanceSummaryPayload()

    family_accumulators: dict[str, _MaintenanceSummaryAccumulator] = {}
    agent_accumulators: dict[str, _MaintenanceSummaryAccumulator] = {}
    agent_families: dict[str, tuple[str, str]] = {}
    family_bucket_totals: dict[str, dict[int, dict[str, int]]] = {}

    for row in maintenance_rows:
        completed_at = row.completed_at
        if completed_at < cutoff or completed_at > generated_at:
            continue
        task_name = row.task_name
        family_key, family_label = _maintenance_family_for_task(task_name)
        family_acc = family_accumulators.setdefault(family_key, _MaintenanceSummaryAccumulator())
        agent_acc = agent_accumulators.setdefault(task_name, _MaintenanceSummaryAccumulator())
        agent_families[task_name] = (family_key, family_label)

        deltas = _run_reported_delta_counts(row.result)
        _update_maintenance_summary_accumulator(family_acc, task_name=task_name, status=row.status, deltas=deltas)
        _update_maintenance_summary_accumulator(agent_acc, task_name=task_name, status=row.status, deltas=deltas)

        bucket_start = int(completed_at // bucket_seconds) * bucket_seconds
        family_bucket = family_bucket_totals.setdefault(family_key, {}).setdefault(
            bucket_start,
            {attribute_name: 0 for _, attribute_name in _RUN_REPORTED_DELTA_KEYS},
        )
        for _key, attribute_name in _RUN_REPORTED_DELTA_KEYS:
            family_bucket[attribute_name] += deltas[attribute_name]

    by_family = [
        NerdMaintenanceSummaryRowPayload(
            key=family_key,
            label=family_label,
            task_names=sorted(accumulator.task_names),
            total_runs=accumulator.total_runs,
            completed_runs=accumulator.completed_runs,
            failed_runs=accumulator.failed_runs,
            retry_runs=accumulator.retry_runs,
            created_count=accumulator.created_count,
            merged_count=accumulator.merged_count,
            updated_count=accumulator.updated_count,
            archived_count=accumulator.archived_count,
            degraded_count=accumulator.degraded_count,
            restored_count=accumulator.restored_count,
            meaningful_actions=accumulator.meaningful_actions,
            lines_compressed=accumulator.lines_compressed,
            delta_total=accumulator.delta_total,
        )
        for family_key, family_label, _task_names in _MAINTENANCE_FAMILY_DEFS
        if (accumulator := family_accumulators.get(family_key)) is not None
    ]

    by_agent = [
        _build_agent_yield_payload(
            task_name=task_name,
            family_key=agent_families[task_name][0],
            family_label=agent_families[task_name][1],
            accumulator=accumulator,
        )
        for task_name, accumulator in sorted(
            agent_accumulators.items(),
            key=lambda item: (-item[1].delta_total, -item[1].meaningful_actions, item[0]),
        )
    ]

    family_delta_series = [
        NerdMaintenanceDeltaSeriesPayload(
            key=family_key,
            label=family_label,
            task_names=sorted(family_accumulators[family_key].task_names),
            buckets=[
                NerdMaintenanceDeltaBucketPayload(
                    bucket_start=float(bucket_start),
                    **family_bucket_totals.get(family_key, {}).get(
                        bucket_start,
                        {attribute_name: 0 for _, attribute_name in _RUN_REPORTED_DELTA_KEYS},
                    ),
                )
                for bucket_start in bucket_starts
            ],
        )
        for family_key, family_label, _task_names in _MAINTENANCE_FAMILY_DEFS
        if family_key in family_accumulators
    ]

    return NerdMaintenanceSummaryPayload(
        by_family=by_family,
        by_agent=by_agent,
        family_delta_series=family_delta_series,
    )


def _bucket_starts(*, cutoff: float, generated_at: float, bucket_seconds: int) -> list[int]:
    if generated_at < cutoff:
        return []
    start_bucket = int(cutoff // bucket_seconds) * bucket_seconds
    end_bucket = int(generated_at // bucket_seconds) * bucket_seconds
    if end_bucket < start_bucket:
        return []
    return list(range(start_bucket, end_bucket + bucket_seconds, bucket_seconds))


def _maintenance_family_for_task(task_name: str) -> tuple[str, str]:
    family = _MAINTENANCE_FAMILY_BY_TASK.get(task_name)
    if family is not None:
        return family
    return ("other", "Other")


def _run_reported_delta_counts(result: TaskResultView) -> dict[str, int]:
    mutation_outcome = result.mutation_outcome
    value_by_result_key = {
        "created": mutation_outcome.created,
        "merged": mutation_outcome.merged,
        "updated": mutation_outcome.updated,
        "archived": mutation_outcome.archived,
        "degraded": mutation_outcome.degraded,
        "restored": mutation_outcome.restored,
        "meaningful_actions": result.meaningful_actions,
        "lines_compressed": result.lines_compressed,
    }
    return {
        attribute_name: max(value_by_result_key[result_key] or 0, 0)
        for result_key, attribute_name in _RUN_REPORTED_DELTA_KEYS
    }


def _update_maintenance_summary_accumulator(
    accumulator: _MaintenanceSummaryAccumulator,
    *,
    task_name: str,
    status: str,
    deltas: dict[str, int],
) -> None:
    accumulator.task_names.add(task_name)
    accumulator.total_runs += 1
    if status == "completed":
        accumulator.completed_runs += 1
    elif status == "failed":
        accumulator.failed_runs += 1
    elif status == "retry":
        accumulator.retry_runs += 1
    for attribute_name, value in deltas.items():
        setattr(accumulator, attribute_name, getattr(accumulator, attribute_name) + value)


def _build_agent_yield_payload(
    *,
    task_name: str,
    family_key: str,
    family_label: str,
    accumulator: _MaintenanceSummaryAccumulator,
) -> NerdMaintenanceAgentYieldPayload:
    completed_runs = max(accumulator.completed_runs, 0)
    denominator = completed_runs if completed_runs > 0 else accumulator.total_runs
    return NerdMaintenanceAgentYieldPayload(
        key=task_name,
        label=task_name,
        family_key=family_key,
        family_label=family_label,
        task_names=sorted(accumulator.task_names),
        total_runs=accumulator.total_runs,
        completed_runs=accumulator.completed_runs,
        failed_runs=accumulator.failed_runs,
        retry_runs=accumulator.retry_runs,
        created_count=accumulator.created_count,
        merged_count=accumulator.merged_count,
        updated_count=accumulator.updated_count,
        archived_count=accumulator.archived_count,
        degraded_count=accumulator.degraded_count,
        restored_count=accumulator.restored_count,
        meaningful_actions=accumulator.meaningful_actions,
        lines_compressed=accumulator.lines_compressed,
        delta_total=accumulator.delta_total,
        actions_per_completed_run=0.0 if denominator == 0 else round(accumulator.meaningful_actions / denominator, 4),
        lines_per_completed_run=0.0 if denominator == 0 else round(accumulator.lines_compressed / denominator, 4),
        delta_per_completed_run=0.0 if denominator == 0 else round(accumulator.delta_total / denominator, 4),
    )
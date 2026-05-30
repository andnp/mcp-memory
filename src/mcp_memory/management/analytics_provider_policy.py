from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mcp_memory.management.models import (
    NerdProviderPolicyPayload,
    NerdProviderPolicyProviderPayload,
    NerdProviderPolicyTaskPayload,
    NerdStatPayload,
)
from mcp_memory.management.reporting_rows import ProviderPolicyEventRow, ProviderUsageRow, RuntimeLogRow


@dataclass
class _ProviderPolicyTaskAccumulator:
    route_exhaustion_count: int = 0
    legacy_fallback_denied_count: int = 0
    admission_skip_count: int = 0
    skip_provider_counts: Counter[tuple[str, str, str | None]] = field(default_factory=Counter)
    active_admission_providers: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class _ProviderPolicyProviderAccumulator:
    provider_name: str
    admission_skip_count: int = 0
    task_counts: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    active_admission_reason: str | None = None
    active_admission_category: str | None = None
    active_retry_delay_seconds: float | None = None


def build_provider_policy_rollups(
    *,
    provider_rows: list[ProviderUsageRow],
    provider_policy_event_rows: list[ProviderPolicyEventRow],
    provider_policy_log_rows: list[RuntimeLogRow],
    provider_usage_repo,
    workspace_id: str | None,
) -> NerdProviderPolicyPayload:
    by_task: dict[str, _ProviderPolicyTaskAccumulator] = {}
    by_provider: dict[tuple[str, str], _ProviderPolicyProviderAccumulator] = {}
    total_route_exhaustion_count = 0
    total_legacy_fallback_denied_count = 0
    total_admission_skip_count = 0
    total_warning_suppressed_count = 0

    if provider_policy_event_rows:
        for row in provider_policy_event_rows:
            task_name = row.task_name or "unknown"
            accumulator = by_task.setdefault(task_name, _ProviderPolicyTaskAccumulator())
            event_kind = row.event_kind
            if event_kind == "route_exhausted":
                accumulator.route_exhaustion_count += 1
                total_route_exhaustion_count += 1
            elif event_kind == "legacy_fallback_denied":
                accumulator.legacy_fallback_denied_count += 1
                total_legacy_fallback_denied_count += 1
            elif event_kind == "route_skipped":
                provider_key = row.provider_key
                model_name = row.model_name
                reason_code = row.reason_code
                if provider_key is not None and model_name is not None:
                    accumulator.skip_provider_counts[(provider_key, model_name, reason_code)] += 1
            if row.warning_suppressed:
                total_warning_suppressed_count += 1
    else:
        for row in provider_policy_log_rows:
            message = row.message
            task_name = row.task_name or "unknown"
            accumulator = by_task.setdefault(task_name, _ProviderPolicyTaskAccumulator())
            if message == "Provider routing exhausted all configured routes":
                accumulator.route_exhaustion_count += 1
                total_route_exhaustion_count += 1
            elif message == "Legacy fallback provider is unavailable due to admission control":
                accumulator.legacy_fallback_denied_count += 1
                total_legacy_fallback_denied_count += 1

    for row in provider_rows:
        if row.status != "skipped":
            continue
        task_name = row.task_name or "unknown"
        provider_key = row.provider_key
        provider_name = row.provider_name
        model_name = row.model_name
        reason_code = row.reason_code

        task_accumulator = by_task.setdefault(task_name, _ProviderPolicyTaskAccumulator())
        task_accumulator.admission_skip_count += 1
        task_accumulator.skip_provider_counts[(provider_key, model_name, reason_code)] += 1

        provider_accumulator = by_provider.setdefault(
            (provider_key, model_name),
            _ProviderPolicyProviderAccumulator(provider_name=provider_name),
        )
        provider_accumulator.admission_skip_count += 1
        provider_accumulator.task_counts[task_name] += 1
        if reason_code is not None:
            provider_accumulator.reason_counts[reason_code] += 1
        total_admission_skip_count += 1

    for summary in provider_usage_repo.summarize_usage(workspace_id=workspace_id):
        if summary.active_admission_reason is None:
            continue
        provider_accumulator = by_provider.setdefault(
            (summary.provider_key, summary.model_name),
            _ProviderPolicyProviderAccumulator(provider_name=summary.provider_name),
        )
        provider_accumulator.active_admission_reason = summary.active_admission_reason
        provider_accumulator.active_admission_category = summary.active_admission_category
        provider_accumulator.active_retry_delay_seconds = summary.active_retry_delay_seconds
        if summary.task_name is not None:
            by_task.setdefault(summary.task_name, _ProviderPolicyTaskAccumulator()).active_admission_providers.add(
                (summary.provider_key, summary.model_name)
            )

    task_rows = [
        NerdProviderPolicyTaskPayload(
            task_name=task_name,
            route_exhaustion_count=accumulator.route_exhaustion_count,
            legacy_fallback_denied_count=accumulator.legacy_fallback_denied_count,
            admission_skip_count=accumulator.admission_skip_count,
            top_skip_provider_key=_counter_top_value(accumulator.skip_provider_counts, index=0),
            top_skip_model_name=_counter_top_value(accumulator.skip_provider_counts, index=1),
            top_skip_reason_code=_counter_top_value(accumulator.skip_provider_counts, index=2),
            active_admission_provider_count=len(accumulator.active_admission_providers),
        )
        for task_name, accumulator in by_task.items()
        if (
            accumulator.route_exhaustion_count > 0
            or accumulator.legacy_fallback_denied_count > 0
            or accumulator.admission_skip_count > 0
            or accumulator.active_admission_providers
        )
    ]
    task_rows.sort(
        key=lambda row: (
            -(row.route_exhaustion_count + row.legacy_fallback_denied_count + row.admission_skip_count),
            row.task_name,
        )
    )

    provider_rows_payload = [
        NerdProviderPolicyProviderPayload(
            provider_key=provider_key,
            provider_name=accumulator.provider_name,
            model_name=model_name,
            admission_skip_count=accumulator.admission_skip_count,
            distinct_task_count=len(accumulator.task_counts),
            top_task_name=_string_counter_top_key(accumulator.task_counts),
            top_reason_code=_string_counter_top_key(accumulator.reason_counts),
            active_admission_reason=accumulator.active_admission_reason,
            active_admission_category=accumulator.active_admission_category,
            active_retry_delay_seconds=accumulator.active_retry_delay_seconds,
        )
        for (provider_key, model_name), accumulator in by_provider.items()
        if accumulator.admission_skip_count > 0 or accumulator.active_admission_reason is not None
    ]
    provider_rows_payload.sort(key=lambda row: (-row.admission_skip_count, row.provider_key, row.model_name))

    return NerdProviderPolicyPayload(
        stats=[
            NerdStatPayload(
                key="provider_policy_route_exhaustion_count",
                label="Route exhaustion warnings",
                value=float(total_route_exhaustion_count),
                unit="count",
            ),
            NerdStatPayload(
                key="provider_policy_legacy_fallback_denied_count",
                label="Legacy fallback denials",
                value=float(total_legacy_fallback_denied_count),
                unit="count",
            ),
            NerdStatPayload(
                key="provider_policy_admission_skip_count",
                label="Admission-control skips",
                value=float(total_admission_skip_count),
                unit="count",
            ),
            NerdStatPayload(
                key="provider_policy_warning_suppressed_count",
                label="Suppressed warning duplicates",
                value=float(total_warning_suppressed_count),
                unit="count",
            ),
        ],
        by_task=task_rows,
        by_provider=provider_rows_payload,
    )


def _counter_top_value(counter: Counter[tuple[str, str, str | None]], *, index: int) -> str | None:
    if not counter:
        return None
    top_key = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
    value = top_key[index]
    return value if isinstance(value, str) and value else None


def _string_counter_top_key(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return max(counter.items(), key=lambda item: (item[1], item[0]))[0]

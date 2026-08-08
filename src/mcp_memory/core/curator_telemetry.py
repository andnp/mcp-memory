"""Authoritative telemetry contracts for curator executions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

from mcp_memory.operational_store_rows import ProviderUsageSample


@dataclass(frozen=True, slots=True)
class CuratorTokenTotals:
    """Tokens observed on uniquely identified provider attempts."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True, slots=True)
class CuratorTelemetryDiscrepancies:
    """Differences between independently persisted telemetry sources."""

    provider_attempts_without_identity: int = 0
    provider_attempts_without_task: int = 0
    provider_attempts_without_execution: int = 0
    reported_tool_calls_delta: int = 0
    reported_mutations_delta: int = 0
    verified_mutations_delta: int = 0


@dataclass(frozen=True, slots=True)
class CuratorTelemetry:
    """Reconciled curator telemetry, separated by evidence authority."""

    internal_tool_calls: int = 0
    provider_attempts: int = 0
    verified_mutations: int = 0
    productive_mutations: int = 0
    quality_evidence: int = 0
    failures: int = 0
    retries: int = 0
    token_totals: CuratorTokenTotals = CuratorTokenTotals()
    unattributed_calls: int = 0
    discrepancies: CuratorTelemetryDiscrepancies = CuratorTelemetryDiscrepancies()

    @property
    def compatibility_mutations(self) -> int:
        """Legacy result projection; authoritative value is productive_mutations."""

        return self.productive_mutations

    @property
    def compatibility_provider_calls(self) -> int:
        """Legacy result projection; authoritative value is provider_attempts."""

        return self.provider_attempts


@dataclass(frozen=True, slots=True)
class CuratorProviderAggregation:
    """Deduplicated provider evidence for one curator reporting window."""

    provider_attempts: int = 0
    failures: int = 0
    retries: int = 0
    token_totals: CuratorTokenTotals = CuratorTokenTotals()
    unattributed_calls: int = 0


def aggregate_provider_attempts(
    samples: Iterable[ProviderUsageSample],
) -> CuratorProviderAggregation:
    """Reconcile provider rows by stable attempt identity.

    A provider row is authoritative only when it carries task, epoch, request,
    and attempt identity. Missing identity is surfaced, never grouped by name.
    """

    identified: dict[str, ProviderUsageSample] = {}
    unattributed_calls = 0
    for sample in samples:
        identity = sample.attempt_identity
        if not identity or not sample.task_id or sample.execution_epoch is None or not sample.request_id or sample.attempt is None:
            unattributed_calls += 1
            continue
        previous = identified.get(identity)
        if previous is None or _sample_precedence(sample) > _sample_precedence(previous):
            identified[identity] = sample

    retry_groups: dict[str, int] = {}
    failures = 0
    tokens = CuratorTokenTotals()
    for sample in identified.values():
        group = f"{sample.task_id}:{sample.execution_epoch}"
        retry_groups[group] = retry_groups.get(group, 0) + 1
        if sample.status not in {"success", "running"}:
            failures += 1
        tokens = _add_tokens(tokens, sample)

    return CuratorProviderAggregation(
        provider_attempts=len(identified),
        failures=failures,
        retries=sum(max(count - 1, 0) for count in retry_groups.values()),
        token_totals=tokens,
        unattributed_calls=unattributed_calls,
    )


def _sample_precedence(sample: ProviderUsageSample) -> tuple[int, float]:
    return (int(sample.status != "running"), sample.created_at)


def _add_tokens(total: CuratorTokenTotals, sample: ProviderUsageSample) -> CuratorTokenTotals:
    return CuratorTokenTotals(
        input_tokens=total.input_tokens + (sample.input_tokens or 0),
        output_tokens=total.output_tokens + (sample.output_tokens or 0),
        cached_input_tokens=total.cached_input_tokens + (sample.cached_input_tokens or 0),
        cache_write_tokens=total.cache_write_tokens + (sample.cache_write_tokens or 0),
        reasoning_tokens=total.reasoning_tokens + (sample.reasoning_tokens or 0),
        total_tokens=total.total_tokens + (sample.total_tokens or 0),
    )

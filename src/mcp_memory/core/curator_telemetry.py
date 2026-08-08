"""Authoritative telemetry contracts for curator executions."""

from __future__ import annotations

from dataclasses import dataclass


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

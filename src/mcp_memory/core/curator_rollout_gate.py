"""Pure operator/canary assessment for curator rollout evidence.

This module does not admit, block, or mutate runtime work.  It evaluates a
completed evidence window so an operator can decide whether a rollout has
earned wider exposure.  Mutation receipts are execution evidence only;
productive yield comes from evaluated quality outcomes.

The default policy mirrors the rollout runbook: regressions may be at most
5%, unobserved outcomes must be below 10%, and productive outcomes must be at
least 80% of evaluated outcomes.  Structural-only outcomes count as
productive only when ``allow_structural_only`` is enabled.  Empty evidence and
any explicit integrity or attribution failure fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CuratorRolloutEvidence:
    """Categorized evidence for one completed curator canary window."""

    verified_mutations: int = 0
    productive_outcomes: int = 0
    structural_only_outcomes: int = 0
    evaluated_outcomes: int = 0
    unobserved_outcomes: int = 0
    regressions: int = 0
    reconciliation_mismatches: int = 0
    destructive_false_positives: int = 0
    provider_attribution_gaps: int = 0
    ledger_invalid_runs: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            name = field.name
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class CuratorRolloutPolicy:
    """Thresholds for an operator's rollout decision."""

    max_regression_rate: float = 0.05
    max_unobserved_rate: float = 0.10
    min_productive_rate: float = 0.80
    allow_structural_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_regression_rate",
            "max_unobserved_rate",
            "min_productive_rate",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class CuratorRolloutDecision:
    """Immutable result of :func:`assess_curator_rollout`."""

    allowed: bool
    stop_reasons: tuple[str, ...]
    regression_rate: float
    unobserved_rate: float
    productive_rate: float


def assess_curator_rollout(
    evidence: CuratorRolloutEvidence,
    policy: CuratorRolloutPolicy | None = None,
) -> CuratorRolloutDecision:
    """Assess evidence without changing runtime admission or stored state.

    Rates use evaluated outcomes for regression and evaluated-plus-unobserved
    outcomes for observation coverage.  Every explicit integrity, destructive
    safety, or attribution issue stops the rollout, even if numeric thresholds
    pass.  Stop reasons are returned in stable evaluation order.
    """

    policy = policy if policy is not None else CuratorRolloutPolicy()
    reasons: list[str] = []
    observed_total = evidence.evaluated_outcomes + evidence.unobserved_outcomes
    productive = evidence.productive_outcomes + (
        evidence.structural_only_outcomes if policy.allow_structural_only else 0
    )
    regression_rate = _rate(evidence.regressions, evidence.evaluated_outcomes)
    unobserved_rate = _rate(evidence.unobserved_outcomes, observed_total)
    productive_rate = _rate(productive, evidence.evaluated_outcomes)

    if observed_total == 0:
        reasons.append("no_evidence")
    if evidence.ledger_invalid_runs:
        reasons.append("ledger_invalid")
    if evidence.reconciliation_mismatches:
        reasons.append("reconciliation_mismatch")
    if evidence.destructive_false_positives:
        reasons.append("destructive_false_positive")
    if evidence.provider_attribution_gaps:
        reasons.append("provider_attribution_gap")
    if evidence.evaluated_outcomes and _exceeds(
        evidence.regressions,
        evidence.evaluated_outcomes,
        policy.max_regression_rate,
    ):
        reasons.append("regression_rate_exceeded")
    if observed_total and _at_or_above(
        evidence.unobserved_outcomes,
        observed_total,
        policy.max_unobserved_rate,
    ):
        reasons.append("unobserved_rate_exceeded")
    if evidence.evaluated_outcomes and _below(
        productive,
        evidence.evaluated_outcomes,
        policy.min_productive_rate,
    ):
        reasons.append("productive_rate_below_threshold")

    return CuratorRolloutDecision(
        allowed=not reasons,
        stop_reasons=tuple(reasons),
        regression_rate=regression_rate,
        unobserved_rate=unobserved_rate,
        productive_rate=productive_rate,
    )


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _exceeds(numerator: int, denominator: int, threshold: float) -> bool:
    return Decimal(numerator) / Decimal(denominator) > Decimal(str(threshold))


def _below(numerator: int, denominator: int, threshold: float) -> bool:
    return Decimal(numerator) / Decimal(denominator) < Decimal(str(threshold))


def _at_or_above(numerator: int, denominator: int, threshold: float) -> bool:
    return Decimal(numerator) / Decimal(denominator) >= Decimal(str(threshold))

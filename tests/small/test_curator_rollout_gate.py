from __future__ import annotations

import pytest

from mcp_memory.core.curator_rollout_gate import (
    CuratorRolloutEvidence,
    CuratorRolloutPolicy,
    assess_curator_rollout,
)

pytestmark = pytest.mark.small


def test_passing_canary_is_allowed_with_verified_quality_evidence() -> None:
    """Allow a canary whose evaluated quality meets every default threshold."""
    decision = assess_curator_rollout(
        CuratorRolloutEvidence(
            verified_mutations=10,
            productive_outcomes=8,
            evaluated_outcomes=10,
        )
    )

    assert decision.allowed
    assert decision.stop_reasons == ()
    assert decision.productive_rate == 0.8


def test_missing_quality_evidence_fails_closed_despite_verified_mutations() -> None:
    """Treat receipts without evaluated outcomes as insufficient evidence."""
    decision = assess_curator_rollout(CuratorRolloutEvidence(verified_mutations=4))

    assert not decision.allowed
    assert decision.stop_reasons == ("no_evidence",)


def test_integrity_and_attribution_failures_stop_rollout() -> None:
    """Expose reconciliation, destructive, provider, and ledger failures."""
    decision = assess_curator_rollout(
        CuratorRolloutEvidence(
            evaluated_outcomes=10,
            productive_outcomes=10,
            reconciliation_mismatches=1,
            destructive_false_positives=1,
            provider_attribution_gaps=1,
            ledger_invalid_runs=1,
        )
    )

    assert decision.stop_reasons == (
        "ledger_invalid",
        "reconciliation_mismatch",
        "destructive_false_positive",
        "provider_attribution_gap",
    )


def test_regression_threshold_is_inclusive_but_exceeding_it_stops() -> None:
    """Permit exactly five percent regressions and reject anything higher."""
    at_boundary = assess_curator_rollout(
        CuratorRolloutEvidence(productive_outcomes=19, evaluated_outcomes=20, regressions=1)
    )
    over_boundary = assess_curator_rollout(
        CuratorRolloutEvidence(productive_outcomes=17, evaluated_outcomes=20, regressions=2)
    )

    assert at_boundary.allowed
    assert over_boundary.stop_reasons == ("regression_rate_exceeded",)


def test_unobserved_threshold_is_strictly_below_ten_percent() -> None:
    """Reject ten percent unobserved outcomes because the runbook is strict."""
    decision = assess_curator_rollout(
        CuratorRolloutEvidence(evaluated_outcomes=9, productive_outcomes=9, unobserved_outcomes=1)
    )

    assert decision.stop_reasons == ("unobserved_rate_exceeded",)


def test_structural_only_counts_only_when_policy_allows_it() -> None:
    """Require explicit policy approval before structural work earns yield."""
    evidence = CuratorRolloutEvidence(structural_only_outcomes=8, evaluated_outcomes=10)

    denied = assess_curator_rollout(evidence)
    allowed = assess_curator_rollout(evidence, CuratorRolloutPolicy(allow_structural_only=True))

    assert denied.stop_reasons == ("productive_rate_below_threshold",)
    assert allowed.allowed


def test_invalid_counts_and_immutable_decision_are_explicit() -> None:
    """Reject malformed counts and keep the assessment result immutable."""
    with pytest.raises(ValueError, match="must be non-negative"):
        CuratorRolloutEvidence(regressions=-1)

    decision = assess_curator_rollout(CuratorRolloutEvidence(evaluated_outcomes=10, productive_outcomes=10))
    with pytest.raises(AttributeError):
        setattr(decision, "allowed", False)

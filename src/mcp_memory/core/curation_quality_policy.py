"""Outcome classification and escalation policy for quality evidence."""

from __future__ import annotations

from enum import StrEnum


class QualityOutcome(StrEnum):
    PRODUCTIVE = "productive"
    VERIFIED_ONLY = "verified_only"
    NEUTRAL = "neutral"
    REGRESSED = "regressed"
    STRUCTURAL_ONLY = "structural_only"
    UNVERIFIED = "unverified"


def classify_quality_outcome(
    *,
    mutation_verified: bool,
    query_trusted: bool,
    consistency_verified: bool,
    structural_only: bool,
    utility_delta: float | None,
    quality_observed: bool,
) -> QualityOutcome:
    if not mutation_verified or not consistency_verified:
        return QualityOutcome.UNVERIFIED
    if structural_only:
        return QualityOutcome.STRUCTURAL_ONLY
    if not query_trusted or not quality_observed:
        return QualityOutcome.NEUTRAL
    if utility_delta is None:
        return QualityOutcome.VERIFIED_ONLY
    if utility_delta > 0.0:
        return QualityOutcome.PRODUCTIVE
    if utility_delta < 0.0:
        return QualityOutcome.REGRESSED
    return QualityOutcome.VERIFIED_ONLY


def should_escalate_quality(outcome: QualityOutcome, *, mutation_verified: bool) -> bool:
    return outcome is QualityOutcome.REGRESSED and mutation_verified

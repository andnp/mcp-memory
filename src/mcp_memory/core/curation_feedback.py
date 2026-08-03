"""Pure feedback classification for iterative curator campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _quality_feedback_payload(result: Any) -> dict[str, object]:
    evidence = _quality_evidence_payload(result)
    unsafe_rejections = {
        "verification_failed",
        "protected_target",
        "policy_denied",
        "disclosure_denied",
        "budget_exhausted",
    }
    rejection_codes = {_enum_value(code) for code in getattr(result.result, "rejection_codes", ())}
    retryable = bool(evidence) and any(
        item.get("neutral_reason") is None for item in evidence
    ) and not rejection_codes.intersection(unsafe_rejections) and not any(
        item.get("wave_status") == "conflict" for item in evidence
    )
    return {
        "retryable": retryable,
        "latest_failed_live_run": {"evidence": evidence},
        "required_response": [
            "Challenge weak split evidence.",
            "Preserve exact search anchors and concrete entities.",
            "Avoid speculative multi-action waves.",
            "Use measured feedback to change strategy rather than repeat.",
        ],
    }


def _feedback_termination_reason(result: Any) -> str | None:
    outcome = _enum_value(getattr(result.result, "outcome", ""))
    if outcome in {"provider_failed", "budget_exhausted"}:
        return outcome
    rejection_codes = {_enum_value(code) for code in getattr(result.result, "rejection_codes", ())}
    if rejection_codes.intersection(
        {"verification_failed", "protected_target", "policy_denied", "disclosure_denied"}
    ):
        return "safety_termination"
    evidence = _quality_evidence_payload(result)
    if any(item.get("wave_status") == "conflict" for item in evidence):
        return "safety_termination"
    if outcome in {"no_op", "quality_override"}:
        return "no_useful_work" if outcome == "no_op" else "converged"
    actionable = [item for item in evidence if item.get("neutral_reason") is None]
    if not actionable:
        return "neutral_evidence" if evidence else "no_quality_evidence"
    if any(_has_measured_negative(item) for item in actionable):
        return None
    if all(item.get("wave_status") == "accepted" for item in evidence):
        return "converged"
    return None


def _quality_evidence_payload(result: Any) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for item in getattr(result.result, "quality_evidence", ()):
        if hasattr(item, "model_dump"):
            evidence.append(item.model_dump(mode="json"))
        elif isinstance(item, Mapping):
            evidence.append(dict(item))
    return evidence


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _has_measured_negative(item: Mapping[str, Any]) -> bool:
    return (
        int(item.get("retrieval_regression_count") or 0) > 0
        or int(item.get("collateral_regression_count") or 0) > 0
        or item.get("content_quality_improved") is False
        or float(item.get("engagement_utility_delta") or 0.0) < 0
    )

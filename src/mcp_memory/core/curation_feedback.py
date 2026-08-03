"""Pure feedback classification for iterative curator campaigns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mcp_memory.core.curation_policy import RejectionCode


_REPAIRABLE_ACTION_ERROR_CODES = frozenset(
    {
        "missing_record_token",
        "invalid_record_token",
        "invalid_merge_targets",
        "invalid_split_child",
        "missing_link_evidence",
        "invalid_absent_link_precondition",
        "missing_absent_link_precondition",
        "stale_precondition",
        "action_transient",
    }
)
_TERMINAL_REJECTION_CODES = frozenset(code.value for code in RejectionCode) | {
    "verification_failed",
    "protected_target",
    "policy_denied",
    "disclosure_denied",
    "budget_exhausted",
    "action_fatal",
    "policy_rejected",
}


def _quality_feedback_payload(result: Any) -> dict[str, object]:
    evidence = _quality_evidence_payload(result)
    action_failures = _action_failure_payload(result)
    repairable_action_failures = [
        item
        for item in action_failures
        if item["error_code"] in _REPAIRABLE_ACTION_ERROR_CODES
    ]
    terminal_action_failures = [
        item
        for item in action_failures
        if item["error_code"] not in _REPAIRABLE_ACTION_ERROR_CODES
    ]
    rejection_codes = {_enum_value(code) for code in getattr(result.result, "rejection_codes", ())}
    retryable = bool(evidence) and any(
        item.get("neutral_reason") is None for item in evidence
    )
    retryable = (
        (retryable or bool(repairable_action_failures))
        and not rejection_codes.intersection(_TERMINAL_REJECTION_CODES)
        and not terminal_action_failures
        and not any(item.get("wave_status") == "conflict" for item in evidence)
    )
    return {
        "retryable": retryable,
        "latest_failed_live_run": {
            "evidence": evidence,
            "rejected_actions": action_failures,
        },
        "required_response": [
            "Challenge weak split evidence.",
            "Preserve exact search anchors and concrete entities.",
            "Avoid speculative multi-action waves.",
            "Use measured feedback to change strategy rather than repeat.",
            "Repair listed contract, stale, or transient action failures instead of repeating the same invalid action shape.",
        ],
    }


def _feedback_termination_reason(result: Any) -> str | None:
    outcome = _enum_value(getattr(result.result, "outcome", ""))
    if outcome in {
        "provider_failed",
        "budget_exhausted",
        "verification_failed",
        "quality_rejected",
    }:
        return outcome
    rejection_codes = {_enum_value(code) for code in getattr(result.result, "rejection_codes", ())}
    if rejection_codes.intersection(_TERMINAL_REJECTION_CODES):
        return "safety_termination"
    action_failures = _action_failure_payload(result)
    if any(
        item["error_code"] not in _REPAIRABLE_ACTION_ERROR_CODES
        for item in action_failures
    ):
        return "safety_termination"
    evidence = _quality_evidence_payload(result)
    if any(
        item["error_code"] in _REPAIRABLE_ACTION_ERROR_CODES
        for item in action_failures
    ):
        return None
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


def _action_failure_payload(result: Any) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for receipt in getattr(result.result, "receipts", ()):
        if _enum_value(getattr(receipt, "status", "")) != "rejected":
            continue
        error_code = getattr(receipt, "error_code", None)
        if not isinstance(error_code, str) or not error_code:
            continue
        affected_ids = [str(value) for value in getattr(receipt, "affected_ids", ())]
        failures.append(
            {
                "action_id": str(getattr(receipt, "action_id", "")),
                "operation": str(getattr(receipt, "operation", "")),
                "affected_ids": affected_ids[:8],
                "affected_id_count": len(affected_ids),
                "error_code": error_code,
            }
        )
    return failures[:12]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _has_measured_negative(item: Mapping[str, Any]) -> bool:
    return (
        int(item.get("retrieval_regression_count") or 0) > 0
        or int(item.get("collateral_regression_count") or 0) > 0
        or item.get("content_quality_improved") is False
        or float(item.get("engagement_utility_delta") or 0.0) < 0
    )

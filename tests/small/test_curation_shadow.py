from types import SimpleNamespace

from mcp_memory.core.curation_feedback import (
    _feedback_termination_reason,
    _quality_feedback_payload,
)


def _result(*evidence: dict[str, object], outcome: str = "applied", rejection_codes=()):
    return SimpleNamespace(
        result=SimpleNamespace(
            outcome=outcome,
            rejection_codes=rejection_codes,
            quality_evidence=list(evidence),
        )
    )


def test_mixed_neutral_feedback_continues_after_measured_regression() -> None:
    result = _result(
        {
            "neutral_reason": None,
            "retrieval_regression_count": 1,
            "wave_status": "accepted",
        },
        {
            "neutral_reason": "no_trusted_query",
            "wave_status": "accepted",
        },
    )

    assert _quality_feedback_payload(result)["retryable"] is True
    assert _feedback_termination_reason(result) is None


def test_neutral_only_feedback_does_not_look_like_safety_failure() -> None:
    result = _result(
        {"neutral_reason": "no_trusted_query", "wave_status": "accepted"},
    )

    assert _quality_feedback_payload(result)["retryable"] is False
    assert _feedback_termination_reason(result) == "neutral_evidence"


def test_unsafe_rejection_still_stops_feedback() -> None:
    result = _result(
        {"neutral_reason": None, "wave_status": "accepted"},
        rejection_codes=("protected_target",),
    )

    assert _quality_feedback_payload(result)["retryable"] is False
    assert _feedback_termination_reason(result) == "safety_termination"


def test_all_accepted_positive_evidence_converges() -> None:
    result = _result(
        {"neutral_reason": None, "wave_status": "accepted"},
    )

    assert _feedback_termination_reason(result) == "converged"

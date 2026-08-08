from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_quality import (
    CurationQualityEvidence,
    quality_productive_mutation_count,
)
from mcp_memory.core.curation_feedback import (
    _feedback_termination_reason,
    _quality_feedback_payload,
)


def _result(
    *evidence: dict[str, object],
    outcome: str = "applied",
    rejection_codes=(),
    receipts=(),
):
    return SimpleNamespace(
        result=SimpleNamespace(
            outcome=outcome,
            rejection_codes=rejection_codes,
            quality_evidence=list(evidence),
            receipts=list(receipts),
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

    assert _quality_feedback_payload(result)["retryable"] is True
    assert _feedback_termination_reason(result) is None


def test_unsafe_rejection_does_not_stop_feedback() -> None:
    result = _result(
        {"neutral_reason": None, "wave_status": "accepted"},
        rejection_codes=("protected_target",),
    )

    assert _quality_feedback_payload(result)["retryable"] is True
    assert _feedback_termination_reason(result) is None


def test_policy_rejection_does_not_stop_feedback_without_quality_evidence() -> None:
    result = _result(
        outcome="deferred",
        rejection_codes=("contradictory_facts",),
    )

    assert _quality_feedback_payload(result)["retryable"] is True
    assert _feedback_termination_reason(result) is None


def test_all_accepted_positive_evidence_still_allows_another_wave() -> None:
    result = _result(
        {"neutral_reason": None, "wave_status": "accepted"},
    )

    assert _feedback_termination_reason(result) is None


def test_repairable_action_failure_retries_without_quality_evidence() -> None:
    action_id = uuid4()
    memory_id = uuid4()
    result = _result(
        outcome="deferred",
        receipts=(
            SimpleNamespace(
                action_id=action_id,
                operation="create_link",
                affected_ids=[memory_id],
                status="rejected",
                error_code="invalid_absent_link_precondition",
            ),
        ),
    )

    feedback = cast(dict[str, Any], _quality_feedback_payload(result))

    assert feedback["retryable"] is True
    assert _feedback_termination_reason(result) is None
    assert feedback["latest_failed_live_run"]["rejected_actions"] == [
        {
            "action_id": str(action_id),
            "operation": "create_link",
            "affected_ids": [str(memory_id)],
            "affected_id_count": 1,
            "error_code": "invalid_absent_link_precondition",
        }
    ]


def test_terminal_action_failure_stops_feedback() -> None:
    result = _result(
        outcome="deferred",
        receipts=(
            SimpleNamespace(
                action_id=uuid4(),
                operation="rewrite_memory",
                affected_ids=[uuid4()],
                status="rejected",
                error_code="action_fatal",
            ),
        ),
    )

    assert _quality_feedback_payload(result)["retryable"] is False
    assert _feedback_termination_reason(result) == "safety_termination"


def test_action_failure_feedback_is_bounded() -> None:
    receipts = tuple(
        SimpleNamespace(
            action_id=uuid4(),
            operation="merge_memories",
            affected_ids=[uuid4() for _ in range(10)],
            status="rejected",
            error_code="action_transient",
        )
        for _ in range(20)
    )

    feedback = cast(
        dict[str, Any],
        _quality_feedback_payload(_result(outcome="deferred", receipts=receipts)),
    )
    failures = cast(
        list[dict[str, Any]],
        feedback["latest_failed_live_run"]["rejected_actions"],
    )

    assert len(failures) == 12
    assert all(len(item["affected_ids"]) == 8 for item in failures)
    assert all(item["affected_id_count"] == 10 for item in failures)


def test_campaign_productive_count_requires_measured_quality_outcome() -> None:
    now = datetime.now(UTC)
    evidence = [
        CurationQualityEvidence(
            run_id=uuid4(),
            action_id=uuid4(),
            operation="rewrite_memory",
            policy_version="1",
            status=status,
            created_at=now,
        )
        for status in ("productive", "structural_only", "neutral", "unverified", "regressed")
    ]

    assert quality_productive_mutation_count(evidence) == 2

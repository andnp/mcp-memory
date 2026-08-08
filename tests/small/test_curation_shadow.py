from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_quality import (
    CurationQualityEvidence,
    quality_productive_mutation_count,
)
from mcp_memory.core.curation_reconciliation import (
    CurationProviderAttribution,
    ProviderAttemptIdentity,
    reconcile_provider_attempts,
)
from mcp_memory.curation_store import CurationRun
from mcp_memory.core.curation_feedback import (
    _feedback_termination_reason,
    _quality_feedback_payload,
)
from mcp_memory.core.curation_shadow import _direct_curator_prompt
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.ports.tasks import TaskRecord


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


def test_direct_curator_prompt_contains_durability_retention_guardrails() -> None:
    prompt = _direct_curator_prompt(
        task=cast(TaskRecord, SimpleNamespace(id="curator-task")),
        seed_records=[],
        campaign_hypothesis=None,
        mutation_budget=CurationMutationBudget(max_accepted_mutations=2),
    )

    assert "durable content" in prompt
    assert "transient content" in prompt
    assert "mixed content" in prompt
    assert "deadline, release, incident, historical, or decision meaning" in prompt
    assert "preserve meaningful dates and qualifiers" in prompt
    assert "work logs, status updates, task-complete summaries, and execution residue" in prompt
    assert "advisory cleanup candidates" in prompt
    assert "Never archive or delete solely due to age, date, or access" in prompt
    assert "preserve every durable claim and its meaningful qualifiers" in prompt


def test_direct_curator_prompt_does_not_direct_blanket_date_deletion() -> None:
    prompt = _direct_curator_prompt(
        task=cast(TaskRecord, SimpleNamespace(id="curator-task")),
        seed_records=[],
        campaign_hypothesis=None,
        mutation_budget=CurationMutationBudget(),
    ).lower()

    assert "delete all dated memories" not in prompt
    assert "delete every dated memory" not in prompt
    assert "archive all dated memories" not in prompt


def test_provider_attempt_reconciliation_requires_exact_run_identity() -> None:
    """Accept provider attempts only when task and execution epoch agree."""
    task_id = uuid4()
    run = CurationRun(
        run_id=uuid4(),
        task_id=task_id,
        execution_epoch=4,
        frontier_key="direct",
        context_fingerprint="context",
    )

    result = reconcile_provider_attempts(
        run,
        [ProviderAttemptIdentity(str(task_id), 4, "task:4:req:1")],
    )

    assert result.disposition is CurationProviderAttribution.EXACT
    assert result.attempt_identities == ("task:4:req:1",)


def test_provider_attempt_reconciliation_fails_closed_for_missing_and_mismatched_rows() -> None:
    """Keep missing and conflicting provider attribution visible at run level."""
    task_id = uuid4()
    run = CurationRun(
        run_id=uuid4(),
        task_id=task_id,
        execution_epoch=4,
        frontier_key="direct",
        context_fingerprint="context",
    )

    missing = reconcile_provider_attempts(run, [SimpleNamespace(task_id=str(task_id), execution_epoch=4)])
    mismatched = reconcile_provider_attempts(
        run,
        [ProviderAttemptIdentity(str(uuid4()), 4, "other:4:req:1")],
    )

    assert missing.disposition is CurationProviderAttribution.MISSING
    assert mismatched.disposition is CurationProviderAttribution.MISMATCHED

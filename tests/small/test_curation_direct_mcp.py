import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_direct_mcp import (
    _direct_curator_prompt,
    _direct_quality_run,
    run_curator_direct_mcp,
)
from mcp_memory.core.curation_feedback import (
    _feedback_termination_reason,
    _quality_feedback_payload,
)
from mcp_memory.core.curation_quality import (
    CurationQualityEvidence,
    quality_productive_mutation_count,
)
from mcp_memory.core.curation_reconciliation import (
    CurationProviderAttribution,
    ProviderAttemptIdentity,
    reconcile_provider_attempts,
)
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.sampling import SamplingBatch
from mcp_memory.core.task_handlers.curator_support import build_curator_context_packet
from mcp_memory.curation_store import CurationRun


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


def _packet_record(memory_id: str, content: str = "Durable full content") -> SimpleNamespace:
    return SimpleNamespace(
        id=memory_id,
        title=f"Title {memory_id}",
        summary=f"Summary {memory_id}",
        content=content,
        type="fact",
        status="active",
        tags=[],
        workspace_ids=["workspace"],
        metadata={},
    )


def _packet_task(data: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(id="packet-task", data=data or {}, execution_epoch=0)


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


def test_curator_packet_identity_is_deterministic() -> None:
    """Give identical ordered candidate context the same stable packet identity."""
    records = [_packet_record("seed"), _packet_record("support")]
    batch = SamplingBatch(
        requested_strategy=None,
        strategy_used="semantic",
        strategy_fallback_reason=None,
        candidate_count=2,
        records=cast(Any, records),
        strategy_selection_reason="selected=semantic",
        strategy_selection_scores={"semantic": 0.8},
        selector_feature_snapshot={"strategy_signals": {"recency": 0.2}},
    )
    first = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)), cast(Any, _packet_task()), batch, records[:1], records
    )
    second = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)), cast(Any, _packet_task()), batch, records[:1], records
    )

    assert first.packet_id == second.packet_id
    assert first.to_mapping()["seeds"] == ["seed"]
    assert first.to_mapping()["support"] == ["support"]
    assert first.to_mapping()["record_tokens"]["seed"].startswith("v1:")


def test_curator_packet_records_omit_overflow_with_bounds_metadata() -> None:
    """Enforce configured record and character bounds while explaining omissions."""
    records = [_packet_record("one"), _packet_record("two"), _packet_record("three")]
    batch = SamplingBatch(None, "semantic", None, 3, cast(Any, records))
    packet = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)),
        cast(Any, _packet_task({"packet_max_records": 2, "packet_max_chars": 30})),
        batch,
        records[:1],
        records,
    ).to_mapping()

    assert len(packet["records"]) == 1
    assert packet["limits"]["records"] == 2
    assert packet["limits"]["characters"] == 30
    assert packet["disclosure"]["content"] == "omitted"
    assert "character_limit" in {item["reason"] for item in packet["omissions"]}


def test_direct_curator_prompt_includes_packet_identity_and_summary_only_context() -> None:
    """Send the canonical packet to the provider without exposing full record content."""
    record = _packet_record("seed", content="secret full record body")
    batch = SamplingBatch(None, "semantic", None, 1, cast(Any, [record]))
    packet = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)), cast(Any, _packet_task()), batch, [record], [record]
    )
    prompt = _direct_curator_prompt(
        task=cast(TaskRecord, SimpleNamespace(id="curator-task")),
        seed_records=[record],
        campaign_hypothesis=None,
        mutation_budget=CurationMutationBudget(),
        context_packet=packet,
    )

    assert packet.packet_id in prompt
    assert '"context_packet"' in prompt
    assert '"content": "omitted"' in prompt
    assert "secret full record body" not in prompt
    assert '"seed_records"' not in prompt


def test_direct_run_persists_packet_identity_and_disclosure_audit() -> None:
    """Use the exact packet identity when constructing the durable curator run."""
    record = _packet_record("seed")
    batch = SamplingBatch(None, "semantic", None, 1, cast(Any, [record]))
    packet = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)),
        cast(Any, _packet_task({"policy_version": "policy-v2"})),
        batch,
        [record],
        [record],
    )
    task = cast(TaskRecord, _packet_task({"policy_version": "policy-v2"}))

    run = _direct_quality_run(task, packet)

    assert run.context_fingerprint == packet.packet_id
    assert run.policy_version == "policy-v2"
    assert run.disclosure_audit["packet_id"] == packet.packet_id
    assert run.disclosure_audit["disclosure"] == packet.to_mapping()["disclosure"]
    assert run.disclosure_audit["limits"] == packet.to_mapping()["limits"]


def test_direct_result_exposes_packet_identity_when_provider_route_is_unavailable() -> None:
    """Expose packet identity in curator results even when no provider session opens."""
    record = _packet_record("seed")
    batch = SamplingBatch(None, "semantic", None, 1, cast(Any, [record]))
    packet = build_curator_context_packet(
        cast(Any, SimpleNamespace(repository=None)), cast(Any, _packet_task()), batch, [record], [record]
    )

    result = asyncio.run(
        run_curator_direct_mcp(
            cast(Any, SimpleNamespace(ai_agent_provider=None)),
            cast(TaskRecord, _packet_task()),
            provider=object(),
            seed_batch=batch,
            sampled_records=[record],
            seed_records=[record],
            claimed_work_item=None,
            work_item_metadata={},
            context_packet=packet,
        )
    )

    assert result["packet_id"] == packet.packet_id


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

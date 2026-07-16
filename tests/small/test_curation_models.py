from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    CurationAction,
    CurationBudgetUsage,
    CurationPlan,
    CurationRunOutcome,
    CurationRunResult,
    ClaimManifest,
    MutationReceipt,
    NormalizeMemoryAction,
    ReceiptStatus,
    RetentionDecision,
    RetentionReason,
)


def test_action_union_rejects_unknown_and_delete_operations() -> None:
    adapter = TypeAdapter(CurationAction)
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "delete_memory", "target_id": str(uuid4())})
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "unknown", "target_id": str(uuid4())})


def test_plan_requires_complete_seed_dispositions() -> None:
    seed = uuid4()
    with pytest.raises(ValidationError, match="missing seed dispositions"):
        CurationPlan(
            plan_id=uuid4(), run_id=uuid4(), frontier_key="frontier", context_fingerprint="context",
            rationale="no safe action", seed_memory_ids=[seed],
        )

    decision = RetentionDecision(memory_id=seed, reason=RetentionReason.ALREADY_FOCUSED, rationale="focused")
    plan = CurationPlan(
        plan_id=uuid4(), run_id=uuid4(), frontier_key="frontier", context_fingerprint="context",
        rationale="no safe action", seed_memory_ids=[seed], retained=[decision],
    )
    assert plan.retained == [decision]


def test_defaults_are_isolated_and_contracts_are_pure() -> None:
    first = CurationBudgetUsage()
    second = CurationBudgetUsage()
    first.seed_records = 1
    assert second.seed_records == 0

    receipt = MutationReceipt(
        run_id=uuid4(), action_id=uuid4(), operation="archive_memory", status=ReceiptStatus.VERIFIED,
    )
    result = CurationRunResult(run_id=uuid4(), outcome=CurationRunOutcome.APPLIED, receipts=[receipt])
    assert result.budget_usage.read_tool_calls == 0


def test_archive_action_is_typed_and_has_no_delete_sibling() -> None:
    action = ArchiveMemoryAction(
        action_id=uuid4(), target_id=uuid4(), confidence=0.9, rationale="preserve lineage",
        claim_manifest=ClaimManifest(preserved_claims=["claim"]),
    )
    assert action.operation == "archive_memory"


def test_normalize_requires_at_least_one_metadata_field() -> None:
    with pytest.raises(ValidationError, match="at least one metadata field"):
        NormalizeMemoryAction(action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="empty")

    action = NormalizeMemoryAction(
        action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="clear tags", tags=[]
    )
    assert action.tags == []

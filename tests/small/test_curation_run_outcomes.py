from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.core.curation_planner import (
    CurationPlannerProviderError,
    PlannerExecutionEnvelope,
    PlannerExecutionStatus,
)
from mcp_memory.core.curation_run_outcomes import (
    budget_usage,
    classify_outcome,
    count_affected_memory_ids,
    count_verified_receipts,
    failure_details,
    project_receipts,
    rejection_codes,
)
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState


def test_classify_outcome_and_rejection_codes() -> None:
    validation = SimpleNamespace(
        valid=True,
        plan=SimpleNamespace(actions=[]),
        issues=[SimpleNamespace(code="duplicate")],
        rejected_actions=[SimpleNamespace(reason_codes=["duplicate", "policy"])],
        accepted_actions=[],
        specialist_routes=[SimpleNamespace(reason_code="policy")],
    )
    assert classify_outcome(validation.plan, cast(Any, validation), None) == (CurationRunOutcome.NO_OP, "valid_no_op")
    assert rejection_codes(cast(Any, validation)) == ["duplicate", "policy"]


def test_budget_and_receipt_projection_counts() -> None:
    run_id, action_id, memory_id = uuid4(), uuid4(), uuid4()
    context = SimpleNamespace(
        usage=SimpleNamespace(
            seed_records=1, support_records=2, context_characters=3,
            read_tool_calls=4, records_returned=5,
        )
    )
    envelopes = [
        SimpleNamespace(token_usage=7, token_usage_source="provider", premium_request=True),
        SimpleNamespace(token_usage=8, token_usage_source=None, premium_request=False),
    ]
    usage = budget_usage(
        cast(Any, context),
        cast(Any, envelopes),
        cast(Any, SimpleNamespace(plan=SimpleNamespace(actions=[1, 2]), accepted_actions=[1])),
    )
    assert usage.token_usage == 15
    assert usage.premium_requests == 1

    receipts = (
        CurationActionReceipt(
            run_id=run_id, action_id=action_id, operation="rewrite_memory",
            affected_ids=[memory_id], status=CurationReceiptState.VERIFIED,
        ),
        CurationActionReceipt(
            run_id=run_id, action_id=uuid4(), operation="archive_memory",
            affected_ids=[memory_id], status=CurationReceiptState.REJECTED,
        ),
    )
    projected = project_receipts(receipts)
    assert projected[0].run_id == run_id
    assert count_verified_receipts(receipts) == 1
    assert count_affected_memory_ids(receipts) == 1


def test_receipt_projection_accepts_lightweight_receipt_double() -> None:
    memory_id = uuid4()

    class ReceiptDouble:
        status = "verified"
        affected_ids = [memory_id]

        def model_dump(self, *, mode: str, exclude: set[str]) -> dict[str, Any]:
            return {
                "run_id": str(uuid4()),
                "action_id": str(uuid4()),
                "operation": "rewrite_memory",
                "affected_ids": [str(memory_id)],
                "status": self.status,
            }

    receipts = (ReceiptDouble(),)
    assert count_verified_receipts(receipts) == 1
    assert count_affected_memory_ids(receipts) == 1
    assert project_receipts(receipts)[0].affected_ids == [memory_id]


def test_failure_details_project_provider_route_diagnostics() -> None:
    failure = CurationPlannerProviderError(
        "provider unavailable",
        reason_code="model_burst_limit_exceeded",
        reason_category="admission",
        retry_delay_seconds=300.0,
        route_available=True,
    )
    envelope = PlannerExecutionEnvelope(
        plan=None,
        provider_key="profile-a",
        model_name="model-a",
        request_id="request-1",
        attempt=2,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        status=PlannerExecutionStatus.PROVIDER_FAILED,
        reason_code="model_burst_limit_exceeded",
        metadata={
            "provider_profile": "profile-a",
            "reason_category": "admission",
            "route_available": True,
            "retry_delay_seconds": 300.0,
        },
    )

    details = failure_details(failure, [envelope])

    assert details["reason_code"] == "model_burst_limit_exceeded"
    assert details["provider_profile"] == "profile-a"
    assert details["model"] == "model-a"
    assert details["retry_count"] == 1
    assert details["another_route_available"] is True

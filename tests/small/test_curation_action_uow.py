from __future__ import annotations

from uuid import uuid4

import pytest

from mcp_memory.curation_action_uow import (
    CurationActionBatchResult,
    CurationActionError,
    CurationActionOutcome,
    CurationActionRequest,
    MutationResult,
)
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState

pytestmark = pytest.mark.small


def test_action_request_freezes_ordered_inputs() -> None:
    """An action request keeps stable target order and token inputs."""
    tokens = {"memory": "before"}
    request = CurationActionRequest(
        run_id=uuid4(),
        action_id=uuid4(),
        target_ids=("memory", "external"),
        expected_tokens=tokens,
        apply=lambda _transaction: MutationResult("normalize_memory"),
    )
    tokens["other"] = "ignored"

    assert request.target_ids == ("memory", "external")
    assert dict(request.expected_tokens) == {"memory": "before"}


def test_batch_result_preserves_outcome_order() -> None:
    """Batch results retain the submitted action order across outcomes."""
    receipt = CurationActionReceipt(
        run_id=uuid4(),
        action_id=uuid4(),
        operation="normalize_memory",
        status=CurationReceiptState.APPLIED_UNVERIFIED,
    )
    error = CurationActionError("temporary failure")
    result = CurationActionBatchResult(
        outcomes=(CurationActionOutcome(receipt=receipt), CurationActionOutcome(error=error))
    )

    assert result.outcomes[0].receipt is receipt
    assert result.outcomes[1].error is error


def test_action_outcome_requires_one_result() -> None:
    """An action outcome cannot represent both success and failure."""
    with pytest.raises(ValueError, match="exactly one receipt or error"):
        CurationActionOutcome()

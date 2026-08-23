from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.curation_action_store import CurationActionStore
from mcp_memory.curation_action_uow import (
    CurationActionError,
    CurationActionRequest,
    CurationTransaction,
    MutationResult,
)
from mcp_memory.curation_batch_executor import SequentialCurationActionExecutor
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState

pytestmark = pytest.mark.small


class _FakeActionStore:
    def __init__(self, results: Sequence[CurationActionReceipt | CurationActionError]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, object]] = []

    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
        preconditions: Any | None = None,
        operation: str | None = None,
        payload: Any | None = None,
    ) -> CurationActionReceipt:
        self.calls.append(
            {
                "run_id": run_id,
                "action_id": action_id,
                "target_ids": tuple(target_ids),
                "expected_tokens": dict(expected_tokens),
                "apply": apply,
                "preconditions": preconditions,
                "operation": operation,
                "payload": payload,
            }
        )
        result = self._results.pop(0)
        if isinstance(result, CurationActionError):
            raise result
        return result


def _request(action_id: UUID) -> CurationActionRequest:
    return CurationActionRequest(
        run_id=uuid4(),
        action_id=action_id,
        target_ids=("memory",),
        expected_tokens={"memory": "before"},
        apply=lambda _transaction: MutationResult("normalize_memory"),
        preconditions={"reason": "test"},
        operation="normalize_memory",
        payload={"title": "Specific"},
    )


def _receipt(request: CurationActionRequest) -> CurationActionReceipt:
    return CurationActionReceipt(
        run_id=request.run_id,
        action_id=request.action_id,
        operation="normalize_memory",
        status=CurationReceiptState.APPLIED_UNVERIFIED,
    )


def test_empty_batch_has_no_store_side_effects() -> None:
    """An empty request sequence returns an empty ordered result."""
    store = _FakeActionStore(())

    result = SequentialCurationActionExecutor(cast(CurationActionStore, store)).execute_actions(())

    assert result.outcomes == ()
    assert store.calls == []


def test_executor_preserves_order_and_continues_after_action_error() -> None:
    """Each action keeps its place even when a prior action reports an error."""
    first = _request(uuid4())
    second = _request(uuid4())
    failure = CurationActionError("temporary failure")
    store = _FakeActionStore((failure, _receipt(second)))

    result = SequentialCurationActionExecutor(cast(CurationActionStore, store)).execute_actions((first, second))

    assert [call["action_id"] for call in store.calls] == [first.action_id, second.action_id]
    assert result.outcomes[0].error is failure
    assert result.outcomes[1].receipt == _receipt(second)


def test_executor_forwards_request_contract_without_reordering() -> None:
    """The coordinator passes request identity and payload fields unchanged."""
    request = _request(uuid4())
    store = _FakeActionStore((_receipt(request),))

    SequentialCurationActionExecutor(cast(CurationActionStore, store)).execute_actions((request,))

    assert store.calls[0]["run_id"] == request.run_id
    assert store.calls[0]["action_id"] == request.action_id
    assert store.calls[0]["target_ids"] == request.target_ids
    assert store.calls[0]["expected_tokens"] == dict(request.expected_tokens)
    assert store.calls[0]["preconditions"] == request.preconditions
    assert store.calls[0]["operation"] == request.operation
    assert store.calls[0]["payload"] == request.payload

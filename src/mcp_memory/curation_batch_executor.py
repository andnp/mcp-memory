"""Sequential coordination for already-decided curation actions."""

from __future__ import annotations

from collections.abc import Sequence

from mcp_memory.curation_action_uow import (
    CurationActionBatchResult,
    CurationActionError,
    CurationActionOutcome,
    CurationActionRequest,
    CurationActionStore,
)


class SequentialCurationActionExecutor:
    """Execute ordered requests through the existing per-action store."""

    def __init__(self, action_store: CurationActionStore) -> None:
        self._action_store = action_store

    def execute_actions(
        self,
        requests: Sequence[CurationActionRequest],
    ) -> CurationActionBatchResult:
        outcomes: list[CurationActionOutcome] = []
        for request in requests:
            try:
                receipt = self._action_store.execute_action(
                    run_id=request.run_id,
                    action_id=request.action_id,
                    target_ids=request.target_ids,
                    expected_tokens=request.expected_tokens,
                    apply=request.apply,
                    preconditions=request.preconditions,
                    operation=request.operation,
                    payload=request.payload,
                )
            except CurationActionError as error:
                outcomes.append(CurationActionOutcome(error=error))
            else:
                outcomes.append(CurationActionOutcome(receipt=receipt))
        return CurationActionBatchResult(tuple(outcomes))


__all__ = ["SequentialCurationActionExecutor"]

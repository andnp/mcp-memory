"""Work-item lifecycle decisions for curation runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from mcp_memory.core.curation_models import CurationRunOutcome
from mcp_memory.core.curation_planner import PlannerExecutionEnvelope
from mcp_memory.work_item_store import WorkItemRecord


class WorkItemAction(StrEnum):
    NONE = "none"
    COMPLETE = "complete"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class CurationWorkItemDecision:
    """The lifecycle decision made from persisted harness evidence."""

    action: WorkItemAction
    reason_code: str
    retry_delay_seconds: float | None = None


class WorkItemRepository(Protocol):
    def enqueue_unique(
        self,
        *,
        family_key: str,
        execution_lane: str,
        payload: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        available_at: float | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[WorkItemRecord, bool]: ...

    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> WorkItemRecord: ...

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecord: ...


class CurationWorkItemService:
    def __init__(self, repository: WorkItemRepository | None = None, *, no_op_cooldown_seconds: float = 3600.0) -> None:
        self._repository = repository
        self._no_op_cooldown_seconds = no_op_cooldown_seconds

    def decide(
        self,
        outcome: CurationRunOutcome,
        reason_code: str,
        envelope: PlannerExecutionEnvelope[Any] | None,
    ) -> CurationWorkItemDecision:
        if outcome is CurationRunOutcome.BUDGET_EXHAUSTED:
            return CurationWorkItemDecision(WorkItemAction.DEFER, reason_code, self._no_op_cooldown_seconds)
        if outcome in (CurationRunOutcome.NO_OP, CurationRunOutcome.APPLIED):
            return CurationWorkItemDecision(WorkItemAction.COMPLETE, reason_code)
        delay = None if envelope is None else envelope.retry_delay_seconds
        return CurationWorkItemDecision(WorkItemAction.DEFER, reason_code, delay)

    def apply(self, work_item_id: str | None, decision: CurationWorkItemDecision) -> None:
        if work_item_id is None or self._repository is None:
            return
        if decision.action is WorkItemAction.COMPLETE:
            self._repository.complete_item(work_item_id)
        elif decision.action is WorkItemAction.DEFER:
            self._repository.defer_item(
                work_item_id,
                error=decision.reason_code,
                retry_delay_seconds=decision.retry_delay_seconds or 0.0,
            )

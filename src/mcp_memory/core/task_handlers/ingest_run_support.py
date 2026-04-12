from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mcp_memory.core.task_handlers.ingest_support import _build_ingest_result


@dataclass
class IngestRunAccumulator:
    created_ids: list[str] = field(default_factory=list)
    claimed_ids: list[int] = field(default_factory=list)
    recoverable_ids: list[int] = field(default_factory=list)
    released_ids: list[int] = field(default_factory=list)
    semantic_entry_dispositions: list[dict[str, Any]] = field(default_factory=list)
    meaningful_actions: int = 0
    batches_processed: int = 0

    def absorb_batch_result(self, batch_result: dict[str, Any]) -> None:
        self.batches_processed += 1
        self.created_ids.extend(batch_result["created_ids"])
        self.claimed_ids.extend(batch_result["claimed_ids"])
        self.recoverable_ids.extend(batch_result["recoverable_ids"])
        self.released_ids.extend(batch_result["released_ids"])
        self.semantic_entry_dispositions.extend(batch_result["entry_dispositions"])
        self.meaningful_actions += int(batch_result["meaningful_actions"])

    def build_handler_result(
        self,
        *,
        requested_grouping_strategy: str | None,
        grouping_strategy_used: str,
        grouping_fallback_reason: str | None,
        pending_remaining: int,
    ) -> dict[str, Any]:
        return _build_ingest_result(
            created_ids=self.created_ids,
            claimed_ids=sorted(set(self.claimed_ids)),
            deleted_ids=[],
            recoverable_ids=sorted(set(self.recoverable_ids)),
            released_ids=sorted(set(self.released_ids)),
            meaningful_actions=self.meaningful_actions,
            semantic_entry_dispositions=self.semantic_entry_dispositions,
        ) | {
            "requested_grouping_strategy": requested_grouping_strategy,
            "grouping_strategy_used": grouping_strategy_used,
            "grouping_fallback_reason": grouping_fallback_reason,
            "batches_processed": self.batches_processed,
            "pending_remaining": pending_remaining,
        }


def should_continue_ingest_run(*, batch_meaningful_actions: int, pending_remaining: int) -> bool:
    if batch_meaningful_actions <= 0:
        return False
    return pending_remaining > 0
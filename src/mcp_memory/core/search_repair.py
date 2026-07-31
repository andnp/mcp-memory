"""Ports and adapters for bounded embedding repair during search."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.core.ports.tasks import TaskQueue
from mcp_memory.core.ports.work_items import (
    EXECUTION_LANE_DETERMINISTIC,
    WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
    WorkItemRepository,
)

EMBEDDING_REPAIR_TASK_NAME = "embedding-repair"
DEFAULT_BACKGROUND_REPAIR_BATCH_SIZE = 32
DEFAULT_BACKGROUND_REPAIR_MAX_BATCHES_PER_RUN = 8
BACKGROUND_REPAIR_POLL_INTERVAL_SECONDS = 0.05


class EmbeddingRepairQueue(Protocol):
    def enqueue_unique(self, *, memory_id: str, workspace_id: str | None, model_name: str, memory_updated_at: str, available_at: float) -> tuple[Any, bool]: ...
    def backlog_snapshot(self) -> Any: ...


@dataclass(frozen=True, slots=True)
class RepairWaitResult:
    elapsed_seconds: float
    pending_count: int


@dataclass(frozen=True, slots=True)
class RepairBacklogSnapshot:
    queued_count: int = 0
    running_count: int = 0
    oldest_queued_age_seconds: float | None = None


class EmbeddingRepairScheduler:
    """Schedule durable repairs and optionally wait for their bounded completion."""

    def __init__(
        self,
        *,
        config: Config,
        task_queue: TaskQueue,
        work_items: WorkItemRepository | None,
        embedding_repair_queue: EmbeddingRepairQueue | None,
        wait_seconds: float,
    ) -> None:
        self._config = config
        self._task_queue = task_queue
        self._work_items = work_items
        self._embedding_repair_queue = embedding_repair_queue
        self._wait_seconds = max(wait_seconds, 0.0)

    def schedule(self, candidates: Sequence[MemoryRecord], model_name: str) -> None:
        from mcp_memory.core.task_handlers.constants import task_priority

        queued_count = 0
        for candidate in candidates:
            if self._embedding_repair_queue is not None:
                _, created = self._embedding_repair_queue.enqueue_unique(
                    memory_id=candidate.id,
                    workspace_id=None,
                    model_name=model_name,
                    memory_updated_at=candidate.updated_at or "",
                    available_at=time.time(),
                )
            else:
                assert self._work_items is not None
                _, created = self._work_items.enqueue_unique(
                    family_key=WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
                    execution_lane=EXECUTION_LANE_DETERMINISTIC,
                    workspace_id=None,
                    priority=task_priority(EMBEDDING_REPAIR_TASK_NAME),
                    idempotency_key=f"{WORK_FAMILY_MEMORY_EMBEDDING_REPAIR}:{model_name}:{candidate.id}:{candidate.updated_at or ''}",
                    payload={
                        "memory_id": candidate.id,
                        "model_name": model_name,
                        "memory_updated_at": candidate.updated_at or "",
                    },
                )
            if created:
                queued_count += 1

        if queued_count > 0:
            batch_size = max(int(getattr(self._config.embeddings, "batch_size", DEFAULT_BACKGROUND_REPAIR_BATCH_SIZE)), 1)
            self._task_queue.enqueue_unique(
                EMBEDDING_REPAIR_TASK_NAME,
                data={
                    "trigger": "search_repair",
                    "batch_size": batch_size,
                    "max_batches_per_run": DEFAULT_BACKGROUND_REPAIR_MAX_BATCHES_PER_RUN,
                },
                workspace_id=None,
                priority=task_priority(EMBEDDING_REPAIR_TASK_NAME),
                available_at=time.time(),
            )

    def wait_for(
        self,
        candidates: Sequence[MemoryRecord],
        stale_check: Callable[[list[MemoryRecord]], list[MemoryRecord]],
    ) -> RepairWaitResult:
        started_at = time.monotonic()
        deadline = started_at + self._wait_seconds
        pending = list(candidates)
        while time.monotonic() < deadline:
            pending = stale_check(list(candidates))
            if not pending:
                break
            time.sleep(BACKGROUND_REPAIR_POLL_INTERVAL_SECONDS)
        return RepairWaitResult(time.monotonic() - started_at, len(pending))

    def backlog_snapshot(self) -> RepairBacklogSnapshot:
        if self._embedding_repair_queue is not None:
            snapshot = self._embedding_repair_queue.backlog_snapshot()
            return RepairBacklogSnapshot(
                queued_count=int(snapshot.queued_count),
                running_count=int(snapshot.running_count),
                oldest_queued_age_seconds=snapshot.oldest_queued_age_seconds,
            )
        if self._work_items is None:
            return RepairBacklogSnapshot()
        pending = self._work_items.list_items(status="pending", limit=10_000)
        deferred = self._work_items.list_items(status="deferred", limit=10_000)
        running = self._work_items.list_items(status="running", limit=10_000)
        queued = [item for item in (*pending, *deferred) if item.family_key == WORK_FAMILY_MEMORY_EMBEDDING_REPAIR]
        running = [item for item in running if item.family_key == WORK_FAMILY_MEMORY_EMBEDDING_REPAIR]
        oldest = min((item.created_at for item in queued), default=None)
        return RepairBacklogSnapshot(
            queued_count=len(queued),
            running_count=len(running),
            oldest_queued_age_seconds=None if oldest is None else max(time.time() - float(oldest), 0.0),
        )

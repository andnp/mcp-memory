from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.core.task_handlers.ingest import build_ingest_preflight_state


class _JournalStub:
    def __init__(self, pending_count: int) -> None:
        self._pending_count = pending_count

    def count_by_status(self, *, workspace_id=None):
        del workspace_id
        return {"pending": self._pending_count}

    def get_pending(self, *, workspace_id=None):
        del workspace_id
        return []


def test_build_ingest_preflight_state_collects_scoping_and_grouping_defaults() -> None:
    ctx = ApplicationContext(
        workspace_id="ctx-workspace",
        journal=_JournalStub(pending_count=3),
        repository=SimpleNamespace(list_memories=lambda **kwargs: []),
    )
    task = SimpleNamespace(
        data={"batch_size": 11, "max_batches_per_run": 4},
        workspace_id="task-workspace",
    )

    preflight = build_ingest_preflight_state(ctx, cast(TaskRecord, cast(Any, task)), provider=None)

    assert preflight.workspace_id == "task-workspace"
    assert preflight.journal_workspace_id == "task-workspace"
    assert preflight.batch_size == 11
    assert preflight.max_batches_per_run == 4
    assert preflight.pending_count_before_run == 3
    assert preflight.provider is None
    assert preflight.grouping_strategy_used in {"lexical-seeded", "semantic-seeded"}
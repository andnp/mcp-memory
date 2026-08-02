from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_support import (
    CuratorCandidateRequest,
    acquire_curator_candidates,
    retrieval_friction_flags,
    select_curator_seed_batch,
    select_curator_support_records,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.relational.repository import RelationalMemoryRecord


pytestmark = pytest.mark.small


def _record(
    memory_id: str,
    *,
    updated_at: str,
    tags: list[str] | None = None,
    summary: str | None = None,
) -> RelationalMemoryRecord:
    return RelationalMemoryRecord(
        id=memory_id,
        title=memory_id,
        content=f"Durable content for {memory_id}.",
        summary=summary or f"Summary for {memory_id}.",
        type="fact",
        status="active",
        created_at=updated_at,
        updated_at=updated_at,
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-other"],
        tags=tags or [],
    )


def _task(strategy: str | None = None) -> TaskRecord:
    return TaskRecord(
        id="curator-whole-corpus-task",
        task_name=CURATOR_TASK_NAME,
        data={} if strategy is None else {"strategy": strategy},
        workspace_id="workspace-task",
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


class _BackendRepository:
    def __init__(self, records_by_strategy: dict[str, list[RelationalMemoryRecord]]) -> None:
        self.records_by_strategy = records_by_strategy
        self.calls: list[dict[str, object]] = []
        self.by_id = {
            record.id: record
            for records in records_by_strategy.values()
            for record in records
        }

    def query_maintenance_candidates(self, strategy: str, **kwargs: object) -> list[RelationalMemoryRecord]:
        self.calls.append({"strategy": strategy, **kwargs})
        records = self.records_by_strategy.get(strategy, [])
        limit = kwargs.get("limit")
        return records[: int(limit)] if isinstance(limit, int) else records

    def get_memory(self, memory_id: str) -> RelationalMemoryRecord | None:
        return self.by_id.get(memory_id)

    def get_links(self, _memory_id: str, direction: str = "outgoing") -> list[object]:
        _ = direction
        return []

    def has_incoming_link(self, _memory_id: str, _link_type: str) -> bool:
        return False

    def count_incoming_links(self, _memory_id: str) -> int:
        return 0


def test_seed_uses_global_bounded_queries_without_task_workspace_filter() -> None:
    old_record = _record("old-global-record", updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [old_record],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
    )

    batch = select_curator_seed_batch(ctx, _task("cold-storage"), seed_limit=1)

    assert [record.id for record in batch.records] == [old_record.id]
    assert {call["strategy"] for call in repository.calls} == {
        "cold-storage",
        "never-surfaced",
        "oversized/thin",
        "orphan/low-support",
        "quality-signal",
        "seeded-random",
    }
    assert all("workspace_id" not in call for call in repository.calls)
    assert all(call["seed"] == "curator-whole-corpus-task" for call in repository.calls)
    assert batch.records[0].workspace_ids == ["workspace-other"]


def test_retrieval_friction_flags_match_generic_summary_forms() -> None:
    covers = _record(
        "covers-summary",
        updated_at="2026-01-01T00:00:00+00:00",
        summary="Covers a broad thing.",
    )
    added = _record(
        "added-summary",
        updated_at="2026-01-01T00:00:00+00:00",
        summary="Added a broad thing.",
    )

    assert "generic_summary" in retrieval_friction_flags(covers)
    assert "generic_summary" in retrieval_friction_flags(added)


def test_support_uses_global_queries_and_keeps_adjacent_metadata() -> None:
    seed = _record("seed", updated_at="2026-01-01T00:00:00+00:00", tags=["auth"])
    adjacent = _record("adjacent", updated_at="2020-01-01T00:00:00+00:00")
    tagged = _record("tagged", updated_at="2020-01-02T00:00:00+00:00", tags=["auth"])
    repository = _BackendRepository({
        "cold-storage": [tagged],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    repository.by_id[adjacent.id] = adjacent
    setattr(repository, "get_links", lambda memory_id, direction="outgoing": (
        [SimpleNamespace(source_id=seed.id, target_id=adjacent.id)]
        if memory_id == seed.id and direction == "outgoing"
        else []
    ))
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
    )

    support = select_curator_support_records(ctx, _task(), [seed], support_limit=2)

    assert {record.id for record in support} == {adjacent.id, tagged.id}
    assert all("workspace_id" not in call for call in repository.calls)
    assert {record.workspace_ids[0] for record in support} == {"workspace-other"}


def test_semantic_candidates_use_global_search_seam() -> None:
    anchor = _record("semantic-anchor", updated_at="2026-01-01T00:00:00+00:00")
    neighbor = _record("semantic-neighbor", updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [anchor],
    })
    repository.by_id[neighbor.id] = neighbor
    search_calls: list[dict[str, object]] = []

    class RetrievalFacade:
        def search_sync(self, query: str, **kwargs: object) -> Any:
            search_calls.append({"query": query, **kwargs})
            result = SimpleNamespace(
                record=SimpleNamespace(source_id=neighbor.id),
            )
            return SimpleNamespace(results=[result])

    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=RetrievalFacade(),
        db_manager=None,
        workspace_id=None,
    )

    batch = select_curator_seed_batch(ctx, _task("semantic"), seed_limit=2)

    assert {record.id for record in batch.records} == {anchor.id, neighbor.id}
    assert search_calls
    assert all(call["workspace_id"] is None for call in search_calls)
    assert all(call["status"] == "active" for call in search_calls)


def test_typed_candidate_service_matches_compatibility_wrapper_with_exclusions() -> None:
    first = _record("first", updated_at="2020-01-01T00:00:00+00:00")
    second = _record("second", updated_at="2020-01-02T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [first, second],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
    )
    task = _task("cold-storage")

    direct = acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id=task.id,
            workspace_id=task.workspace_id,
            requested_strategy="cold-storage",
            limit=1,
            exclude_memory_ids=frozenset({first.id}),
        ),
    )
    compatibility = select_curator_seed_batch(
        ctx,
        task,
        seed_limit=1,
        exclude_memory_ids={first.id},
    )

    assert [record.id for record in direct.records] == [record.id for record in compatibility.records]
    assert direct.candidate_count == compatibility.candidate_count
    assert direct.strategy_used == compatibility.strategy_used
    assert direct.strategy_fallback_reason == compatibility.strategy_fallback_reason
    assert direct.records[0].id == second.id
    assert direct.candidate_count == 1
    assert direct.candidate_count == len(direct.records)
    assert {call["limit"] for call in repository.calls[:6]} == {2}
    assert {call["limit"] for call in repository.calls[6:]} == {50}


def test_legacy_task_record_handles_large_candidate_pool_without_oversized_records() -> None:
    records = [
        _record(f"candidate-{index:02d}", updated_at=f"2020-01-{index + 1:02d}T00:00:00+00:00")
        for index in range(17)
    ]
    repository = _BackendRepository({
        "cold-storage": records,
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
    )

    batch = select_curator_seed_batch(ctx, _task("cold-storage"), seed_limit=1)

    assert len(batch.records) == 1


def test_legacy_task_limit_sizes_backend_queries_without_overriding_typed_limit() -> None:
    record = _record("candidate", updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [record],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "seeded-random": [],
    })
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
    )
    legacy_task = _task("cold-storage")
    legacy_task.data["limit"] = 7

    select_curator_seed_batch(ctx, legacy_task, seed_limit=1)
    legacy_limits = {call["limit"] for call in repository.calls}
    repository.calls.clear()

    acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id=legacy_task.id,
            requested_strategy="cold-storage",
            limit=1,
        ),
    )
    typed_limits = {call["limit"] for call in repository.calls}

    assert legacy_limits == {7}
    assert typed_limits == {1}

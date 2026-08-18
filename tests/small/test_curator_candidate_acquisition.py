from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_models import CampaignHypothesis, CampaignRetrievalProblem
from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME
from mcp_memory.core.task_handlers.curator_support import (
    CuratorCandidateRequest,
    acquire_curator_candidates,
    curator_candidate_revision_token,
    retrieval_friction_flags,
    select_curator_seed_batch,
    select_curator_support_records,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.curation_store import (
    CandidateDisposition,
    CurationCandidateState,
    SQLiteCurationStore,
)
from mcp_memory.mutation_history import MutationActorKind
from mcp_memory.relational.repository import MemoryRecord

pytestmark = pytest.mark.small


def _record(
    memory_id: str,
    *,
    updated_at: str,
    title: str | None = None,
    tags: list[str] | None = None,
    summary: str | None = None,
    content: str | None = None,
    memory_type: str = "fact",
    metadata: dict[str, object] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=title or memory_id,
        content=content or f"Durable content for {memory_id}.",
        summary=summary or f"Summary for {memory_id}.",
        type=memory_type,
        status="active",
        created_at=updated_at,
        updated_at=updated_at,
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-other"],
        tags=tags or [],
        metadata=metadata or {},
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
    def __init__(self, records_by_strategy: dict[str, list[MemoryRecord]]) -> None:
        self.records_by_strategy = records_by_strategy
        self.calls: list[dict[str, object]] = []
        self.by_id = {
            record.id: record
            for records in records_by_strategy.values()
            for record in records
        }

    def query_maintenance_candidates(self, strategy: str, **kwargs: object) -> list[MemoryRecord]:
        self.calls.append({"strategy": strategy, **kwargs})
        records = self.records_by_strategy.get(strategy, [])
        limit = kwargs.get("limit")
        return records[: int(limit)] if isinstance(limit, int) else records

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
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
        "retrieval-quality": [],
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
        "retrieval-quality",
        "seeded-random",
    }
    assert all("workspace_id" not in call for call in repository.calls)
    assert all(call["seed"] == "curator-whole-corpus-task" for call in repository.calls)
    assert batch.records[0].workspace_ids == ["workspace-other"]


def test_quality_signal_route_can_revisit_cooled_failed_repair(db_manager) -> None:
    candidate_id = str(uuid4())
    record = _record(candidate_id, updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "retrieval-quality": [],
        "cold-storage": [],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [record],
        "seeded-random": [],
    })
    curation = SQLiteCurationStore(db_manager)
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        db_manager=None,
        workspace_id=None,
        curation=curation,
        mutation_history=None,
    )
    curation.put_candidate_state(
        CurationCandidateState(
            memory_id=UUID(candidate_id),
            last_observed_revision_token=curator_candidate_revision_token(ctx, record),
            disposition=CandidateDisposition.ESCALATED,
            cooldown_until=datetime.now(UTC) + timedelta(hours=6),
            last_disposition_reason="retrieval_regression",
        )
    )

    assert select_curator_seed_batch(ctx, _task("semantic"), seed_limit=1).records == []
    quality_batch = select_curator_seed_batch(ctx, _task("quality-signal"), seed_limit=1)

    assert [item.id for item in quality_batch.records] == [candidate_id]


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


def test_retrieval_friction_flags_mark_raw_ingress_quality() -> None:
    record = _record(
        "raw-ingress",
        updated_at="2026-01-01T00:00:00+00:00",
        content="x" * 1_601,
        tags=["specific"],
        memory_type="observation",
        metadata={"created_via_ingest": True},
    )

    assert "raw_ingress" in retrieval_friction_flags(record)


def test_retrieval_friction_flags_low_conversion_high_exposure() -> None:
    record = _record(
        "low-conversion",
        updated_at="2026-01-01T00:00:00+00:00",
        metadata={
            "retrieval_engagement": {
                "search_count": 6,
                "converted_search_count": 1,
            }
        },
    )

    assert "low_read_conversion" in retrieval_friction_flags(record)


def test_retrieval_friction_flags_mark_dated_work_logs() -> None:
    record = _record(
        "dated-log",
        updated_at="2026-01-01T00:00:00+00:00",
        memory_type="journal",
        title="2026-01-15 work log",
        content="Worked on the migration and recorded the progress update.",
    )

    assert "dated_work_log" in retrieval_friction_flags(record)


def test_retrieval_friction_flags_preserve_durable_date_context() -> None:
    record = _record(
        "durable-date",
        updated_at="2026-01-01T00:00:00+00:00",
        title="2026-01-15 release deadline",
        content="The release deadline is 2026-01-15; this is a durable planning constraint.",
    )

    assert "dated_work_log" not in retrieval_friction_flags(record)


def test_retrieval_friction_flags_mark_status_execution_and_mixed_content() -> None:
    record = _record(
        "mixed-status",
        updated_at="2026-01-01T00:00:00+00:00",
        content="Decision: keep the boundary. Completed the migration after debugging; ran tests with a temporary workaround.",
    )

    flags = retrieval_friction_flags(record)
    assert "task_completion_residue" in flags
    assert "transient_execution_detail" in flags
    assert "mixed_durability_content" in flags


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


def test_semantic_candidates_do_not_prefetch_remote_search() -> None:
    anchor = _record("semantic-anchor", updated_at="2026-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [anchor],
    })
    search_calls: list[dict[str, object]] = []

    class RetrievalFacade:
        def search_sync(self, query: str, **kwargs: object) -> Any:
            search_calls.append({"query": query, **kwargs})
            return SimpleNamespace(results=[])

    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=RetrievalFacade(),
        db_manager=None,
        workspace_id=None,
    )

    batch = select_curator_seed_batch(ctx, _task("semantic"), seed_limit=2)

    assert [record.id for record in batch.records] == [anchor.id]
    assert search_calls == []


def test_explicit_campaign_hypothesis_prioritizes_expected_and_query_hits() -> None:
    expected_id = uuid4()
    expected = _record(str(expected_id), updated_at="2020-01-01T00:00:00+00:00")
    query_hit = _record("query-hit", updated_at="2020-01-02T00:00:00+00:00")
    heuristic = _record("heuristic", updated_at="2020-01-03T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [heuristic],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    repository.by_id.update({expected.id: expected, query_hit.id: query_hit})

    class RetrievalFacade:
        def search_sync(self, query: str, **kwargs: object) -> Any:
            assert query == "find the goal"
            return SimpleNamespace(
                results=[SimpleNamespace(record=SimpleNamespace(source_id=query_hit.id))]
            )

    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=RetrievalFacade(),
        db_manager=None,
        workspace_id=None,
    )
    hypothesis = CampaignHypothesis(
        query="find the goal",
        retrieval_problem=CampaignRetrievalProblem.RETRIEVAL_QUALITY,
        expected_memory_ids=[expected_id],
    )

    batch = acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id="goal-task",
            requested_strategy="cold-storage",
            limit=2,
            campaign_hypothesis=hypothesis,
        ),
    )

    assert [record.id for record in batch.records] == [expected.id, query_hit.id]


def test_explicit_goal_anchors_survive_suppression_into_planner_batch(db_manager) -> None:
    expected_id = uuid4()
    expected = _record(str(expected_id), updated_at="2020-01-01T00:00:00+00:00")
    query_hit_id = uuid4()
    query_hit = _record(str(query_hit_id), updated_at="2020-01-02T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    repository.by_id.update({expected.id: expected, query_hit.id: query_hit})
    now = datetime.now(UTC)
    curation = SQLiteCurationStore(db_manager)
    history = SimpleNamespace(
        list_events=lambda *, memory_id, limit: [
            SimpleNamespace(
                created_at=now - timedelta(minutes=5),
                actor_kind=MutationActorKind.USER,
                family="curator",
            )
        ]
    )

    class RetrievalFacade:
        def search_sync(self, query: str, **kwargs: object) -> Any:
            assert query == "find the goal"
            return SimpleNamespace(
                results=[SimpleNamespace(record=SimpleNamespace(source_id=query_hit.id))]
            )

    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=RetrievalFacade(),
        curation=curation,
        mutation_history=history,
        db_manager=None,
        workspace_id=None,
    )
    curation.put_candidate_state(
        CurationCandidateState(
            memory_id=expected_id,
            last_observed_revision_token=curator_candidate_revision_token(ctx, expected),
            disposition=CandidateDisposition.COOLDOWN,
            cooldown_until=now + timedelta(hours=1),
        )
    )

    batch = acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id="goal-cooldown-task",
            requested_strategy="cold-storage",
            limit=2,
            campaign_hypothesis=CampaignHypothesis(
                query="find the goal",
                retrieval_problem=CampaignRetrievalProblem.RETRIEVAL_QUALITY,
                expected_memory_ids=[expected_id],
            ),
        ),
    )

    assert [record.id for record in batch.records] == [expected.id, query_hit.id]


@pytest.mark.parametrize("suppression", ["cooldown", "recent-edit"])
def test_legacy_candidates_remain_filtered_during_acquisition(db_manager, suppression: str) -> None:
    candidate_id = uuid4()
    candidate = _record(str(candidate_id), updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [candidate],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    curation = SQLiteCurationStore(db_manager)
    now = datetime.now(UTC)
    history = None
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=None,
        curation=curation,
        mutation_history=None,
        db_manager=None,
        workspace_id=None,
    )
    if suppression == "cooldown":
        curation.put_candidate_state(
            CurationCandidateState(
                memory_id=candidate_id,
                last_observed_revision_token=curator_candidate_revision_token(ctx, candidate),
                disposition=CandidateDisposition.COOLDOWN,
                cooldown_until=now + timedelta(hours=1),
            )
        )
    else:
        history = SimpleNamespace(
            list_events=lambda *, memory_id, limit: [
                SimpleNamespace(
                    created_at=now - timedelta(minutes=5),
                    actor_kind=MutationActorKind.USER,
                    family="curator",
                )
            ]
        )
    ctx.mutation_history = history

    batch = acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id=f"legacy-{suppression}-task",
            requested_strategy="cold-storage",
            limit=1,
        ),
    )

    assert batch.records == []


def test_legacy_campaign_hypothesis_keeps_heuristic_acquisition() -> None:
    heuristic = _record("heuristic", updated_at="2020-01-01T00:00:00+00:00")
    repository = _BackendRepository({
        "cold-storage": [heuristic],
        "never-surfaced": [],
        "oversized/thin": [],
        "orphan/low-support": [],
        "quality-signal": [],
        "seeded-random": [],
    })
    ctx: Any = SimpleNamespace(
        repository=repository,
        relational_search=None,
        memory_retrieval=None,
        db_manager=None,
        workspace_id=None,
    )

    batch = acquire_curator_candidates(
        ctx,
        CuratorCandidateRequest(
            task_id="legacy-task",
            requested_strategy="cold-storage",
            limit=1,
            campaign_hypothesis=CampaignHypothesis.legacy(),
        ),
    )

    assert [record.id for record in batch.records] == [heuristic.id]


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
    assert {call["limit"] for call in repository.calls[:7]} == {2}
    assert {call["limit"] for call in repository.calls[7:]} == {50}


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

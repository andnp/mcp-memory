from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from searchkernel.runtime import clear_query_embedding_cache
from searchkernel.search.record_pipeline import RecordSearchOutcome, RecordSearchResult

from mcp_memory.config import Config
from mcp_memory.core.ports import (
    EmbeddingMaintenancePort,
    MemoryIDResolutionPort,
    ReadCacheValidationPort,
    SearchHealthPort,
    StartupHealthPort,
)
from mcp_memory.core.search_ranking import RankingEngine, ScoringWeights
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.integrations.memory_retrieval import (
    MemoryRetrievalPort,
    build_search_execution_diagnostics,
)
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import (
    RelationalMemorySearchService,
    SearchExecutionDiagnostics,
    _to_relational_search_result,
)
from tests.small.maintenance_read_repository_contract import (
    assert_maintenance_read_preserves_telemetry,
)

pytestmark = pytest.mark.small


class _FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _clear_query_embedding_cache() -> Iterator[None]:
    clear_query_embedding_cache()
    yield
    clear_query_embedding_cache()


def test_core_ranking_engine_fuses_vector_and_keyword_lists() -> None:
    engine = RankingEngine(Config(), weights=ScoringWeights(rrf_k=60.0))

    fused = engine.fuse_reciprocal_rank(
        ["memory-a", "memory-b"],
        ["memory-b", "memory-c"],
    )

    assert fused["memory-b"] > fused["memory-a"]
    assert fused["memory-b"] > fused["memory-c"]
    assert fused["memory-a"] == pytest.approx(1.0 / 61.0)


def test_core_ranking_engine_applies_authority_from_ranked_candidates(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    config = Config()
    engine = RankingEngine(config)
    authority = repository.create_memory(
        title="Auth decision",
        content="Authoritative auth decision.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    supporter = repository.create_memory(
        title="Auth plan",
        content="Depends on the auth decision.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    assert authority is not None and supporter is not None
    repository.add_link(supporter.id, authority.id, "DEPENDS_ON")

    candidates = repository.get_ranking_candidates([authority.id, supporter.id])
    ranked = engine.rank_records(
        candidates,
        {authority.id: 0.04, supporter.id: 0.04},
        workspace_id="workspace-alpha",
    )

    scores = {record.id: score for record, score in ranked}
    assert scores[authority.id] > scores[supporter.id]
    assert config.search_ranking.authority_link_step > 0


def test_repository_keyword_candidates_hide_superseded_records(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    old_plan = repository.create_memory(
        title="Legacy auth rollout",
        content="Old auth rollout plan.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    current_plan = repository.create_memory(
        title="Current auth rollout",
        content="Current auth rollout plan.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert old_plan is not None and current_plan is not None
    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES")

    assert repository.search_keyword_memory_ids("auth rollout", limit=10) == [
        current_plan.id
    ]


def test_repository_keyword_candidates_normalize_punctuation(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Auth rollout plan",
        content="Roll out auth for all services.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth", "rollout"],
    )
    assert record is not None

    ids = repository.search_keyword_memory_ids(
        'auth, rollout!!! "phase-1"',
        limit=10,
    )

    assert ids[0] == record.id


def test_service_search_delegates_to_kernel_with_compatibility_payload(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    old_plan = repository.create_memory(
        title="Legacy auth plan",
        content="Old auth plan for workspace alpha.",
        summary="Old plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    current_plan = repository.create_memory(
        title="Current auth plan",
        content="Current auth plan for workspace alpha.",
        summary="Current plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    cross_workspace = repository.create_memory(
        title="Cross workspace auth fact",
        content="Shared auth fact for another workspace.",
        summary="Shared fact summary.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
    )
    assert old_plan is not None and current_plan is not None
    assert cross_workspace is not None
    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES")

    results = service.search_memories(
        "auth plan",
        workspace_id="workspace-alpha",
        limit=5,
    )

    assert [result.memory_id for result in results] == [current_plan.id]
    assert results[0].summary == "Current plan summary."
    assert results[0].workspace_ids == ["workspace-alpha"]
    assert results[0].ranking_debug is not None
    assert "provenance" in results[0].ranking_debug


def test_service_search_preserves_content_summary_fallback(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Fallback summary memory",
        content=(
            "This memory has no formal summary. "
            "It should fall back to the content before the cutoff. "
            "Trailing details should not appear in the result."
        ),
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None
    assert repository.update_memory(record.id, summary="") is not None

    results = service.search_memories(
        "formal summary",
        workspace_id="workspace-alpha",
        limit=5,
    )

    assert results
    assert results[0].summary.startswith("This memory has no formal summary.")


def test_search_operation_reuses_canonical_result_mapping() -> None:
    status = SimpleNamespace(value="active")
    record = SimpleNamespace(
        source_id="memory-id",
        storage_key="workspace:memory-id",
        title="Mapped result",
        body="Body fallback for a result without summary metadata.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        status=status,
        metadata={
            "memory_ref": "mem-42",
            "summary": "",
            "memory_type": 123,
            "memory_status": None,
            "tags": ["useful", 123],
            "workspace_ids": ["workspace-a", 456],
        },
    )
    search_result = SimpleNamespace(
        record=record,
        score=0.1234567,
        provenance=SimpleNamespace(
            strategies=(),
            to_dict=lambda: {"matched_by_keyword": True},
        ),
    )

    class _Retrieval:
        def search_sync(self, request):
            return SimpleNamespace(results=[search_result])

    expected = _to_relational_search_result(cast(RecordSearchResult, search_result))
    outcome = cast(MemoryRetrievalPort, _Retrieval()).search_sync("mapped result")
    actual = _to_relational_search_result(outcome.results[0])

    assert actual == expected
    assert actual.memory_ref == 42
    assert actual.summary == record.body
    assert actual.memory_type == ""
    assert actual.status == "active"
    assert actual.created_at == "2026-01-01T00:00:00+00:00"
    assert actual.updated_at == "2026-01-02T00:00:00+00:00"
    assert actual.tags == ["useful"]
    assert actual.workspace_ids == ["workspace-a"]
    assert actual.ranking_debug == {
        "provenance": {"matched_by_keyword": True},
        "canonical_id": "workspace:memory-id",
        "duplicate_candidate": False,
        "multi_lane_provenance": False,
        "final_duplicate": False,
    }


def test_search_diagnostics_signal_multi_strategy_candidates_without_merging() -> None:
    record = SimpleNamespace(
        source_id="memory-a",
        storage_key="workspace:memory-a",
        title="Duplicate candidate",
        body="Body",
        status=SimpleNamespace(value="active"),
        metadata={"memory_type": "fact", "workspace_ids": []},
    )
    result = SimpleNamespace(
        record=record,
        score=0.5,
        provenance=SimpleNamespace(
            strategies=("keyword", "vector"),
            to_dict=lambda: {"strategies": ["keyword", "vector"]},
        ),
    )
    mapped = _to_relational_search_result(cast(RecordSearchResult, result))
    diagnostics = SearchExecutionDiagnostics(
        duplicate_candidate_ids=[],
        multi_lane_candidate_ids=["memory-a"],
    )

    assert mapped.ranking_debug is not None
    assert mapped.ranking_debug["duplicate_candidate"] is True
    assert mapped.ranking_debug["multi_lane_provenance"] is True
    assert mapped.ranking_debug["final_duplicate"] is False
    assert diagnostics.to_payload()["duplicate_candidate_ids"] == []
    assert diagnostics.to_payload()["multi_lane_candidate_ids"] == ["memory-a"]
    assert diagnostics.to_payload()["final_duplicate_ids"] == []


def test_service_search_diagnostics_separate_final_duplicates(db_manager) -> None:
    """Report repeated final IDs separately from multi-lane provenance."""
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = SimpleNamespace(
        source_id="memory-a",
        storage_key="workspace:memory-a",
        title="Repeated result",
        body="Body",
        status=SimpleNamespace(value="active"),
        metadata={"memory_type": "fact", "workspace_ids": []},
    )
    results = [
        SimpleNamespace(
            record=record,
            record_id="memory-a",
            score=0.5,
            provenance=SimpleNamespace(
                strategies=("keyword", "vector"),
                to_dict=lambda: {"strategies": ["keyword", "vector"]},
            ),
        ),
        SimpleNamespace(
            record=record,
            record_id="memory-a",
            score=0.4,
            provenance=SimpleNamespace(
                strategies=("keyword",),
                to_dict=lambda: {"strategies": ["keyword"]},
            ),
        ),
    ]
    outcome = cast(
        RecordSearchOutcome,
        SimpleNamespace(
            results=results,
            candidate_count=2,
            stage_timings_ms={},
            candidate_counts={},
            diagnostics=(),
            cache_diagnostics=(),
            failures=(),
            missing_record_ids=(),
            degraded=False,
            trace=None,
        ),
    )
    diagnostic_payload = build_search_execution_diagnostics(
        outcome,
        workspace_id=None,
        ranking_workspace_id=None,
    )
    service._retrieval_facade.search_sync_with_diagnostics = cast(
        Callable[..., tuple[RecordSearchOutcome, SearchExecutionDiagnostics]],
        lambda *args, **kwargs: (outcome, diagnostic_payload),
    )

    mapped, diagnostics = service.search_memories_with_diagnostics(
        "repeated result",
        side_effect_free=True,
    )

    assert diagnostics.multi_lane_candidate_ids == ["memory-a"]
    assert diagnostics.duplicate_candidate_ids == []
    assert diagnostics.final_duplicate_ids == ["memory-a"]
    assert diagnostics.raw_lane_overlap_count is None
    assert diagnostics.to_payload()["duplicate_candidate_ids"] == []
    assert diagnostics.to_payload()["overlap"] == {
        "raw_lane_overlap_count": None,
        "multi_lane_result_count": 1,
        "final_duplicate_count": 1,
    }
    assert all(
        result.ranking_debug is not None
        and result.ranking_debug.get("final_duplicate") is True
        for result in mapped
    )


def test_service_search_filters_archived_records_by_default(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    active = repository.create_memory(
        title="Active architecture",
        content="Focused active architecture note.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    archived = repository.create_memory(
        title="Archived architecture",
        content="Archived architecture note.",
        memory_type="fact",
        status="archived",
        workspace_ids=["workspace-alpha"],
    )
    assert active is not None and archived is not None

    assert [
        result.memory_id
        for result in service.search_memories(
            "architecture",
            workspace_id="workspace-alpha",
            limit=10,
        )
    ] == [active.id]
    assert [
        result.memory_id
        for result in service.search_memories(
            "architecture",
            workspace_id="workspace-alpha",
            status="archived",
            limit=10,
        )
    ] == [archived.id]


def test_service_search_diagnostics_expose_kernel_stage_timing(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Kernel search note",
        content="Canonical kernel retrieval diagnostics.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    results, diagnostics = service.search_memories_with_diagnostics(
        "kernel retrieval",
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert results
    assert diagnostics.timing_ms["total"] >= 0.0
    assert diagnostics.to_payload()["semantic_candidate_strategy"] == "kernel"
    assert diagnostics.timing_ms["search"] >= 0.0


def test_service_search_diagnostics_describe_scope(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Scope diagnostics note",
        content="Global and filtered searches expose their scope.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    _, global_diagnostics = service.search_memories_with_diagnostics(
        "scope diagnostics",
        ranking_workspace_id="workspace-alpha",
        debug=True,
    )
    _, filtered_diagnostics = service.search_memories_with_diagnostics(
        "scope diagnostics",
        workspace_id="workspace-alpha",
        ranking_workspace_id="workspace-beta",
        debug=True,
    )

    assert global_diagnostics.to_payload()["scope"] == {
        "mode": "global",
        "workspace_filter": None,
        "ranking_workspace_id": "workspace-alpha",
    }
    assert filtered_diagnostics.to_payload()["scope"] == {
        "mode": "filtered",
        "workspace_filter": "workspace-alpha",
        "ranking_workspace_id": "workspace-beta",
    }


def test_service_search_is_global_by_default_with_deterministic_workspace_boost(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    timestamp = "2026-01-01T00:00:00+00:00"
    local = repository.create_memory(
        title="Shared workspace ranking topic",
        content="Shared workspace ranking topic.",
        summary="Shared workspace ranking topic.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        created_at=timestamp,
        updated_at=timestamp,
    )
    cross_workspace = repository.create_memory(
        title="Shared workspace ranking topic",
        content="Shared workspace ranking topic.",
        summary="Shared workspace ranking topic.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        created_at=timestamp,
        updated_at=timestamp,
    )
    assert local is not None and cross_workspace is not None

    first = service.search_memories(
        "shared workspace ranking topic",
        ranking_workspace_id="workspace-alpha",
        limit=2,
        side_effect_free=True,
    )
    second = service.search_memories(
        "shared workspace ranking topic",
        ranking_workspace_id="workspace-alpha",
        limit=2,
        side_effect_free=True,
    )
    beta_context = service.search_memories(
        "shared workspace ranking topic",
        ranking_workspace_id="workspace-beta",
        limit=2,
        side_effect_free=True,
    )

    assert [result.memory_id for result in first] == [local.id, cross_workspace.id]
    assert [result.memory_id for result in second] == [local.id, cross_workspace.id]
    assert [result.memory_id for result in beta_context] == [cross_workspace.id, local.id]


def test_service_search_explicit_workspace_filter_isolates_results(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    alpha = repository.create_memory(
        title="Filtered workspace topic",
        content="Filtered workspace topic.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    beta = repository.create_memory(
        title="Filtered workspace topic",
        content="Filtered workspace topic.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
    )
    assert alpha is not None and beta is not None

    results = service.search_memories(
        "filtered workspace topic",
        workspace_id="workspace-alpha",
        ranking_workspace_id="workspace-beta",
        limit=5,
        side_effect_free=True,
    )

    assert [result.memory_id for result in results] == [alpha.id]


def test_service_search_diagnostics_preserve_kernel_outcome_details(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    repository.create_memory(
        title="Kernel diagnostics note",
        content="A diagnostic result should preserve kernel cache and failure details.",
        memory_type="fact",
        workspace_ids=["workspace-a"],
    )
    service = RelationalMemorySearchService(repository, Config())

    outcome = service._retrieval_facade.search_sync("diagnostic result")
    results, diagnostics = service.search_memories_with_diagnostics(
        "diagnostic result",
        workspace_id="workspace-a",
        debug=True,
    )

    assert results
    assert diagnostics.kernel_diagnostics == list(outcome.diagnostics)
    assert diagnostics.cache_diagnostics == list(outcome.cache_diagnostics)
    assert diagnostics.failure_count == len(outcome.failures)
    assert diagnostics.missing_record_count == len(outcome.missing_record_ids)
    assert diagnostics.degraded is outcome.degraded
    assert diagnostics.candidate_counts == {
        str(stage): int(count)
        for stage, count in outcome.candidate_counts.items()
    }
    assert diagnostics.to_payload()["candidate_counts"] == diagnostics.candidate_counts
    assert diagnostics.to_payload()["kernel_diagnostics"] == list(outcome.diagnostics)


@pytest.mark.parametrize(
    (
        "diagnostic",
        "degraded",
        "expected_count",
        "expected_rate",
        "expected_abstained",
    ),
    [
        (
            "semantic_abstention:semantic_candidates=1;"
            "semantic_only_candidates=1;rejected=0",
            False,
            0,
            0.0,
            False,
        ),
        (
            "semantic_abstention:semantic_candidates=1;"
            "semantic_only_candidates=1;rejected=1",
            False,
            1,
            1.0,
            True,
        ),
        (
            "semantic_abstention:semantic_candidates=1;"
            "semantic_only_candidates=0;rejected=0",
            False,
            0,
            None,
            False,
        ),
        (
            "semantic_abstention:semantic_candidates=2;"
            "semantic_only_candidates=2;rejected=1",
            True,
            1,
            0.5,
            None,
        ),
    ],
)
def test_service_search_diagnostics_expose_semantic_abstention(
    db_manager,
    diagnostic: str,
    degraded: bool,
    expected_count: int,
    expected_rate: float | None,
    expected_abstained: bool | None,
) -> None:
    """Map bounded outcome diagnostics to benchmark-safe semantic signals."""
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    outcome = cast(
        RecordSearchOutcome,
        SimpleNamespace(
            results=(),
            candidate_count=0,
            stage_timings_ms={},
            candidate_counts={},
            diagnostics=(diagnostic,),
            cache_diagnostics=(),
            failures=(RuntimeError("degraded"),) if degraded else (),
            missing_record_ids=(),
            degraded=degraded,
            trace=None,
        ),
    )
    diagnostic_payload = build_search_execution_diagnostics(
        outcome,
        workspace_id=None,
        ranking_workspace_id=None,
    )
    service._retrieval_facade.search_sync_with_diagnostics = cast(
        Callable[..., tuple[RecordSearchOutcome, SearchExecutionDiagnostics]],
        lambda *args, **kwargs: (outcome, diagnostic_payload),
    )

    _, diagnostics = service.search_memories_with_diagnostics(
        "semantic diagnostics",
        debug=True,
        side_effect_free=True,
    )

    assert diagnostics.semantic_abstention_count == expected_count
    assert diagnostics.semantic_abstention_rate == expected_rate
    assert diagnostics.semantic_abstained is expected_abstained
    assert diagnostics.to_payload()["semantic_abstention_count"] == expected_count


def test_read_memory_preserves_access_and_superseded_breadcrumbs(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    old_fact = repository.create_memory(
        title="SQLite fact",
        content="Use SQLite locally.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    current_fact = repository.create_memory(
        title="SQLite fact refined",
        content="Use SQLite locally with WAL mode.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert old_fact is not None and current_fact is not None
    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES")
    repository.record_access(
        current_fact.id,
        access_score=4.0,
        accessed_at=(datetime.now(UTC) - timedelta(days=7)).isoformat(),
    )

    result = service.read_memory(current_fact.id)

    assert result is not None
    assert result.record.id == current_fact.id
    assert result.record.access_score > 2.9
    assert result.record.read_count == 1
    assert [memory.id for memory in result.superseded] == [old_fact.id]


def test_maintenance_peek_preserves_telemetry(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Maintenance note",
        content="Authoritative maintenance content.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None
    repository.record_access(
        record.id,
        access_score=4.0,
        accessed_at="2026-07-14T12:00:00+00:00",
        increment_read_count=True,
    )
    repository.touch_last_surfaced([record.id], "2026-07-14T12:01:00+00:00")

    result = assert_maintenance_read_preserves_telemetry(
        repository,
        [record.id],
        lambda: service.peek_memory(record.id),
    )

    assert result is not None
    assert result.record == repository.get_memory(record.id)


def test_maintenance_search_returns_full_context_without_telemetry_changes(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Maintenance search candidate",
        content="Authoritative maintenance investigation content.",
        summary="Maintenance candidate summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    results = assert_maintenance_read_preserves_telemetry(
        repository,
        [record.id],
        lambda: service.search_memories_for_maintenance(
            "authoritative investigation",
            workspace_id="workspace-alpha",
            memory_type="fact",
            limit=10,
        ),
    )

    assert [result.record.id for result in results] == [record.id]
    assert results[0].relationships == {
        "outgoing": repository.get_links(record.id, "outgoing"),
        "incoming": repository.get_links(record.id, "incoming"),
    }


def test_maintenance_search_batches_context_hydration_and_preserves_filters(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    old = repository.create_memory(
        title="Legacy batch maintenance fact",
        content="Legacy batch maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["legacy"],
        created_at="2026-07-14T12:01:00+00:00",
        updated_at="2026-07-14T12:01:00+00:00",
    )
    current = repository.create_memory(
        title="Current batch maintenance fact",
        content="Current batch maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["current", "maintenance"],
        created_at="2026-07-14T12:03:00+00:00",
        updated_at="2026-07-14T12:03:00+00:00",
    )
    peer = repository.create_memory(
        title="Peer batch maintenance fact",
        content="Peer batch maintenance details.",
        memory_type="fact",
        workspace_ids=["workspace-alpha", "workspace-beta"],
        tags=["maintenance", "peer"],
        created_at="2026-07-14T12:02:00+00:00",
        updated_at="2026-07-14T12:02:00+00:00",
    )
    cross_workspace = repository.create_memory(
        title="Cross workspace batch maintenance fact",
        content="This should be filtered out.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
    )
    other_type = repository.create_memory(
        title="Batch maintenance plan",
        content="This plan should be filtered out.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    stale = repository.create_memory(
        title="Stale batch maintenance fact",
        content="This stale fact should be filtered out.",
        memory_type="fact",
        status="stale",
        workspace_ids=["workspace-alpha"],
    )
    supporter = repository.create_memory(
        title="Batch maintenance supporter",
        content="Supports the current fact.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert (
        old is not None
        and current is not None
        and peer is not None
        and cross_workspace is not None
        and other_type is not None
        and stale is not None
        and supporter is not None
    )
    repository.add_link(current.id, old.id, "SUPERSEDES", "Current replaces legacy")
    repository.add_link(supporter.id, current.id, "DEPENDS_ON", "Support for current")

    expected_ids = repository._search_keyword_memory_ids(
        "batch maintenance",
        workspace_id="workspace-alpha",
        memory_type="fact",
        status="active",
        include_superseded=True,
        limit=10,
    )
    queries: list[str] = []
    connection = db_manager.get_connection()
    connection.set_trace_callback(queries.append)
    try:
        contexts = repository.search_memories_for_maintenance(
            "batch maintenance",
            workspace_id="workspace-alpha",
            memory_type="fact",
            status="active",
            include_superseded=True,
            limit=10,
        )
    finally:
        connection.set_trace_callback(None)

    assert [context.record.id for context in contexts] == expected_ids
    assert cross_workspace.id not in expected_ids
    assert other_type.id not in expected_ids
    assert stale.id not in expected_ids
    assert contexts == [repository.peek_memory(memory_id) for memory_id in expected_ids]
    assert contexts[0].record.workspace_ids == ["workspace-alpha"]
    assert contexts[0].record.tags == ["current", "maintenance"]
    assert [record.id for record in contexts[0].superseded] == [old.id]
    assert contexts[0].superseded[0].workspace_ids == ["workspace-alpha"]
    assert contexts[0].superseded[0].tags == ["legacy"]
    assert contexts[0].relationships["incoming"] == repository.get_links(
        current.id,
        "incoming",
    )
    application_queries = [query for query in queries if not query.startswith("--")]
    assert len(application_queries) <= 8
    assert not any("WHERE memory_id = ?" in query for query in application_queries)


def test_maintenance_search_skips_missing_and_empty_hydration_ids(
    db_manager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Hydration presence check",
        content="A record used to check missing hydration rows.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    monkeypatch.setattr(
        repository,
        "_search_keyword_memory_ids",
        lambda *args, **kwargs: ["missing-memory", record.id],
    )
    assert [context.record.id for context in repository.search_memories_for_maintenance("ignored")] == [
        record.id
    ]

    monkeypatch.setattr(repository, "_search_keyword_memory_ids", lambda *args, **kwargs: [])
    assert repository.search_memories_for_maintenance("ignored") == []


def test_service_satisfies_search_health_capability(db_manager) -> None:
    """Expose semantic health through the typed search health capability."""
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    assert isinstance(service, SearchHealthPort)
    health = cast(SearchHealthPort, service).get_health()

    assert health.available is False


def test_service_satisfies_startup_health_capability(db_manager) -> None:
    """Expose startup integrity checks through the typed startup capability."""
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    assert isinstance(service, StartupHealthPort)
    health = cast(StartupHealthPort, service).run_startup_health_check()

    assert health.available is False
    assert health.last_integrity_check_at is not None


def test_embedding_health_and_rebuild_remain_service_owned(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
        db_manager=db_manager,
    )
    record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    assert isinstance(service, EmbeddingMaintenancePort)
    result = cast(EmbeddingMaintenancePort, service).rebuild_semantic_index()
    health = service.get_health()

    assert result["rebuilt"] is True
    records_indexed = result["records_indexed"]
    assert isinstance(records_indexed, int) and records_indexed >= 1
    assert vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name=_FakeEmbedder.model_name,
    ) is not None
    assert health.rebuild_count == 1
    assert health.available is True


def test_service_resolves_ids_and_cache_tokens_through_authoritative_repository(
    db_manager,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())
    record = repository.create_memory(
        title="Cache token note",
        content="Read cache validation.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    assert isinstance(service, ReadCacheValidationPort)
    assert isinstance(service, MemoryIDResolutionPort)
    tokens = cast(ReadCacheValidationPort, service).get_read_cache_validation_tokens(
        [record.id]
    )

    assert cast(MemoryIDResolutionPort, service).resolve_memory_id(record.id) == record.id
    assert record.id in tokens

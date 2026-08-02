from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mcp_memory.config import Config
from mcp_memory.core.search_ranking import RankingEngine, ScoringWeights
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.operations import SearchMemoryRecordsOperation
from mcp_memory.relational.search import (
    RelationalMemorySearchService,
    _to_relational_search_result,
)
from searchkernel.runtime import clear_query_embedding_cache
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
        provenance=SimpleNamespace(to_dict=lambda: {"matched_by_keyword": True}),
    )

    class _Retrieval:
        def search_sync(self, request):
            return SimpleNamespace(results=[search_result])

    expected = _to_relational_search_result(search_result)
    actual = SearchMemoryRecordsOperation(_Retrieval()).execute(
        query="mapped result",
        workspace_id=None,
        limit=5,
        adaptive_limit=False,
        memory_type=None,
        status=None,
        include_superseded=False,
    )[0]

    assert actual == expected
    assert actual.memory_ref == 42
    assert actual.summary == record.body
    assert actual.memory_type == ""
    assert actual.status == "active"
    assert actual.tags == ["useful"]
    assert actual.workspace_ids == ["workspace-a"]
    assert actual.ranking_debug == {
        "provenance": {"matched_by_keyword": True},
        "canonical_id": "workspace:memory-id",
    }


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


def test_service_search_diagnostics_are_kernel_compatible_total_only(db_manager) -> None:
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
    assert set(diagnostics.timing_ms) == {"total"}


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
    assert diagnostics.to_payload()["kernel_diagnostics"] == list(outcome.diagnostics)


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
        accessed_at=(datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
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

    result = service.rebuild_semantic_index()
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

    tokens = service.get_read_cache_validation_tokens([record.id])

    assert service.resolve_memory_id(record.id) == record.id
    assert record.id in tokens

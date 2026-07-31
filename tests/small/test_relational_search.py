import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.relational.repository import RelationalMemoryRecord, RelationalMemoryRepository
from mcp_memory.relational.search import (
    INLINE_EMBEDDING_REPAIR_LIMIT,
    RankingEngine,
    RelationalMemorySearchService,
    RelationalSearchResult,
    SearchExecutionDiagnostics,
    ScoringWeights,
    _is_technical_single_token_query,
    _keyword_token_coverage,
    _query_tokens,
    _rank_semantic_candidate_ids,
    _strong_keyword_bounded_candidate_cap,
)
from mcp_memory.mcp.services import search_memory_records_service
from mcp_memory.context import ApplicationContext
from mcp_memory.work_item_store import SQLiteWorkItemRepository
from tests.small.maintenance_read_repository_contract import (
    assert_maintenance_read_preserves_telemetry,
)
from searchkernel.runtime import clear_query_embedding_cache


pytestmark = pytest.mark.small


@pytest.fixture(autouse=True)
def _clear_query_embedding_cache() -> Iterator[None]:
    clear_query_embedding_cache()
    yield
    clear_query_embedding_cache()


def test_ranking_engine_fuses_rrf_across_vector_and_keyword_lists(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    engine = RankingEngine(repository, Config(), weights=ScoringWeights(rrf_k=60.0))

    fused = engine.fuse_reciprocal_rank(
        ["memory-a", "memory-b"],
        ["memory-b", "memory-c"],
    )

    assert fused["memory-b"] > fused["memory-a"]
    assert fused["memory-b"] > fused["memory-c"]
    assert fused["memory-a"] == pytest.approx(1.0 / 61.0)


def test_ranking_engine_calibrates_rrf_scores_around_threshold(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    engine = RankingEngine(
        repository,
        Config(),
        weights=ScoringWeights(calibration_threshold=0.035, calibration_steepness=150.0),
    )

    assert engine.calibrate_score(0.035) == pytest.approx(0.5)
    assert engine.calibrate_score(0.06) > engine.calibrate_score(0.035)
    assert engine.calibrate_score(0.01) < 0.5


def test_ranking_engine_uses_configured_search_weights(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    config = Config(search_ranking=SearchRankingConfig(rrf_k=42.0, workspace_multiplier=1.15))
    engine = RankingEngine(repository, config)

    assert engine._weights.rrf_k == 42.0
    assert engine._weights.workspace_multiplier == 1.15


def test_ranking_engine_applies_stage_boosts_and_penalties_in_order(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    engine = RankingEngine(repository, Config())

    evergreen = repository.create_memory(
        title="Evergreen fact",
        content="Evergreen memory content.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        created_at="2025-01-01T00:00:00+00:00",
        updated_at="2025-01-01T00:00:00+00:00",
    )
    fresh = repository.create_memory(
        title="Fresh journal",
        content="Fresh journal content.",
        memory_type="journal",
        workspace_ids=["workspace-alpha"],
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    stale = repository.create_memory(
        title="Stale plan",
        content="Stale plan content.",
        memory_type="plan",
        status="stale",
        workspace_ids=["workspace-alpha"],
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    working = repository.create_memory(
        title="Working memory",
        content="Working memory content.",
        memory_type="observation",
        workspace_ids=["workspace-alpha"],
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    assert evergreen is not None and fresh is not None and stale is not None and working is not None

    repository.record_access(
        working.id,
        access_score=25.0,
        accessed_at=datetime.now(timezone.utc).isoformat(),
    )
    repository.add_link(fresh.id, evergreen.id, "DEPENDS_ON")
    repository.add_link(working.id, evergreen.id, "DEPENDS_ON")

    scores = {
        record.id: score
        for record, score in engine.rank_records(
            [evergreen, fresh, stale, working],
            {
                evergreen.id: 0.04,
                fresh.id: 0.04,
                stale.id: 0.04,
                working.id: 0.04,
            },
            workspace_id="workspace-alpha",
        )
    }

    assert scores[working.id] > scores[stale.id]
    assert scores[fresh.id] > scores[stale.id]
    assert scores[evergreen.id] > scores[stale.id]


def test_repository_keyword_candidates_use_fts_and_hide_superseded(db_manager) -> None:
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
    cross_workspace = repository.create_memory(
        title="Auth notes",
        content="Shared auth notes.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
    )
    assert old_plan is not None and current_plan is not None and cross_workspace is not None

    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES")

    ids = repository.search_keyword_memory_ids(
        "auth rollout",
        workspace_id="workspace-alpha",
        limit=10,
    )

    assert ids == [current_plan.id]


def test_repository_keyword_candidates_normalize_punctuation_heavy_queries(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    record = repository.create_memory(
        title="Auth rollout plan",
        content="Roll out auth for all services.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth", "rollout"],
    )
    assert record is not None

    ids = repository.search_keyword_memory_ids('auth, rollout!!! "phase-1"', limit=10)

    assert ids[0] == record.id


def test_repository_searchable_memories_preserve_lexical_inputs_for_bounded_semantic_eval(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    primary = repository.create_memory(
        title="Postgres backend rollout",
        content="Postgres backend guidance for shared deployments and storage latency.",
        summary="Postgres backend summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["postgres", "latency"],
    )
    superseded = repository.create_memory(
        title="Postgres backend rollout legacy",
        content="Legacy Postgres backend note.",
        summary="Legacy backend summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["postgres"],
    )
    assert primary is not None and superseded is not None

    repository.add_link(primary.id, superseded.id, "SUPERSEDES")

    query_tokens = _query_tokens("postgres backend storage latency")
    searchable = repository.get_searchable_memories([primary.id, superseded.id])
    ranking = repository.get_ranking_candidates([primary.id, superseded.id])

    assert [record.id for record in searchable] == [primary.id]
    assert [candidate.record.id for candidate in ranking] == [primary.id]
    assert _keyword_token_coverage(query_tokens, searchable[0]) == pytest.approx(
        _keyword_token_coverage(query_tokens, ranking[0].record)
    )


def test_search_memories_prioritizes_workspace_and_hides_superseded(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    old_plan = repository.create_memory(
        title="Legacy auth plan",
        content="Old auth plan for workspace alpha.",
        summary="Old plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
        created_at="2026-02-01T10:00:00+00:00",
        updated_at="2026-02-01T10:00:00+00:00",
    )
    assert old_plan is not None

    current_plan = repository.create_memory(
        title="Current auth plan",
        content="Current auth plan for workspace alpha.",
        summary="Current plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
        created_at="2026-03-10T10:00:00+00:00",
        updated_at="2026-03-10T10:00:00+00:00",
    )
    assert current_plan is not None

    cross_workspace = repository.create_memory(
        title="Cross workspace auth fact",
        content="Shared auth fact for another workspace.",
        summary="Shared fact summary.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
        created_at="2026-03-09T10:00:00+00:00",
        updated_at="2026-03-09T10:00:00+00:00",
    )
    assert cross_workspace is not None

    repository.add_link(current_plan.id, old_plan.id, "SUPERSEDES", "Replaced during redesign")

    results = service.search_memories("auth plan", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results] == [current_plan.id, cross_workspace.id]
    assert results[0].summary == "Current plan summary."
    assert results[0].workspace_ids == ["workspace-alpha"]


def test_search_memories_falls_back_to_truncated_content_summary(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    content = (
        "This memory does not have a formal summary. "
        "It should fall back to the content and stop at the last complete sentence before the cutoff. "
        "The remaining text is intentionally long so that the helper must truncate rather than return the full body unchanged. "
        "Additional trailing details should not appear in the summary output."
    )
    record = repository.create_memory(
        title="Fallback summary memory",
        content=content,
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["summary"],
    )
    assert record is not None
    cleared = repository.update_memory(record.id, summary="")
    assert cleared is not None

    results = service.search_memories("formal summary", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id
    assert results[0].summary == (
        "This memory does not have a formal summary. "
        "It should fall back to the content and stop at the last complete sentence before the cutoff."
    )


def test_search_memories_falls_back_to_ellipsis_when_no_sentence_boundary_exists(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    content = "x" * 240
    record = repository.create_memory(
        title="No punctuation summary memory",
        content=content,
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["summary"],
    )
    assert record is not None
    cleared = repository.update_memory(record.id, summary="")
    assert cleared is not None

    results = service.search_memories("No punctuation summary memory", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id
    assert results[0].summary == ("x" * 200) + "…"


def test_read_memory_returns_superseded_breadcrumbs_and_updates_access_score(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    old_fact = repository.create_memory(
        title="SQLite fact",
        content="Use SQLite locally.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["database"],
        created_at="2026-03-01T08:00:00+00:00",
        updated_at="2026-03-01T08:00:00+00:00",
    )
    current_fact = repository.create_memory(
        title="SQLite fact refined",
        content="Use SQLite locally with WAL mode.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["database"],
        created_at="2026-03-05T08:00:00+00:00",
        updated_at="2026-03-05T08:00:00+00:00",
    )
    assert old_fact is not None and current_fact is not None

    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES", "Refined after testing")
    repository.record_access(
        current_fact.id,
        access_score=4.0,
        accessed_at=(datetime.now(timezone.utc) - timedelta(days=7)).isoformat(),
    )

    result = service.read_memory(current_fact.id)
    refreshed = repository.get_memory(current_fact.id)

    assert result is not None
    assert result.record.id == current_fact.id
    assert result.record.access_score > 2.9
    assert result.record.read_count == 1
    assert refreshed is not None
    assert refreshed.read_count == 1
    assert [memory.id for memory in result.superseded] == [old_fact.id]
    assert result.relationships["outgoing"][0].link_type == "SUPERSEDES"


def test_maintenance_peek_matches_read_context_without_changing_telemetry(db_manager) -> None:
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
    repository.add_link(current_fact.id, old_fact.id, "SUPERSEDES", "Refined after testing")
    repository.record_access(
        current_fact.id,
        access_score=4.0,
        accessed_at="2026-07-14T12:00:00+00:00",
        increment_read_count=True,
    )
    repository.touch_last_surfaced([current_fact.id], "2026-07-14T12:01:00+00:00")

    result = assert_maintenance_read_preserves_telemetry(
        repository,
        [current_fact.id, old_fact.id],
        lambda: service.peek_memory(current_fact.id),
    )
    ordinary_record = repository.get_memory(current_fact.id)
    assert result is not None and ordinary_record is not None
    assert result.record == ordinary_record
    assert result.relationships["outgoing"] == repository.get_links(current_fact.id, "outgoing")
    assert result.relationships["incoming"] == repository.get_links(current_fact.id, "incoming")
    assert [record.id for record in result.superseded] == [old_fact.id]


def test_maintenance_search_returns_full_context_without_changing_telemetry(db_manager) -> None:
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
    other_type = repository.create_memory(
        title="Maintenance search plan",
        content="Authoritative maintenance investigation plan.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    assert other_type is not None
    repository.record_access(
        record.id,
        access_score=2.5,
        accessed_at="2026-07-14T12:00:00+00:00",
        increment_read_count=True,
    )
    repository.touch_last_surfaced([record.id], "2026-07-14T12:01:00+00:00")

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
    assert results[0].record == repository.get_memory(record.id)
    assert results[0].relationships == {
        "outgoing": repository.get_links(record.id, "outgoing"),
        "incoming": repository.get_links(record.id, "incoming"),
    }


def test_list_most_read_memories_orders_by_explicit_read_count(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    alpha = repository.create_memory(
        title="Alpha memory",
        content="Alpha content.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    beta = repository.create_memory(
        title="Beta memory",
        content="Beta content.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    gamma = repository.create_memory(
        title="Gamma memory",
        content="Gamma content.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
    )
    assert alpha is not None and beta is not None and gamma is not None

    service.read_memory(beta.id)
    service.read_memory(alpha.id)
    service.read_memory(alpha.id)
    service.read_memory(gamma.id)

    workspace_results = repository.list_most_read_memories(workspace_id="workspace-alpha", limit=10)
    global_results = repository.list_most_read_memories(limit=10)
    active_results = repository.list_most_read_memories(limit=10, status="active")

    repository.update_memory(gamma.id, status="archived")
    active_results_after_archive = repository.list_most_read_memories(limit=10, status="active")

    assert [record.id for record in workspace_results] == [alpha.id, beta.id]
    assert [record.read_count for record in workspace_results] == [2, 1]
    assert global_results[0].id == alpha.id
    assert global_results[0].read_count == 2
    assert {record.id for record in global_results[1:3]} == {beta.id, gamma.id}
    assert {record.read_count for record in global_results[1:3]} == {1}
    assert {record.id for record in active_results} == {alpha.id, beta.id, gamma.id}
    assert [record.id for record in active_results_after_archive] == [alpha.id, beta.id]


def test_search_memories_penalizes_stale_records_and_updates_last_surfaced(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    active = repository.create_memory(
        title="Search pipeline active",
        content="Active search pipeline note.",
        summary="Active summary.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        tags=["search"],
    )
    stale = repository.create_memory(
        title="Search pipeline stale",
        content="Stale search pipeline note.",
        summary="Stale summary.",
        memory_type="plan",
        status="stale",
        workspace_ids=["workspace-alpha"],
        tags=["search"],
    )
    assert active is not None and stale is not None

    results = service.search_memories("search pipeline", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results] == [active.id, stale.id]
    refreshed_active = repository.get_memory(active.id)
    refreshed_stale = repository.get_memory(stale.id)
    assert refreshed_active is not None and refreshed_active.last_surfaced_at is not None
    assert refreshed_stale is not None and refreshed_stale.last_surfaced_at is not None


def test_search_memories_hides_archived_by_default_but_allows_explicit_archived_queries(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    active = repository.create_memory(
        title="RLCore architecture active",
        content="Focused active RLCore architecture note.",
        summary="Active RLCore architecture summary.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
        tags=["architecture"],
    )
    archived = repository.create_memory(
        title="RLCore architecture archived",
        content="Archived oversized RLCore architecture blob.",
        summary="Archived RLCore architecture summary.",
        memory_type="fact",
        status="archived",
        workspace_ids=["workspace-alpha"],
        tags=["architecture"],
    )
    assert active is not None and archived is not None

    default_results = service.search_memories("RLCore architecture", workspace_id="workspace-alpha", limit=10)
    archived_results = service.search_memories(
        "RLCore architecture",
        workspace_id="workspace-alpha",
        limit=10,
        status="archived",
    )

    assert [result.memory_id for result in default_results] == [active.id]
    assert [result.memory_id for result in archived_results] == [archived.id]


def test_search_memories_applies_graph_authority_boost(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    authority = repository.create_memory(
        title="Auth decision authority",
        content="Authoritative auth decision.",
        summary="Authority summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    peer = repository.create_memory(
        title="Auth decision peer",
        content="Peer auth decision.",
        summary="Peer summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    supporter_one = repository.create_memory(
        title="Auth plan one",
        content="Depends on auth decision authority.",
        workspace_ids=["workspace-alpha"],
        memory_type="plan",
    )
    supporter_two = repository.create_memory(
        title="Auth plan two",
        content="Also depends on auth decision authority.",
        workspace_ids=["workspace-alpha"],
        memory_type="plan",
    )
    assert authority is not None and peer is not None and supporter_one is not None and supporter_two is not None

    repository.add_link(supporter_one.id, authority.id, "DEPENDS_ON")
    repository.add_link(supporter_two.id, authority.id, "DEPENDS_ON")

    results = service.search_memories(
        "summary",
        workspace_id="workspace-alpha",
        limit=5,
    )

    assert results[0].memory_id == authority.id
    assert peer.id in {result.memory_id for result in results}


def test_search_memories_weights_supporting_links_above_contradictions(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    supporting = repository.create_memory(
        title="Auth architecture authority",
        content="Auth architecture summary.",
        summary="Auth architecture summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    contradictory = repository.create_memory(
        title="Auth architecture contradictory",
        content="Auth architecture summary.",
        summary="Auth architecture summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    supporter = repository.create_memory(
        title="Auth rollout supporter",
        content="Depends on auth architecture authority.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    supporter_two = repository.create_memory(
        title="Auth rollout supporter two",
        content="Also depends on auth architecture authority.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    challenger = repository.create_memory(
        title="Auth rollout challenger",
        content="Contradicts auth architecture contradictory.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    assert supporting is not None and contradictory is not None and supporter is not None and supporter_two is not None and challenger is not None

    repository.add_link(supporter.id, supporting.id, "DEPENDS_ON")
    repository.add_link(supporter_two.id, supporting.id, "DEPENDS_ON")
    repository.add_link(challenger.id, contradictory.id, "CONTRADICTS")

    results = service.search_memories("auth architecture summary", workspace_id="workspace-alpha", limit=5)
    positions = {result.memory_id: index for index, result in enumerate(results)}

    assert positions[supporting.id] < positions[contradictory.id]
    assert contradictory.id in {result.memory_id for result in results}


def test_search_memories_prefers_graph_supported_canonical_memory_over_unsupported_working_plan(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    working_plan = repository.create_memory(
        title="Ingest migration working plan",
        content="Agentic ingest final shape execution plan.",
        summary="Working plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["ingest"],
    )
    canonical_observation = repository.create_memory(
        title="Ingest final-shape canonical note",
        content="Agentic ingest final shape with durable claim finalization.",
        summary="Canonical ingest summary.",
        memory_type="observation",
        workspace_ids=["workspace-alpha"],
        tags=["ingest"],
    )
    supporting_fact = repository.create_memory(
        title="Ingest implementation detail",
        content="Depends on the canonical ingest final shape memory.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert working_plan is not None and canonical_observation is not None and supporting_fact is not None

    repository.record_access(
        working_plan.id,
        access_score=64.0,
        accessed_at=datetime.now(timezone.utc).isoformat(),
    )
    repository.add_link(supporting_fact.id, canonical_observation.id, "DEPENDS_ON")

    results = service.search_memories("agentic ingest final shape", workspace_id="workspace-alpha", limit=5, debug=True)
    positions = {result.memory_id: index for index, result in enumerate(results)}
    canonical_debug = next(result.ranking_debug for result in results if result.memory_id == canonical_observation.id)
    plan_debug = next(result.ranking_debug for result in results if result.memory_id == working_plan.id)

    assert positions[canonical_observation.id] < positions[working_plan.id]
    assert canonical_debug is not None
    assert plan_debug is not None
    assert float(canonical_debug["graph_support_bonus"]) > 0
    assert float(plan_debug["access_bonus"]) < 0.1


def test_search_memories_memory_type_hint_does_not_filter_results(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    preferred_fact = repository.create_memory(
        title="Auth decision fact",
        content="Authoritative auth decision for token rotation.",
        summary="Fact summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    related_plan = repository.create_memory(
        title="Auth decision rollout plan",
        content="Plan for rolling out the auth decision to all clients.",
        summary="Plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert preferred_fact is not None and related_plan is not None

    results = service.search_memories(
        "auth decision",
        workspace_id="workspace-alpha",
        memory_type="fact",
        limit=5,
    )

    assert {result.memory_id for result in results[:2]} == {preferred_fact.id, related_plan.id}
    assert {result.memory_type for result in results[:2]} == {"fact", "plan"}


def test_rank_semantic_candidate_ids_prefers_workspace_matches() -> None:
    local = RelationalMemoryRecord(
        id="local",
        title="Local",
        content="Local",
        summary="Local",
        type="fact",
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-03-01T00:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-alpha"],
        tags=[],
    )
    remote = RelationalMemoryRecord(
        id="remote",
        title="Remote",
        content="Remote",
        summary="Remote",
        type="fact",
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-03-01T00:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-beta"],
        tags=[],
    )

    ranked = _rank_semantic_candidate_ids(
        [remote, local],
        {"remote": 0.51, "local": 0.5},
        workspace_id="workspace-alpha",
        limit=2,
        workspace_multiplier=1.2,
    )

    assert ranked == ["local", "remote"]


def test_search_memories_can_expand_graph_neighbors_from_primary_match(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    rollout_plan = repository.create_memory(
        title="Token rollout checklist",
        content="Token rollout checklist for every service.",
        summary="Rollout checklist.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
        tags=["token"],
    )
    canonical_fact = repository.create_memory(
        title="Canonical auth policy",
        content="Use short-lived service tokens and rotation windows.",
        summary="Canonical auth fact.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert rollout_plan is not None and canonical_fact is not None

    repository.add_link(rollout_plan.id, canonical_fact.id, "DEPENDS_ON")

    results = service.search_memories("token rollout checklist", workspace_id="workspace-alpha", limit=5, debug=True)
    result_ids = [result.memory_id for result in results]
    debug_by_id = {result.memory_id: result.ranking_debug for result in results}
    canonical_debug = debug_by_id[canonical_fact.id]

    assert rollout_plan.id == result_ids[0]
    assert canonical_fact.id in result_ids
    assert canonical_debug is not None
    assert canonical_debug["expanded_by_graph"] is True
    assert canonical_debug["graph_link_type"] == "DEPENDS_ON"


class _FakeEmbedder:
    model_name = "fake-mini"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["auth", "permission", "token", "security", "identity"]):
                vectors.append([1.0, 0.0])
            elif any(token in lowered for token in ["sqlite", "database", "wal"]):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


class _CountingFakeEmbedder(_FakeEmbedder):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return super().embed(texts)


class _BlockedFallbackEmbedder:
    @property
    def model_name(self) -> str:
        return "hash:sentence-transformers/all-MiniLM-L6-v2"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _CandidateFilteringVectorStore:
    supports_candidate_filtering = True


class _NamedCountingEmbedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.call_count = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        return [[float(len(self.model_name)), float(len(text))] for text in texts]


class _BlockingEmbedder:
    def __init__(self, *, model_name: str = "blocking-mini", fail_first_call: bool = False) -> None:
        self.model_name = model_name
        self.fail_first_call = fail_first_call
        self.call_count = 0
        self.started = threading.Event()
        self.release = threading.Event()

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.call_count += len(texts)
        self.started.set()
        self.release.wait(timeout=5.0)
        if self.fail_first_call and self.call_count == 1:
            raise RuntimeError("query embed boom")
        return [[1.0, float(len(text))] for text in texts]


def test_query_embedding_cache_reuses_exact_query_per_model_key(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    alpha_embedder = _NamedCountingEmbedder("alpha-mini")
    beta_embedder = _NamedCountingEmbedder("beta-mini")
    alpha_service = RelationalMemorySearchService(repository, Config(), embedder=alpha_embedder, vector_store=object())
    beta_service = RelationalMemorySearchService(repository, Config(), embedder=beta_embedder, vector_store=object())

    alpha_first = alpha_service._get_query_embedding("permissions security")
    alpha_second = alpha_service._get_query_embedding("permissions security")
    beta_first = beta_service._get_query_embedding("permissions security")

    assert alpha_first == alpha_second
    assert alpha_embedder.call_count == 1
    assert beta_embedder.call_count == 1
    assert alpha_first != beta_first


def test_query_embedding_coalescing_shares_inflight_work_across_threads(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    embedder = _BlockingEmbedder()
    service = RelationalMemorySearchService(repository, Config(), embedder=embedder, vector_store=object())
    results: list[list[float]] = []
    errors: list[Exception] = []

    def _load_embedding() -> None:
        try:
            results.append(service._get_query_embedding("permissions security"))
        except Exception as exc:  # pragma: no cover - defensive failure capture for the thread
            errors.append(exc)

    leader = threading.Thread(target=_load_embedding)
    follower = threading.Thread(target=_load_embedding)
    leader.start()
    assert embedder.started.wait(timeout=5.0)
    follower.start()
    embedder.release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    assert embedder.call_count == 1


def test_query_embedding_failures_do_not_cache_or_leak_waiters(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    embedder = _BlockingEmbedder(fail_first_call=True)
    service = RelationalMemorySearchService(repository, Config(), embedder=embedder, vector_store=object())
    errors: list[str] = []

    def _load_embedding() -> None:
        try:
            service._get_query_embedding("permissions security")
        except Exception as exc:
            errors.append(str(exc))

    leader = threading.Thread(target=_load_embedding)
    follower = threading.Thread(target=_load_embedding)
    leader.start()
    assert embedder.started.wait(timeout=5.0)
    follower.start()
    embedder.release.set()
    leader.join(timeout=5.0)
    follower.join(timeout=5.0)

    assert errors == ["query embed boom", "query embed boom"]
    assert embedder.call_count == 1

    embedder.started.clear()
    embedder.release = threading.Event()
    retry_results: list[list[float]] = []

    def _retry_load() -> None:
        retry_results.append(service._get_query_embedding("permissions security"))

    retry = threading.Thread(target=_retry_load)
    retry.start()
    assert embedder.started.wait(timeout=5.0)
    embedder.release.set()
    retry.join(timeout=5.0)

    assert retry_results == [[1.0, float(len("permissions security"))]]
    assert embedder.call_count == 2


def test_search_memories_supports_semantic_candidates_without_lexical_overlap(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=SQLiteVectorStore(db_manager),
    )

    auth_record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    database_record = repository.create_memory(
        title="Storage settings",
        content="SQLite WAL tuning guidance.",
        summary="Database tuning.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["database"],
    )
    assert auth_record is not None and database_record is not None

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == auth_record.id


def test_search_memories_debug_diagnostics_include_semantic_selection_subtimings(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=SQLiteVectorStore(db_manager),
    )

    record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    _, diagnostics = service.search_memories_with_diagnostics(
        "permissions security",
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert diagnostics.timing_ms["semantic_selection"] >= 0.0
    assert diagnostics.timing_ms["semantic_candidate_pool"] >= 0.0
    assert diagnostics.timing_ms["semantic_embedding_stale_check"] >= 0.0
    assert diagnostics.timing_ms["semantic_embedding_refresh"] >= 0.0
    assert diagnostics.timing_ms["semantic_query_embedding"] >= 0.0
    assert diagnostics.timing_ms["semantic_vector_search"] >= 0.0
    assert diagnostics.timing_ms["semantic_ranking"] >= 0.0
    assert diagnostics.timing_ms["candidate_hydration"] >= 0.0
    assert diagnostics.timing_ms["candidate_hydration_query_execution"] >= 0.0
    assert diagnostics.timing_ms["candidate_hydration_row_fetch"] >= 0.0
    assert diagnostics.timing_ms["candidate_hydration_candidate_build"] >= 0.0
    assert "semantic_speculative_fallback" not in diagnostics.timing_ms


def test_search_memory_records_service_debug_payload_includes_search_diagnostics() -> None:
    class FakeSearchService:
        def search_memories(self, **_kwargs):
            raise AssertionError("debug path should use diagnostics-aware execution")

        def search_memories_with_diagnostics(self, **_kwargs):
            return (
                [
                    RelationalSearchResult(
                        memory_id="memory-1",
                        title="Memory one",
                        summary="Summary",
                        memory_type="fact",
                        status="active",
                        score=0.9,
                    )
                ],
                SearchExecutionDiagnostics(
                    timing_ms={
                        "total": 12.0,
                        "candidate_hydration": 4.0,
                        "candidate_hydration_query_execution": 1.0,
                        "candidate_hydration_row_fetch": 1.5,
                        "candidate_hydration_candidate_build": 1.5,
                    },
                    keyword_candidate_count=5,
                    semantic_candidate_count=3,
                    semantic_candidate_strategy="speculative-bounded",
                    vector_search={
                        "backend": "postgres",
                        "row_count": 42,
                        "raw_type_counts": {"str": 42},
                        "fetch_ms": 1.0,
                        "decode_ms": 2.0,
                        "score_ms": 3.0,
                        "sort_ms": 4.0,
                    },
                ),
            )

    payload = search_memory_records_service(
        ApplicationContext(relational_search=FakeSearchService()),
        {"query": "memory one", "debug": True},
    )

    assert payload["status"] == "ok"
    assert float(payload["timing_ms"]["total"]) >= 0.0
    assert payload["search_diagnostics"]["timing_ms"] == {
        "total": 12.0,
        "candidate_hydration": 4.0,
        "candidate_hydration_query_execution": 1.0,
        "candidate_hydration_row_fetch": 1.5,
        "candidate_hydration_candidate_build": 1.5,
    }
    assert payload["search_diagnostics"]["semantic_candidate_strategy"] == "speculative-bounded"
    assert payload["search_diagnostics"]["vector_search"]["raw_type_counts"] == {"str": 42}


def test_search_memories_uses_speculative_bounded_semantic_scores_for_dense_keyword_neighborhoods(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    records = []
    for index in range(5):
        record = repository.create_memory(
            title=f"Postgres backend note {index}",
            content="Postgres backend guidance for shared deployments.",
            summary="Postgres backend summary.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["postgres"],
        )
        assert record is not None
        records.append(record)

    call_modes: list[str] = []
    score_by_id = {
        record.id: score
        for record, score in zip(records, [0.92, 0.89, 0.86, 0.83, 0.8], strict=False)
    }

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        if candidate_ids is None:
            raise AssertionError("global semantic fallback should not run for dense speculative bounded results")
        call_modes.append("bounded")
        assert set(candidate_ids) == {record.id for record in records}
        return score_by_id

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results = service.search_memories(
        "postgres backend storage latency",
        workspace_id="workspace-alpha",
        limit=3,
    )

    assert call_modes == ["bounded"]
    assert len(results) == 3
    assert {result.memory_id for result in results}.issubset({record.id for record in records})


def test_search_memories_caps_strong_keyword_bounded_candidate_ids_conservatively(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    for index in range(30):
        record = repository.create_memory(
            title=f"Postgres backend storage latency note {index:02d}",
            content="Postgres backend storage latency guidance for shared deployments.",
            summary=f"Postgres backend storage latency summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["postgres", "backend", "storage", "latency"],
        )
        assert record is not None

    query = "postgres backend storage latency"
    keyword_ids = repository.search_keyword_memory_ids(query, limit=50)
    assert len(keyword_ids) >= 20

    observed_candidate_ids: list[str] = []

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        assert candidate_ids is not None
        observed_candidate_ids.extend(candidate_ids)
        return {
            memory_id: 1.0 - (rank * 0.01)
            for rank, memory_id in enumerate(candidate_ids)
        }

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results = service.search_memories(
        query,
        workspace_id="workspace-alpha",
        limit=5,
    )

    assert observed_candidate_ids == keyword_ids[:20]
    assert len(observed_candidate_ids) == 20
    assert [result.memory_id for result in results] == keyword_ids[:5]


def test_bounded_semantic_candidates_use_lightweight_searchable_memory_projection(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    record = repository.create_memory(
        title="Postgres backend note",
        content="Postgres backend guidance for shared deployments.",
        summary="Postgres backend summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["postgres"],
    )
    assert record is not None

    def _fail_if_ranking_candidates_used(*_args, **_kwargs):
        raise AssertionError("bounded semantic setup should not require ranking candidate hydration")

    monkeypatch.setattr(repository, "get_ranking_candidates", _fail_if_ranking_candidates_used)

    bounded, speculative = service._bounded_semantic_candidates(
        [record.id],
        _query_tokens("postgres backend guidance"),
        status=None,
        include_superseded=False,
        requested_limit=1,
    )

    assert bounded is not None
    assert [candidate.id for candidate in bounded] == [record.id]
    assert speculative is False


def test_technical_single_token_query_detection_is_narrow() -> None:
    assert _is_technical_single_token_query(["daemon_request_timed_out"]) is True
    assert _is_technical_single_token_query(["daemon-request-timeout"]) is True
    assert _is_technical_single_token_query(["timeout"]) is False
    assert _is_technical_single_token_query(["daemon", "request"]) is False


def test_search_memories_skips_semantic_scoring_for_supported_technical_single_token_queries(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    for index in range(5):
        record = repository.create_memory(
            title=f"Operational incident note {index:02d}",
            content="Observed daemon_request_timed_out while servicing the live memory search request.",
            summary=f"Operational incident summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["incident"],
        )
        assert record is not None

    query = "daemon_request_timed_out"
    keyword_ids = repository.search_keyword_memory_ids(query, limit=50)
    semantic_calls = 0

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        nonlocal semantic_calls
        _ = _query, candidates, _workspace_id, candidate_ids, limit
        semantic_calls += 1
        raise AssertionError("technical single-token keyword-supported queries should skip semantic scoring")

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results, diagnostics = service.search_memories_with_diagnostics(
        query,
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert len(keyword_ids) == 5
    assert semantic_calls == 0
    assert [result.memory_id for result in results] == keyword_ids
    assert len(results) == 5
    assert diagnostics.semantic_candidate_count == 0
    assert diagnostics.semantic_candidate_strategy == "keyword-only-bounded"
    assert diagnostics.timing_ms["semantic_query_embedding"] == 0.0


def test_search_memories_preserves_global_semantic_search_for_nontechnical_single_token_queries(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    for index in range(25):
        record = repository.create_memory(
            title=f"Operational incident note {index:02d}",
            content="Observed timeout while servicing the live memory search request.",
            summary=f"Operational incident summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["incident"],
        )
        assert record is not None

    call_modes: list[str] = []

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        call_modes.append("bounded" if candidate_ids is not None else "global")
        candidate_ids = candidate_ids or [candidate.id for candidate in candidates]
        return {
            memory_id: 1.0 - (rank * 0.01)
            for rank, memory_id in enumerate(candidate_ids)
        }

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    _, diagnostics = service.search_memories_with_diagnostics(
        "timeout",
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert call_modes == ["bounded"]
    assert diagnostics.semantic_candidate_strategy == "global"


def test_search_memories_global_semantic_pool_uses_id_listing_instead_of_list_memories(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    for index in range(12):
        record = repository.create_memory(
            title=f"Operational incident note {index:02d}",
            content="Observed timeout while servicing the live memory search request.",
            summary=f"Operational incident summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["incident"],
        )
        assert record is not None

    observed_list_memory_ids_calls: list[tuple[str | None, str | None, str | None, int]] = []
    original_list_memory_ids = repository.list_memory_ids

    def _fail_list_memories(*_args, **_kwargs):
        raise AssertionError("global semantic pool should source IDs via list_memory_ids, not list_memories")

    def _record_list_memory_ids(*, workspace_id=None, memory_type=None, status=None, limit=100):
        observed_list_memory_ids_calls.append((workspace_id, memory_type, status, limit))
        return original_list_memory_ids(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        candidate_ids = candidate_ids or [candidate.id for candidate in candidates]
        return {
            memory_id: 1.0 - (rank * 0.01)
            for rank, memory_id in enumerate(candidate_ids)
        }

    monkeypatch.setattr(repository, "list_memories", _fail_list_memories)
    monkeypatch.setattr(repository, "list_memory_ids", _record_list_memory_ids)
    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results, diagnostics = service.search_memories_with_diagnostics(
        "timeout",
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert results
    assert diagnostics.semantic_candidate_strategy == "global"
    assert (None, None, None, 500) in observed_list_memory_ids_calls


def test_search_memories_preserves_global_semantic_search_when_technical_single_token_keyword_support_is_insufficient(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    for index in range(4):
        record = repository.create_memory(
            title=f"Operational incident note {index:02d}",
            content="Observed daemon_request_timed_out while servicing the live memory search request.",
            summary=f"Operational incident summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["incident"],
        )
        assert record is not None

    call_modes: list[str] = []

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        call_modes.append("bounded" if candidate_ids is not None else "global")
        candidate_ids = candidate_ids or [candidate.id for candidate in candidates]
        return {
            memory_id: 1.0 - (rank * 0.01)
            for rank, memory_id in enumerate(candidate_ids)
        }

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    _, diagnostics = service.search_memories_with_diagnostics(
        "daemon_request_timed_out",
        workspace_id="workspace-alpha",
        limit=5,
        debug=True,
    )

    assert call_modes == ["bounded"]
    assert diagnostics.semantic_candidate_strategy == "global"


def test_repository_list_memory_ids_matches_list_memories_ordering_and_filters(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)

    oldest = repository.create_memory(
        title="Old alpha fact",
        content="Old alpha fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-01T00:00:00+00:00",
        created_at="2026-03-01T00:00:00+00:00",
    )
    newest = repository.create_memory(
        title="New alpha fact",
        content="New alpha fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-03T00:00:00+00:00",
        created_at="2026-03-03T00:00:00+00:00",
    )
    other_type = repository.create_memory(
        title="Alpha plan",
        content="Alpha plan content.",
        memory_type="plan",
        status="active",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-02T00:00:00+00:00",
        created_at="2026-03-02T00:00:00+00:00",
    )
    other_workspace = repository.create_memory(
        title="Beta fact",
        content="Beta fact content.",
        memory_type="fact",
        status="active",
        workspace_ids=["workspace-beta"],
        updated_at="2026-03-04T00:00:00+00:00",
        created_at="2026-03-04T00:00:00+00:00",
    )
    stale = repository.create_memory(
        title="Stale alpha fact",
        content="Stale alpha fact content.",
        memory_type="fact",
        status="stale",
        workspace_ids=["workspace-alpha"],
        updated_at="2026-03-05T00:00:00+00:00",
        created_at="2026-03-05T00:00:00+00:00",
    )
    assert oldest is not None and newest is not None and other_type is not None
    assert other_workspace is not None and stale is not None

    expected = repository.list_memories(
        workspace_id="workspace-alpha",
        memory_type="fact",
        status="active",
        limit=10,
    )
    observed_ids = repository.list_memory_ids(
        workspace_id="workspace-alpha",
        memory_type="fact",
        status="active",
        limit=10,
    )

    assert observed_ids == [record.id for record in expected] == [newest.id, oldest.id]


def test_speculative_fallback_sources_ids_without_list_memories_hydration(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    lexical_records = []
    for index in range(3):
        record = repository.create_memory(
            title=f"Postgres backend note {index}",
            content="Postgres backend guidance for shared deployments.",
            summary="Postgres backend summary.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["postgres"],
        )
        assert record is not None
        lexical_records.append(record)

    for index in range(22):
        filler = repository.create_memory(
            title=f"Filler note {index:02d}",
            content="Background note without lexical overlap.",
            summary=f"Filler summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["filler"],
        )
        assert filler is not None

    semantic_only = repository.create_memory(
        title="Deployment incident diagnosis",
        content="Operational investigation note for shared production incidents.",
        summary="Incident diagnosis summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["incident"],
    )
    assert semantic_only is not None

    fallback_cap = _strong_keyword_bounded_candidate_cap(3)
    expected_fallback_ids = repository.list_memory_ids(limit=fallback_cap)
    assert semantic_only.id in expected_fallback_ids

    def _fail_list_memories(*_args, **_kwargs):
        raise AssertionError("speculative fallback should source IDs via list_memory_ids, not list_memories")

    observed_list_memory_ids_calls: list[tuple[str | None, str | None, str | None, int]] = []
    original_list_memory_ids = repository.list_memory_ids

    def _record_list_memory_ids(*, workspace_id=None, memory_type=None, status=None, limit=100):
        observed_list_memory_ids_calls.append((workspace_id, memory_type, status, limit))
        return original_list_memory_ids(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        if candidate_ids is not None:
            candidate_id_set = set(candidate_ids)
            lexical_id_set = {record.id for record in lexical_records}
            if candidate_id_set == lexical_id_set:
                return {
                    lexical_records[0].id: 0.91,
                    lexical_records[1].id: 0.9,
                }
            assert candidate_id_set == set(expected_fallback_ids)
            assert semantic_only.id in candidate_id_set
        return {
            semantic_only.id: 0.99,
            lexical_records[0].id: 0.91,
            lexical_records[1].id: 0.9,
            lexical_records[2].id: 0.89,
        }

    monkeypatch.setattr(repository, "list_memories", _fail_list_memories)
    monkeypatch.setattr(repository, "list_memory_ids", _record_list_memory_ids)
    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results, diagnostics = service.search_memories_with_diagnostics(
        "postgres backend storage latency",
        workspace_id="workspace-alpha",
        limit=3,
        debug=True,
    )

    assert len(results) == 3
    assert diagnostics.semantic_candidate_strategy == "global-fallback"
    assert (None, None, None, fallback_cap) in observed_list_memory_ids_calls


def test_search_memories_broadens_to_candidate_filtered_fallback_semantic_scores_when_speculative_bounded_scores_are_sparse(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=_CandidateFilteringVectorStore(),
    )

    lexical_records = []
    for index in range(3):
        record = repository.create_memory(
            title=f"Postgres backend note {index}",
            content="Postgres backend guidance for shared deployments.",
            summary="Postgres backend summary.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["postgres"],
        )
        assert record is not None
        lexical_records.append(record)

    for index in range(30):
        filler = repository.create_memory(
            title=f"Unrelated filler {index:02d}",
            content="Background note without lexical overlap.",
            summary=f"Filler summary {index:02d}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["filler"],
        )
        assert filler is not None

    semantic_only = repository.create_memory(
        title="Deployment incident diagnosis",
        content="Operational investigation note for shared production incidents.",
        summary="Incident diagnosis summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["incident"],
    )
    assert semantic_only is not None

    fallback_cap = _strong_keyword_bounded_candidate_cap(3)
    expected_fallback_ids = repository.list_memory_ids(limit=fallback_cap)
    assert fallback_cap == 20
    assert fallback_cap < 500
    assert semantic_only.id in expected_fallback_ids

    fallback_candidate_ids: list[str] = []

    def _semantic_scores(_query, candidates, _workspace_id, *, candidate_ids=None, limit):
        _ = candidates, limit
        if candidate_ids is not None:
            candidate_id_set = set(candidate_ids)
            lexical_id_set = {record.id for record in lexical_records}
            if candidate_id_set == lexical_id_set:
                return {
                    lexical_records[0].id: 0.91,
                    lexical_records[1].id: 0.9,
                }
            fallback_candidate_ids.extend(candidate_ids)
            assert candidate_id_set == set(expected_fallback_ids)
            assert semantic_only.id in candidate_id_set
        return {
            semantic_only.id: 0.99,
            lexical_records[0].id: 0.91,
            lexical_records[1].id: 0.9,
            lexical_records[2].id: 0.89,
        }

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results, diagnostics = service.search_memories_with_diagnostics(
        "postgres backend storage latency",
        workspace_id="workspace-alpha",
        limit=3,
        debug=True,
    )

    assert fallback_candidate_ids == expected_fallback_ids
    assert len(results) == 3
    assert diagnostics.semantic_candidate_strategy == "global-fallback"
    assert diagnostics.timing_ms["semantic_speculative_fallback"] >= 0.0


def test_workspace_scoped_search_keeps_global_semantic_candidates(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=SQLiteVectorStore(db_manager),
    )

    local_record = repository.create_memory(
        title="Workspace alpha identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls for alpha.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    remote_record = repository.create_memory(
        title="Workspace beta identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls for beta.",
        memory_type="fact",
        workspace_ids=["workspace-beta"],
        tags=["auth"],
    )
    assert local_record is not None and remote_record is not None

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)

    assert [result.memory_id for result in results[:2]] == [local_record.id, remote_record.id]


def test_search_memories_refreshes_stale_embeddings_for_current_model(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
    )

    record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
        created_at="2026-03-15T18:00:00+00:00",
        updated_at="2026-03-15T18:00:00+00:00",
    )
    assert record is not None

    vector_store.upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id="workspace-alpha",
        model_name=_FakeEmbedder.model_name,
        embedding=[0.0, 1.0],
    )
    stale_record = vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name=_FakeEmbedder.model_name,
    )
    assert stale_record is not None

    refreshed = repository.update_memory(
        record.id,
        content="Authentication token rotation and credential policy with permissions hardening.",
    )
    assert refreshed is not None

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)
    updated_record = vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name=_FakeEmbedder.model_name,
    )

    assert results
    assert results[0].memory_id == record.id
    assert updated_record is not None
    assert updated_record.updated_at > stale_record.updated_at
    assert updated_record.workspace_id is None
    assert updated_record.embedding == [1.0, 0.0]


def test_search_memories_caps_inline_embedding_repairs_to_recent_backlog_slice(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    embedder = _CountingFakeEmbedder()
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=embedder,
        vector_store=vector_store,
    )

    total_records = INLINE_EMBEDDING_REPAIR_LIMIT + 6
    newest_record_id: str | None = None
    oldest_record_id: str | None = None
    for index in range(total_records):
        record = repository.create_memory(
            title=f"Identity policy {index}",
            content="Authentication token rotation and credential policy.",
            summary=f"Identity controls {index}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["auth"],
            created_at=f"2026-03-{(index % 28) + 1:02d}T12:00:00+00:00",
            updated_at=f"2026-03-{(index % 28) + 1:02d}T12:00:00+00:00",
        )
        assert record is not None
        if index == 0:
            oldest_record_id = record.id
        if index == total_records - 1:
            newest_record_id = record.id

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)

    assert results
    assert embedder.batch_sizes[0] == INLINE_EMBEDDING_REPAIR_LIMIT
    assert newest_record_id is not None and oldest_record_id is not None
    newest_embedding = vector_store.get(
        source_kind="memory",
        source_id=newest_record_id,
        model_name=embedder.model_name,
    )
    oldest_embedding = vector_store.get(
        source_kind="memory",
        source_id=oldest_record_id,
        model_name=embedder.model_name,
    )

    assert newest_embedding is not None
    assert oldest_embedding is None


def test_search_memories_abstains_on_low_confidence_semantic_only_query(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(search_ranking=SearchRankingConfig(semantic_only_abstain_threshold=0.8)),
    )

    record = repository.create_memory(
        title="Provider routing note",
        content="Provider routing and admission control summary.",
        summary="Provider routing summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    def _low_confidence_semantic_scores(*args, **kwargs):
        return {record.id: 0.72}

    monkeypatch.setattr(service, "_semantic_scores", _low_confidence_semantic_scores)

    results = service.search_memories("quantum zebra croissant", workspace_id="workspace-alpha", limit=5)

    assert results == []
    refreshed = repository.get_memory(record.id)
    assert refreshed is not None
    assert refreshed.last_surfaced_at is None


def test_search_memories_preserves_strong_semantic_only_matches_above_abstain_threshold(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(search_ranking=SearchRankingConfig(semantic_only_abstain_threshold=0.8)),
    )

    record = repository.create_memory(
        title="Identity policy",
        content="Authentication token rotation and credential policy.",
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert record is not None

    def _high_confidence_semantic_scores(*args, **kwargs):
        return {record.id: 0.96}

    monkeypatch.setattr(service, "_semantic_scores", _high_confidence_semantic_scores)

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id


def test_search_memories_penalizes_semantic_only_candidates_when_keyword_matches_exist(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(search_ranking=SearchRankingConfig(semantic_only_keyword_penalty=0.4)),
    )

    exact = repository.create_memory(
        title="Ripgrep ban",
        content="Avoid broad ripgrep scans in large repositories.",
        summary="Ripgrep ban summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["grep"],
    )
    distractor = repository.create_memory(
        title="Provider routing note",
        content="Admission control and provider routing details.",
        summary="Provider routing summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert exact is not None and distractor is not None

    def _semantic_scores(_query, candidates, _workspace_id, *, limit):
        _ = candidates, limit
        return {
            distractor.id: 0.99,
            exact.id: 0.55,
        }

    monkeypatch.setattr(service, "_semantic_scores", _semantic_scores)

    results = service.search_memories("ripgrep ban", workspace_id="workspace-alpha", limit=5, debug=True)
    debug_by_id = {result.memory_id: result.ranking_debug for result in results}
    distractor_debug = debug_by_id[distractor.id]

    assert results
    assert results[0].memory_id == exact.id
    assert distractor_debug is not None
    assert float(distractor_debug["ranking_signal_multiplier"]) == pytest.approx(0.4)


def test_search_memories_penalizes_graph_only_expansions_against_direct_matches(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(search_ranking=SearchRankingConfig(graph_expansion_only_penalty=0.2)),
    )

    primary = repository.create_memory(
        title="Ripgrep ban",
        content="Avoid broad ripgrep scans in large repositories.",
        summary="Ripgrep ban summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["grep"],
    )
    direct = repository.create_memory(
        title="Ripgrep scanning guidelines",
        content="Ripgrep guidance for repository scans.",
        summary="Ripgrep guidance summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["grep"],
    )
    graph_only = repository.create_memory(
        title="Dashboard plan",
        content="Management dashboard planning note.",
        summary="Dashboard plan summary.",
        memory_type="plan",
        workspace_ids=["workspace-alpha"],
    )
    assert primary is not None and direct is not None and graph_only is not None

    repository.add_link(primary.id, graph_only.id, "DEPENDS_ON")

    results = service.search_memories("ripgrep ban", workspace_id="workspace-alpha", limit=5, debug=True)
    positions = {result.memory_id: index for index, result in enumerate(results)}
    graph_debug = next(result.ranking_debug for result in results if result.memory_id == graph_only.id)

    assert positions[primary.id] < positions[graph_only.id]
    assert positions[direct.id] < positions[graph_only.id]
    assert graph_debug is not None
    assert float(graph_debug["ranking_signal_multiplier"]) == pytest.approx(0.2)


def test_search_memories_prefers_summary_and_title_keyword_quality_over_content_only_overlap(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(repository, Config())

    strong = repository.create_memory(
        title="Ripgrep scanning guidance",
        content="Use targeted alternatives for large-repo scans.",
        summary="Ripgrep ban guidance for repository scans.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["grep"],
    )
    content_only = repository.create_memory(
        title="Ingest robustness note",
        content="This note mentions ripgrep once but mostly discusses unrelated queue behavior.",
        summary="Queue robustness summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
    )
    assert strong is not None and content_only is not None

    results = service.search_memories("ripgrep ban", workspace_id="workspace-alpha", limit=5, debug=True)
    debug_by_id = {result.memory_id: result.ranking_debug for result in results}
    strong_debug = debug_by_id[strong.id]
    content_only_debug = debug_by_id[content_only.id]

    assert results[0].memory_id == strong.id
    assert strong_debug is not None
    assert content_only_debug is not None
    assert float(strong_debug["keyword_token_coverage"]) > float(content_only_debug["keyword_token_coverage"])


def test_search_memories_can_expand_results_when_scores_stay_dense(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    config = Config(
        search_ranking=SearchRankingConfig(
            adaptive_result_max=6,
            adaptive_result_score_ratio_floor=0.7,
            adaptive_result_min_score=0.0,
            adaptive_result_max_score_gap=0.08,
        )
    )
    service = RelationalMemorySearchService(repository, config)

    records = []
    for index in range(6):
        record = repository.create_memory(
            title=f"Dense result {index}",
            content=f"Dense result content {index}.",
            summary=f"Dense result summary {index}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["dense"],
        )
        assert record is not None
        records.append(record)

    score_by_id = {
        record.id: score
        for record, score in zip(
            records,
            [0.92, 0.89, 0.86, 0.84, 0.81, 0.78],
            strict=False,
        )
    }

    monkeypatch.setattr(
        service,
        "_semantic_scores",
        lambda *_args, **_kwargs: score_by_id,
    )

    results = service.search_memories(
        "semantic neighborhood",
        workspace_id="workspace-alpha",
        limit=3,
        adaptive_limit=True,
    )

    assert [result.memory_id for result in results] == [record.id for record in records]


def test_search_memories_stops_adaptive_expansion_on_large_score_drop(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    config = Config(
        search_ranking=SearchRankingConfig(
            adaptive_result_max=6,
            adaptive_result_score_ratio_floor=0.96,
            adaptive_result_min_score=0.0,
            adaptive_result_max_score_gap=0.08,
        )
    )
    service = RelationalMemorySearchService(repository, config)

    records = []
    for index in range(6):
        record = repository.create_memory(
            title=f"Band result {index}",
            content=f"Band result content {index}.",
            summary=f"Band result summary {index}.",
            memory_type="fact",
            workspace_ids=["workspace-alpha"],
            tags=["band"],
        )
        assert record is not None
        records.append(record)

    score_by_id = {
        record.id: score
        for record, score in zip(
            records,
            [0.92, 0.89, 0.86, 0.75, 0.74, 0.73],
            strict=False,
        )
    }

    monkeypatch.setattr(
        service,
        "_semantic_scores",
        lambda *_args, **_kwargs: score_by_id,
    )

    results = service.search_memories(
        "semantic neighborhood",
        workspace_id="workspace-alpha",
        limit=3,
        adaptive_limit=True,
    )

    assert [result.memory_id for result in results] == [record.id for record in records[:3]]


def test_resolved_result_limit_keeps_requested_limit_when_adaptive_mode_is_disabled(db_manager) -> None:
    service = RelationalMemorySearchService(RelationalMemoryRepository(db_manager), Config())
    ranked = [
        RelationalSearchResult("memory-1", "Title", "Summary", "fact", "active", score=0.9),
        RelationalSearchResult("memory-2", "Title", "Summary", "fact", "active", score=0.8),
    ]

    assert service._resolved_result_limit(ranked, requested_limit=1, adaptive_limit=False) == 1


def test_resolved_result_limit_stops_at_absolute_score_floor(db_manager) -> None:
    service = RelationalMemorySearchService(
        RelationalMemoryRepository(db_manager),
        Config(
            search_ranking=SearchRankingConfig(
                adaptive_result_max=6,
                adaptive_result_score_ratio_floor=0.0,
                adaptive_result_min_score=0.7,
                adaptive_result_max_score_gap=1.0,
            )
        ),
    )
    ranked = [
        RelationalSearchResult("memory-1", "Title", "Summary", "fact", "active", score=0.9),
        RelationalSearchResult("memory-2", "Title", "Summary", "fact", "active", score=0.8),
        RelationalSearchResult("memory-3", "Title", "Summary", "fact", "active", score=0.6),
    ]

    assert service._resolved_result_limit(ranked, requested_limit=1, adaptive_limit=True) == 2


def test_resolved_result_limit_respects_adaptive_cap(db_manager) -> None:
    service = RelationalMemorySearchService(
        RelationalMemoryRepository(db_manager),
        Config(
            search_ranking=SearchRankingConfig(
                adaptive_result_max=3,
                adaptive_result_score_ratio_floor=0.0,
                adaptive_result_min_score=0.0,
                adaptive_result_max_score_gap=1.0,
            )
        ),
    )
    ranked = [
        RelationalSearchResult(f"memory-{index}", "Title", "Summary", "fact", "active", score=score)
        for index, score in enumerate([0.9, 0.89, 0.88, 0.87])
    ]

    assert service._resolved_result_limit(ranked, requested_limit=1, adaptive_limit=True) == 3


@pytest.mark.parametrize("requested_limit", [0, -3])
def test_resolved_result_limit_bounds_non_positive_requested_limits(db_manager, requested_limit: int) -> None:
    service = RelationalMemorySearchService(RelationalMemoryRepository(db_manager), Config())
    ranked = [
        RelationalSearchResult("memory-1", "Title", "Summary", "fact", "active", score=0.9),
        RelationalSearchResult("memory-2", "Title", "Summary", "fact", "active", score=0.8),
    ]

    assert service._resolved_result_limit(ranked, requested_limit=requested_limit, adaptive_limit=False) == 1


def test_search_memories_falls_back_to_keyword_results_when_vector_search_fails(db_manager, monkeypatch, caplog) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
    )

    record = repository.create_memory(
        title="Auth rollout note",
        content="Auth rollout note with lexical search terms.",
        summary="Auth rollout summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    original_search = vector_store.search

    def _failing_search(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(vector_store, "search", _failing_search)

    with caplog.at_level("WARNING"):
        results = service.search_memories("auth rollout", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id
    assert any("Semantic search unavailable; falling back to keyword-only ranking" in message for message in caplog.messages)

    monkeypatch.setattr(vector_store, "search", original_search)


def test_search_memories_falls_back_to_keyword_results_when_embedding_refresh_fails(db_manager, monkeypatch, caplog) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
    )

    record = repository.create_memory(
        title="Permission rollout note",
        content="Permission rollout note with lexical search terms.",
        summary="Permission rollout summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    def _failing_upsert(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(vector_store, "upsert", _failing_upsert)

    with caplog.at_level("WARNING"):
        results = service.search_memories("permission rollout", workspace_id="workspace-alpha", limit=5)

    assert results
    assert results[0].memory_id == record.id
    assert any("Semantic search unavailable; falling back to keyword-only ranking" in message for message in caplog.messages)


def test_search_startup_health_check_marks_semantic_path_unavailable_on_io_error(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
        db_manager=db_manager,
    )

    def _failing_search(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(vector_store, "search", _failing_search)

    health = service.run_startup_health_check()

    assert health.semantic_enabled is True
    assert health.available is False
    assert health.integrity_check_error == "disk I/O error"
    assert health.last_integrity_check_at is not None


def test_search_memories_retries_after_reopening_db_connection(db_manager, monkeypatch) -> None:
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
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    original_search = vector_store.search
    call_count = 0

    def _flaky_search(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return original_search(*args, **kwargs)

    monkeypatch.setattr(vector_store, "search", _flaky_search)

    results = service.search_memories("permissions security", workspace_id="workspace-alpha", limit=5)
    health = service.get_health()

    assert results
    assert results[0].memory_id == record.id
    assert call_count == 2
    assert health.available is True
    assert health.degraded is False
    assert health.last_recovery_at is not None


def test_rebuild_semantic_index_recreates_embeddings_for_current_model(db_manager) -> None:
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
        summary="Identity controls.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["auth"],
    )
    assert record is not None

    result = service.rebuild_semantic_index()
    stored = vector_store.get(
        source_kind="memory",
        source_id=record.id,
        model_name=_FakeEmbedder.model_name,
    )
    health = service.get_health()

    assert result["rebuilt"] is True
    assert isinstance(result["records_indexed"], int)
    assert result["records_indexed"] >= 1
    assert stored is not None
    assert health.rebuild_count == 1
    assert health.available is True


def test_search_memories_falls_back_without_queueing_repairs_when_fallback_persistence_is_blocked(db_manager, caplog) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    work_items = SQLiteWorkItemRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_BlockedFallbackEmbedder(),
        vector_store=SimpleNamespace(
            get_write_policy_state=lambda: SimpleNamespace(
                fallback_persistence_policy="blocked",
                blocked_fallback_write_count=0,
                last_blocked_fallback_model_name=None,
            )
        ),
        task_queue=task_queue,
        work_items=work_items,
        background_repair_wait_seconds=1.0,
    )

    record = repository.create_memory(
        title="Postgres shared-mode note",
        content="Fallback models should degrade to keyword search without scheduling repair work.",
        summary="Shared-mode fallback handling.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["postgres"],
    )
    assert record is not None

    with caplog.at_level("WARNING"):
        results = service.search_memories("shared-mode fallback", workspace_id="workspace-alpha", limit=5)

    queued_work_item_count = db_manager.get_connection().execute("SELECT COUNT(*) FROM work_items").fetchone()[0]

    assert [result.memory_id for result in results] == [record.id]
    assert task_queue.list_tasks(status=None, workspace_id=None, limit=10) == []
    assert queued_work_item_count == 0
    assert any("Semantic search unavailable; falling back to keyword-only ranking" in message for message in caplog.messages)


def test_rebuild_semantic_index_reports_blocked_fallback_persistence_policy(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_BlockedFallbackEmbedder(),
        vector_store=SimpleNamespace(
            get_write_policy_state=lambda: SimpleNamespace(
                fallback_persistence_policy="blocked",
                blocked_fallback_write_count=0,
                last_blocked_fallback_model_name=None,
            )
        ),
        db_manager=db_manager,
    )

    record = repository.create_memory(
        title="Blocked rebuild note",
        content="Rebuild should report the policy block instead of raising.",
        summary="Rebuild policy block.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["postgres"],
    )
    assert record is not None

    result = service.rebuild_semantic_index()
    health = service.get_health()

    assert result == {
        "semantic_enabled": True,
        "rebuilt": False,
        "records_indexed": 0,
        "reason": "fallback_embedding_persistence_blocked",
    }
    assert health.available is False
    assert health.degraded is True
    assert health.fallback_count == 1
    assert health.last_error == (
        "Fallback/hash embeddings cannot be persisted in Postgres/shared mode; "
        "blocked model_name 'hash:sentence-transformers/all-MiniLM-L6-v2'"
    )


def test_stale_embedding_detection_uses_bulk_updated_at_lookup_when_available(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    vector_store = SQLiteVectorStore(db_manager)
    service = RelationalMemorySearchService(
        repository,
        Config(),
        embedder=_FakeEmbedder(),
        vector_store=vector_store,
    )

    record = repository.create_memory(
        title="Bulk embedding lookup",
        content="Use one query instead of N+1 embedding lookups.",
        summary="Bulk lookup summary.",
        memory_type="fact",
        workspace_ids=["workspace-alpha"],
        tags=["search"],
        updated_at="2026-03-28T00:00:00+00:00",
    )
    assert record is not None
    vector_store.upsert(
        source_kind="memory",
        source_id=record.id,
        workspace_id=None,
        model_name=_FakeEmbedder.model_name,
        embedding=[0.1, 0.2],
    )

    get_called = False

    def _fail_if_called(**_kwargs):
        nonlocal get_called
        get_called = True
        raise AssertionError("per-record get should not be used when bulk lookup exists")

    monkeypatch.setattr(vector_store, "get", _fail_if_called)

    stale = service._stale_or_missing_embedding_candidates([record])

    assert stale == []
    assert get_called is False

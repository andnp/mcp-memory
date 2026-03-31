import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRecord, RelationalMemoryRepository
from mcp_memory.relational.search import (
    INLINE_EMBEDDING_REPAIR_LIMIT,
    RankingEngine,
    RelationalMemorySearchService,
    RelationalSearchResult,
    SearchExecutionDiagnostics,
    ScoringWeights,
    _keyword_token_coverage,
    _query_tokens,
    _rank_semantic_candidate_ids,
    _strong_keyword_bounded_candidate_cap,
)
from mcp_memory.mcp.services import search_memory_records_service
from mcp_memory.context import ApplicationContext


pytestmark = pytest.mark.small


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


class _CandidateFilteringVectorStore:
    supports_candidate_filtering = True


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
    expected_fallback_ids = [record.id for record in repository.list_memories(limit=fallback_cap)]
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

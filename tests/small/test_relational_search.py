import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRecord, RelationalMemoryRepository
from mcp_memory.relational.search import RankingEngine, RelationalMemorySearchService, ScoringWeights, _rank_semantic_candidate_ids


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
    assert updated_record.embedding == [1.0, 0.0]


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

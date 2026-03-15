from datetime import datetime, timedelta, timezone

import pytest

from mcp_memory.config import Config
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RankingEngine, RelationalMemorySearchService, ScoringWeights


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

    assert [record.id for record in workspace_results] == [alpha.id, beta.id]
    assert [record.read_count for record in workspace_results] == [2, 1]
    assert global_results[0].id == alpha.id
    assert global_results[0].read_count == 2
    assert {record.id for record in global_results[1:3]} == {beta.id, gamma.id}
    assert {record.read_count for record in global_results[1:3]} == {1}


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
        "auth decision",
        workspace_id="workspace-alpha",
        memory_type="fact",
        limit=5,
    )

    assert [result.memory_id for result in results[:2]] == [authority.id, peer.id]


def test_search_memories_upweights_memory_type_without_filtering(db_manager) -> None:
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

    assert [result.memory_id for result in results[:2]] == [preferred_fact.id, related_plan.id]
    assert {result.memory_type for result in results[:2]} == {"fact", "plan"}


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

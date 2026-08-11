from __future__ import annotations

from collections import Counter
from typing import cast

import pytest
from searchkernel.runtime import QueryEmbeddingCache
from searchkernel.search.record_pipeline import RecordSearchConfig

import mcp_memory.integrations.searchkernel_record_pipeline as record_pipeline
from mcp_memory.config import Config, SearchKernelConfig, SearchRankingConfig
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryReadContext,
    MemoryRecord,
    MemoryRepositoryPort,
    RankedMemoryCandidate,
)
from mcp_memory.core.search_ranking import RankingEngine, RankingSignals
from mcp_memory.integrations.memory_retrieval import build_search_execution_diagnostics
from mcp_memory.integrations.searchkernel_adapters import MemoryVectorBackend
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX,
    build_memory_record_pipeline,
)

pytestmark = pytest.mark.small


def _memory(
    memory_id: str,
    *,
    status: str = "active",
    workspace_ids: list[str] | None = None,
    title: str | None = None,
    content: str | None = None,
    summary: str | None = "summary",
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=title or memory_id,
        content=content or f"body {memory_id}",
        summary=summary,
        type="fact",
        status=status,
        created_at="2026-07-30T12:00:00+00:00",
        updated_at="2026-07-30T13:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=workspace_ids or ["workspace-1"],
        tags=tags or [],
    )


class FakeRepository:
    def __init__(
        self,
        records: dict[str, MemoryRecord] | None = None,
        keyword_ids: list[str] | None = None,
    ) -> None:
        self.records = records or {
            "active": _memory("active"),
            "other-workspace": _memory("other-workspace", workspace_ids=["workspace-2"]),
            "archived": _memory("archived", status="archived"),
            "superseded": _memory("superseded"),
        }
        self.keyword_ids = keyword_ids
        self.ranking_candidate_calls = 0
        self.links = [
            MemoryLink("active", "superseded", "SUPERSEDES", ""),
        ]

    def search_keyword_memory_ids(self, query: str, **kwargs: object) -> list[str]:
        return list(self.records) if self.keyword_ids is None else list(self.keyword_ids)

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.records.get(memory_id)

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RankedMemoryCandidate]:
        self.ranking_candidate_calls += 1
        candidates: list[RankedMemoryCandidate] = []
        for memory_id in memory_ids:
            record = self.records.get(memory_id)
            if record is None:
                continue
            incoming = [link for link in self.links if link.target_id == memory_id]
            counts: dict[str, int] = {}
            for link in incoming:
                counts[link.link_type] = counts.get(link.link_type, 0) + 1
            candidates.append(
                RankedMemoryCandidate(
                    record=record,
                    incoming_links_count=len(incoming),
                    has_incoming_supersedes=any(
                        link.link_type == "SUPERSEDES" for link in incoming
                    ),
                    incoming_link_type_counts=counts,
                )
            )
        return candidates

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        links = [
            link
            for link in self.links
            if (link.source_id if direction == "outgoing" else link.target_id) == memory_id
        ]
        if link_type is not None:
            links = [link for link in links if link.link_type == link_type]
        return links

    def peek_memory(self, memory_id: str) -> MemoryReadContext | None:
        record = self.get_memory(memory_id)
        if record is None:
            return None
        return MemoryReadContext(record, {"outgoing": self.get_links(memory_id)}, [])


class EpochRepository(FakeRepository):
    def get_search_epochs(self) -> dict[str, int]:
        return {"keyword": 1, "vector": 1, "graph": 1}


class CountingRepository(FakeRepository):
    def __init__(self) -> None:
        super().__init__(keyword_ids=["active"])
        self.memory_calls: Counter[str] = Counter()
        self.link_calls: Counter[tuple[str, str, str | None]] = Counter()

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        self.memory_calls[memory_id] += 1
        return super().get_memory(memory_id)

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        self.link_calls[(memory_id, direction, link_type)] += 1
        return super().get_links(memory_id, direction, link_type)


class RankingRepository(FakeRepository):
    def __init__(self) -> None:
        super().__init__()
        accessed = _memory("accessed")
        accessed.access_score = 25.0
        accessed.last_accessed_at = "2026-07-30T12:00:00+00:00"
        self.records = {
            "plain": _memory("plain"),
            "accessed": accessed,
            "authority": _memory("authority"),
            "stale": _memory("stale", status="stale"),
        }
        self.links = [
            MemoryLink("supporter", "authority", "DEPENDS_ON", ""),
        ]


class FakeVectorStore:
    supports_candidate_filtering = True

    def __init__(self, results: list[tuple[str, float]] | None = None) -> None:
        self.search_count = 0
        self.write_count = 0
        self.search_limits: list[int] = []
        self.search_filters: list[dict[str, object] | None] = []
        self.search_candidate_ids: list[list[str] | None] = []
        self.results = results

    def upsert(self, **kwargs: object) -> bool:
        self.write_count += 1
        return True

    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        self.search_count += 1
        limit = kwargs.get("limit")
        assert isinstance(limit, int)
        self.search_limits.append(limit)
        filters = kwargs.get("filters")
        assert filters is None or isinstance(filters, dict)
        self.search_filters.append(filters)
        candidate_ids = kwargs.get("candidate_ids")
        assert candidate_ids is None or isinstance(candidate_ids, list)
        self.search_candidate_ids.append(candidate_ids)
        return self.results or [
            ("active", 1.0),
            ("other-workspace", 0.9),
            ("archived", 0.8),
            ("superseded", 0.7),
        ]

    def delete(self, **kwargs: object) -> int:
        self.write_count += 1
        return 1


class FailingVectorStore(FakeVectorStore):
    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        raise RuntimeError("vector backend unavailable")


class FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


class FailingQueryEmbeddingCache:
    async def async_get_or_compute(self, **kwargs: object) -> list[float]:
        del kwargs
        raise RuntimeError("cache unavailable")


@pytest.mark.asyncio
async def test_pipeline_reuses_injected_query_embedding_cache() -> None:
    """Share query vectors across repeated searches on one record pipeline."""
    repository = FakeRepository()
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    cache = QueryEmbeddingCache()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=embedder,
        query_embedding_cache=cache,
    )

    await pipeline.search("repeated query", limit=1)
    await pipeline.search("repeated query", limit=1)

    assert embedder.queries == ["repeated query"]
    assert cache.metrics.misses == 1
    assert cache.metrics.hits == 1


@pytest.mark.asyncio
async def test_pipeline_bypasses_query_embedding_cache_failure() -> None:
    """Keep semantic search available when the query cache is unavailable."""
    repository = FakeRepository()
    embedder = FakeEmbedder()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FakeVectorStore()),
        embedder=embedder,
        query_embedding_cache=cast(QueryEmbeddingCache, FailingQueryEmbeddingCache()),
    )

    outcome = await pipeline.search("cache failure query", limit=1)

    assert [result.record_id for result in outcome.results] == ["active"]
    assert embedder.queries == ["cache failure query"]


@pytest.mark.asyncio
async def test_composition_applies_memory_policy_without_writes() -> None:
    repository = FakeRepository()
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=embedder,
    )

    outcome = await pipeline.search(
        "query",
        limit=10,
        filters={
            "workspace_id": "workspace-1",
            "status": "active",
            "include_superseded": False,
        },
    )

    assert [result.record_id for result in outcome.results] == ["active"]
    assert vector_store.search_count == 1
    assert vector_store.search_limits == [50]
    assert vector_store.write_count == 0
    assert embedder.queries == ["query"]


@pytest.mark.asyncio
async def test_artifact_queries_keep_vector_lane_for_memory_eligibility() -> None:
    """Retain semantic retrieval while workspace and lifecycle filters apply."""
    repository = FakeRepository()
    vector_store = FakeVectorStore()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search(
        "release v1",
        limit=1,
        filters={
            "workspace_id": "workspace-1",
            "status": "active",
            "include_superseded": False,
        },
    )

    assert [result.record_id for result in outcome.results] == ["active"]
    assert vector_store.search_count == 1
    assert "vector:artifact_keyword_confident" not in outcome.diagnostics


@pytest.mark.asyncio
async def test_candidate_cache_requires_authoritative_epochs() -> None:
    pipeline_without_epochs = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", FakeRepository())
    )
    without_epochs = await pipeline_without_epochs.search("query", limit=1)
    assert any(
        diagnostic.startswith("candidate_cache:bypass:")
        for diagnostic in without_epochs.cache_diagnostics
    )

    pipeline = build_memory_record_pipeline(cast("MemoryRepositoryPort", EpochRepository()))
    first = await pipeline.search("query", limit=1)
    second = await pipeline.search("query", limit=1)

    assert "candidate_cache:miss" in first.cache_diagnostics
    assert "candidate_cache:hit" in second.cache_diagnostics


@pytest.mark.asyncio
async def test_strong_keyword_matches_bound_vector_candidates() -> None:
    """Bound semantic candidates when the keyword match is strong."""
    repository = FakeRepository(keyword_ids=["active"])
    vector_store = FakeVectorStore()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
    )

    await pipeline.search(
        "active",
        limit=3,
        filters={"workspace_id": "workspace-1"},
    )

    assert vector_store.search_candidate_ids == [["active"]]


@pytest.mark.asyncio
async def test_near_token_keyword_match_does_not_bound_vector_candidates() -> None:
    near_token = _memory(
        "near-token",
        title="authentication policy",
        summary="authentication policy",
        content="Authentication policy details.",
    )
    semantic = _memory(
        "semantic",
        title="credential controls",
        summary="credential controls",
        content="Credential controls for auth.",
    )
    repository = FakeRepository(
        records={near_token.id: near_token, semantic.id: semantic},
        keyword_ids=[near_token.id],
    )
    vector_store = FakeVectorStore([(semantic.id, 0.95)])
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
    )

    await pipeline.search("auth", limit=1)

    assert vector_store.search_candidate_ids == [None]


@pytest.mark.asyncio
async def test_superseded_policy_is_expressible() -> None:
    repository = FakeRepository()
    pipeline = build_memory_record_pipeline(cast("MemoryRepositoryPort", repository))

    excluded = await pipeline.search("query", limit=10)
    included = await pipeline.search(
        "query",
        limit=10,
        filters={"include_superseded": True},
    )

    assert "superseded" not in {result.record_id for result in excluded.results}
    assert "superseded" in {result.record_id for result in included.results}


@pytest.mark.asyncio
async def test_missing_vector_or_embedder_degrades_to_keyword_pipeline() -> None:
    repository = FakeRepository()
    without_vector = build_memory_record_pipeline(cast("MemoryRepositoryPort", repository))
    without_embedder = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FakeVectorStore()),
    )

    assert without_vector.diagnostics.reasons
    assert without_embedder.diagnostics.reasons
    assert (await without_vector.search("query", limit=1)).results
    assert (await without_embedder.search("query", limit=1)).results


@pytest.mark.asyncio
async def test_diagnostic_planner_reports_keyword_only_lane() -> None:
    """Expose the planner's keyword-only decision when semantic retrieval is absent."""
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", FakeRepository(keyword_ids=["active"]))
    )

    outcome = await pipeline.search("active", limit=1)
    diagnostics = build_search_execution_diagnostics(
        outcome,
        workspace_id=None,
        ranking_workspace_id=None,
    )

    assert diagnostics.lane_decisions is not None
    assert diagnostics.lane_decisions["enabled"] == ["keyword"]
    assert diagnostics.lane_decisions["budgets"] == {
        "keyword": 50,
        "vector": 50,
        "graph_seeds": 3,
        "rerank": 0,
    }
    assert diagnostics.lane_decisions["skipped"] == ["vector:unavailable"]
    assert diagnostics.candidate_counts == {"keyword": 1}


@pytest.mark.asyncio
async def test_diagnostic_planner_reports_vector_only_lane() -> None:
    """Expose the forced semantic-only decision and its disabled keyword lane."""
    repository = FakeRepository(
        records={"semantic": _memory("semantic")},
        keyword_ids=[],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FakeVectorStore([("semantic", 0.95)])),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search(
        "semantic query",
        limit=1,
        filters={"retrieval_mode": "semantic_only"},
    )
    diagnostics = build_search_execution_diagnostics(
        outcome,
        workspace_id=None,
        ranking_workspace_id=None,
    )

    assert diagnostics.lane_decisions is not None
    assert diagnostics.lane_decisions["enabled"] == ["vector"]
    skipped = diagnostics.lane_decisions["skipped"]
    assert isinstance(skipped, list)
    assert "keyword:unavailable" in skipped
    assert diagnostics.candidate_counts == {"vector": 1}


@pytest.mark.asyncio
async def test_diagnostic_planner_reports_hybrid_lanes() -> None:
    """Expose both retrieval lanes for a query that does not force a single lane."""
    repository = FakeRepository(
        records={
            "active": _memory("active"),
            "semantic": _memory("semantic"),
        },
        keyword_ids=["active"],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FakeVectorStore([("semantic", 0.95)])),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search("unmatched query", limit=2)
    diagnostics = build_search_execution_diagnostics(
        outcome,
        workspace_id=None,
        ranking_workspace_id=None,
    )

    assert diagnostics.lane_decisions is not None
    enabled = diagnostics.lane_decisions["enabled"]
    assert isinstance(enabled, list)
    assert set(enabled) >= {"keyword", "vector"}
    assert diagnostics.candidate_counts is not None
    assert set(diagnostics.candidate_counts) >= {"keyword", "vector"}


@pytest.mark.asyncio
async def test_score_adjustment_matches_relational_ranking_engine() -> None:
    repository = RankingRepository()
    config = Config(
        search_ranking=SearchRankingConfig(
            workspace_multiplier=1.4,
            degradation_multiplier=0.25,
        )
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        config=config,
    )

    results = (await pipeline.search(
        "query",
        limit=10,
        filters={"workspace_id": "workspace-1"},
    )).results
    engine = RankingEngine(config)
    expected = {
        record.id: score
        for record, score in engine.rank_records(
            repository.get_ranking_candidates(list(repository.records)),
            {
                memory_id: 1.0 / (config.search_ranking.rrf_k + rank + 1)
                for rank, memory_id in enumerate(
                    ["plain", "accessed", "authority", "stale"]
                )
            },
            "workspace-1",
            ranking_signals={
                memory_id: RankingSignals(
                    matched_by_keyword=True,
                    keyword_token_coverage=0.0,
                )
                for memory_id in repository.records
            },
            keyword_candidates_present=True,
        )
    }

    assert {result.record_id: result.score for result in results} == pytest.approx(expected)
    scores = {result.record_id: result.score for result in results}
    assert scores["accessed"] > scores["plain"]
    assert scores["authority"] > scores["plain"]
    assert scores["stale"] < scores["plain"]


@pytest.mark.asyncio
async def test_pipeline_ranking_is_read_only() -> None:
    repository = RankingRepository()
    before = {
        memory_id: (
            record.access_score,
            record.last_accessed_at,
            record.last_surfaced_at,
        )
        for memory_id, record in repository.records.items()
    }

    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
    )
    await pipeline.search("query", limit=10)

    after = {
        memory_id: (
            record.access_score,
            record.last_accessed_at,
            record.last_surfaced_at,
        )
        for memory_id, record in repository.records.items()
    }
    assert after == before


def test_pipeline_matches_native_graph_expansion_bounds() -> None:
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", FakeRepository()),
    )

    kernel_config = pipeline._pipeline._config

    assert kernel_config.max_graph_seeds == 3
    assert kernel_config.max_neighbors_per_seed == 10
    assert kernel_config.adaptive_graph_enabled is True


def test_pipeline_uses_searchkernel_failure_mode() -> None:
    """Keep failure-mode wiring compatible with existing searchkernel policy."""
    repository = FakeRepository()

    lenient = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
    )
    strict = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        config=Config(searchkernel=SearchKernelConfig(failure_mode="strict")),
    )

    assert lenient._pipeline._config.failure_mode == "lenient"
    assert strict._pipeline._config.failure_mode == "strict"


def test_pipeline_keeps_advanced_searchkernel_policies_disabled() -> None:
    """Preserve baseline fusion, expansion, reranking, and routing identity."""
    pipeline = build_memory_record_pipeline(cast("MemoryRepositoryPort", FakeRepository()))

    kernel_config = pipeline._pipeline._config

    assert kernel_config.fusion_mode == "rrf"
    assert kernel_config.expansion_enabled is False
    assert kernel_config.synonym_expansion_enabled is False
    assert kernel_config.rerank_budget == 0
    assert pipeline._pipeline._routing_fingerprint == "record-search-v1"
    assert kernel_config.artifact_confidence_threshold == (
        RecordSearchConfig().artifact_confidence_threshold
    )


@pytest.mark.asyncio
async def test_pipeline_uses_candidate_eligibility_hook_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass memory eligibility to SearchKernel's released policy seam."""
    captured: dict[str, object] = {}

    class CompatiblePolicy:
        def __init__(
            self,
            *,
            query_candidate_set_eligible: object = None,
            **kwargs: object,
        ) -> None:
            captured["query_candidate_set_eligible"] = query_candidate_set_eligible
            for name, value in kwargs.items():
                setattr(self, name, value)
            self.query_candidate_filter = None
            self.query_candidate_set_eligible = query_candidate_set_eligible
            self.score_adjuster = None
            self.parent_expander = None

    monkeypatch.setattr(record_pipeline, "RecordSearchPolicy", CompatiblePolicy)
    repository = FakeRepository(keyword_ids=["active"])
    vector_store = FakeVectorStore()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
    )
    outcome = await pipeline.search(
        "src/searchkernel.py",
        limit=1,
        filters={
            "workspace_id": "workspace-1",
            "status": "active",
            "include_superseded": False,
        },
    )

    assert callable(captured["query_candidate_set_eligible"])
    assert vector_store.search_count == 0
    assert "vector:artifact_keyword_confident" in outcome.diagnostics


def test_pipeline_applies_enabled_advanced_searchkernel_policies() -> None:
    """Thread enabled policies without weakening the eligibility safeguard."""
    config = Config(
        searchkernel=SearchKernelConfig(
            calibrated_fusion_enabled=True,
            query_expansion_enabled=True,
            query_expansion_policy="synonym",
            rerank_policy="default",
            rerank_budget=4,
        )
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", FakeRepository()),
        config=config,
    )

    kernel_config = pipeline._pipeline._config

    assert kernel_config.fusion_mode == "calibrated"
    assert kernel_config.expansion_enabled is False
    assert kernel_config.synonym_expansion_enabled is True
    assert kernel_config.rerank_budget == 4
    assert pipeline._pipeline._routing_fingerprint == (
        "record-search-v1:calibrated-fusion;query-expansion:synonym;rerank:default:4"
    )
    assert kernel_config.artifact_confidence_threshold == (
        RecordSearchConfig().artifact_confidence_threshold
    )


@pytest.mark.asyncio
async def test_policy_lookups_are_cached_for_one_search() -> None:
    repository = CountingRepository()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
    )

    outcome = await pipeline.search("depends query", limit=1)

    assert [result.record_id for result in outcome.results] == ["active"]
    assert repository.ranking_candidate_calls == 1
    assert repository.memory_calls["active"] == 0
    assert repository.link_calls[("active", "incoming", None)] == 0
    assert repository.link_calls[("active", "outgoing", None)] == 1


@pytest.mark.asyncio
async def test_policy_prefetch_avoids_rehydrating_overlapping_rankings() -> None:
    repository = CountingRepository()
    vector_store = FakeVectorStore([("active", 1.0)])
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search("query", limit=1)

    assert [result.record_id for result in outcome.results] == ["active"]
    assert repository.ranking_candidate_calls == 1


@pytest.mark.asyncio
async def test_hydration_reuses_search_scoped_record_cache() -> None:
    repository = CountingRepository()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
    )

    outcome = await pipeline.search("query", limit=1)

    assert [result.record_id for result in outcome.results] == ["active"]
    assert repository.ranking_candidate_calls == 1
    assert repository.memory_calls["active"] == 0


@pytest.mark.asyncio
async def test_keyword_signal_uses_partial_and_full_token_coverage() -> None:
    partial = _memory(
        "partial",
        title="alpha",
        summary="alpha",
        content="alpha",
    )
    full = _memory(
        "full",
        title="alpha beta",
        summary="alpha beta",
        content="alpha beta",
        tags=["alpha", "beta"],
    )
    repository = FakeRepository(
        records={"partial": partial, "full": full},
        keyword_ids=["partial", "full"],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        config=Config(
            search_ranking=SearchRankingConfig(
                keyword_coverage_floor=0.6,
                keyword_low_coverage_penalty=0.4,
            )
        ),
    )
    results = await pipeline.search("alpha beta", limit=2)

    scores = {result.record_id: result.score for result in results.results}
    assert scores["full"] > scores["partial"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("semantic_score", "expected_ids", "expected_rejected"),
    [
        (0.72, [], 1),
        (0.96, ["semantic"], 0),
    ],
)
async def test_semantic_only_abstention_uses_vector_raw_score(
    semantic_score: float,
    expected_ids: list[str],
    expected_rejected: int,
) -> None:
    """Report semantic-only candidates rejected by the raw-score policy."""
    repository = FakeRepository(
        records={"semantic": _memory("semantic")},
        keyword_ids=[],
    )
    vector_store = FakeVectorStore([("semantic", semantic_score)])
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=FakeEmbedder(),
        config=Config(
            search_ranking=SearchRankingConfig(
                semantic_only_abstain_threshold=0.8,
            )
        ),
    )
    outcome = await pipeline.search("unmatched query", limit=2)

    assert [result.record_id for result in outcome.results] == expected_ids
    assert (
        f"{MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX}"
        f"semantic_candidates=1;semantic_only_candidates=1;rejected={expected_rejected}"
    ) in outcome.diagnostics
    assert vector_store.write_count == 0


@pytest.mark.asyncio
async def test_semantic_only_filter_drops_low_confidence_tail() -> None:
    repository = FakeRepository(
        records={
            "strong": _memory("strong"),
            "weak": _memory("weak"),
        },
        keyword_ids=[],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast(
            "MemoryVectorBackend",
            FakeVectorStore([("strong", 0.96), ("weak", 0.72)]),
        ),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search("unmatched query", limit=5)

    assert [result.record_id for result in outcome.results] == ["strong"]


@pytest.mark.asyncio
async def test_keyword_match_survives_low_confidence_semantic_control() -> None:
    relevant = _memory(
        "relevant",
        title="auth policy",
        summary="auth policy",
        content="Auth policy details.",
    )
    distractor = _memory("distractor")
    repository = FakeRepository(
        records={"relevant": relevant, "distractor": distractor},
        keyword_ids=["relevant"],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast(
            "MemoryVectorBackend",
            FakeVectorStore([("distractor", 0.72), ("relevant", 0.55)]),
        ),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search("auth policy rollout", limit=5)

    assert [result.record_id for result in outcome.results] == ["relevant"]


@pytest.mark.asyncio
async def test_mixed_keyword_and_semantic_results_keep_keyword_match() -> None:
    exact = _memory(
        "exact",
        title="ripgrep ban",
        summary="ripgrep ban",
        content="Avoid broad ripgrep scans.",
        tags=["ripgrep", "ban"],
    )
    distractor = _memory("distractor")
    repository = FakeRepository(
        records={"exact": exact, "distractor": distractor},
        keyword_ids=["exact"],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast(
            "MemoryVectorBackend",
            FakeVectorStore([("distractor", 0.99), ("exact", 0.55)]),
        ),
        embedder=FakeEmbedder(),
        config=Config(
            search_ranking=SearchRankingConfig(
                semantic_only_keyword_penalty=0.4,
            )
        ),
    )
    results = await pipeline.search("ripgrep ban", limit=2)

    assert {result.record_id for result in results.results} == {"exact", "distractor"}
    assert results.results[0].record_id == "exact"


@pytest.mark.asyncio
async def test_keyword_match_survives_semantic_only_abstention_threshold() -> None:
    """Exclude keyword-supported semantic candidates from abstention counts."""
    exact = _memory(
        "exact",
        title="ripgrep ban",
        summary="ripgrep ban",
        content="Avoid broad ripgrep scans.",
        tags=["ripgrep", "ban"],
    )
    repository = FakeRepository(
        records={"exact": exact},
        keyword_ids=["exact"],
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast(
            "MemoryVectorBackend",
            FakeVectorStore([("exact", 0.72)]),
        ),
        embedder=FakeEmbedder(),
        config=Config(
            search_ranking=SearchRankingConfig(
                semantic_only_abstain_threshold=0.8,
            )
        ),
    )
    outcome = await pipeline.search("ripgrep ban", limit=1)

    assert [result.record_id for result in outcome.results] == ["exact"]
    assert (
        f"{MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX}"
        "semantic_candidates=1;semantic_only_candidates=0;rejected=0"
    ) in outcome.diagnostics


@pytest.mark.asyncio
async def test_degraded_semantic_search_does_not_report_abstention() -> None:
    """Keep backend degradation separate from semantic abstention outcomes."""
    repository = FakeRepository(keyword_ids=[])
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FailingVectorStore()),
        embedder=FakeEmbedder(),
    )

    outcome = await pipeline.search("unmatched query", limit=2)

    assert outcome.degraded is True
    assert (
        f"{MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX}"
        "semantic_candidates=0;semantic_only_candidates=0;rejected=0"
    ) in outcome.diagnostics

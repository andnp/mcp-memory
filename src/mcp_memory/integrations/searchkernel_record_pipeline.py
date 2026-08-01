"""Memory-owned composition for searchkernel's generic record pipeline."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, cast
from searchkernel.domain import Vector
from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.runtime import get_or_compute_query_embedding
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
    RecordSearchResult,
)

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryReadPort,
    MemoryRecord,
    MemoryRepositoryPort,
    RankedMemoryCandidate,
)
from mcp_memory.core.search_ranking import (
    RankingEngine,
    RankingSignals,
    _keyword_token_coverage,
    _query_tokens,
)
from mcp_memory.integrations.searchkernel_adapters import (
    MemoryGraphStore,
    MemoryHydrator,
    MemoryKeywordStore,
    MemoryVectorBackend,
    MemoryVectorStore,
)


class MemoryQueryEmbeddingProvider:
    """Adapt mcp-memory's batch embedder to the query seam."""

    def __init__(self, embedder: EmbeddingBatchProvider, dim: int) -> None:
        self._embedder = embedder
        self.model_name = embedder.model_name
        self.dim = dim

    async def embed_query(self, text: str) -> Vector:
        def compute() -> Vector:
            embeddings = self._embedder.embed([text])
            if len(embeddings) != 1:
                raise ValueError("memory embedder must return one query vector")
            return list(embeddings[0])

        vector = await asyncio.to_thread(
            get_or_compute_query_embedding,
            model_name=self.model_name,
            query=text,
            compute=compute,
        )
        if len(vector) != self.dim:
            raise ValueError(
                f"memory query embedding has dimension {len(vector)}, "
                f"expected {self.dim}"
            )
        return vector


@dataclass(frozen=True, slots=True)
class MemoryRecordPipelineDiagnostics:
    """Composition-time degradation that remains visible to shadow logging."""

    reasons: tuple[str, ...] = ()


@dataclass
class _MemorySearchSignalContext:
    query_tokens: tuple[str, ...]
    semantic_only_abstain_threshold: float
    keyword_candidates_present: bool = False
    top_semantic_score: float = 0.0
    top_score: float = float("-inf")
    top_record_id: str = ""


_ACTIVE_FILTERS: ContextVar[dict[str, object]] = ContextVar(
    "mcp_memory_searchkernel_filters",
    default={},
)
_ACTIVE_POLICY_CONTEXT: ContextVar["_MemorySearchPolicyContext | None"] = ContextVar(
    "mcp_memory_searchkernel_policy_context",
    default=None,
)


@dataclass(slots=True)
class _MemorySearchPolicyContext:
    repository: MemoryRepositoryPort
    records: dict[str, MemoryRecord | None] = field(default_factory=dict)
    links: dict[tuple[str, str], tuple[MemoryLink, ...]] = field(default_factory=dict)

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        if memory_id not in self.records:
            self.records[memory_id] = self.repository.get_memory(memory_id)
        return self.records[memory_id]

    def get_links(
        self,
        memory_id: str,
        *,
        direction: str,
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        key = (memory_id, direction)
        if key not in self.links:
            self.links[key] = tuple(
                self.repository.get_links(memory_id, direction=direction)
            )
        links = self.links[key]
        if link_type is not None:
            return [link for link in links if link.link_type == link_type]
        return list(links)

    def prefetch(self, memory_ids: Sequence[str]) -> None:
        requested_ids = list(dict.fromkeys(memory_ids))
        if not requested_ids:
            return
        missing_ids = [
            memory_id
            for memory_id in requested_ids
            if memory_id not in self.records
            or (memory_id, "incoming") not in self.links
        ]
        if not missing_ids:
            return
        candidates = self.repository.get_ranking_candidates(
            missing_ids,
            include_superseded=True,
        )
        candidates_by_id = {candidate.record.id: candidate for candidate in candidates}
        for memory_id in missing_ids:
            candidate = candidates_by_id.get(memory_id)
            if candidate is None:
                self.records[memory_id] = None
                self.links[(memory_id, "incoming")] = ()
                continue
            self.records[memory_id] = candidate.record
            incoming_links = tuple(
                MemoryLink("", memory_id, link_type, "")
                for link_type, count in candidate.incoming_link_type_counts.items()
                for _ in range(count)
            )
            self.links[(memory_id, "incoming")] = incoming_links

    def ranking_candidate(self, memory_id: str) -> RankedMemoryCandidate | None:
        self.prefetch([memory_id])
        record = self.records.get(memory_id)
        if record is None:
            return None
        incoming_links = self.links.get((memory_id, "incoming"), ())
        counts: dict[str, int] = {}
        for link in incoming_links:
            counts[link.link_type] = counts.get(link.link_type, 0) + 1
        return RankedMemoryCandidate(
            record=record,
            incoming_links_count=len(incoming_links),
            has_incoming_supersedes=counts.get("SUPERSEDES", 0) > 0,
            incoming_link_type_counts=counts,
        )

class _PolicyRepositoryView:
    def __init__(self, repository: MemoryRepositoryPort) -> None:
        self._repository = repository

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return _cached_memory(self._repository, memory_id)

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        return _cached_links(
            self._repository,
            memory_id,
            direction=direction,
            link_type=link_type,
        )


class MemoryRecordSearchPipeline:
    """Bind per-search memory policy to a standalone record pipeline."""

    def __init__(
        self,
        pipeline: RecordSearchPipeline,
        *,
        repository: MemoryRepositoryPort,
        semantic_only_abstain_threshold: float = 0.8,
        diagnostics: MemoryRecordPipelineDiagnostics = MemoryRecordPipelineDiagnostics(),
    ) -> None:
        self._pipeline = pipeline
        self._repository = repository
        self._semantic_only_abstain_threshold = semantic_only_abstain_threshold
        self.diagnostics = diagnostics

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> RecordSearchOutcome:
        signal_context = _MemorySearchSignalContext(
            query_tokens=tuple(_query_tokens(query)),
            semantic_only_abstain_threshold=self._semantic_only_abstain_threshold,
        )
        active_filters = dict(filters or {})
        active_filters.setdefault("source_kind", "memory")
        active_filters["_mcp_memory_signal_context"] = signal_context
        active_filters["_mcp_memory_requested_limit"] = limit
        token = _ACTIVE_FILTERS.set(active_filters)
        policy_token = _ACTIVE_POLICY_CONTEXT.set(
            _MemorySearchPolicyContext(self._repository)
        )
        try:
            outcome = self._pipeline.search(
                query,
                limit=limit,
                filters=active_filters,
            )
            if inspect.isawaitable(outcome):
                return await outcome
            return outcome
        finally:
            _ACTIVE_POLICY_CONTEXT.reset(policy_token)
            _ACTIVE_FILTERS.reset(token)


def build_memory_record_pipeline(
    repository: MemoryRepositoryPort,
    *,
    vector_store: MemoryVectorBackend | None = None,
    embedder: EmbeddingBatchProvider | None = None,
    embedding_dim: int | None = None,
    embedding_maintenance: Any | None = None,
    adaptive_enabled: bool = False,
    config: Config | None = None,
) -> MemoryRecordSearchPipeline:
    """Compose a read-only memory pipeline with memory-owned ranking hooks.

    The generic kernel owns retrieval orchestration while policy hooks retain
    lifecycle, workspace, type, supersession, and ranking authority. Memory's
    core RankingEngine supplies the
    recency, workspace, access, authority, degradation, and signal adjustments
    without moving policy into searchkernel. The memory-owned signal context
    carries query-wide keyword presence and semantic abstention state.
    """
    resolved_config = config or Config()
    cutover_config = resolved_config.searchkernel_cutover

    def prefetch(memory_ids: Sequence[str]) -> None:
        _prefetch_memory_records(repository, memory_ids)

    policy_repository = _PolicyRepositoryView(repository)
    ranking_engine = RankingEngine(resolved_config)
    diagnostics: list[str] = []
    embedding_provider: MemoryQueryEmbeddingProvider | None = None
    adapted_vector_store: MemoryVectorStore | None = None
    if vector_store is None:
        diagnostics.append("vector store unavailable; using keyword and graph retrieval")
    elif embedder is None:
        diagnostics.append("embedder unavailable; using keyword and graph retrieval")
    else:
        resolved_dim = embedding_dim or _embedding_dimension(embedder)
        if resolved_dim is None:
            diagnostics.append("embedding dimension unavailable; using keyword and graph retrieval")
        else:
            embedding_provider = MemoryQueryEmbeddingProvider(embedder, resolved_dim)
            adapted_vector_store = MemoryVectorStore(
                vector_store,
                prefetch=prefetch,
                ensure_embeddings=(
                    lambda candidate_ids, filters: _ensure_searchable_embeddings(
                        repository,
                        embedding_maintenance,
                        candidate_ids,
                        filters,
                    )
                    if embedding_maintenance is not None
                    else None
                ),
            )

    policy = RecordSearchPolicy(
        candidate_filter=lambda candidate: _candidate_allowed(repository, candidate),
        vector_candidate_ids=lambda ranking, filters: _vector_candidate_ids(
            repository,
            ranking,
            filters,
            resolved_config,
        ),
        vector_ranking_order=lambda ranking, filters: _order_vector_ranking(
            repository,
            ranking,
            filters,
            resolved_config,
        ),
        score_adjuster=lambda candidate: _adjust_score(
            ranking_engine,
            repository,
            candidate,
        ),
        result_filter=lambda result: _result_allowed(repository, result),
        post_process=_sort_results,
    )
    hydrator = MemoryHydrator(cast(MemoryReadPort, policy_repository))
    pipeline = RecordSearchPipeline(
        hydrator=hydrator,
        keyword_store=MemoryKeywordStore(repository, prefetch=prefetch),
        vector_store=adapted_vector_store,
        graph_store=MemoryGraphStore(cast(MemoryRepositoryPort, policy_repository)),
        embedding_provider=embedding_provider,
        config=RecordSearchConfig(
            minimum_candidate_limit=50,
            graph_fusion="max",
            max_graph_seeds=3,
            max_neighbors_per_seed=10,
            adaptive_enabled=adaptive_enabled,
            maximum_limit=resolved_config.search_ranking.adaptive_result_max,
            score_ratio_floor=resolved_config.search_ranking.adaptive_result_score_ratio_floor,
            minimum_score=resolved_config.search_ranking.adaptive_result_min_score,
            maximum_score_gap=resolved_config.search_ranking.adaptive_result_max_score_gap,
            failure_mode=(
                "strict"
                if cutover_config.failure_mode == "strict"
                else "lenient"
            ),
        ),
        policy=policy,
        continue_on_error=None,
    )
    return MemoryRecordSearchPipeline(
        pipeline,
        repository=repository,
        semantic_only_abstain_threshold=resolved_config.search_ranking.semantic_only_abstain_threshold,
        diagnostics=MemoryRecordPipelineDiagnostics(tuple(diagnostics)),
    )


def _embedding_dimension(embedder: EmbeddingBatchProvider) -> int | None:
    for attribute in ("embedding_dimension", "dim", "_dimensions"):
        value = getattr(embedder, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    try:
        embeddings = embedder.embed(["searchkernel dimension probe"])
    except Exception:
        return None
    if len(embeddings) != 1 or not embeddings[0]:
        return None
    return len(embeddings[0])


def _ensure_searchable_embeddings(
    repository: MemoryRepositoryPort,
    embedding_maintenance: Any,
    candidate_ids: Sequence[str],
    filters: Mapping[str, Any],
) -> None:
    requested_status = filters.get("status")
    status = requested_status if isinstance(requested_status, str) else None
    if status is None and filters.get("statuses") == ["active"]:
        status = "active"
    if candidate_ids:
        memory_ids = list(candidate_ids)
    else:
        memory_ids = repository.list_memory_ids(
            workspace_id=(
                filters["workspace_id"]
                if isinstance(filters.get("workspace_id"), str)
                else None
            ),
            status=status,
            limit=500,
        )
    candidates = repository.get_searchable_memories(
        memory_ids,
        status=status,
        include_superseded=bool(filters.get("include_superseded", False)),
    )
    embedding_maintenance.ensure_searchable_memory_embeddings(candidates)


def _cached_memory(
    repository: MemoryRepositoryPort,
    memory_id: str,
) -> MemoryRecord | None:
    context = _ACTIVE_POLICY_CONTEXT.get()
    if context is not None and context.repository is repository:
        return context.get_memory(memory_id)
    return repository.get_memory(memory_id)


def _prefetch_memory_records(
    repository: MemoryRepositoryPort,
    memory_ids: Sequence[str],
) -> None:
    context = _ACTIVE_POLICY_CONTEXT.get()
    if context is not None and context.repository is repository:
        context.prefetch(memory_ids)


def _cached_links(
    repository: MemoryRepositoryPort,
    memory_id: str,
    *,
    direction: str,
    link_type: str | None = None,
) -> list[MemoryLink]:
    context = _ACTIVE_POLICY_CONTEXT.get()
    if context is not None and context.repository is repository:
        return context.get_links(
            memory_id,
            direction=direction,
            link_type=link_type,
        )
    return repository.get_links(
        memory_id,
        direction=direction,
        link_type=link_type,
    )


def _candidate_allowed(
    repository: MemoryRepositoryPort,
    candidate: RecordSearchCandidate,
) -> bool:
    record = _cached_memory(repository, candidate.record_id)
    if record is None or not _memory_allowed(repository, record):
        return False
    signal_context = _signal_context()
    if signal_context is not None and "keyword" in candidate.provenance.strategies:
        signal_context.keyword_candidates_present = True
    return True


def _vector_candidate_ids(
    repository: MemoryRepositoryPort,
    keyword_ranking: Sequence[tuple[str, float]],
    filters: dict[str, object],
    config: Config,
) -> Sequence[str] | None:
    if not keyword_ranking:
        return None
    signal_context = filters.get("_mcp_memory_signal_context")
    if not isinstance(signal_context, _MemorySearchSignalContext):
        return None
    keyword_records = [
        record
        for record_id, _ in keyword_ranking
        if (record := _cached_memory(repository, record_id)) is not None
    ]
    strongest_coverage = max(
        (
            _keyword_token_coverage(signal_context.query_tokens, record)
            for record in keyword_records[:5]
        ),
        default=0.0,
    )
    if strongest_coverage < config.search_ranking.keyword_coverage_floor:
        return None
    requested_limit = filters.get("_mcp_memory_requested_limit")
    effective_limit = requested_limit if isinstance(requested_limit, int) else 1
    candidate_cap = max(effective_limit * 4, 20)
    return [record_id for record_id, _ in keyword_ranking[:candidate_cap]]


def _order_vector_ranking(
    repository: MemoryRepositoryPort,
    ranking: Sequence[tuple[str, float]],
    filters: dict[str, object],
    config: Config,
) -> Sequence[tuple[str, float]]:
    workspace_id = filters.get("_ranking_workspace_id", filters.get("workspace_id"))
    workspace = workspace_id if isinstance(workspace_id, str) else None

    def rank_key(item: tuple[str, float]) -> tuple[float, float, str, str]:
        record_id, score = item
        record = _cached_memory(repository, record_id)
        semantic_score = max(min((score + 1.0) / 2.0, 1.0), 0.0)
        workspace_boost = (
            config.search_ranking.workspace_multiplier
            if record is not None
            and workspace is not None
            and workspace in record.workspace_ids
            else 1.0
        )
        updated_at = "" if record is None else record.updated_at
        return (
            semantic_score * workspace_boost,
            semantic_score,
            updated_at,
            record_id,
        )

    return sorted(ranking, key=rank_key, reverse=True)


def _result_allowed(
    repository: MemoryRepositoryPort,
    result: RecordSearchResult,
) -> bool:
    record = _cached_memory(repository, result.record_id)
    if record is None or not _memory_allowed(repository, record):
        return False
    signal_context = _signal_context()
    if signal_context is None:
        return True
    return (
        signal_context.keyword_candidates_present
        or signal_context.top_semantic_score
        >= signal_context.semantic_only_abstain_threshold
    )


def _adjust_score(
    ranking_engine: RankingEngine,
    repository: MemoryRepositoryPort,
    candidate: RecordSearchCandidate,
) -> float:
    record = _cached_memory(repository, candidate.record_id)
    if record is None:
        return 0.0
    ranked_candidate = _ranking_candidate(repository, candidate.record_id)
    if ranked_candidate is None:
        return 0.0

    workspace_id = _ACTIVE_FILTERS.get().get(
        "_ranking_workspace_id",
        _ACTIVE_FILTERS.get().get("workspace_id"),
    )
    workspace = workspace_id if isinstance(workspace_id, str) else None
    provenance = candidate.provenance
    keyword = provenance.strategy_details.get("keyword")
    semantic = provenance.strategy_details.get("vector")
    signal_context = _signal_context()
    keyword_candidates_present = (
        signal_context.keyword_candidates_present
        if signal_context is not None
        else keyword is not None
    )
    keyword_token_coverage = (
        _keyword_token_coverage(signal_context.query_tokens, record)
        if signal_context is not None and keyword is not None
        else 0.0
    )
    semantic_score = 0.0 if semantic is None else semantic.raw_score
    signals = RankingSignals(
        matched_by_keyword=keyword is not None,
        matched_by_semantic=semantic is not None,
        semantic_score=semantic_score,
        keyword_token_coverage=keyword_token_coverage,
        expanded_by_graph="graph" in provenance.strategies,
    )
    ranked = ranking_engine.rank_records(
        [ranked_candidate],
        {record.id: candidate.score},
        workspace,
        ranking_signals={record.id: signals},
        keyword_candidates_present=keyword_candidates_present,
    )
    adjusted_score = ranked[0][1] if ranked else 0.0
    if signal_context is not None and (
        adjusted_score > signal_context.top_score
        or (
            adjusted_score == signal_context.top_score
            and record.id < signal_context.top_record_id
        )
    ):
        signal_context.top_score = adjusted_score
        signal_context.top_record_id = record.id
        signal_context.top_semantic_score = semantic_score
    return adjusted_score


def _ranking_candidate(
    repository: MemoryRepositoryPort,
    memory_id: str,
) -> RankedMemoryCandidate | None:
    context = _ACTIVE_POLICY_CONTEXT.get()
    if context is not None and context.repository is repository:
        return context.ranking_candidate(memory_id)
    record = repository.get_memory(memory_id)
    if record is None:
        return None
    incoming_links = repository.get_links(memory_id, direction="incoming")
    counts: dict[str, int] = {}
    for link in incoming_links:
        counts[link.link_type] = counts.get(link.link_type, 0) + 1
    return RankedMemoryCandidate(
        record=record,
        incoming_links_count=len(incoming_links),
        has_incoming_supersedes=counts.get("SUPERSEDES", 0) > 0,
        incoming_link_type_counts=counts,
    )


def _signal_context() -> _MemorySearchSignalContext | None:
    context = _ACTIVE_FILTERS.get().get("_mcp_memory_signal_context")
    return context if isinstance(context, _MemorySearchSignalContext) else None


def _sort_results(
    results: list[RecordSearchResult],
) -> list[RecordSearchResult]:
    return sorted(results, key=lambda result: (-result.score, result.record_id))


def _memory_allowed(repository: MemoryRepositoryPort, memory: MemoryRecord) -> bool:
    filters = _ACTIVE_FILTERS.get()
    workspace_id = filters.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id not in memory.workspace_ids:
        return False
    requested_status = filters.get("status")
    if isinstance(requested_status, str):
        if memory.status != requested_status:
            return False
    elif memory.status == "archived":
        return False

    memory_type = filters.get("memory_type")
    if isinstance(memory_type, str) and memory.type != memory_type:
        return False

    if not bool(filters.get("include_superseded", False)):
        incoming = _cached_links(
            repository,
            memory.id,
            direction="incoming",
            link_type="SUPERSEDES",
        )
        if incoming:
            return False
    return True
__all__ = [
    "MemoryQueryEmbeddingProvider",
    "MemoryRecordPipelineDiagnostics",
    "MemoryRecordSearchPipeline",
    "build_memory_record_pipeline",
]

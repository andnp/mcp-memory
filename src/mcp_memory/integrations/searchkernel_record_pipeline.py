"""Memory-owned composition for searchkernel's generic record pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from searchkernel.domain import Vector
from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchConfig,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
    RecordSearchResult,
)

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord, MemoryRepositoryPort
from mcp_memory.relational.search import (
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
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._embedder.model_name

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, query: str) -> Vector:
        embeddings = self._embedder.embed([query])
        if len(embeddings) != 1:
            raise ValueError("memory embedder must return one query vector")
        vector = list(embeddings[0])
        if len(vector) != self._dim:
            raise ValueError(
                f"memory query embedding has dimension {len(vector)}, "
                f"expected {self._dim}"
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


class MemoryRecordSearchPipeline:
    """Bind per-search memory policy to a standalone record pipeline."""

    def __init__(
        self,
        pipeline: RecordSearchPipeline,
        *,
        semantic_only_abstain_threshold: float = 0.8,
        diagnostics: MemoryRecordPipelineDiagnostics = MemoryRecordPipelineDiagnostics(),
    ) -> None:
        self._pipeline = pipeline
        self._semantic_only_abstain_threshold = semantic_only_abstain_threshold
        self.diagnostics = diagnostics

    def search(
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
        active_filters["_mcp_memory_signal_context"] = signal_context
        active_filters["_mcp_memory_requested_limit"] = limit
        token = _ACTIVE_FILTERS.set(active_filters)
        try:
            return self._pipeline.search(query, limit=limit, filters=active_filters)
        finally:
            _ACTIVE_FILTERS.reset(token)


def build_memory_record_pipeline(
    repository: MemoryRepositoryPort,
    *,
    vector_store: MemoryVectorBackend | None = None,
    embedder: EmbeddingBatchProvider | None = None,
    embedding_dim: int | None = None,
    config: Config | None = None,
) -> MemoryRecordSearchPipeline:
    """Compose a read-only memory pipeline with memory-owned ranking hooks.

    The generic kernel can express lifecycle, workspace, type, and supersession
    filtering through policy hooks. Memory's RankingEngine supplies the
    recency, workspace, access, authority, degradation, and signal adjustments
    without moving policy into searchkernel. The memory-owned signal context
    carries query-wide keyword presence and semantic abstention state.
    """
    resolved_config = config or Config()
    ranking_engine = RankingEngine(repository, resolved_config)
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
            adapted_vector_store = MemoryVectorStore(vector_store)

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
    hydrator = MemoryHydrator(repository)
    pipeline = RecordSearchPipeline(
        hydrator=lambda record_id: hydrator.hydrate_record(record_id),
        keyword_store=MemoryKeywordStore(repository),
        vector_store=adapted_vector_store,
        graph_store=MemoryGraphStore(repository),
        embedding_provider=embedding_provider,
        config=RecordSearchConfig(
            minimum_candidate_limit=50,
            graph_fusion="max",
        ),
        policy=policy,
        continue_on_error=True,
    )
    return MemoryRecordSearchPipeline(
        pipeline,
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


def _candidate_allowed(
    repository: MemoryRepositoryPort,
    candidate: RecordSearchCandidate,
) -> bool:
    record = repository.get_memory(candidate.record_id)
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
        if (record := repository.get_memory(record_id)) is not None
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
    workspace_id = filters.get("workspace_id")
    workspace = workspace_id if isinstance(workspace_id, str) else None

    def rank_key(item: tuple[str, float]) -> tuple[float, float, str, str]:
        record_id, score = item
        record = repository.get_memory(record_id)
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
    record = repository.get_memory(result.record_id)
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
    record = repository.get_memory(candidate.record_id)
    if record is None:
        return 0.0

    workspace_id = _ACTIVE_FILTERS.get().get("workspace_id")
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
        [record],
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


def _signal_context() -> _MemorySearchSignalContext | None:
    context = _ACTIVE_FILTERS.get().get("_mcp_memory_signal_context")
    return context if isinstance(context, _MemorySearchSignalContext) else None


def _sort_results(
    results: list[RecordSearchResult],
) -> list[RecordSearchResult]:
    return sorted(results, key=lambda result: (-result.score, result.record_id))


def _memory_allowed(repository: MemoryRepositoryPort, memory: MemoryRecord) -> bool:
    filters = _ACTIVE_FILTERS.get()
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
        incoming = repository.get_links(
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

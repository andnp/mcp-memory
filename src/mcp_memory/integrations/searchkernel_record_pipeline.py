"""Memory-owned composition for searchkernel's generic record pipeline."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from searchkernel.domain import Vector
from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.search.record_pipeline import (
    RecordSearchCandidate,
    RecordSearchOutcome,
    RecordSearchPipeline,
    RecordSearchPolicy,
    RecordSearchResult,
)

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord, MemoryRepositoryPort
from mcp_memory.relational.search import RankingEngine, RankingSignals
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
        diagnostics: MemoryRecordPipelineDiagnostics = MemoryRecordPipelineDiagnostics(),
    ) -> None:
        self._pipeline = pipeline
        self.diagnostics = diagnostics

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> RecordSearchOutcome:
        token = _ACTIVE_FILTERS.set(dict(filters or {}))
        try:
            return self._pipeline.search(query, limit=limit, filters=filters)
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
    without moving policy into searchkernel. Keyword coverage and query-wide
    semantic-abstention context are unavailable from per-candidate provenance,
    so those native signal decisions remain a parity gap.
    """
    resolved_config = config or Config()
    ranking_engine = RankingEngine(repository, resolved_config)
    diagnostics: list[str] = []
    diagnostics.append(
        "ranking signal gap: keyword coverage and query-wide semantic abstention "
        "remain native-only"
    )
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
        policy=policy,
        continue_on_error=True,
    )
    return MemoryRecordSearchPipeline(
        pipeline,
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
    return record is not None and _memory_allowed(repository, record)


def _result_allowed(
    repository: MemoryRepositoryPort,
    result: RecordSearchResult,
) -> bool:
    record = repository.get_memory(result.record_id)
    return record is not None and _memory_allowed(repository, record)


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
    signals = RankingSignals(
        matched_by_keyword=keyword is not None,
        matched_by_semantic=semantic is not None,
        semantic_score=0.0 if semantic is None else semantic.raw_score,
        keyword_token_coverage=1.0 if keyword is not None else 0.0,
        expanded_by_graph="graph" in provenance.strategies,
    )
    ranked = ranking_engine.rank_records(
        [record],
        {record.id: candidate.score},
        workspace,
        ranking_signals={record.id: signals},
        keyword_candidates_present=keyword is not None,
    )
    return ranked[0][1] if ranked else 0.0


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

    workspace_id = filters.get("workspace_id")
    if isinstance(workspace_id, str) and workspace_id not in memory.workspace_ids:
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

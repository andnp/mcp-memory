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

from mcp_memory.core.ports.memory import MemoryRecord, MemoryRepositoryPort
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
) -> MemoryRecordSearchPipeline:
    """Compose a read-only memory pipeline without copying native ranking.

    The generic kernel can express lifecycle, workspace, type, and supersession
    filtering through policy hooks. It cannot express memory's authority and
    access-score ranking stages, so native relational results remain authoritative
    during shadow mode.
    """
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
        result_filter=lambda result: _result_allowed(repository, result),
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

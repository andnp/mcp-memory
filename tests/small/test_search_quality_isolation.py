from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar, cast

import pytest
from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import (
    LocalGraphStore,
    LocalKeywordStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.search.record_pipeline import RecordSearchPipeline

from mcp_memory.config import Config, SearchRankingConfig
from mcp_memory.core.ports.memory import (
    MemoryReadContext,
    MemoryRecord,
    MemoryRepositoryPort,
    RankedMemoryCandidate,
)
from mcp_memory.integrations.searchkernel_adapters import MemoryVectorBackend
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MemoryRecordSearchPipeline,
    build_memory_record_pipeline,
)

pytestmark = pytest.mark.small


class RecallEmbedder:
    model_name = "recall-test"
    dim = 4

    _axes: ClassVar[dict[str, int]] = {
        "quality": 0,
        "summary": 0,
        "specificity": 0,
        "wording": 0,
        "finding": 0,
        "memories": 0,
        "tags": 1,
        "discoverability": 1,
        "search": 2,
        "retrieval": 2,
        "ranking": 2,
        "slow": 3,
        "requests": 3,
        "provider": 3,
        "overlap": 3,
        "concurrency": 3,
        "latency": 3,
        "tracing": 3,
    }

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            axis = self._axes.get(token.strip(".,?!:;"))
            if axis is not None:
                vector[axis] += 1.0
        return vector


def _record(
    source_id: str,
    title: str,
    body: str,
    *,
    embedding: list[float] | None = None,
    embedding_model: str | None = None,
) -> Record:
    timestamp = datetime(2026, 8, 1, 12, tzinfo=UTC)
    return Record(
        source_kind="memory",
        source_id=source_id,
        title=title,
        body=body,
        created_at=timestamp,
        updated_at=timestamp,
        metadata={},
        status=RecordStatus.ACTIVE,
        embedding=embedding,
        embedding_model=embedding_model,
        workspace_id="workspace",
    )


@dataclass
class NativeKernelRecallHarness:
    pipeline: RecordSearchPipeline
    backend: LocalRecordBackend
    records: tuple[Record, ...]

    @classmethod
    def build(cls) -> NativeKernelRecallHarness:
        embedder = RecallEmbedder()
        base_records = (
            _record(
                "quality",
                "Search quality guidance",
                "Specific summaries improve retrieval and generic tags weaken discoverability.",
            ),
            _record(
                "operations",
                "Search operations",
                "Provider concurrency can affect search latency and request tracing.",
            ),
            _record(
                "unrelated",
                "Generic architecture note",
                "A broad implementation note without search quality details.",
            ),
        )
        backend = LocalRecordBackend()
        records = tuple(
            _record(
                record.source_id,
                record.title,
                record.body,
                embedding=embedder.embed([record.body])[0],
                embedding_model=embedder.model_name,
            )
            for record in base_records
        )
        backend.upsert(list(records), embedder.model_name, embedder.dim)
        pipeline = RecordSearchPipeline(
            hydrator=backend,
            keyword_store=LocalKeywordStore(backend),
            vector_store=LocalVectorStore(backend),
            graph_store=LocalGraphStore(backend),
            embedding_provider=embedder,
            embedding_model_name=embedder.model_name,
            embedding_dim=embedder.dim,
        )
        return cls(pipeline, backend, records)

    async def search(self, query: str):
        return await self.pipeline.async_search(
            query,
            limit=3,
            filters={"workspace_id": "workspace"},
        )

    def close(self) -> None:
        self.backend.db_manager.get_connection().close()


@pytest.mark.asyncio
async def test_native_kernel_harness_retrieves_quality_record() -> None:
    harness = NativeKernelRecallHarness.build()
    try:
        outcome = await harness.search("summary specificity retrieval")
    finally:
        harness.close()

    assert outcome.results
    assert outcome.results[0].record_id == "quality"


def _memory_record(record: Record) -> MemoryRecord:
    return MemoryRecord(
        id=record.source_id,
        title=record.title,
        content=record.body,
        summary=record.body,
        type="fact",
        status="active",
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace"],
        tags=[],
    )


class MemoryRecallRepository:
    def __init__(self, records: tuple[MemoryRecord, ...]) -> None:
        self.records = {record.id: record for record in records}

    def search_keyword_memory_ids(self, query: str, **kwargs: object) -> list[str]:
        query_tokens = set(query.lower().split())
        ranked = []
        for record in self.records.values():
            searchable = " ".join(
                (record.title, record.content, record.summary or "")
            ).lower()
            matched = len(query_tokens.intersection(searchable.split()))
            if matched:
                ranked.append((matched, record.id))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [record_id for _matched, record_id in ranked]

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.records.get(memory_id)

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RankedMemoryCandidate]:
        del status, include_superseded
        return [
            RankedMemoryCandidate(
                record=self.records[memory_id],
                incoming_links_count=0,
            )
            for memory_id in memory_ids
            if memory_id in self.records
        ]

    def get_links(self, memory_id: str, direction: str = "outgoing", link_type=None):
        del memory_id, direction, link_type
        return []

    def peek_memory(self, memory_id: str) -> MemoryReadContext | None:
        record = self.get_memory(memory_id)
        return None if record is None else MemoryReadContext(record, {}, [])


class MemoryRecallVectorBackend:
    supports_candidate_filtering = True

    def __init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}

    def upsert(self, **kwargs: object) -> bool:
        source_id = kwargs.get("source_id")
        embedding = kwargs.get("embedding")
        if not isinstance(source_id, str) or not isinstance(embedding, list):
            raise TypeError("source_id and embedding are required")
        self.vectors[source_id] = cast("list[float]", embedding)
        return True

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        candidate_ids: list[str] | None = None,
        diagnostics: dict[str, object] | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        del source_kind, model_name, diagnostics, workspace_id
        ids = self.vectors if candidate_ids is None else {
            memory_id: self.vectors[memory_id]
            for memory_id in candidate_ids
            if memory_id in self.vectors
        }
        results = [
            (memory_id, _cosine(query_embedding, embedding))
            for memory_id, embedding in ids.items()
        ]
        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:limit]

    def delete(self, **kwargs: object) -> int:
        del kwargs
        return 0


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


@dataclass
class MemoryRecallHarness:
    pipeline: MemoryRecordSearchPipeline

    @classmethod
    def build(cls) -> MemoryRecallHarness:
        native = NativeKernelRecallHarness.build()
        try:
            records = tuple(_memory_record(record) for record in native.records)
        finally:
            native.close()
        repository = MemoryRecallRepository(records)
        embedder = RecallEmbedder()
        vector_backend = MemoryRecallVectorBackend()
        for record in records:
            vector_backend.upsert(
                source_id=record.id,
                embedding=embedder.embed([record.content])[0],
            )
        pipeline = build_memory_record_pipeline(
            cast("MemoryRepositoryPort", repository),
            vector_store=cast("MemoryVectorBackend", vector_backend),
            embedder=embedder,
            config=Config(
                search_ranking=SearchRankingConfig(
                    semantic_only_abstain_threshold=0.0,
                )
            ),
        )
        return cls(pipeline)

    async def search(self, query: str):
        return await self.pipeline.search(
            query,
            limit=3,
            filters={"workspace_id": "workspace"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_id"),
    [
        ("how does wording affect finding memories", "quality"),
        ("what happens to slow requests during provider overlap", "operations"),
    ],
)
async def test_paraphrase_recall_matches_native_kernel(
    query: str,
    expected_id: str,
) -> None:
    native_harness = NativeKernelRecallHarness.build()
    try:
        native = await native_harness.search(query)
    finally:
        native_harness.close()
    memory = await MemoryRecallHarness.build().search(query)

    assert expected_id in [result.record_id for result in native.results]
    assert expected_id in [result.record_id for result in memory.results]

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from searchkernel.domain import Record, RecordStatus
from searchkernel.indices import (
    LocalGraphStore,
    LocalKeywordStore,
    LocalRecordBackend,
    LocalVectorStore,
)
from searchkernel.search.record_pipeline import RecordSearchPipeline

pytestmark = pytest.mark.small


class RecallEmbedder:
    model_name = "recall-test"
    dim = 3

    _axes: ClassVar[dict[str, int]] = {
        "quality": 0,
        "summary": 0,
        "specificity": 0,
        "tags": 1,
        "discoverability": 1,
        "search": 2,
        "retrieval": 2,
        "ranking": 2,
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


@pytest.mark.asyncio
async def test_native_kernel_harness_retrieves_quality_record() -> None:
    harness = NativeKernelRecallHarness.build()

    outcome = await harness.search("summary specificity retrieval")

    assert outcome.results
    assert outcome.results[0].record_id == "quality"

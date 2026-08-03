from __future__ import annotations

import pytest
from searchkernel.ports.content_source import IngestionError

from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.integrations.searchkernel_ingestion import MemoryRecordIngestor


pytestmark = pytest.mark.small


def _memory(memory_id: str = "memory-1") -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title="Memory title",
        content="Memory body",
        summary="Memory summary",
        type="fact",
        status="active",
        created_at="2026-08-01T12:00:00+00:00",
        updated_at="2026-08-01T13:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=["workspace-a"],
        tags=["ingestion"],
    )


class _Embedder:
    model_name = "test-model"
    dim = 2

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


class _VectorStore:
    def __init__(self, *, failing_ids: set[str] | None = None) -> None:
        self.failing_ids = failing_ids or set()
        self.upserts: list[dict[str, object]] = []

    def upsert(self, **kwargs: object) -> bool:
        source_id = str(kwargs["source_id"])
        if source_id in self.failing_ids:
            raise RuntimeError(f"cannot write {source_id}")
        self.upserts.append(kwargs)
        return True

    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        del kwargs
        return []

    def delete(self, **kwargs: object) -> int:
        del kwargs
        return 0


@pytest.mark.asyncio
async def test_memory_ingestor_returns_per_record_receipt_and_checkpoint() -> None:
    embedder = _Embedder()
    vector_store = _VectorStore()
    ingestor = MemoryRecordIngestor(object(), vector_store, embedder)

    receipt = await ingestor.index_records([_memory()], checkpoint="cursor-7")

    assert receipt.checkpoint == "cursor-7"
    assert receipt.attempted == 1
    assert receipt.committed == 1
    assert receipt.failed == 0
    assert receipt.records[0].source_kind == "memory"
    assert receipt.records[0].source_id == "memory-1"
    assert receipt.records[0].workspace_id == "workspace-a"
    assert embedder.calls == [["Title: Memory title\n\nMemory summary\nMemory body\ningestion"]]
    assert vector_store.upserts[0]["source_updated_at"] == "2026-08-01T13:00:00+00:00"


@pytest.mark.asyncio
async def test_memory_ingestor_preserves_strict_and_lenient_failure_modes() -> None:
    records = [_memory("ok-1"), _memory("fail"), _memory("ok-2")]

    strict = MemoryRecordIngestor(object(), _VectorStore(failing_ids={"fail"}), _Embedder())
    strict_receipt = await strict.index_records(records, failure_mode="strict")
    assert [result.status for result in strict_receipt.records] == ["committed", "failed"]

    lenient = MemoryRecordIngestor(object(), _VectorStore(failing_ids={"fail"}), _Embedder())
    lenient_receipt = await lenient.index_records(records, failure_mode="lenient")
    assert [result.status for result in lenient_receipt.records] == [
        "committed",
        "failed",
        "committed",
    ]


def test_embedding_maintenance_uses_receipt_ingestion_for_inline_repair() -> None:
    embedder = _Embedder()
    vector_store = _VectorStore()
    maintenance = MemoryEmbeddingMaintenance(
        object(),
        Config(),
        embedder=embedder,
        vector_store=vector_store,
    )

    maintenance._ensure_memory_embeddings([_memory()])

    assert [item["source_id"] for item in vector_store.upserts] == ["memory-1"]


def test_embedding_maintenance_surfaces_receipt_failure() -> None:
    maintenance = MemoryEmbeddingMaintenance(
        object(),
        Config(),
        embedder=_Embedder(),
        vector_store=_VectorStore(failing_ids={"memory-1"}),
    )

    with pytest.raises(IngestionError, match="Failed to ingest 1 record"):
        maintenance._ensure_memory_embeddings([_memory()])

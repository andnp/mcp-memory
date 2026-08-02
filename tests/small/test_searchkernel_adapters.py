from typing import cast

import pytest
from searchkernel.domain import GraphNeighbor, Record, RecordHit, RecordIdentity, RecordStatus

from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryReadContext,
    MemoryReadPort,
    MemoryRecord,
)
from mcp_memory.integrations.searchkernel_adapters import (
    MemoryEmbeddingSink,
    MemoryGraphStore,
    MemoryHydrator,
    MemoryKeywordStore,
    MemoryRecordAdapter,
    MemoryRepairQueue,
    MemoryVectorStore,
)


pytestmark = pytest.mark.small


def _memory_record(
    status: str = "active",
    *,
    memory_id: str = "memory-1",
    workspace_ids: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title="Title",
        content="Body",
        summary="Summary",
        type="fact",
        status=status,
        created_at="2026-07-30T12:00:00+00:00",
        updated_at="2026-07-30T13:00:00+00:00",
        read_count=2,
        access_score=1.5,
        last_accessed_at=None,
        last_surfaced_at=None,
        metadata={"custom": "value"},
        workspace_ids=workspace_ids or ["workspace-1"],
        tags=["tag"],
    )


class _Repository:
    def __init__(self) -> None:
        self.links = {
            "memory-1": [MemoryLink("memory-1", "memory-2", "DEPENDS_ON", "")]
        }

    def search_keyword_memory_ids(self, query: str, **kwargs: object) -> list[str]:
        self.keyword_kwargs = kwargs
        return ["memory-1", "memory-2"]

    def get_links(self, memory_id: str, direction: str = "outgoing", link_type=None):
        return self.links.get(memory_id, [])

    def get_memory(self, memory_id: str):
        if memory_id != "memory-1":
            return None
        return _memory_record()

    def peek_memory(self, memory_id: str):
        if memory_id != "memory-1":
            return None
        return MemoryReadContext(_memory_record(), {"outgoing": self.links["memory-1"]}, [])


class _BatchRepository(_Repository):
    def __init__(self) -> None:
        super().__init__()
        self.batch_memory_calls: list[tuple[list[str], str | None, bool]] = []
        self.batch_link_calls: list[tuple[list[str], str]] = []

    def get_searchable_memories(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        self.batch_memory_calls.append((memory_ids, status, include_superseded))
        return [
            _memory_record(memory_id=memory_id, workspace_ids=["workspace-1"])
            for memory_id in memory_ids
        ]

    def get_links_many(self, memory_ids: list[str], *, direction: str = "outgoing"):
        self.batch_link_calls.append((memory_ids, direction))
        return {memory_id: self.links.get(memory_id, []) for memory_id in memory_ids}


class _VectorBackend:
    supports_candidate_filtering = True

    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None
        self.upserts: list[dict[str, object]] = []

    def upsert(self, **kwargs: object) -> bool:
        self.upserts.append(kwargs)
        return True

    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        self.search_kwargs = kwargs
        return [("memory-1", 0.9)]

    def delete(self, **kwargs: object) -> int:
        return 1


def test_record_adapter_preserves_memory_metadata_and_maps_degraded_status():
    record = MemoryRecordAdapter.to_record(_memory_record("degraded"))

    assert record.source_kind == "memory"
    assert record.source_id == "memory-1"
    assert record.status is RecordStatus.STALE
    assert record.metadata["memory_status"] == "degraded"
    assert record.metadata["workspace_ids"] == ["workspace-1"]


def test_record_adapter_uses_composite_workspace_identity():
    first = MemoryRecordAdapter.to_record(
        _memory_record(memory_id="same", workspace_ids=["workspace-1"])
    )
    second = MemoryRecordAdapter.to_record(
        _memory_record(memory_id="same", workspace_ids=["workspace-2"])
    )

    assert first.storage_key != second.storage_key
    assert first.metadata["canonical_id"] == first.storage_key
    assert second.metadata["canonical_id"] == second.storage_key


@pytest.mark.asyncio
async def test_keyword_store_delegates_relational_policy():
    repository = _Repository()
    store = MemoryKeywordStore(cast(MemoryReadPort, repository))

    assert callable(store.search)
    hits = await store.search("query", 2, {"workspace_id": "workspace-1"})

    assert [
        (hit.identity, hit.score)
        for hit in cast(list[RecordHit], hits)
    ] == [
        (RecordIdentity("workspace-1", "memory", "memory-1"), 1.0),
        (RecordIdentity("workspace-1", "memory", "memory-2"), 0.5),
    ]
    assert repository.keyword_kwargs["workspace_id"] == "workspace-1"


@pytest.mark.asyncio
async def test_vector_store_uses_formal_candidate_filter_capability():
    backend = _VectorBackend()
    store = MemoryVectorStore(backend)

    assert callable(store.search)
    hits = await store.search(
        [1.0, 0.0],
        3,
        model_name="model",
        dim=2,
        filters={"candidate_ids": ["memory-1"]},
    )
    assert hits == [
        RecordHit(RecordIdentity(None, "memory", "memory-1"), 0.9)
    ]
    assert backend.search_kwargs is not None
    assert backend.search_kwargs["candidate_ids"] == ["memory-1"]


@pytest.mark.asyncio
async def test_vector_store_hydrates_canonical_workspace_identity_when_available():
    backend = _VectorBackend()
    repository = _BatchRepository()
    store = MemoryVectorStore(
        backend,
        repository=cast(MemoryReadPort, repository),
    )

    hits = await store.search(
        [1.0, 0.0],
        1,
        model_name="model",
        dim=2,
    )

    assert hits == [
        RecordHit(RecordIdentity("workspace-1", "memory", "memory-1"), 0.9)
    ]
    assert repository.batch_memory_calls == [
        (["memory-1"], None, True)
    ]


def test_vector_store_upsert_preserves_version_guard():
    backend = _VectorBackend()
    store = MemoryVectorStore(backend)
    record = MemoryRecordAdapter.to_record(_memory_record())
    record.embedding = [1.0, 0.0]

    store.upsert([record], "model", 2)

    assert backend.upserts[0]["source_updated_at"] == record.updated_at.isoformat()
    assert store.epoch() == 1


@pytest.mark.asyncio
async def test_graph_store_reads_links_without_writing_memory_schema():
    repository = _Repository()
    store = MemoryGraphStore(cast(MemoryReadPort, repository))

    assert callable(store.neighbors)
    assert await store.neighbors("memory-1") == [
        GraphNeighbor(
            RecordIdentity("workspace-1", "memory", "memory-2"),
            "DEPENDS_ON",
            0.7,
        )
    ]
    with pytest.raises(NotImplementedError):
        store.upsert_edges([("memory-1", "memory-2", "DEPENDS_ON", 1.0)])


@pytest.mark.asyncio
async def test_hydrator_returns_kernel_record_without_access_telemetry():
    hydrator = MemoryHydrator(cast(MemoryReadPort, _Repository()))

    record = await hydrator.hydrate_record("memory-1")

    assert isinstance(record, Record)
    assert record.source_id == "memory-1"
    assert record.body == "Body"


@pytest.mark.asyncio
async def test_hydrator_preserves_requested_workspace_identity():
    hydrator = MemoryHydrator(cast(MemoryReadPort, _Repository()))
    identity = RecordIdentity("workspace-2", "memory", "memory-1")

    record = await hydrator.hydrate_record(identity)

    assert record is not None
    assert record.storage_key == identity.storage_key
    assert record.metadata["canonical_id"] == identity.storage_key


@pytest.mark.asyncio
async def test_hydrator_uses_batch_repository_api_for_canonical_keys():
    repository = _BatchRepository()
    hydrator = MemoryHydrator(cast(MemoryReadPort, repository))
    identities = [
        RecordIdentity("workspace-1", "memory", "memory-1"),
        RecordIdentity("workspace-1", "memory", "memory-2"),
    ]

    records = await hydrator.hydrate_records(identities)

    assert list(records) == [identity.storage_key for identity in identities]
    assert all(records[identity.storage_key] is not None for identity in identities)
    assert repository.batch_memory_calls == [
        (["memory-1", "memory-2"], None, True)
    ]


@pytest.mark.asyncio
async def test_graph_store_uses_batch_links_and_preserves_seed_workspace():
    repository = _BatchRepository()
    store = MemoryGraphStore(cast(MemoryReadPort, repository))
    identity = RecordIdentity("workspace-1", "memory", "memory-1")

    neighbors = await store.neighbors_many([identity], depth=1)

    assert neighbors[identity.storage_key] == [
        GraphNeighbor(
            RecordIdentity("workspace-1", "memory", "memory-2"),
            "DEPENDS_ON",
            0.7,
        )
    ]
    assert repository.batch_link_calls == [
        (["memory-1"], "outgoing")
    ]


def test_embedding_sink_restricts_source_namespace():
    backend = _VectorBackend()
    sink = MemoryEmbeddingSink(backend)

    assert sink.upsert(
        source_kind="memory",
        source_id="memory-1",
        workspace_id=None,
        model_name="model",
        embedding=[1.0],
    )
    with pytest.raises(ValueError):
        sink.upsert(
            source_kind="other",
            source_id="other-1",
            workspace_id=None,
            model_name="model",
            embedding=[1.0],
        )


def test_repair_queue_adapter_preserves_queue_methods():
    class Queue:
        def enqueue_unique(self, **kwargs: object):
            return kwargs

        def backlog_snapshot(self, **kwargs: object):
            return kwargs

    adapter = MemoryRepairQueue(Queue())

    assert adapter.enqueue_unique(memory_id="memory-1") == {"memory_id": "memory-1"}
    assert adapter.backlog_snapshot(now=1.0) == {"now": 1.0}

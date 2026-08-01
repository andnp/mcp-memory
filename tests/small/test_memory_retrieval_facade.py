from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest

from mcp_memory.core.ports.memory import MemoryRepositoryPort
from mcp_memory.integrations.memory_retrieval import MemoryRetrievalFacade


pytestmark = pytest.mark.small


@dataclass
class FakeOutcome:
    query: str
    limit: int
    filters: dict[str, object]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, dict[str, object]]] = []

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> FakeOutcome:
        self.calls.append((query, limit, filters))
        await asyncio.sleep(0)
        return FakeOutcome(query, limit, filters)


class FakeNativeSearch:
    def read_memory(self, memory_id: str) -> tuple[str, str]:
        return ("read", memory_id)

    def peek_memory(self, memory_id: str) -> tuple[str, str]:
        return ("peek", memory_id)

    def search_memories_for_maintenance(self, query: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        return (query, kwargs)

    def resolve_memory_id(self, memory_id: str) -> str:
        return f"resolved:{memory_id}"


def _facade(pipeline: FakePipeline) -> MemoryRetrievalFacade:
    return MemoryRetrievalFacade(
        cast(MemoryRepositoryPort, object()),
        pipeline=cast(Any, pipeline),
        native_search=FakeNativeSearch(),
    )


@pytest.mark.asyncio
async def test_async_search_is_canonical_and_preserves_filters() -> None:
    pipeline = FakePipeline()
    facade = _facade(pipeline)

    outcome = await facade.search(
        "query",
        limit=3,
        filters={
            "workspace_id": "workspace-1",
            "memory_type": "fact",
            "status": "active",
            "include_superseded": False,
        },
    )

    assert outcome == FakeOutcome(
        "query",
        3,
        {
            "workspace_id": "workspace-1",
            "memory_type": "fact",
            "status": "active",
            "include_superseded": False,
        },
    )
    assert len(pipeline.calls) == 1


def test_sync_search_works_without_running_loop() -> None:
    pipeline = FakePipeline()
    outcome = _facade(pipeline).search_sync("query", limit=2)

    assert isinstance(outcome, FakeOutcome)
    assert outcome.query == "query"
    assert outcome.limit == 2


def test_adaptive_search_uses_adaptive_pipeline_factory() -> None:
    pipelines = {False: FakePipeline(), True: FakePipeline()}
    requested: list[bool] = []

    def pipeline_factory(adaptive_limit: bool) -> Any:
        requested.append(adaptive_limit)
        return pipelines[adaptive_limit]

    facade = MemoryRetrievalFacade(
        cast(MemoryRepositoryPort, object()),
        pipeline_factory=pipeline_factory,
    )

    outcome = facade.search_sync("query", adaptive_limit=True)

    assert isinstance(outcome, FakeOutcome)
    assert requested == [True]


@pytest.mark.asyncio
async def test_sync_search_is_safe_inside_running_loop() -> None:
    pipeline = FakePipeline()

    outcome = _facade(pipeline).search_sync("query", limit=4)

    assert isinstance(outcome, FakeOutcome)
    assert outcome.query == "query"
    assert outcome.limit == 4


def test_read_peek_and_maintenance_delegate_to_native_service() -> None:
    facade = _facade(FakePipeline())

    assert facade.read_memory("memory-1") == ("read", "memory-1")
    assert facade.peek_memory("memory-1") == ("peek", "memory-1")
    assert facade.search_memories_for_maintenance(
        "query",
        workspace_id="workspace-1",
    ) == ("query", {"workspace_id": "workspace-1"})
    assert facade.resolve_memory_id("mem-1") == "resolved:mem-1"

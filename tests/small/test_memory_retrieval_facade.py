from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, cast

import pytest

import mcp_memory.integrations.memory_retrieval as memory_retrieval
from mcp_memory.core.ports.memory import MemoryRepositoryPort
from mcp_memory.integrations.memory_retrieval import MemoryRetrievalFacade, MemorySearchRequest
from searchkernel.search.record_pipeline import RecordSearchOutcome


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


class DiagnosticPipeline:
    def __init__(self, outcome: RecordSearchOutcome) -> None:
        self.outcome = outcome

    async def search(
        self,
        query: str,
        *,
        limit: int,
        filters: dict[str, object],
    ) -> RecordSearchOutcome:
        del query, limit, filters
        return self.outcome


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
    """Select the adaptive pipeline without changing the facade contract."""
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


def test_standard_and_adaptive_pipelines_share_query_embedding_cache_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one query cache owner when lazily creating both pipeline variants."""
    pipelines = {False: FakePipeline(), True: FakePipeline()}
    caches: list[object] = []

    def build_pipeline(*args: Any, **kwargs: Any) -> FakePipeline:
        del args
        caches.append(kwargs["query_embedding_cache"])
        return pipelines[kwargs["adaptive_enabled"]]

    monkeypatch.setattr(memory_retrieval, "build_memory_record_pipeline", build_pipeline)
    facade = MemoryRetrievalFacade(cast(MemoryRepositoryPort, object()))

    facade.search_sync("standard query")
    facade.search_sync("adaptive query", adaptive_limit=True)

    assert len(caches) == 2
    assert caches[0] is caches[1]


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


def test_search_request_keeps_scope_filter_separate_from_ranking_context() -> None:
    pipeline = FakePipeline()
    facade = _facade(pipeline)

    facade.search_sync(
        MemorySearchRequest(
            query="global",
            workspace_id=None,
            ranking_workspace_id="workspace-1",
        )
    )

    assert pipeline.calls[0][2]["_ranking_workspace_id"] == "workspace-1"
    assert "workspace_id" not in pipeline.calls[0][2]


@pytest.mark.asyncio
async def test_async_and_sync_diagnostics_share_the_same_projection() -> None:
    """Keep diagnostic fields equivalent across both retrieval entry points."""
    outcome = RecordSearchOutcome(
        diagnostics=(
            "query_plan:lanes:keyword,vector",
            "query_plan:budgets:keyword=5,vector=5,graph_seeds=10,rerank=0",
            "query_plan:skip:graph:awaiting_seed_confidence",
        ),
        cache_diagnostics=("candidate_cache:hit",),
        candidate_count=3,
        candidate_counts={"keyword": 2, "vector": 1},
        stage_timings_ms={"search": 1.25},
    )
    facade = _facade(cast(FakePipeline, DiagnosticPipeline(outcome)))

    async_outcome, async_diagnostics = await facade.search_with_diagnostics(
        "query",
        workspace_id="workspace-1",
        ranking_workspace_id="workspace-2",
    )
    sync_outcome, sync_diagnostics = facade.search_sync_with_diagnostics(
        "query",
        workspace_id="workspace-1",
        ranking_workspace_id="workspace-2",
    )

    async_payload = async_diagnostics.to_payload()
    sync_payload = sync_diagnostics.to_payload()
    async_payload["timing_ms"] = {}
    sync_payload["timing_ms"] = {}
    assert async_outcome == sync_outcome == outcome
    assert async_payload == sync_payload
    assert async_payload["lane_decisions"] == {
        "enabled": ["keyword", "vector"],
        "budgets": {"keyword": 5, "vector": 5, "graph_seeds": 10, "rerank": 0},
        "skipped": ["graph:awaiting_seed_confidence"],
    }
    overlap = cast(dict[str, object], async_payload["overlap"])
    assert overlap["raw_lane_overlap_count"] is None


def test_diagnostic_projection_failure_falls_back_to_successful_outcome(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep successful retrieval intact when diagnostics cannot be projected."""
    outcome = RecordSearchOutcome(results=())
    facade = _facade(cast(FakePipeline, DiagnosticPipeline(outcome)))

    def fail_projection(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("projection failed")

    monkeypatch.setattr(
        "mcp_memory.integrations.memory_retrieval.build_search_execution_diagnostics",
        fail_projection,
    )

    with caplog.at_level("WARNING"):
        actual_outcome, diagnostics = facade.search_sync_with_diagnostics(
            "query"
        )

    assert actual_outcome is outcome
    assert diagnostics.degraded is False
    assert diagnostics.failure_count == 0
    assert "diagnostics projection failed" in caplog.text


def test_diagnostics_preserve_missing_hydration_and_degraded_state() -> None:
    """Expose missing hydrated IDs without collapsing the successful outcome."""
    outcome = RecordSearchOutcome(missing_record_ids=("missing",))
    facade = _facade(cast(FakePipeline, DiagnosticPipeline(outcome)))

    actual_outcome, diagnostics = facade.search_sync_with_diagnostics("query")

    assert actual_outcome is outcome
    assert diagnostics.degraded is True
    assert diagnostics.missing_record_ids == ["missing"]
    assert diagnostics.missing_record_count == 1
    assert diagnostics.to_payload()["missing_record_ids"] == ["missing"]

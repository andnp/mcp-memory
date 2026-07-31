from __future__ import annotations

from typing import cast

import pytest

from mcp_memory.application import memory_use_cases
from mcp_memory.config import Config, SearchKernelShadowConfig, SearchRankingConfig
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryReadContext,
    MemoryRecord,
    MemoryRepositoryPort,
)
from mcp_memory.relational.search import RankingEngine
from mcp_memory.context import ApplicationContext
from mcp_memory.integrations.searchkernel_adapters import MemoryVectorBackend
from mcp_memory.integrations.searchkernel_record_pipeline import (
    build_memory_record_pipeline,
)
from mcp_memory.integrations.searchkernel_shadow import run_searchkernel_shadow


pytestmark = pytest.mark.small


def _memory(
    memory_id: str,
    *,
    status: str = "active",
    workspace_ids: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title=memory_id,
        content=f"body {memory_id}",
        summary="summary",
        type="fact",
        status=status,
        created_at="2026-07-30T12:00:00+00:00",
        updated_at="2026-07-30T13:00:00+00:00",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        workspace_ids=workspace_ids or ["workspace-1"],
        tags=[],
    )


class FakeRepository:
    def __init__(self) -> None:
        self.records = {
            "active": _memory("active"),
            "other-workspace": _memory("other-workspace", workspace_ids=["workspace-2"]),
            "archived": _memory("archived", status="archived"),
            "superseded": _memory("superseded"),
        }
        self.links = [
            MemoryLink("active", "superseded", "SUPERSEDES", ""),
        ]

    def search_keyword_memory_ids(self, query: str, **kwargs: object) -> list[str]:
        return list(self.records)

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        return self.records.get(memory_id)

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        links = [
            link
            for link in self.links
            if (link.source_id if direction == "outgoing" else link.target_id) == memory_id
        ]
        if link_type is not None:
            links = [link for link in links if link.link_type == link_type]
        return links

    def peek_memory(self, memory_id: str) -> MemoryReadContext | None:
        record = self.get_memory(memory_id)
        if record is None:
            return None
        return MemoryReadContext(record, {"outgoing": self.get_links(memory_id)}, [])


class RankingRepository(FakeRepository):
    def __init__(self) -> None:
        super().__init__()
        accessed = _memory("accessed")
        accessed.access_score = 25.0
        accessed.last_accessed_at = "2026-07-30T12:00:00+00:00"
        self.records = {
            "plain": _memory("plain"),
            "accessed": accessed,
            "authority": _memory("authority"),
            "stale": _memory("stale", status="stale"),
        }
        self.links = [
            MemoryLink("supporter", "authority", "DEPENDS_ON", ""),
        ]


class FakeVectorStore:
    supports_candidate_filtering = True

    def __init__(self) -> None:
        self.search_count = 0
        self.write_count = 0

    def upsert(self, **kwargs: object) -> bool:
        self.write_count += 1
        return True

    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        self.search_count += 1
        return [
            ("active", 1.0),
            ("other-workspace", 0.9),
            ("archived", 0.8),
            ("superseded", 0.7),
        ]

    def delete(self, **kwargs: object) -> int:
        self.write_count += 1
        return 1


class FakeEmbedder:
    model_name = "fake-model"
    dim = 2

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[1.0, 0.0] for _ in texts]


def test_composition_applies_memory_policy_without_writes() -> None:
    repository = FakeRepository()
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", vector_store),
        embedder=embedder,
    )

    outcome = pipeline.search(
        "query",
        limit=10,
        filters={
            "workspace_id": "workspace-1",
            "status": "active",
            "include_superseded": False,
        },
    )

    assert [result.record_id for result in outcome.results] == ["active"]
    assert vector_store.search_count == 1
    assert vector_store.write_count == 0
    assert embedder.queries == ["query"]


def test_superseded_policy_is_expressible() -> None:
    repository = FakeRepository()
    pipeline = build_memory_record_pipeline(cast("MemoryRepositoryPort", repository))

    excluded = pipeline.search("query", limit=10)
    included = pipeline.search(
        "query",
        limit=10,
        filters={"include_superseded": True},
    )

    assert "superseded" not in {result.record_id for result in excluded.results}
    assert "superseded" in {result.record_id for result in included.results}


def test_missing_vector_or_embedder_degrades_to_keyword_pipeline() -> None:
    repository = FakeRepository()
    without_vector = build_memory_record_pipeline(cast("MemoryRepositoryPort", repository))
    without_embedder = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        vector_store=cast("MemoryVectorBackend", FakeVectorStore()),
    )

    assert without_vector.diagnostics.reasons
    assert without_embedder.diagnostics.reasons
    assert without_vector.search("query", limit=1).results
    assert without_embedder.search("query", limit=1).results


def test_score_adjustment_matches_relational_ranking_engine() -> None:
    repository = RankingRepository()
    config = Config(
        search_ranking=SearchRankingConfig(
            workspace_multiplier=1.4,
            degradation_multiplier=0.25,
        )
    )
    pipeline = build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
        config=config,
    )

    results = pipeline.search(
        "query",
        limit=10,
        filters={"workspace_id": "workspace-1"},
    ).results
    engine = RankingEngine(cast("MemoryRepositoryPort", repository), config)
    expected = {
        memory_id: engine.score_from_rrf(
            repository.records[memory_id],
            1.0 / (config.search_ranking.rrf_k + rank + 1),
            "workspace-1",
        )
        for rank, memory_id in enumerate(
            ["plain", "accessed", "authority", "stale"]
        )
    }

    assert {result.record_id: result.score for result in results} == pytest.approx(expected)
    scores = {result.record_id: result.score for result in results}
    assert scores["accessed"] > scores["plain"]
    assert scores["authority"] > scores["plain"]
    assert scores["stale"] < scores["plain"]


def test_pipeline_ranking_is_read_only() -> None:
    repository = RankingRepository()
    before = {
        memory_id: (
            record.access_score,
            record.last_accessed_at,
            record.last_surfaced_at,
        )
        for memory_id, record in repository.records.items()
    }

    build_memory_record_pipeline(
        cast("MemoryRepositoryPort", repository),
    ).search("query", limit=10)

    after = {
        memory_id: (
            record.access_score,
            record.last_accessed_at,
            record.last_surfaced_at,
        )
        for memory_id, record in repository.records.items()
    }
    assert after == before


@pytest.mark.asyncio
async def test_shadow_compares_pipeline_without_replacing_native_results() -> None:
    repository = FakeRepository()
    pipeline = build_memory_record_pipeline(cast("MemoryRepositoryPort", repository))

    diagnostics = await run_searchkernel_shadow(
        pipeline,
        query="query",
        requested_limit=2,
        native_results=[type("Native", (), {"memory_id": "native", "score": 1.0})()],
        filters={"workspace_id": "workspace-1"},
    )

    assert diagnostics.native_ids == ("native",)
    assert diagnostics.kernel_ids == ("active",)
    assert diagnostics.error is not None


def test_enabled_shadow_path_builds_record_pipeline(monkeypatch) -> None:
    runtime_logs: list[dict[str, object]] = []
    context = ApplicationContext(
        config=Config(searchkernel_shadow=SearchKernelShadowConfig(enabled=True)),
        repository=FakeRepository(),
        relational_search=object(),
        runtime_logs=type(
            "RuntimeLogs",
            (),
            {"write_log": lambda _self, **kwargs: runtime_logs.append(kwargs)},
        )(),
    )
    pipeline = object()
    called = False
    captured_config = None

    def build_pipeline(*args: object, **kwargs: object) -> object:
        nonlocal called, captured_config
        called = True
        captured_config = kwargs["config"]
        return pipeline

    async def run_shadow(kernel: object, **kwargs: object):
        assert kernel is pipeline
        return type(
            "Diagnostics",
            (),
            {
                "error": None,
                "to_payload": lambda _self: {
                    "native_ids": ["native"],
                    "kernel_ids": ["kernel"],
                },
            },
        )()

    monkeypatch.setattr(
        memory_use_cases,
        "build_memory_record_pipeline",
        build_pipeline,
    )
    monkeypatch.setattr(memory_use_cases, "run_searchkernel_shadow", run_shadow)
    monkeypatch.setattr(
        memory_use_cases,
        "build_memory_search_kernel",
        lambda *args, **kwargs: pytest.fail("legacy source wrapper was used"),
    )

    memory_use_cases._run_searchkernel_shadow(
        context,
        query="query",
        limit=2,
        native_results=[],
        workspace_id=None,
        memory_type=None,
        status=None,
        include_superseded=False,
    )

    assert called
    assert captured_config is context.config
    payload = runtime_logs[0]["data"]
    assert isinstance(payload, dict)
    assert payload["native_ids"] == ["native"]

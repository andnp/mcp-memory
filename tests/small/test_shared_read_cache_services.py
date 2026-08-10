from __future__ import annotations

import gc
from pathlib import Path
import sqlite3
from threading import Event, Thread
from types import SimpleNamespace
from typing import cast
import warnings

import pytest

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import JournalEntry
from mcp_memory.mcp import services as services_module
from mcp_memory.mcp.services import read_memory_record_service, record_thought_service, search_memory_records_service
from mcp_memory.storage.shared_read_cache import (
    SharedReadCache,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheSearchRequest,
)
from mcp_memory.utils.db import SQLITE_BUSY_TIMEOUT_MILLISECONDS


pytestmark = pytest.mark.small


def test_shared_read_cache_closes_sqlite_connections(tmp_path: Path) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
        cache.invalidate_for_mutation(["memory-1"])
        cache.close()
        del cache
        gc.collect()

    assert not [warning for warning in caught if warning.category is ResourceWarning]


class FakeTelemetryRepository:
    def __init__(self) -> None:
        self.search_calls: list[dict[str, object]] = []
        self.read_calls: list[dict[str, object]] = []

    def record_search(self, **kwargs) -> None:
        self.search_calls.append(kwargs)

    def record_read(self, **kwargs) -> None:
        self.read_calls.append(kwargs)


class FakeRuntimeLogs:
    def write_log(self, **kwargs) -> None:
        return None


class SuccessfulSearchService:
    def search_memories(self, **kwargs):
        del kwargs
        return [
            SimpleNamespace(
                memory_id="memory-1",
                title="Warm cache",
                summary="Fresh authoritative result",
                memory_type="fact",
                status="active",
                tags=["cache"],
                workspace_ids=["workspace-123"],
                score=0.9,
            )
        ]


class LegacyRetrievalAdapter:
    """Test-only facade seam for lightweight native-service doubles."""

    def __init__(self, context) -> None:
        self.context = context

    def search_sync(self, query: str, **kwargs):
        results = self.context.relational_search.search_memories(query=query, **kwargs)
        return SimpleNamespace(results=[self._result(result) for result in results])

    async def search(self, query: str, **kwargs):
        return self.search_sync(query, **kwargs)

    @staticmethod
    def _result(result):
        record = SimpleNamespace(
            source_id=result.memory_id,
            title=result.title,
            body=result.summary,
            metadata={
                "summary": result.summary,
                "memory_type": result.memory_type,
                "memory_status": result.status,
                "tags": result.tags,
                "workspace_ids": result.workspace_ids,
                "memory_ref": getattr(result, "memory_ref", None),
            },
            status=SimpleNamespace(value=result.status),
            storage_key=f"memory:memory:{result.memory_id}",
            workspace_id=result.workspace_ids[0] if result.workspace_ids else None,
        )
        return SimpleNamespace(
            record=record,
            score=result.score,
            provenance=SimpleNamespace(to_dict=lambda: {}),
        )


class CountingSearchService:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls = 0

    def search_memories(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.should_fail:
            raise TimeoutError("authoritative search timed out")
        return [
            SimpleNamespace(
                memory_id="memory-1",
                title="Warm cache",
                summary="Fresh authoritative result",
                memory_type="fact",
                status="active",
                tags=["cache"],
                workspace_ids=["workspace-123"],
                score=0.9,
                ranking_debug=None,
            )
        ]


class BlockingProjectionAwareSearchService:
    def __init__(self, *, token_by_memory_id: dict[str, str] | None = None) -> None:
        self.calls = 0
        self.started = Event()
        self.release = Event()
        self.token_by_memory_id = token_by_memory_id or {}
        self.validation_calls: list[list[str]] = []

    def search_memories(self, **kwargs):
        del kwargs
        self.calls += 1
        self.started.set()
        assert self.release.wait(timeout=5.0)
        return [
            SimpleNamespace(
                memory_id="memory-1",
                title="Warm cache",
                summary="Fresh authoritative result",
                memory_type="fact",
                status="active",
                tags=["cache"],
                workspace_ids=["workspace-123"],
                score=0.9,
                ranking_debug=None,
            )
        ]

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        self.validation_calls.append(list(memory_ids))
        return {
            memory_id: token
            for memory_id, token in self.token_by_memory_id.items()
            if memory_id in memory_ids
        }


class FailingSearchService:
    def search_memories(self, **kwargs):
        del kwargs
        raise TimeoutError("authoritative search timed out")


class FailingProjectionAwareSearchService(FailingSearchService):
    def __init__(self, token_by_memory_id: dict[str, str] | None = None) -> None:
        self.token_by_memory_id = token_by_memory_id or {}
        self.validation_calls: list[list[str]] = []

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        self.validation_calls.append(list(memory_ids))
        return {
            memory_id: token
            for memory_id, token in self.token_by_memory_id.items()
            if memory_id in memory_ids
        }


class ProjectionAwareSearchService:
    def __init__(
        self,
        *,
        token_by_memory_id: dict[str, str] | None = None,
        status: str = "active",
    ) -> None:
        self.token_by_memory_id = token_by_memory_id or {}
        self.status = status
        self.validation_calls: list[list[str]] = []

    def search_memories(self, **kwargs):
        del kwargs
        return [
            SimpleNamespace(
                memory_id="memory-1",
                title="Warm cache",
                summary="Fresh authoritative result",
                memory_type="fact",
                status=self.status,
                tags=["cache"],
                workspace_ids=["workspace-123"],
                score=0.9,
                ranking_debug=None,
            )
        ]

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        self.validation_calls.append(list(memory_ids))
        return {
            memory_id: token
            for memory_id, token in self.token_by_memory_id.items()
            if memory_id in memory_ids
        }


class DeletingSearchService:
    def __init__(self) -> None:
        self.deleted = False
        self.calls = 0
        self.validation_calls: list[list[str]] = []

    def search_memories(self, **kwargs):
        del kwargs
        self.calls += 1
        if self.deleted:
            return []
        return [
            SimpleNamespace(
                memory_id="memory-1",
                title="Warm cache",
                summary="Fresh authoritative result",
                memory_type="fact",
                status="active",
                tags=["cache"],
                workspace_ids=["workspace-123"],
                score=0.9,
                ranking_debug=None,
            )
        ]

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        self.validation_calls.append(list(memory_ids))
        if self.deleted:
            return {}
        return {"memory-1": "token-1"}


class WritebackCapableJournal:
    def __init__(self, *, should_fail: bool = False, block_on_record: Event | None = None) -> None:
        self.should_fail = should_fail
        self.block_on_record = block_on_record
        self.completed = Event()
        self.entries: list[JournalEntry] = []
        self._next_id = 1

    def record_with_timestamp(
        self,
        content: str,
        workspace_id: str | None = None,
        *,
        timestamp: float | None = None,
    ) -> JournalEntry:
        try:
            if self.block_on_record is not None:
                self.block_on_record.wait(timeout=1.0)
            if self.should_fail:
                raise TimeoutError("authoritative postgres write timed out")
            normalized_content = content.strip()
            resolved_timestamp = 0.0 if timestamp is None else float(timestamp)
            for entry in self.entries:
                if (
                    entry.content == normalized_content
                    and entry.workspace_id == workspace_id
                    and entry.timestamp == resolved_timestamp
                ):
                    return entry
            entry = JournalEntry(
                id=self._next_id,
                content=normalized_content,
                workspace_id=workspace_id,
                timestamp=resolved_timestamp,
                status="pending",
            )
            self._next_id += 1
            self.entries.append(entry)
            return entry
        finally:
            self.completed.set()


def _build_read_result(*, summary: str = "Fresh authoritative record") -> SimpleNamespace:
    record = SimpleNamespace(
        id="memory-1",
        title="Warm cache",
        content="Cached content",
        summary=summary,
        type="fact",
        status="active",
        created_at="2026-03-28T00:00:00Z",
        updated_at="2026-03-28T00:00:00Z",
        read_count=1,
        access_score=0.5,
        last_accessed_at=None,
        last_surfaced_at=None,
        metadata={},
        workspace_ids=["workspace-123"],
        tags=["cache"],
    )
    relationship = SimpleNamespace(
        source_id="memory-1",
        target_id="memory-2",
        link_type="related_to",
        context=None,
    )
    superseded = SimpleNamespace(
        id="memory-0",
        title="Old cache",
        content="Old content",
        summary="Old summary",
        type="fact",
        status="superseded",
        created_at="2026-03-27T00:00:00Z",
        updated_at="2026-03-27T00:00:00Z",
        read_count=0,
        access_score=0.0,
        last_accessed_at=None,
        last_surfaced_at=None,
        metadata={},
        workspace_ids=["workspace-123"],
        tags=["cache"],
    )
    return SimpleNamespace(
        record=record,
        relationships={"outbound": [relationship], "inbound": []},
        superseded=[superseded],
    )


def _build_cached_read_payload(*, summary: str) -> dict[str, object]:
    return {
        "status": "ok",
        "record": {
            "id": "memory-1",
            "title": "Warm cache",
            "content": "Cached content",
            "summary": summary,
            "type": "fact",
            "status": "active",
            "created_at": "2026-03-28T00:00:00Z",
            "updated_at": "2026-03-28T00:00:00Z",
            "read_count": 1,
            "access_score": 0.5,
            "last_accessed_at": None,
            "last_surfaced_at": None,
            "metadata": {},
            "workspace_ids": ["workspace-123"],
            "tags": ["cache"],
        },
        "relationships": {
            "outbound": [
                {
                    "source_id": "memory-1",
                    "target_id": "memory-2",
                    "link_type": "related_to",
                    "context": None,
                }
            ],
            "inbound": [],
        },
        "superseded": [
            {
                "id": "memory-0",
                "title": "Old cache",
                "content": "Old content",
                "summary": "Old summary",
                "type": "fact",
                "status": "superseded",
                "created_at": "2026-03-27T00:00:00Z",
                "updated_at": "2026-03-27T00:00:00Z",
                "read_count": 0,
                "access_score": 0.0,
                "last_accessed_at": None,
                "last_surfaced_at": None,
                "metadata": {},
                "workspace_ids": ["workspace-123"],
                "tags": ["cache"],
            }
        ],
    }


class ConfigurableReadService:
    def __init__(
        self,
        *,
        read_result: object = None,
        token_responses: list[dict[str, str] | Exception] | None = None,
    ) -> None:
        self._read_result = read_result
        self._token_responses = list(token_responses or [])
        self.read_calls = 0
        self.validation_calls = 0

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        assert memory_ids == ["memory-1"]
        self.validation_calls += 1
        if not self._token_responses:
            return {}
        response = self._token_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def read_memory(self, memory_id: str):
        assert memory_id == "memory-1"
        self.read_calls += 1
        if isinstance(self._read_result, Exception):
            raise self._read_result
        return self._read_result


def _build_context(*, read_cache: SharedReadCache, relational_search: object) -> ApplicationContext:
    context = ApplicationContext(
        workspace_id="workspace-123",
        storage_backend="postgres",
        relational_search=cast(MemorySearchPort, relational_search),
        read_cache=read_cache,
        retrieval_telemetry=FakeTelemetryRepository(),
        runtime_logs=FakeRuntimeLogs(),
    )
    setattr(context, "memory_retrieval", LegacyRetrievalAdapter(context))
    return context


def _projection_payload(
    memory_id: str,
    *,
    title: str,
    summary: str,
    memory_type: str = "fact",
    status: str = "active",
    tags: list[str] | None = None,
    workspace_ids: list[str] | None = None,
    score: float = 0.5,
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "title": title,
        "summary": summary,
        "memory_type": memory_type,
        "status": status,
        "tags": tags or [],
        "workspace_ids": workspace_ids or ["workspace-123"],
        "score": score,
    }


def _compact_projection_payload(payload: dict[str, object] | dict[str, str | float | list]) -> dict[str, object]:
    return {
        key: payload[key]
        for key in ("memory_id", "title", "summary")
        if key in payload
    }


def test_search_memory_records_service_warms_shared_read_cache(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    ctx = _build_context(read_cache=cache, relational_search=SuccessfulSearchService())

    response = search_memory_records_service(ctx, {"query": "warm cache"})

    cached_payload = cache.load_search_response(
        SharedReadCacheSearchRequest(
            query="warm cache",
            workspace_id="workspace-123",
            limit=5,
            adaptive_limit=True,
            memory_type=None,
            status=None,
            include_superseded=False,
        )
    )

    assert response["status"] == "ok"
    assert cached_payload is not None
    assert cached_payload["results"] == response["results"]
    assert cached_payload["_cache_validation_tokens"] == {}
    assert "_cache_validation_tokens" not in response


def test_shared_read_cache_policy_identity_isolates_fresh_stale_and_inflight_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Policy changes cannot reuse fresh, stale, or coalesced responses."""
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    baseline = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
        policy_version="policy-v1",
        feature_fingerprint="baseline",
    )
    policy_variant = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
        policy_version="policy-v2",
        feature_fingerprint="baseline",
    )
    feature_variant = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
        policy_version="policy-v1",
        feature_fingerprint="rerank",
    )
    payload = {"status": "ok", "results": [{"memory_id": "baseline"}]}

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 10.0)
    cache.store_search_response(baseline, payload)
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 20.0)

    assert cache.load_search_response(baseline) == payload
    assert cache.load_search_response(policy_variant) is None
    assert cache.load_search_response(feature_variant) is None
    assert cache.load_fresh_search_response(baseline, ttl_seconds=5.0) is None
    assert cache.load_search_response(baseline) == payload
    assert cache.load_fresh_search_response(policy_variant, ttl_seconds=30.0) is None
    assert cache.load_search_response(policy_variant) is None

    baseline_leader = cache.begin_inflight_search(baseline)
    baseline_follower = cache.begin_inflight_search(baseline)
    variant_leader = cache.begin_inflight_search(policy_variant)

    assert baseline_leader.is_leader is True
    assert baseline_follower.is_leader is False
    assert variant_leader.is_leader is True

    cache.finish_inflight_search(baseline_leader, payload=payload)
    cache.finish_inflight_search(
        variant_leader,
        payload={"status": "ok", "results": [{"memory_id": "variant"}]},
    )

    assert cache.wait_for_inflight_search(baseline_follower) == payload


def test_shared_read_cache_invalidation_clears_exact_query_and_affected_entries(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="old query",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cache.store_read_response("memory-1", {"status": "ok"}, validation_token="v1")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload={"memory_id": "memory-1"},
                validation_token="v1",
            )
        ]
    )
    cache.store_search_response(request, {"status": "ok", "results": []})

    cache.invalidate_for_mutation(["memory-1"])

    assert cache.load_read_response("memory-1") is None
    assert cache.load_projection_entry("memory-1") is None
    assert cache.load_search_response(request) is None


def test_record_thought_service_queues_writeback_entry_on_authoritative_failure(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "writeback"
    config.storage.cache.max_outbox_entries = 4
    journal = WritebackCapableJournal(should_fail=True)
    ctx = ApplicationContext(
        config=config,
        workspace_id="workspace-123",
        storage_backend="postgres",
        journal=journal,
        read_cache=cache,
    )

    response = record_thought_service(ctx, {"content": "queue this thought"})

    outbox_entries = cache.list_record_thought_outbox_entries(limit=10)
    assert response == {"status": "recorded"}
    assert [entry.content for entry in outbox_entries] == ["queue this thought"]
    assert [entry.workspace_id for entry in outbox_entries] == ["workspace-123"]


def test_record_thought_service_leaves_writeback_queue_for_background_flush(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "writeback"
    config.storage.cache.max_outbox_entries = 4
    journal = WritebackCapableJournal(should_fail=True)
    ctx = ApplicationContext(
        config=config,
        workspace_id="workspace-123",
        storage_backend="postgres",
        journal=journal,
        read_cache=cache,
    )

    first = record_thought_service(ctx, {"content": "queue this thought"})
    journal.should_fail = False
    second = record_thought_service(ctx, {"content": "write the current thought"})

    assert first == {"status": "recorded"}
    assert second == {"status": "recorded"}
    assert [entry.content for entry in cache.list_record_thought_outbox_entries(limit=10)] == [
        "queue this thought"
    ]
    assert [entry.content for entry in journal.entries] == ["write the current thought"]


def test_record_thought_service_queues_writeback_entry_on_authoritative_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "writeback"
    config.storage.cache.max_outbox_entries = 4
    unblock_record = Event()
    journal = WritebackCapableJournal(should_fail=True, block_on_record=unblock_record)
    ctx = ApplicationContext(
        config=config,
        workspace_id="workspace-123",
        storage_backend="postgres",
        journal=journal,
        read_cache=cache,
    )

    monkeypatch.setattr(
        "mcp_memory.core.journal_operations._RECORD_THOUGHT_AUTHORITATIVE_TIMEOUT_SECONDS",
        0.01,
    )

    response = record_thought_service(ctx, {"content": "slow thought"})
    unblock_record.set()
    assert journal.completed.wait(timeout=1.0)

    outbox_entries = cache.list_record_thought_outbox_entries(limit=10)
    assert response == {"status": "recorded"}
    assert [entry.content for entry in outbox_entries] == ["slow thought"]


def test_search_memory_records_service_warms_projection_rows_and_persists_tokens(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = ProjectionAwareSearchService(token_by_memory_id={"memory-1": "token-1"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    response = search_memory_records_service(ctx, {"query": "warm cache"})

    projection_entry = cache.load_projection_entry("memory-1")

    assert response["status"] == "ok"
    assert search_service.validation_calls == [["memory-1"]]
    assert projection_entry is not None
    assert projection_entry.payload == response["results"][0]
    assert projection_entry.validation_token == "token-1"
    assert projection_entry.cached_at > 0.0


def test_load_validated_cached_projection_entries_filters_and_invalidates_stale_rows(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload={"memory_id": "memory-1", "title": "Valid projection"},
                validation_token="token-1",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-2",
                payload={"memory_id": "memory-2", "title": "Stale projection"},
                validation_token="stale-token",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-3",
                payload={"memory_id": "memory-3", "title": "Tokenless projection"},
                validation_token=None,
            ),
        ]
    )
    search_service = ProjectionAwareSearchService(
        token_by_memory_id={"memory-1": "token-1", "memory-2": "fresh-token"}
    )
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    entries = services_module._load_validated_cached_projection_entries(  # noqa: SLF001
        ctx,
        ["memory-1", "memory-2", "memory-3"],
        caller_kind="external",
    )

    assert [entry.memory_id for entry in entries] == ["memory-1"]
    assert search_service.validation_calls == [["memory-1", "memory-2"]]
    assert cache.load_projection_entry("memory-1") is not None
    assert cache.load_projection_entry("memory-2") is None
    assert cache.load_projection_entry("memory-3") is None


def test_search_memory_records_service_serves_stale_cache_on_authoritative_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cached_payload = {
        "status": "ok",
        "results": [
            {
                "memory_id": "memory-1",
                "title": "Warm cache",
                "summary": "Cached result",
                "memory_type": "fact",
                "status": "active",
                "tags": ["cache"],
                "workspace_ids": ["workspace-123"],
                "score": 0.9,
            }
        ],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, cached_payload)
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["status"] == "ok"
    assert response["cache_status"] == "stale_fallback"
    assert response["degraded"] is True
    assert response["results"] == [_compact_projection_payload(cached_payload["results"][0])]
    assert ctx.retrieval_telemetry.search_calls == []


def test_read_memory_record_service_validated_cached_hit_short_circuits_authoritative_read(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cached_payload = _build_cached_read_payload(summary="Cached validated record")
    cache.store_read_response("memory-1", cached_payload, validation_token="token-1")
    read_service = ConfigurableReadService(token_responses=[{"memory-1": "token-1"}])
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    assert response["status"] == "ok"
    assert response["record"] == {
        "id": "memory-1",
        "title": "Warm cache",
        "content": "Cached content",
    }
    assert "related_counts" not in response
    assert "relationships" not in response
    assert "superseded" not in response
    assert read_service.validation_calls == 1
    assert read_service.read_calls == 0
    assert len(ctx.retrieval_telemetry.read_calls) == 1
    assert ctx.retrieval_telemetry.read_calls[0]["caller_kind"] == "external"
    assert ctx.retrieval_telemetry.read_calls[0]["memory_id"] == "memory-1"


def test_read_memory_record_service_external_compacts_large_record_and_hides_superseded_by_default(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    read_result = _build_read_result(summary="Large authoritative record")
    read_result.record.content = "A" * 1500
    read_result.superseded[0].content = "B" * 900
    read_service = ConfigurableReadService(read_result=read_result)
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    assert response["status"] == "ok"
    assert response["record"]["content"] == "A" * 1500
    assert "related_counts" not in response
    assert "superseded" not in response


def test_read_memory_record_service_external_can_expand_superseded(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    read_result = _build_read_result(summary="Large authoritative record")
    read_result.superseded[0].content = "B" * 900
    read_service = ConfigurableReadService(read_result=read_result)
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1", "include_superseded": True})

    assert response["status"] == "ok"
    assert response["superseded"][0]["content"] == "B" * 900


def test_read_memory_record_service_mismatched_token_forces_authoritative_refresh(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Stale cached record"),
        validation_token="stale-token",
    )
    read_service = ConfigurableReadService(
        read_result=_build_read_result(summary="Fresh authoritative record"),
        token_responses=[{"memory-1": "fresh-token"}],
    )
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    cached_entry = cache.load_read_entry("memory-1")
    assert response["status"] == "ok"
    assert response["record"]["content"] == "Cached content"
    assert read_service.validation_calls == 1
    assert read_service.read_calls == 1
    assert cached_entry is not None
    assert cached_entry.payload == response
    assert cached_entry.validation_token == "fresh-token"


def test_read_memory_record_service_missing_authoritative_token_forces_authoritative_read(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Stale cached record"),
        validation_token="stale-token",
    )
    read_service = ConfigurableReadService(
        read_result=_build_read_result(summary="Fresh authoritative record"),
        token_responses=[{}, {"memory-1": "fresh-token"}],
    )
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    cached_entry = cache.load_read_entry("memory-1")
    assert response["status"] == "ok"
    assert response["record"]["content"] == "Cached content"
    assert read_service.validation_calls == 2
    assert read_service.read_calls == 1
    assert cached_entry is not None
    assert cached_entry.validation_token == "fresh-token"


def test_read_memory_record_service_validator_failure_and_read_failure_serve_stale_fallback(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cached_payload = _build_cached_read_payload(summary="Cached stale fallback")
    cache.store_read_response("memory-1", cached_payload, validation_token="token-1")
    read_service = ConfigurableReadService(
        read_result=TimeoutError("authoritative read timed out"),
        token_responses=[TimeoutError("authoritative validation timed out")],
    )
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    assert response["status"] == "ok"
    assert response["cache_status"] == "stale_fallback"
    assert response["degraded"] is True
    cached_record = cached_payload["record"]
    assert isinstance(cached_record, dict)
    assert response["record"]["content"] == cached_record["content"]
    assert "metadata" not in response["record"]
    assert "workspace_ids" not in response["record"]
    assert read_service.validation_calls == 1
    assert read_service.read_calls == 1
    assert ctx.retrieval_telemetry.read_calls == []


def test_read_memory_record_service_internal_calls_bypass_validated_cache_hit_path(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Cached validated record"),
        validation_token="token-1",
    )
    read_service = ConfigurableReadService(
        read_result=_build_read_result(summary="Fresh internal authoritative record"),
        token_responses=[{"memory-1": "token-1"}],
    )
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"}, caller_kind="internal")

    assert response["status"] == "ok"
    assert response["record"]["content"] == "Cached content"
    assert read_service.validation_calls == 0
    assert read_service.read_calls == 1


def test_read_memory_record_service_internal_calls_return_full_content(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    read_result = _build_read_result(summary="Fresh internal authoritative record")
    read_result.record.content = "A" * 1500
    read_result.superseded[0].content = "B" * 900
    read_service = ConfigurableReadService(read_result=read_result)
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"}, caller_kind="internal")

    assert response["status"] == "ok"
    assert response["record"]["content"] == "A" * 1500
    assert response["superseded"][0]["content"] == "B" * 900


def test_read_memory_record_service_does_not_fallback_on_memory_not_found(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_read_response(
        "memory-1",
        {
            "status": "ok",
            "record": {"id": "memory-1"},
            "relationships": {},
            "superseded": [],
        },
    )
    ctx = _build_context(
        read_cache=cache,
        relational_search=ConfigurableReadService(read_result=None),
    )

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    assert response == {"status": "error", "error": "memory_not_found"}


def test_search_memory_records_service_skips_cache_for_internal_calls(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cache.store_search_response(
        request,
        {
            "status": "ok",
            "results": [],
            "recommended_follow_up_tool": "read_memory_record",
            "guidance": "cached guidance",
        },
    )
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    with pytest.raises(TimeoutError):
        search_memory_records_service(ctx, {"query": "warm cache"}, caller_kind="internal")


def test_search_memory_records_service_internal_call_does_not_warm_projection_rows(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = ProjectionAwareSearchService(token_by_memory_id={"memory-1": "token-1"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    response = search_memory_records_service(ctx, {"query": "warm cache"}, caller_kind="internal")

    assert response["status"] == "ok"
    assert search_service.validation_calls == []
    assert cache.load_projection_entry("memory-1") is None


def test_search_memory_records_service_short_circuits_on_fresh_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cached_payload = {
        "status": "ok",
        "results": [
            {
                "memory_id": "memory-1",
                "title": "Warm cache",
                "summary": "Cached result",
                "memory_type": "fact",
                "status": "active",
                "tags": ["cache"],
                "workspace_ids": ["workspace-123"],
                "score": 0.9,
            }
        ],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, cached_payload)
    search_service = CountingSearchService()
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response == {
        "status": "ok",
        "results": [_compact_projection_payload(cached_payload["results"][0])],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    assert search_service.calls == 0
    assert ctx.retrieval_telemetry.search_calls == []


def test_search_memory_records_service_refreshes_fresh_cache_when_record_was_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = DeletingSearchService()
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    warm_response = search_memory_records_service(ctx, {"query": "warm cache"})

    search_service.deleted = True
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    refreshed_response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert warm_response["results"][0]["memory_ref"] == "memory-1"
    assert refreshed_response["results"] == []
    assert search_service.calls == 2
    assert search_service.validation_calls == [["memory-1"], ["memory-1"]]


def test_search_memory_records_service_refreshes_fresh_cache_when_record_token_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = ProjectionAwareSearchService(token_by_memory_id={"memory-1": "token-1"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    warm_response = search_memory_records_service(ctx, {"query": "warm cache"})

    search_service.token_by_memory_id["memory-1"] = "token-2"
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    refreshed_response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert warm_response["results"][0]["summary"] == "Fresh authoritative result"
    assert refreshed_response["results"][0]["summary"] == "Fresh authoritative result"
    assert search_service.validation_calls == [["memory-1"], ["memory-1"], ["memory-1"]]


def test_search_memory_records_service_invalidates_archive_status_token_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = ProjectionAwareSearchService(token_by_memory_id={"memory-1": "active-token"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    search_memory_records_service(ctx, {"query": "warm cache"})

    search_service.status = "archived"
    search_service.token_by_memory_id["memory-1"] = "archived-token"
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["status"] == "ok"
    assert search_service.validation_calls == [["memory-1"], ["memory-1"], ["memory-1"]]


def test_search_memory_records_service_refreshes_expired_cache_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cached_payload = {
        "status": "ok",
        "results": [],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, cached_payload)
    search_service = CountingSearchService()
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["status"] == "ok"
    assert response["results"][0]["summary"] == "Fresh authoritative result"
    assert search_service.calls == 1
    assert len(ctx.retrieval_telemetry.search_calls) == 1


def test_search_memory_records_service_debug_bypasses_fresh_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(
        request,
        {
            "status": "ok",
            "results": [],
            "recommended_follow_up_tool": "read_memory_record",
            "guidance": "cached guidance",
        },
    )
    search_service = CountingSearchService()
    ctx = _build_context(read_cache=cache, relational_search=search_service)
    original_execute_with_diagnostics = services_module.SearchMemoryRecordsOperation.execute_with_diagnostics

    def _fake_execute_with_diagnostics(self, **kwargs):
        del self
        results = search_service.search_memories(**kwargs)
        diagnostics = SimpleNamespace(timing_ms={"total": 1.0}, to_payload=lambda: {"mode": "debug"})
        return results, diagnostics

    monkeypatch.setattr(
        services_module.SearchMemoryRecordsOperation,
        "execute_with_diagnostics",
        _fake_execute_with_diagnostics,
    )
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)

    try:
        response = search_memory_records_service(ctx, {"query": "warm cache", "debug": True})
    finally:
        monkeypatch.setattr(
            services_module.SearchMemoryRecordsOperation,
            "execute_with_diagnostics",
            original_execute_with_diagnostics,
        )

    assert response["status"] == "ok"
    assert response["search_diagnostics"] == {"mode": "debug"}
    assert search_service.calls == 1


def test_search_memory_records_service_debug_call_does_not_warm_projection_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = ProjectionAwareSearchService(token_by_memory_id={"memory-1": "token-1"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)
    original_execute_with_diagnostics = services_module.SearchMemoryRecordsOperation.execute_with_diagnostics

    def _fake_execute_with_diagnostics(self, **kwargs):
        del self
        results = search_service.search_memories(**kwargs)
        diagnostics = SimpleNamespace(timing_ms={"total": 1.0}, to_payload=lambda: {"mode": "debug"})
        return results, diagnostics

    monkeypatch.setattr(
        services_module.SearchMemoryRecordsOperation,
        "execute_with_diagnostics",
        _fake_execute_with_diagnostics,
    )

    try:
        response = search_memory_records_service(ctx, {"query": "warm cache", "debug": True})
    finally:
        monkeypatch.setattr(
            services_module.SearchMemoryRecordsOperation,
            "execute_with_diagnostics",
            original_execute_with_diagnostics,
        )

    assert response["status"] == "ok"
    assert response["search_diagnostics"] == {"mode": "debug"}
    assert search_service.validation_calls == []
    assert cache.load_projection_entry("memory-1") is None


def test_search_memory_records_service_uses_stale_fallback_after_hot_hit_ttl_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cached_payload = {
        "status": "ok",
        "results": [
            {
                "memory_id": "memory-1",
                "title": "Warm cache",
                "summary": "Cached result",
                "memory_type": "fact",
                "status": "active",
                "tags": ["cache"],
                "workspace_ids": ["workspace-123"],
                "score": 0.9,
            }
        ],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, cached_payload)
    search_service = CountingSearchService(should_fail=True)
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["status"] == "ok"
    assert response["cache_status"] == "stale_fallback"
    assert response["degraded"] is True
    assert response["results"] == [_compact_projection_payload(cached_payload["results"][0])]
    assert search_service.calls == 1


def test_search_memory_records_service_stale_exact_fallback_takes_precedence_over_projection_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    stale_payload = {
        "status": "ok",
        "results": [_projection_payload("memory-stale", title="Exact stale hit", summary="Exact cached response")],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, stale_payload)
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-projection",
                payload=_projection_payload(
                    "memory-projection",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                ),
                validation_token="token-1",
            )
        ]
    )
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["cache_status"] == "stale_fallback"
    assert response["results"] == [_compact_projection_payload(stale_payload["results"][0])]


def test_search_memory_records_service_projection_fallback_activates_when_authoritative_fails_without_stale_exact_cache(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-2",
                payload=_projection_payload(
                    "memory-2",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                    tags=["cache", "warm"],
                ),
                validation_token="token-2",
            )
        ]
    )
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    with caplog.at_level("WARNING"):
        response = search_memory_records_service(ctx, {"query": "warm cache"})

    assert response["status"] == "ok"
    assert response["cache_status"] == "projection_fallback"
    assert response["degraded"] is True
    assert [result["memory_id"] for result in response["results"]] == ["memory-2"]
    assert any("projection-backed degraded search response" in message for message in caplog.messages)
    assert ctx.retrieval_telemetry.search_calls == []


@pytest.mark.parametrize(
    ("tokens", "expected_fallback", "expected_cached"),
    [
        ({"memory-1": "token-1"}, True, True),
        ({"memory-1": "changed-token"}, False, False),
        ({}, False, False),
    ],
    ids=["valid-token", "changed-token", "deleted-record"],
)
def test_projection_fallback_validates_projection_tokens(
    tmp_path: Path,
    tokens: dict[str, str],
    expected_fallback: bool,
    expected_cached: bool,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload=_projection_payload(
                    "memory-1",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                ),
                validation_token="token-1",
            )
        ]
    )
    search_service = FailingProjectionAwareSearchService(tokens)
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    if expected_fallback:
        response = search_memory_records_service(ctx, {"query": "warm cache"})
        assert response["cache_status"] == "projection_fallback"
        assert [result["memory_id"] for result in response["results"]] == ["memory-1"]
    else:
        with pytest.raises(TimeoutError, match="authoritative search timed out"):
            search_memory_records_service(ctx, {"query": "warm cache"})

    assert search_service.validation_calls == [["memory-1"]]
    assert (cache.load_projection_entry("memory-1") is not None) is expected_cached


def test_projection_fallback_backfills_limit_after_stale_projection(
    tmp_path: Path,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-stale",
                payload=_projection_payload(
                    "memory-stale",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                ),
                validation_token="stale-token",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-valid",
                payload=_projection_payload(
                    "memory-valid",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                ),
                validation_token="valid-token",
            ),
        ]
    )
    search_service = FailingProjectionAwareSearchService(
        {"memory-valid": "valid-token"}
    )
    ctx = _build_context(read_cache=cache, relational_search=search_service)

    response = search_memory_records_service(ctx, {"query": "warm cache", "limit": 1})

    assert response["cache_status"] == "projection_fallback"
    assert [result["memory_id"] for result in response["results"]] == ["memory-valid"]


def test_search_memory_records_service_projection_fallback_does_not_return_empty_success(
    tmp_path: Path,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload=_projection_payload(
                    "memory-1",
                    title="Unrelated projection",
                    summary="Does not match query tokens",
                ),
                validation_token="token-1",
            )
        ]
    )
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    with pytest.raises(TimeoutError, match="authoritative search timed out"):
        search_memory_records_service(ctx, {"query": "warm cache"})


def test_search_memory_records_service_projection_fallback_is_disabled_for_internal_and_debug_callers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload=_projection_payload(
                    "memory-1",
                    title="Warm cache projection",
                    summary="Projection fallback result",
                ),
                validation_token="token-1",
            )
        ]
    )
    ctx = _build_context(read_cache=cache, relational_search=FailingSearchService())

    with pytest.raises(TimeoutError, match="authoritative search timed out"):
        search_memory_records_service(ctx, {"query": "warm cache"}, caller_kind="internal")

    original_execute_with_diagnostics = services_module.SearchMemoryRecordsOperation.execute_with_diagnostics

    def _failing_execute_with_diagnostics(self, **kwargs):
        del self, kwargs
        raise TimeoutError("authoritative search timed out")

    monkeypatch.setattr(
        services_module.SearchMemoryRecordsOperation,
        "execute_with_diagnostics",
        _failing_execute_with_diagnostics,
    )
    try:
        with pytest.raises(TimeoutError, match="authoritative search timed out"):
            search_memory_records_service(ctx, {"query": "warm cache", "debug": True})
    finally:
        monkeypatch.setattr(
            services_module.SearchMemoryRecordsOperation,
            "execute_with_diagnostics",
            original_execute_with_diagnostics,
        )


def test_shared_read_cache_projection_search_respects_filters_and_ignores_tokenless_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 50.0)
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload=_projection_payload(
                    "memory-1",
                    title="Warm cache fact",
                    summary="Workspace-filtered active fact",
                    tags=["warm", "cache"],
                    workspace_ids=["workspace-123"],
                ),
                validation_token="token-1",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-2",
                payload=_projection_payload(
                    "memory-2",
                    title="Warm cache plan",
                    summary="Filtered by memory type",
                    memory_type="plan",
                    tags=["warm"],
                    workspace_ids=["workspace-123"],
                ),
                validation_token="token-2",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-3",
                payload=_projection_payload(
                    "memory-3",
                    title="Warm cache stale",
                    summary="Filtered by status",
                    status="stale",
                    tags=["warm"],
                    workspace_ids=["workspace-123"],
                ),
                validation_token="token-3",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-4",
                payload=_projection_payload(
                    "memory-4",
                    title="Warm cache superseded",
                    summary="Filtered by include_superseded",
                    status="superseded",
                    tags=["warm"],
                    workspace_ids=["workspace-123"],
                ),
                validation_token="token-4",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-5",
                payload=_projection_payload(
                    "memory-5",
                    title="Warm cache other workspace",
                    summary="Filtered by workspace",
                    tags=["warm"],
                    workspace_ids=["workspace-999"],
                ),
                validation_token="token-5",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-6",
                payload=_projection_payload(
                    "memory-6",
                    title="Warm cache tokenless",
                    summary="Should be ignored entirely",
                    tags=["warm", "cache"],
                    workspace_ids=["workspace-123"],
                ),
                validation_token=None,
            ),
        ]
    )

    base_request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=10,
        adaptive_limit=False,
        memory_type="fact",
        status="active",
        include_superseded=False,
    )

    default_results = cache.search_projection_payloads(base_request)
    include_superseded_results = cache.search_projection_payloads(
        SharedReadCacheSearchRequest(
            query="warm cache",
            workspace_id="workspace-123",
            limit=10,
            adaptive_limit=False,
            memory_type=None,
            status=None,
            include_superseded=True,
        )
    )

    assert [payload["memory_id"] for payload in default_results] == ["memory-1"]
    assert [payload["memory_id"] for payload in include_superseded_results] == [
        "memory-1",
        "memory-2",
        "memory-3",
        "memory-4",
    ]


def test_shared_read_cache_projection_search_ranks_by_hit_count_then_cached_at_then_memory_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 10.0)
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-c",
                payload=_projection_payload(
                    "memory-c",
                    title="Warm cache alpha beta",
                    summary="Hits both tokens",
                    tags=["alpha"],
                ),
                validation_token="token-c",
            )
        ]
    )
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 20.0)
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-b",
                payload=_projection_payload(
                    "memory-b",
                    title="Warm cache alpha",
                    summary="One token with newer cache time",
                ),
                validation_token="token-b",
            ),
            SharedReadCacheProjectionUpsert(
                memory_id="memory-a",
                payload=_projection_payload(
                    "memory-a",
                    title="Warm cache alpha",
                    summary="One token same cache time",
                ),
                validation_token="token-a",
            ),
        ]
    )

    results = cache.search_projection_payloads(
        SharedReadCacheSearchRequest(
            query="alpha beta",
            workspace_id="workspace-123",
            limit=10,
            adaptive_limit=False,
            memory_type=None,
            status=None,
            include_superseded=True,
        )
    )

    assert [payload["memory_id"] for payload in results] == ["memory-c", "memory-a", "memory-b"]


def test_shared_read_cache_metrics_snapshot_tracks_inventory_and_zero_safe_rates(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    empty_snapshot = cache.get_metrics_snapshot()

    cache.store_search_response(
        SharedReadCacheSearchRequest(
            query="warm cache",
            workspace_id="workspace-123",
            limit=5,
            adaptive_limit=True,
            memory_type=None,
            status=None,
            include_superseded=False,
        ),
        {
            "status": "ok",
            "results": [],
            "recommended_follow_up_tool": "read_memory_record",
            "guidance": "cached guidance",
        },
    )
    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Cached read"),
        validation_token="token-1",
    )
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-1",
                payload=_projection_payload(
                    "memory-1",
                    title="Warm cache projection",
                    summary="Projection cache row",
                ),
                validation_token="token-1",
            )
        ]
    )
    cache.increment_metric("search_requests", amount=4)
    cache.increment_metric("fresh_exact_search_hits", amount=2)
    cache.increment_metric("projection_fallbacks", amount=1)
    cache.increment_metric("read_requests", amount=3)
    cache.increment_metric("validated_read_hits", amount=1)
    cache.increment_metric("read_validation_failures", amount=1)
    cache.increment_metric("warmed_projection_rows", amount=5)

    snapshot = cache.get_metrics_snapshot()

    assert empty_snapshot.cached_search_result_count == 0
    assert empty_snapshot.cached_read_record_count == 0
    assert empty_snapshot.cached_projection_count == 0
    assert empty_snapshot.fresh_exact_search_hit_rate == 0.0
    assert empty_snapshot.validated_read_hit_rate == 0.0
    assert empty_snapshot.recent.window_minutes == 15
    assert empty_snapshot.recent.search_requests == 0
    assert empty_snapshot.recent.read_requests == 0
    assert empty_snapshot.recent.fresh_exact_search_hit_rate == 0.0
    assert empty_snapshot.recent.validated_read_hit_rate == 0.0

    assert snapshot.cached_search_result_count == 1
    assert snapshot.cached_read_record_count == 1
    assert snapshot.cached_projection_count == 1
    assert snapshot.search_requests == 4
    assert snapshot.fresh_exact_search_hits == 2
    assert snapshot.projection_fallbacks == 1
    assert snapshot.read_requests == 3
    assert snapshot.validated_read_hits == 1
    assert snapshot.read_validation_failures == 1
    assert snapshot.warmed_projection_rows == 5
    assert snapshot.fresh_exact_search_hit_rate == 0.5
    assert snapshot.projection_fallback_rate == 0.25
    assert snapshot.validated_read_hit_rate == pytest.approx(0.3333)
    assert snapshot.read_validation_failure_rate == pytest.approx(0.3333)
    assert snapshot.recent.window_minutes == 15
    assert snapshot.recent.search_requests == 4
    assert snapshot.recent.fresh_exact_search_hits == 2
    assert snapshot.recent.projection_fallbacks == 1
    assert snapshot.recent.read_requests == 3
    assert snapshot.recent.validated_read_hits == 1
    assert snapshot.recent.read_validation_failures == 1
    assert snapshot.recent.warmed_projection_rows == 5
    assert snapshot.recent.fresh_exact_search_hit_rate == 0.5
    assert snapshot.recent.projection_fallback_rate == 0.25
    assert snapshot.recent.validated_read_hit_rate == pytest.approx(0.3333)
    assert snapshot.recent.read_validation_failure_rate == pytest.approx(0.3333)


def test_shared_read_cache_recent_metrics_snapshot_aggregates_across_bucket_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    minute_offsets = [0, 5, 14, 15, 16]
    for offset in minute_offsets:
        now = 1_000.0 + (offset * 60.0)
        monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda now=now: now)
        cache.increment_metric("search_requests", amount=1)
        cache.increment_metric("fresh_exact_search_hits", amount=1)
        cache.increment_metric("read_requests", amount=2)
        cache.increment_metric("validated_read_hits", amount=1)
        cache.increment_metric("warmed_projection_rows", amount=3)

    snapshot = cache.get_metrics_snapshot(now=1_000.0 + (16 * 60.0) + 59.0)

    assert snapshot.search_requests == 5
    assert snapshot.read_requests == 10
    assert snapshot.warmed_projection_rows == 15
    assert snapshot.recent.window_minutes == 15
    assert snapshot.recent.search_requests == 4
    assert snapshot.recent.fresh_exact_search_hits == 4
    assert snapshot.recent.read_requests == 8
    assert snapshot.recent.validated_read_hits == 4
    assert snapshot.recent.warmed_projection_rows == 12
    assert snapshot.recent.fresh_exact_search_hit_rate == 1.0
    assert snapshot.recent.validated_read_hit_rate == 0.5


def test_search_memory_records_service_records_cache_metrics_for_external_non_debug_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    ctx = _build_context(read_cache=cache, relational_search=CountingSearchService())

    warm_response = search_memory_records_service(ctx, {"query": "warm cache"})

    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, warm_response)
    ctx.relational_search = cast(MemorySearchPort, CountingSearchService())

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    fresh_hit_response = search_memory_records_service(ctx, {"query": "warm cache"})

    ctx.relational_search = cast(MemorySearchPort, FailingSearchService())
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    stale_response = search_memory_records_service(ctx, {"query": "warm cache"})

    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id="memory-2",
                payload=_projection_payload(
                    "memory-2",
                    title="Projection cache path",
                    summary="Projection cache fallback",
                    tags=["projection", "cache"],
                ),
                validation_token="token-2",
            )
        ]
    )
    projection_response = search_memory_records_service(ctx, {"query": "projection cache"})

    snapshot = cache.get_metrics_snapshot()

    assert warm_response["status"] == "ok"
    assert fresh_hit_response["status"] == "ok"
    assert stale_response["cache_status"] == "stale_fallback"
    assert projection_response["cache_status"] == "projection_fallback"
    assert snapshot.search_requests == 4
    assert snapshot.fresh_exact_search_hits == 1
    assert snapshot.stale_exact_search_fallbacks == 1
    assert snapshot.projection_fallbacks == 1
    assert snapshot.warmed_projection_rows == 1
    assert snapshot.fresh_exact_search_hit_rate == 0.25
    assert snapshot.stale_exact_search_fallback_rate == 0.25
    assert snapshot.projection_fallback_rate == 0.25


def test_search_memory_records_service_coalesces_concurrent_identical_external_requests(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    search_service = BlockingProjectionAwareSearchService(token_by_memory_id={"memory-1": "token-1"})
    ctx = _build_context(read_cache=cache, relational_search=search_service)
    responses: list[dict[str, object] | None] = [None, None, None]
    failures: list[BaseException] = []
    follower_joined = Event()
    original_begin_inflight_search = cache.begin_inflight_search

    def _begin_inflight_search(request: SharedReadCacheSearchRequest):
        entry = original_begin_inflight_search(request)
        if not entry.is_leader:
            follower_joined.set()
        return entry

    cache.begin_inflight_search = _begin_inflight_search

    def _invoke(index: int) -> None:
        try:
            responses[index] = search_memory_records_service(ctx, {"query": "warm cache"})
        except BaseException as error:  # pragma: no cover - test harness capture
            failures.append(error)

    threads = [Thread(target=_invoke, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    assert search_service.started.wait(timeout=5.0)
    assert follower_joined.wait(timeout=5.0)
    search_service.release.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert failures == []
    assert search_service.calls == 1
    assert search_service.validation_calls
    assert all(memory_ids == ["memory-1"] for memory_ids in search_service.validation_calls)
    assert responses[0] == responses[1] == responses[2]
    assert responses[0] is not None
    assert responses[0]["status"] == "ok"
    assert "_cache_validation_tokens" not in responses[0]
    assert len(ctx.retrieval_telemetry.search_calls) >= 2

    snapshot = cache.get_metrics_snapshot()
    assert snapshot.search_requests == 3
    assert snapshot.warmed_projection_rows == 1


def test_shared_read_cache_wait_for_inflight_search_times_out_for_unfinished_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    monkeypatch.setattr(
        "mcp_memory.storage.shared_read_cache._INFLIGHT_SEARCH_FOLLOWER_WAIT_TIMEOUT_SECONDS",
        0.01,
    )

    leader = cache.begin_inflight_search(request)
    follower = cache.begin_inflight_search(request)

    assert leader.is_leader is True
    assert follower.is_leader is False
    with pytest.raises(TimeoutError, match="Timed out waiting for in-flight shared read-cache search"):
        cache.wait_for_inflight_search(follower)


def test_shared_read_cache_timeout_evicts_stale_inflight_entry_so_later_request_can_lead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    monkeypatch.setattr(
        "mcp_memory.storage.shared_read_cache._INFLIGHT_SEARCH_FOLLOWER_WAIT_TIMEOUT_SECONDS",
        0.01,
    )

    original_leader = cache.begin_inflight_search(request)
    stale_follower = cache.begin_inflight_search(request)

    with pytest.raises(TimeoutError, match="Timed out waiting for in-flight shared read-cache search"):
        cache.wait_for_inflight_search(stale_follower)

    replacement_leader = cache.begin_inflight_search(request)
    replacement_follower = cache.begin_inflight_search(request)
    payload = {"status": "ok", "results": []}

    assert replacement_leader.is_leader is True
    assert replacement_follower.is_leader is False

    cache.finish_inflight_search(replacement_leader, payload=payload)

    assert cache.wait_for_inflight_search(replacement_follower) == payload

    cache.finish_inflight_search(original_leader, payload={"status": "ok", "results": ["stale"]})
    assert cache.begin_inflight_search(request).is_leader is True


def test_search_memory_records_service_coalesced_authoritative_failure_preserves_stale_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    request = SharedReadCacheSearchRequest(
        query="warm cache",
        workspace_id="workspace-123",
        limit=5,
        adaptive_limit=True,
        memory_type=None,
        status=None,
        include_superseded=False,
    )
    cached_payload = {
        "status": "ok",
        "results": [_projection_payload("memory-1", title="Warm cache", summary="Cached result")],
        "recommended_follow_up_tool": "read_memory_record",
        "guidance": "cached guidance",
    }
    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 100.0)
    cache.store_search_response(request, cached_payload)
    search_service = BlockingProjectionAwareSearchService()
    original_search_memories = search_service.search_memories

    def _failing_search_memories(**kwargs):
        original_search_memories(**kwargs)
        raise TimeoutError("authoritative search timed out")

    search_service.search_memories = _failing_search_memories
    ctx = _build_context(read_cache=cache, relational_search=search_service)
    responses: list[dict[str, object] | None] = [None, None]
    failures: list[BaseException] = []
    follower_joined = Event()
    original_begin_inflight_search = cache.begin_inflight_search

    def _begin_inflight_search(request: SharedReadCacheSearchRequest):
        entry = original_begin_inflight_search(request)
        if not entry.is_leader:
            follower_joined.set()
        return entry

    cache.begin_inflight_search = _begin_inflight_search

    def _invoke(index: int) -> None:
        try:
            responses[index] = search_memory_records_service(ctx, {"query": "warm cache"})
        except BaseException as error:  # pragma: no cover - test harness capture
            failures.append(error)

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 106.0)
    threads = [Thread(target=_invoke, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    assert search_service.started.wait(timeout=5.0)
    assert follower_joined.wait(timeout=5.0)
    search_service.release.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert failures == []
    assert search_service.calls == 1
    assert responses[0] == responses[1]
    assert responses[0] is not None
    assert responses[0]["cache_status"] == "stale_fallback"


def test_read_memory_record_service_records_cache_metrics_for_validation_paths(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Validated cached record"),
        validation_token="token-1",
    )
    validated_ctx = _build_context(
        read_cache=cache,
        relational_search=ConfigurableReadService(token_responses=[{"memory-1": "token-1"}]),
    )
    validated_response = read_memory_record_service(validated_ctx, {"memory_id": "memory-1"})

    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Mismatched cached record"),
        validation_token="stale-token",
    )
    mismatch_ctx = _build_context(
        read_cache=cache,
        relational_search=ConfigurableReadService(
            read_result=_build_read_result(summary="Authoritative mismatch refresh"),
            token_responses=[{"memory-1": "fresh-token"}],
        ),
    )
    mismatch_response = read_memory_record_service(mismatch_ctx, {"memory_id": "memory-1"})

    cache.store_read_response(
        "memory-1",
        _build_cached_read_payload(summary="Validation failure cached record"),
        validation_token="token-2",
    )
    failure_ctx = _build_context(
        read_cache=cache,
        relational_search=ConfigurableReadService(
            read_result=_build_read_result(summary="Authoritative after validation failure"),
            token_responses=[TimeoutError("validation timeout")],
        ),
    )
    failure_response = read_memory_record_service(failure_ctx, {"memory_id": "memory-1"})

    snapshot = cache.get_metrics_snapshot()

    assert validated_response["record"]["content"] == "Cached content"
    assert mismatch_response["record"]["content"] == "Cached content"
    assert failure_response["record"]["content"] == "Cached content"
    assert snapshot.read_requests == 3
    assert snapshot.validated_read_hits == 1
    assert snapshot.read_validation_mismatches == 1
    assert snapshot.read_validation_failures == 1
    assert snapshot.validated_read_hit_rate == pytest.approx(0.3333)
    assert snapshot.read_validation_mismatch_rate == pytest.approx(0.3333)
    assert snapshot.read_validation_failure_rate == pytest.approx(0.3333)


def test_shared_read_cache_configures_wal_and_busy_timeout(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")

    with cache._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == SQLITE_BUSY_TIMEOUT_MILLISECONDS


def test_shared_read_cache_reads_during_concurrent_write_lock(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cache.store_read_response("memory-1", {"record": {"id": "memory-1"}})

    blocker = sqlite3.connect(cache._db_path, timeout=0.1)
    blocker.execute("BEGIN EXCLUSIVE")
    completed = Event()
    responses: list[dict[str, object] | None] = []

    def read_cached_record() -> None:
        try:
            responses.append(cache.load_read_response("memory-1"))
        finally:
            completed.set()

    reader = Thread(target=read_cached_record)
    reader.start()
    assert completed.wait(timeout=1.0)
    reader.join(timeout=1.0)
    blocker.rollback()
    blocker.close()

    assert responses == [{"record": {"id": "memory-1"}}]

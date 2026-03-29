from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp import services as services_module
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.storage.shared_read_cache import (
    SharedReadCache,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheSearchRequest,
)


pytestmark = pytest.mark.small


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


class FailingSearchService:
    def search_memories(self, **kwargs):
        del kwargs
        raise TimeoutError("authoritative search timed out")


class ProjectionAwareSearchService:
    def __init__(self, *, token_by_memory_id: dict[str, str] | None = None) -> None:
        self.token_by_memory_id = token_by_memory_id or {}
        self.validation_calls: list[list[str]] = []

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
    return ApplicationContext(
        workspace_id="workspace-123",
        storage_backend="postgres",
        relational_search=relational_search,
        read_cache=read_cache,
        retrieval_telemetry=FakeTelemetryRepository(),
        runtime_logs=FakeRuntimeLogs(),
    )


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
    assert cached_payload == response


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
    assert response["results"] == cached_payload["results"]
    assert ctx.retrieval_telemetry.search_calls == []


def test_read_memory_record_service_validated_cached_hit_short_circuits_authoritative_read(tmp_path: Path) -> None:
    cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
    cached_payload = _build_cached_read_payload(summary="Cached validated record")
    cache.store_read_response("memory-1", cached_payload, validation_token="token-1")
    read_service = ConfigurableReadService(token_responses=[{"memory-1": "token-1"}])
    ctx = _build_context(read_cache=cache, relational_search=read_service)

    response = read_memory_record_service(ctx, {"memory_id": "memory-1"})

    assert response == cached_payload
    assert read_service.validation_calls == 1
    assert read_service.read_calls == 0
    assert ctx.retrieval_telemetry.read_calls == []


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
    assert response["record"]["summary"] == "Fresh authoritative record"
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
    assert response["record"]["summary"] == "Fresh authoritative record"
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
    assert response["record"] == cached_payload["record"]
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
    assert response["record"]["summary"] == "Fresh internal authoritative record"
    assert read_service.validation_calls == 0
    assert read_service.read_calls == 1


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

    assert response == cached_payload
    assert search_service.calls == 0
    assert ctx.retrieval_telemetry.search_calls == []


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
    assert response["results"] == cached_payload["results"]
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
    assert response["results"] == stale_payload["results"]


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
    ctx.relational_search = CountingSearchService()

    monkeypatch.setattr("mcp_memory.storage.shared_read_cache.time", lambda: 104.0)
    fresh_hit_response = search_memory_records_service(ctx, {"query": "warm cache"})

    ctx.relational_search = FailingSearchService()
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

    assert validated_response["record"]["summary"] == "Validated cached record"
    assert mismatch_response["record"]["summary"] == "Authoritative mismatch refresh"
    assert failure_response["record"]["summary"] == "Authoritative after validation failure"
    assert snapshot.read_requests == 3
    assert snapshot.validated_read_hits == 1
    assert snapshot.read_validation_mismatches == 1
    assert snapshot.read_validation_failures == 1
    assert snapshot.validated_read_hit_rate == pytest.approx(0.3333)
    assert snapshot.read_validation_mismatch_rate == pytest.approx(0.3333)
    assert snapshot.read_validation_failure_rate == pytest.approx(0.3333)

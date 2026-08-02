from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest
from searchkernel.ports.federation import (
    CallerAuthorizationContext,
    SearchRequest,
)

from mcp_memory.daemon_dispatch import dispatch_federation_request
from mcp_memory.daemon_app import create_daemon_app
from mcp_memory.integrations.federation_source import MemoryFederationSource


pytestmark = pytest.mark.small


def _result(
    memory_id: str = "memory-1",
    *,
    status: str = "active",
    summary: str = "safe summary",
) -> SimpleNamespace:
    record = SimpleNamespace(
        source_id=memory_id,
        workspace_id="workspace-1",
        title="Memory title",
        body="private body must not cross federation",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 1, 2, tzinfo=UTC),
        status=SimpleNamespace(value=status),
        metadata={
            "memory_ref": 42,
            "summary": summary,
            "memory_type": "fact",
            "memory_status": status,
            "tags": ["tag"],
            "workspace_ids": ["workspace-1"],
        },
    )
    return SimpleNamespace(record=record, score=0.91)


class RetrievalDouble:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.request: dict[str, object] | None = None

    async def search(self, query: str, **kwargs: object) -> SimpleNamespace:
        self.request = {"query": query, **kwargs}
        return SimpleNamespace(
            results=self.results,
            diagnostics=("vector unavailable",),
            cache_diagnostics=(),
            failures=(),
            degraded=True,
        )


class RepositoryDouble:
    def get_search_epochs(self) -> dict[str, int]:
        return {"links": 3, "records": 7}


@pytest.mark.asyncio
async def test_source_uses_canonical_retrieval_and_preserves_provenance() -> None:
    retrieval = RetrievalDouble([_result()])
    source = MemoryFederationSource(cast(Any, retrieval), cast(Any, RepositoryDouble()))
    request = SearchRequest(
        query="facts",
        top_k=4,
        filters={"workspace_id": "workspace-1"},
        caller=CallerAuthorizationContext("caller-1"),
        request_id="request-1",
    )

    response = await source.search(request)

    assert retrieval.request == {
        "query": "facts",
        "limit": 4,
        "workspace_id": "workspace-1",
        "memory_type": None,
        "status": None,
        "include_superseded": False,
    }
    assert response.partial is True
    assert response.index_epoch == '{"links":3,"records":7}'
    assert response.warnings == ("vector unavailable",)
    hit = response.hits[0]
    assert hit.source_kind == "memory"
    assert hit.source_id == "memory-1"
    assert hit.uri == "memory://42"
    assert hit.provenance.request_id == "request-1"
    assert hit.provenance.source == source.source_identity
    assert hit.rerank_text is None
    assert "private body" not in hit.snippet


@pytest.mark.asyncio
async def test_source_authorization_filters_hits_before_text_exposure() -> None:
    retrieval = RetrievalDouble([_result("allowed"), _result("denied")])
    source = MemoryFederationSource(
        cast(Any, retrieval),
        authorize=lambda request, result: result.record.source_id == "allowed",
    )

    response = await source.search(SearchRequest(query="query"))

    assert [hit.source_id for hit in response.hits] == ["allowed"]
    assert response.hits[0].source_rank == 1


def test_federation_dispatch_exposes_contract_capabilities_and_health() -> None:
    retrieval = RetrievalDouble([_result()])
    source = MemoryFederationSource(cast(Any, retrieval), cast(Any, RepositoryDouble()))
    service = SimpleNamespace(_retrieval=retrieval, _repository=RepositoryDouble())
    routes = SimpleNamespace(service=service, management=SimpleNamespace())

    capabilities = asyncio.run(
        dispatch_federation_request(routes, "/v1/search/capabilities", {})
    )
    health = asyncio.run(dispatch_federation_request(routes, "/v1/health", {}))
    search = asyncio.run(
        dispatch_federation_request(
            routes,
            "/v1/search",
            cast(
                dict[str, object],
                SearchRequest(query="facts", request_id="dispatch-request").to_dict(),
            ),
        )
    )

    assert capabilities["contract_versions"] == ["v1"]
    assert health["source"] == source.source_identity.to_dict()
    assert health["index_epoch"] == '{"links":3,"records":7}'
    assert search["hits"][0]["provenance"]["request_id"] == "dispatch-request"


def test_federation_dispatch_prefers_composed_route_source() -> None:
    source = MemoryFederationSource(cast(Any, RetrievalDouble([_result()])))
    routes = SimpleNamespace(
        federation_source=source,
        service=SimpleNamespace(_retrieval=None, _repository=None),
    )

    capabilities = asyncio.run(
        dispatch_federation_request(routes, "/v1/search/capabilities", {})
    )

    assert capabilities["contract_versions"] == ["v1"]


def test_http_endpoint_registers_v1_route() -> None:
    app = create_daemon_app(port=0)

    assert any(
        getattr(route, "path", None) == "/v1/{v1_path:path}"
        for route in app.routes
    )

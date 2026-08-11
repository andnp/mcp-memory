from __future__ import annotations

from dataclasses import dataclass
from threading import get_ident

import pytest

from mcp_memory.core.retrieval import (
    BoundedRetrievalService,
    RetrievalDiagnostics,
    RetrievalFailure,
    RetrievalRequest,
    RetrievalResult,
)


pytestmark = pytest.mark.small


def test_retrieval_request_normalizes_fields_and_bounds_limit() -> None:
    """Normalize query fields while enforcing the service result bound."""
    request = RetrievalRequest.normalize(
        {
            "query": "  specific query  ",
            "limit": 500,
            "workspace_id": "  workspace-a ",
            "tags": [" alpha ", "", "beta"],
            "include_superseded": True,
        },
        max_limit=25,
    )

    assert request == RetrievalRequest(
        query="specific query",
        limit=25,
        workspace_id="workspace-a",
        tags=("alpha", "beta"),
        include_superseded=True,
    )


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"query": "   "}, "query is required"),
        ({"query": "query", "limit": 0}, "limit must be at least 1"),
        ({"query": "query", "tags": ["ok", 3]}, "tags must be a sequence"),
    ],
)
def test_retrieval_request_rejects_invalid_fields(
    payload: dict[str, object], error: str
) -> None:
    """Reject malformed requests before an adapter receives them."""
    with pytest.raises((TypeError, ValueError), match=error):
        RetrievalRequest.normalize(payload)


@dataclass(frozen=True)
class _FakeBackend:
    calls: list[RetrievalRequest]
    thread_ids: list[int] | None = None

    async def search(self, request: RetrievalRequest) -> RetrievalResult[str]:
        self.calls.append(request)
        if self.thread_ids is not None:
            self.thread_ids.append(get_ident())
        return RetrievalResult(
            items=("first", "second", "third"),
            diagnostics=RetrievalDiagnostics(
                candidate_count=3,
                returned_count=3,
                failures=(RetrievalFailure("vector", "degraded"),),
            ),
        )

    def search_sync(self, request: RetrievalRequest) -> RetrievalResult[str]:
        self.calls.append(request)
        if self.thread_ids is not None:
            self.thread_ids.append(get_ident())
        return RetrievalResult(
            items=("first", "second", "third"),
            diagnostics=RetrievalDiagnostics(
                candidate_count=3,
                returned_count=3,
                failures=(RetrievalFailure("vector", "degraded"),),
            ),
        )


@pytest.mark.asyncio
async def test_async_and_sync_results_share_bounded_contract() -> None:
    """Keep async and sync results equivalent for one normalized request."""
    backend = _FakeBackend([])
    service = BoundedRetrievalService(backend.search, sync_search=backend.search_sync)

    async_result = await service.search({"query": " query ", "limit": 2})
    sync_result = service.search_sync({"query": " query ", "limit": 2})

    assert async_result.items == sync_result.items == ("first", "second")
    assert async_result.diagnostics == sync_result.diagnostics
    assert async_result.diagnostics.returned_count == 2
    assert backend.calls == [
        RetrievalRequest(query="query", limit=2),
        RetrievalRequest(query="query", limit=2),
    ]


@pytest.mark.asyncio
async def test_sync_entry_point_is_safe_inside_running_loop() -> None:
    """Allow sync callers to run from an active event loop.

    Keep the coroutine isolated from the caller's event-loop thread.
    """
    caller_thread_id = get_ident()
    thread_ids: list[int] = []
    backend = _FakeBackend([], thread_ids)
    service = BoundedRetrievalService(backend.search)

    result = service.search_sync("inside loop")

    assert result.items == ("first", "second", "third")
    assert backend.calls == [RetrievalRequest(query="inside loop")]
    assert thread_ids and thread_ids[0] != caller_thread_id


@pytest.mark.asyncio
async def test_nested_sync_entry_point_preserves_async_exception() -> None:
    """Propagate adapter exceptions through the nested sync bridge.

    Preserve the original exception type and message for callers.
    """

    async def fail(request: RetrievalRequest) -> RetrievalResult[str]:
        del request
        raise LookupError("backend unavailable")

    service = BoundedRetrievalService(fail)

    with pytest.raises(LookupError, match="backend unavailable"):
        service.search_sync("inside loop")

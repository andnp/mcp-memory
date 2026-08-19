"""Provider-neutral bounded retrieval contracts."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

DEFAULT_RETRIEVAL_LIMIT = 10
MAX_RETRIEVAL_LIMIT = 100
MAX_RETRIEVAL_QUERY_LENGTH = 4_096
MAX_RETRIEVAL_DIAGNOSTICS = 32

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Normalized input shared by asynchronous and synchronous retrieval."""

    query: str
    limit: int = DEFAULT_RETRIEVAL_LIMIT
    workspace_id: str | None = None
    memory_type: str | None = None
    status: str | None = None
    tags: tuple[str, ...] = ()
    include_superseded: bool = False
    adaptive_limit: bool = False

    @classmethod
    def normalize(
        cls,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        default_limit: int = DEFAULT_RETRIEVAL_LIMIT,
        max_limit: int = MAX_RETRIEVAL_LIMIT,
    ) -> RetrievalRequest:
        """Normalize supported request forms and enforce retrieval bounds."""
        if isinstance(request, cls):
            values = {
                "query": request.query,
                "limit": request.limit,
                "workspace_id": request.workspace_id,
                "memory_type": request.memory_type,
                "status": request.status,
                "tags": request.tags,
                "include_superseded": request.include_superseded,
                "adaptive_limit": request.adaptive_limit,
            }
        elif isinstance(request, str):
            values = {"query": request, "limit": default_limit}
        elif isinstance(request, Mapping):
            values = dict(request)
            values.setdefault("limit", default_limit)
        else:
            raise TypeError("request must be a RetrievalRequest, string, or mapping")

        query = values.get("query")
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query is required")
        if len(query) > MAX_RETRIEVAL_QUERY_LENGTH:
            raise ValueError(
                f"query must be at most {MAX_RETRIEVAL_QUERY_LENGTH} characters"
            )

        limit = values["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if max_limit < 1:
            raise ValueError("max_limit must be at least 1")
        bounded_limit = min(limit, max_limit)

        tags_value = values.get("tags", ())
        if isinstance(tags_value, str) or not isinstance(tags_value, Sequence):
            raise TypeError("tags must be a sequence of strings")
        if any(not isinstance(tag, str) for tag in tags_value):
            raise TypeError("tags must be a sequence of strings")
        tags = tuple(tag.strip() for tag in tags_value if tag.strip())

        return cls(
            query=query,
            limit=bounded_limit,
            workspace_id=_optional_string(values.get("workspace_id"), "workspace_id"),
            memory_type=_optional_string(values.get("memory_type"), "memory_type"),
            status=_optional_string(values.get("status"), "status"),
            tags=tags,
            include_superseded=_bool_value(
                values.get("include_superseded", False), "include_superseded"
            ),
            adaptive_limit=_bool_value(
                values.get("adaptive_limit", False), "adaptive_limit"
            ),
        )


@dataclass(frozen=True, slots=True)
class RetrievalFailure:
    """Bounded, transport-safe description of one retrieval failure."""

    stage: str
    message: str
    exception_type: str = ""


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Stable diagnostics envelope independent of a storage implementation."""

    elapsed_ms: float = 0.0
    candidate_count: int = 0
    returned_count: int = 0
    degraded: bool = False
    failures: tuple[RetrievalFailure, ...] = ()
    missing_ids: tuple[str, ...] = ()
    details: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and non-negative")
        if self.candidate_count < 0 or self.returned_count < 0:
            raise ValueError("diagnostic counts must be non-negative")
        object.__setattr__(self, "failures", tuple(self.failures[:MAX_RETRIEVAL_DIAGNOSTICS]))
        object.__setattr__(self, "missing_ids", tuple(self.missing_ids[:MAX_RETRIEVAL_DIAGNOSTICS]))
        object.__setattr__(self, "details", dict(self.details))


@dataclass(frozen=True, slots=True)
class RetrievalResult(Generic[T]):
    """Results plus bounded diagnostics returned by a retrieval service."""

    items: tuple[T, ...] = ()
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)


class MemoryRetrievalPort(Protocol):
    async def search(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> object: ...

    async def search_with_diagnostics(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
        debug: bool = False,
    ) -> tuple[object, object]: ...

    def search_sync(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> object: ...

    def search_sync_with_diagnostics(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
        debug: bool = False,
    ) -> tuple[object, object]: ...


AsyncSearch = Callable[[RetrievalRequest], Awaitable[RetrievalResult[T] | Sequence[T]]]
SyncSearch = Callable[[RetrievalRequest], RetrievalResult[T] | Sequence[T]]


class BoundedRetrievalService(Generic[T]):
    """Adapt retrieval implementations to one bounded async/sync contract."""

    def __init__(
        self,
        async_search: AsyncSearch[T],
        *,
        sync_search: SyncSearch[T] | None = None,
        max_limit: int = MAX_RETRIEVAL_LIMIT,
    ) -> None:
        if max_limit < 1:
            raise ValueError("max_limit must be at least 1")
        self._async_search = async_search
        self._sync_search = sync_search
        self._max_limit = max_limit

    async def search(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
    ) -> RetrievalResult[T]:
        normalized = RetrievalRequest.normalize(request, max_limit=self._max_limit)
        raw = await self._async_search(normalized)
        return _bounded_result(raw, normalized.limit)

    def search_sync(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
    ) -> RetrievalResult[T]:
        normalized = RetrievalRequest.normalize(request, max_limit=self._max_limit)
        if self._sync_search is not None:
            raw = self._sync_search(normalized)
        else:
            raw = run_coroutine_sync(
                lambda: _invoke_async_search(self._async_search, normalized)
            )
        return _bounded_result(raw, normalized.limit)


def _bounded_result(
    result: RetrievalResult[T] | Sequence[T],
    limit: int,
) -> RetrievalResult[T]:
    if isinstance(result, RetrievalResult):
        items = tuple(result.items[:limit])
        diagnostics = result.diagnostics
    else:
        items = tuple(result[:limit])
        diagnostics = RetrievalDiagnostics()
    if diagnostics.returned_count != len(items):
        diagnostics = RetrievalDiagnostics(
            elapsed_ms=diagnostics.elapsed_ms,
            candidate_count=diagnostics.candidate_count,
            returned_count=len(items),
            degraded=diagnostics.degraded,
            failures=diagnostics.failures,
            missing_ids=diagnostics.missing_ids,
            details=diagnostics.details,
        )
    return RetrievalResult(items=items, diagnostics=diagnostics)


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    return normalized or None


def _bool_value(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


R = TypeVar("R")


def run_coroutine_sync(factory: Callable[[], Coroutine[object, object, R]]) -> R:
    """Run a coroutine from sync code, isolating nested event loops."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_coroutine, factory).result()


async def _invoke_async_search(
    async_search: AsyncSearch[T],
    request: RetrievalRequest,
) -> RetrievalResult[T] | Sequence[T]:
    return await async_search(request)


def _run_coroutine(factory: Callable[[], Coroutine[object, object, R]]) -> R:
    return asyncio.run(factory())


__all__ = [
    "DEFAULT_RETRIEVAL_LIMIT",
    "MAX_RETRIEVAL_DIAGNOSTICS",
    "MAX_RETRIEVAL_LIMIT",
    "MAX_RETRIEVAL_QUERY_LENGTH",
    "BoundedRetrievalService",
    "MemoryRetrievalPort",
    "RetrievalDiagnostics",
    "RetrievalFailure",
    "RetrievalRequest",
    "RetrievalResult",
    "run_coroutine_sync",
]

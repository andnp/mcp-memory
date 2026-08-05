"""Application-facing memory retrieval contract."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from searchkernel.search.record_pipeline import RecordSearchOutcome

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRepositoryPort
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MemoryRecordSearchPipeline,
    build_memory_record_pipeline,
)


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    query: str
    limit: int = 10
    adaptive_limit: bool = False
    workspace_id: str | None = None
    memory_type: str | None = None
    status: str | None = None
    include_superseded: bool = False
    ranking_workspace_id: str | None = None


class MemoryRetrievalPort(Protocol):
    async def search(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome: ...

    def search_sync(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome: ...


class MemoryRetrievalFacade:
    """Expose canonical async memory search with a safe synchronous bridge."""

    def __init__(
        self,
        repository: MemoryRepositoryPort,
        *,
        config: Config | None = None,
        vector_store: Any | None = None,
        embedder: Any | None = None,
        embedding_maintenance: Any | None = None,
        native_search: Any | None = None,
        pipeline: MemoryRecordSearchPipeline | None = None,
        pipeline_factory: Callable[[bool], MemoryRecordSearchPipeline] | None = None,
    ) -> None:
        self._native_search = native_search
        self._provided_pipeline = pipeline
        self._pipeline = pipeline
        factory = pipeline_factory
        if pipeline_factory is None:
            resolved_config = config or Config()

            def default_pipeline_factory(
                adaptive_enabled: bool,
            ) -> MemoryRecordSearchPipeline:
                return build_memory_record_pipeline(
                    repository,
                    vector_store=vector_store,
                    embedder=embedder,
                    embedding_maintenance=embedding_maintenance,
                    adaptive_enabled=adaptive_enabled,
                    config=resolved_config,
                )

            factory = default_pipeline_factory
        assert factory is not None
        self._pipeline_factory = factory
        self._adaptive_pipeline: MemoryRecordSearchPipeline | None = None

    async def search(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome:
        if isinstance(request, str):
            request = MemorySearchRequest(
                query=request,
                limit=limit,
                adaptive_limit=adaptive_limit,
                workspace_id=workspace_id,
                memory_type=memory_type,
                status=status,
                include_superseded=include_superseded,
                ranking_workspace_id=ranking_workspace_id,
            )
        pipeline = self._resolve_pipeline(request.adaptive_limit)
        active_filters = dict(filters or {})
        if request.workspace_id is not None:
            active_filters["workspace_id"] = request.workspace_id
        if request.memory_type is not None:
            active_filters["memory_type"] = request.memory_type
        if request.status is not None:
            active_filters["status"] = request.status
        if request.ranking_workspace_id is not None:
            active_filters["_ranking_workspace_id"] = request.ranking_workspace_id
        if request.include_superseded or "include_superseded" not in active_filters:
            active_filters["include_superseded"] = request.include_superseded
        if side_effect_free:
            active_filters["_side_effect_free"] = True
        return await pipeline.search(
            request.query,
            limit=request.limit,
            filters=active_filters,
        )

    def search_sync(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome:
        return _run_async_safely(
            lambda: self.search(
                request,
                limit=limit,
                adaptive_limit=adaptive_limit,
                workspace_id=workspace_id,
                memory_type=memory_type,
                status=status,
                include_superseded=include_superseded,
                ranking_workspace_id=ranking_workspace_id,
                filters=filters,
                side_effect_free=side_effect_free,
            )
        )

    def read_memory(self, memory_id: str) -> Any:
        return self._native_method("read_memory")(memory_id)

    def peek_memory(self, memory_id: str) -> Any:
        return self._native_method("peek_memory")(memory_id)

    def search_memories_for_maintenance(self, *args: Any, **kwargs: Any) -> Any:
        return self._native_method("search_memories_for_maintenance")(
            *args,
            **kwargs,
        )

    def resolve_memory_id(self, memory_id: str) -> str | None:
        return self._native_method("resolve_memory_id")(memory_id)

    def _resolve_pipeline(self, adaptive_limit: bool) -> MemoryRecordSearchPipeline:
        if self._provided_pipeline is not None:
            return self._provided_pipeline
        if adaptive_limit:
            if self._adaptive_pipeline is None:
                self._adaptive_pipeline = self._pipeline_factory(True)
            return self._adaptive_pipeline
        if self._pipeline is None:
            self._pipeline = self._pipeline_factory(False)
        return self._pipeline

    def _native_method(self, name: str) -> Callable[..., Any]:
        if self._native_search is None:
            raise ValueError("relational_search_not_initialized")
        method = getattr(self._native_search, name, None)
        if not callable(method):
            raise AttributeError(f"native search does not provide {name}")
        return method


def _run_async_safely(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_in_thread, factory).result()


def _run_in_thread(factory: Callable[[], Coroutine[Any, Any, Any]]) -> Any:
    return asyncio.run(factory())


def build_memory_retrieval_facade(
    repository: MemoryRepositoryPort,
    *,
    config: Config | None = None,
    vector_store: Any | None = None,
    embedder: Any | None = None,
    embedding_maintenance: Any | None = None,
    native_search: Any | None = None,
) -> MemoryRetrievalFacade:
    """Compose the canonical retrieval boundary for an application caller."""
    return MemoryRetrievalFacade(
        repository,
        config=config,
        vector_store=vector_store,
        embedder=embedder,
        embedding_maintenance=embedding_maintenance,
        native_search=native_search,
    )


__all__ = [
    "MemoryRetrievalFacade",
    "MemorySearchRequest",
    "MemoryRetrievalPort",
    "build_memory_retrieval_facade",
]

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mcp_memory.core.ports import SearchHealthPort
from mcp_memory.management.models import (
    CacheHealthPayload,
    EmbeddingIntegrityEventSummaryPayload,
    OverviewPayload,
)
from mcp_memory.management.overview_reporting import build_overview
from mcp_memory.mcp.telemetry import search_diagnostics_snapshot


@dataclass(frozen=True)
class OverviewServiceDependencies:
    memory_queries: Any
    repository: Any
    task_queue: Any
    runtime_info: Any
    db_manager: Any
    provider_usage_repo: Any
    runtime_logs_repo: Any
    embedder: Any
    storage_backend: str | None
    vector_store: Any
    relational_search: Any
    retrieval_telemetry: Any
    cache_health: Callable[[], CacheHealthPayload]
    embedding_integrity_summary: Callable[[], EmbeddingIntegrityEventSummaryPayload]
    search_health: SearchHealthPort | None = None


class OverviewService:
    def __init__(self, dependencies: OverviewServiceDependencies) -> None:
        self._dependencies = dependencies

    def get_overview(
        self,
        *,
        recent_limit: int = 10,
        failed_limit: int = 10,
    ) -> OverviewPayload:
        dependencies = self._dependencies
        return build_overview(
            memory_queries=dependencies.memory_queries,
            repository=dependencies.repository,
            task_queue=dependencies.task_queue,
            runtime_info=dependencies.runtime_info,
            db_manager=dependencies.db_manager,
            provider_usage_repo=dependencies.provider_usage_repo,
            runtime_logs_repo=dependencies.runtime_logs_repo,
            embedder=dependencies.embedder,
            storage_backend=dependencies.storage_backend,
            vector_store=dependencies.vector_store,
            relational_search=dependencies.search_health or dependencies.relational_search,
            search_diagnostics=search_diagnostics_snapshot(dependencies.retrieval_telemetry),
            cache=dependencies.cache_health(),
            embedding_integrity_summary=dependencies.embedding_integrity_summary(),
            recent_limit=recent_limit,
            failed_limit=failed_limit,
        )

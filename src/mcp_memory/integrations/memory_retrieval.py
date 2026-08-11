"""Application-facing memory retrieval contract."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from searchkernel.search.record_pipeline import RecordSearchOutcome

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRepositoryPort
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX,
    MemoryRecordSearchPipeline,
    build_memory_record_pipeline,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    query: str
    limit: int = 10
    adaptive_limit: bool = False
    workspace_id: str | None = None
    memory_type: str | None = None
    status: str | None = None
    tags: tuple[str, ...] = ()
    include_superseded: bool = False
    ranking_workspace_id: str | None = None


@dataclass(slots=True)
class SearchExecutionDiagnostics:
    """Application-facing diagnostics shared by async and sync retrieval."""

    timing_ms: dict[str, float] = field(default_factory=dict)
    keyword_candidate_count: int = 0
    semantic_candidate_count: int = 0
    semantic_only_candidate_count: int = 0
    semantic_abstention_count: int = 0
    semantic_abstention_rate: float | None = None
    semantic_abstained: bool | None = None
    semantic_candidate_strategy: str = "kernel"
    candidate_count: int = 0
    candidate_counts: dict[str, int] | None = None
    vector_search: dict[str, object] | None = None
    kernel_diagnostics: list[str] | None = None
    cache_diagnostics: list[str] | None = None
    failures: list[dict[str, object]] | None = None
    missing_record_ids: list[str] | None = None
    failure_count: int = 0
    missing_record_count: int = 0
    degraded: bool = False
    trace: dict[str, object] | None = None
    scope: dict[str, object] | None = None
    lane_decisions: dict[str, object] | None = None
    raw_lane_overlap_count: int | None = None
    multi_lane_candidate_ids: list[str] | None = None
    duplicate_candidate_ids: list[str] | None = None
    final_duplicate_ids: list[str] | None = None

    def to_payload(self) -> dict[str, object]:
        candidate_counts = dict(self.candidate_counts or {})
        kernel_diagnostics = list(self.kernel_diagnostics or [])
        cache_diagnostics = list(self.cache_diagnostics or [])
        failures = [dict(failure) for failure in self.failures or []]
        missing_record_ids = list(self.missing_record_ids or [])
        multi_lane_ids = list(self.multi_lane_candidate_ids or [])
        duplicate_ids = list(self.duplicate_candidate_ids or [])
        final_duplicate_ids = list(self.final_duplicate_ids or [])
        lane_decisions = self.lane_decisions or {}
        enabled = lane_decisions.get("enabled")
        budgets = lane_decisions.get("budgets")
        skipped = lane_decisions.get("skipped")
        semantic = {
            "candidate_count": self.semantic_candidate_count,
            "semantic_only_candidate_count": self.semantic_only_candidate_count,
            "abstention_count": self.semantic_abstention_count,
            "abstention_rate": self.semantic_abstention_rate,
        }
        return {
            "timing_ms": dict(self.timing_ms),
            "keyword_candidate_count": self.keyword_candidate_count,
            "semantic_candidate_count": self.semantic_candidate_count,
            "semantic_only_candidate_count": self.semantic_only_candidate_count,
            "semantic_abstention_count": self.semantic_abstention_count,
            "semantic_abstention_rate": self.semantic_abstention_rate,
            "semantic_abstained": self.semantic_abstained,
            "semantic_candidate_strategy": self.semantic_candidate_strategy,
            "candidate_count": self.candidate_count,
            "candidate_counts": candidate_counts,
            "vector_search": (
                None if self.vector_search is None else dict(self.vector_search)
            ),
            "kernel_diagnostics": kernel_diagnostics,
            "cache_diagnostics": cache_diagnostics,
            "failures": failures,
            "missing_record_ids": missing_record_ids,
            "failure_count": self.failure_count,
            "missing_record_count": self.missing_record_count,
            "degraded": self.degraded,
            "trace": None if self.trace is None else dict(self.trace),
            "scope": dict(self.scope or {}),
            "lane_decisions": {
                "enabled": list(enabled) if isinstance(enabled, list) else [],
                "budgets": dict(budgets) if isinstance(budgets, dict) else {},
                "skipped": list(skipped) if isinstance(skipped, list) else [],
            },
            "overlap": {
                "raw_lane_overlap_count": self.raw_lane_overlap_count,
                "multi_lane_result_count": len(multi_lane_ids),
                "final_duplicate_count": len(final_duplicate_ids),
            },
            "duplicate_candidate_ids": duplicate_ids,
            "multi_lane_candidate_ids": multi_lane_ids,
            "final_duplicate_ids": final_duplicate_ids,
            "semantic": semantic,
        }


def build_search_execution_diagnostics(
    outcome: RecordSearchOutcome,
    *,
    workspace_id: str | None,
    ranking_workspace_id: str | None,
    total_ms: float | None = None,
    debug: bool = False,
) -> SearchExecutionDiagnostics:
    """Project one kernel outcome into the application diagnostic contract."""
    timing_ms = {
        stage: round(float(duration), 3)
        for stage, duration in outcome.stage_timings_ms.items()
    }
    if total_ms is not None:
        timing_ms = {"total": round(total_ms, 3), **timing_ms}
    semantic_candidate_count, semantic_only_count, abstention_count = (
        _semantic_abstention_counts(outcome.diagnostics)
    )
    semantic_abstention_rate = (
        abstention_count / semantic_only_count if semantic_only_count else None
    )
    semantic_abstained = (
        None
        if outcome.degraded or semantic_candidate_count == 0
        else abstention_count > 0
    )
    result_counts = Counter(result.record_id for result in outcome.results)
    final_duplicate_ids = sorted(
        record_id for record_id, count in result_counts.items() if count > 1
    )
    multi_lane_candidate_ids = sorted(
        {
            result.record_id
            for result in outcome.results
            if len(result.provenance.strategies) > 1
        }
    )
    trace = None
    if debug and outcome.trace is not None:
        try:
            trace = outcome.trace.to_dict()
        except Exception as error:  # noqa: BLE001 - diagnostics are best effort
            logger.warning(
                "search diagnostics trace projection failed: %s",
                type(error).__name__,
            )
    return SearchExecutionDiagnostics(
        timing_ms=timing_ms,
        semantic_candidate_count=semantic_candidate_count,
        semantic_only_candidate_count=semantic_only_count,
        semantic_abstention_count=abstention_count,
        semantic_abstention_rate=semantic_abstention_rate,
        semantic_abstained=semantic_abstained,
        candidate_count=int(outcome.candidate_count),
        candidate_counts={
            str(stage): int(count) for stage, count in outcome.candidate_counts.items()
        },
        kernel_diagnostics=list(outcome.diagnostics),
        cache_diagnostics=list(outcome.cache_diagnostics),
        failures=[
            {
                "stage": getattr(failure, "stage", "unknown"),
                "message": getattr(failure, "message", str(failure)),
                "exception_type": getattr(
                    failure, "exception_type", type(failure).__name__
                ),
            }
            for failure in outcome.failures
        ],
        missing_record_ids=list(outcome.missing_record_ids),
        failure_count=len(outcome.failures),
        missing_record_count=len(outcome.missing_record_ids),
        degraded=outcome.degraded,
        trace=trace,
        scope={
            "mode": "filtered" if workspace_id is not None else "global",
            "workspace_filter": workspace_id,
            "ranking_workspace_id": ranking_workspace_id,
        },
        lane_decisions=_lane_decisions(outcome.diagnostics),
        raw_lane_overlap_count=None,
        multi_lane_candidate_ids=multi_lane_candidate_ids,
        duplicate_candidate_ids=[],
        final_duplicate_ids=final_duplicate_ids,
    )


def _semantic_abstention_counts(
    diagnostics: tuple[str, ...],
) -> tuple[int, int, int]:
    diagnostic = next(
        (
            value
            for value in diagnostics
            if value.startswith(MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX)
        ),
        None,
    )
    if diagnostic is None:
        return 0, 0, 0
    values: dict[str, int] = {}
    for field_value in diagnostic[
        len(MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX) :
    ].split(";"):
        name, separator, raw_value = field_value.partition("=")
        if not separator:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value >= 0:
            values[name] = value
    return (
        values.get("semantic_candidates", 0),
        values.get("semantic_only_candidates", 0),
        values.get("rejected", 0),
    )


def _lane_decisions(diagnostics: tuple[str, ...]) -> dict[str, object]:
    enabled: list[str] = []
    budgets: dict[str, int] = {}
    skipped: list[str] = []
    for diagnostic in diagnostics:
        if diagnostic.startswith("query_plan:lanes:"):
            enabled = [
                lane
                for lane in diagnostic.removeprefix("query_plan:lanes:").split(",")
                if lane and lane != "none"
            ]
        elif diagnostic.startswith("query_plan:budgets:"):
            for value in diagnostic.removeprefix("query_plan:budgets:").split(","):
                name, separator, raw_budget = value.partition("=")
                if separator:
                    try:
                        budgets[name] = int(raw_budget)
                    except ValueError:
                        continue
        elif diagnostic.startswith("query_plan:skip:"):
            skipped.append(diagnostic.removeprefix("query_plan:skip:"))
    return {"enabled": enabled, "budgets": budgets, "skipped": skipped}


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
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome: ...

    async def search_with_diagnostics(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
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
    ) -> tuple[RecordSearchOutcome, SearchExecutionDiagnostics]: ...

    def search_sync(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome: ...

    def search_sync_with_diagnostics(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
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
    ) -> tuple[RecordSearchOutcome, SearchExecutionDiagnostics]: ...


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
        tags: tuple[str, ...] | None = None,
        include_superseded: bool = False,
        ranking_workspace_id: str | None = None,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RecordSearchOutcome:
        request = self._normalize_request(
            request,
            limit=limit,
            adaptive_limit=adaptive_limit,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            tags=tags,
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
        if request.tags:
            active_filters["tags"] = request.tags
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

    async def search_with_diagnostics(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
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
    ) -> tuple[RecordSearchOutcome, SearchExecutionDiagnostics]:
        normalized_request = self._normalize_request(
            request,
            limit=limit,
            adaptive_limit=adaptive_limit,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            tags=tags,
            include_superseded=include_superseded,
            ranking_workspace_id=ranking_workspace_id,
        )
        started_at = perf_counter()
        outcome = await self.search(
            normalized_request,
            filters=filters,
            side_effect_free=side_effect_free,
        )
        try:
            diagnostics = build_search_execution_diagnostics(
                outcome,
                workspace_id=normalized_request.workspace_id,
                ranking_workspace_id=normalized_request.ranking_workspace_id,
                total_ms=(perf_counter() - started_at) * 1000.0,
                debug=debug,
            )
        except Exception as error:  # noqa: BLE001 - diagnostics are best effort
            logger.warning(
                "search diagnostics projection failed: %s",
                type(error).__name__,
            )
            diagnostics = SearchExecutionDiagnostics(
                timing_ms={"total": round((perf_counter() - started_at) * 1000.0, 3)},
                failure_count=len(outcome.failures),
                missing_record_count=len(outcome.missing_record_ids),
                degraded=outcome.degraded,
            )
        return outcome, diagnostics

    def search_sync(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
        adaptive_limit: bool = False,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: tuple[str, ...] | None = None,
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
                tags=tags,
                include_superseded=include_superseded,
                ranking_workspace_id=ranking_workspace_id,
                filters=filters,
                side_effect_free=side_effect_free,
            )
        )

    def search_sync_with_diagnostics(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int = 10,
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
    ) -> tuple[RecordSearchOutcome, SearchExecutionDiagnostics]:
        return _run_async_safely(
            lambda: self.search_with_diagnostics(
                request,
                limit=limit,
                adaptive_limit=adaptive_limit,
                workspace_id=workspace_id,
                memory_type=memory_type,
                status=status,
                tags=tags,
                include_superseded=include_superseded,
                ranking_workspace_id=ranking_workspace_id,
                filters=filters,
                side_effect_free=side_effect_free,
                debug=debug,
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

    def _normalize_request(
        self,
        request: MemorySearchRequest | str,
        *,
        limit: int,
        adaptive_limit: bool,
        workspace_id: str | None,
        memory_type: str | None,
        status: str | None,
        tags: tuple[str, ...] | None,
        include_superseded: bool,
        ranking_workspace_id: str | None,
    ) -> MemorySearchRequest:
        if isinstance(request, MemorySearchRequest):
            return request
        return MemorySearchRequest(
            query=request,
            limit=limit,
            adaptive_limit=adaptive_limit,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            tags=tuple(tags or ()),
            include_superseded=include_superseded,
            ranking_workspace_id=ranking_workspace_id,
        )

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
    "SearchExecutionDiagnostics",
    "build_search_execution_diagnostics",
    "build_memory_retrieval_facade",
]

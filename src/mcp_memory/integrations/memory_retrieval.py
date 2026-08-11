"""Application-facing memory retrieval contract."""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Callable, Coroutine, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol

from searchkernel.runtime import QueryEmbeddingCache
from searchkernel.search.record_pipeline import RecordSearchOutcome

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRepositoryPort
from mcp_memory.core.retrieval import (
    RetrievalDiagnostics,
    RetrievalFailure,
    RetrievalRequest,
    RetrievalResult,
)
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX,
    _ACTIVE_QUERY_EMBEDDING_CACHE,
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
    final_duplicate_count: int | None = None

    def to_payload(self) -> dict[str, object]:
        candidate_counts = dict(self.candidate_counts or {})
        kernel_diagnostics = list(self.kernel_diagnostics or [])
        cache_diagnostics = list(self.cache_diagnostics or [])
        failures = [dict(failure) for failure in self.failures or []]
        missing_record_ids = list(self.missing_record_ids or [])
        multi_lane_ids = list(self.multi_lane_candidate_ids or [])
        duplicate_ids = list(self.duplicate_candidate_ids or [])
        final_duplicate_ids = list(self.final_duplicate_ids or [])
        final_duplicate_count = (
            len(final_duplicate_ids)
            if self.final_duplicate_count is None
            else self.final_duplicate_count
        )
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
                "final_duplicate_count": final_duplicate_count,
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
    evidence = getattr(outcome, "diagnostic_evidence", None)
    typed_timing_ms = _typed_timing_ms(evidence)
    timing_ms = {
        stage: round(float(duration), 3) for stage, duration in (
            typed_timing_ms
            if typed_timing_ms is not None
            else outcome.stage_timings_ms
        ).items()
    }
    if total_ms is not None:
        timing_ms = {"total": round(total_ms, 3), **timing_ms}
    typed_failures = _typed_sequence(evidence, "failures")
    failures = outcome.failures if typed_failures is None else typed_failures
    typed_missing_record_ids = _typed_string_sequence(evidence, "missing_record_ids")
    missing_record_ids = (
        outcome.missing_record_ids
        if typed_missing_record_ids is None
        else typed_missing_record_ids
    )
    degraded = outcome.degraded or bool(failures or missing_record_ids)
    semantic_candidate_count, semantic_only_count, abstention_count = (
        _semantic_abstention_counts(outcome.diagnostics)
    )
    semantic_abstention_rate = (
        abstention_count / semantic_only_count if semantic_only_count else None
    )
    semantic_abstained = (
        None
        if degraded or semantic_candidate_count == 0
        else abstention_count > 0
    )
    result_counts = Counter(result.record_id for result in outcome.results)
    final_duplicate_ids = sorted(
        record_id for record_id, count in result_counts.items() if count > 1
    )
    result_ids_by_storage_key = {
        _result_storage_key(result): result.record_id for result in outcome.results
    }
    result_provenance = _typed_mapping(evidence, "result_provenance")
    multi_lane_candidate_ids = sorted(
        {
            result_ids_by_storage_key.get(storage_key, storage_key)
            for storage_key, strategies in (
                result_provenance.items()
                if result_provenance is not None
                else (
                    (_result_storage_key(result), result.provenance.strategies)
                    for result in outcome.results
                )
            )
            if _has_multiple_strategies(strategies)
        }
    )
    final_duplicate_count = _typed_nonnegative_int(
        evidence, "final_duplicate_count", len(final_duplicate_ids)
    )
    raw_lane_overlap_count = _typed_overlap_count(evidence)
    lane_decisions = _typed_lane_decisions(evidence) or _lane_decisions(
        outcome.diagnostics
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
            for failure in failures
        ],
        missing_record_ids=list(missing_record_ids),
        failure_count=len(failures),
        missing_record_count=len(missing_record_ids),
        degraded=degraded,
        trace=trace,
        scope={
            "mode": "filtered" if workspace_id is not None else "global",
            "workspace_filter": workspace_id,
            "ranking_workspace_id": ranking_workspace_id,
        },
        lane_decisions=lane_decisions,
        raw_lane_overlap_count=raw_lane_overlap_count,
        multi_lane_candidate_ids=multi_lane_candidate_ids,
        duplicate_candidate_ids=[],
        final_duplicate_ids=final_duplicate_ids,
        final_duplicate_count=final_duplicate_count,
    )


def _typed_mapping(evidence: object, name: str) -> dict[str, object] | None:
    value = getattr(evidence, name, None)
    if not isinstance(value, Mapping):
        return None
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
    }


def _typed_timing_ms(evidence: object) -> dict[str, float] | None:
    value = _typed_mapping(evidence, "stage_timings_ms")
    if value is None:
        return None
    timing_ms: dict[str, float] = {}
    for stage, duration in value.items():
        if not isinstance(duration, (int, float)):
            return None
        timing_ms[stage] = float(duration)
    return timing_ms


def _typed_sequence(evidence: object, name: str) -> tuple[object, ...] | None:
    value = getattr(evidence, name, None)
    if not isinstance(value, (list, tuple)):
        return None
    return tuple(value)


def _typed_string_sequence(evidence: object, name: str) -> tuple[str, ...] | None:
    value = _typed_sequence(evidence, name)
    if value is None:
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        items.append(item)
    return tuple(items)


def _typed_nonnegative_int(evidence: object, name: str, fallback: int) -> int:
    value = getattr(evidence, name, None)
    return value if isinstance(value, int) and value >= 0 else fallback


def _typed_overlap_count(evidence: object) -> int | None:
    capability = getattr(evidence, "raw_pre_fusion_overlap", None)
    if not getattr(capability, "available", False):
        return None
    count = getattr(capability, "count", None)
    return count if isinstance(count, int) and count >= 0 else None


def _has_multiple_strategies(strategies: object) -> bool:
    if not isinstance(strategies, (list, tuple)):
        return False
    return len(strategies) > 1


def _result_storage_key(result: object) -> str:
    storage_key = getattr(result, "storage_key", None)
    if isinstance(storage_key, str):
        return storage_key
    record = getattr(result, "record", None)
    record_storage_key = getattr(record, "storage_key", None)
    return str(record_storage_key)


def _typed_lane_decisions(evidence: object) -> dict[str, object] | None:
    if evidence is None:
        return None
    enabled = _typed_sequence(evidence, "enabled_lanes")
    budgets = _typed_mapping(evidence, "lane_budgets")
    skipped = _typed_sequence(evidence, "skipped_lanes")
    if enabled is None or budgets is None or skipped is None:
        return None
    typed_budgets: dict[str, int] = {}
    for name, budget in budgets.items():
        if not isinstance(budget, int):
            return None
        typed_budgets[name] = budget
    skipped_values = [
        f"{lane}:{reason}"
        for skip in skipped
        if (lane := getattr(skip, "lane", None)) is not None
        and (reason := getattr(skip, "reason", None)) is not None
    ]
    return {
        "enabled": [str(lane) for lane in enabled],
        "budgets": typed_budgets,
        "skipped": skipped_values,
    }


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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        query_embedding_cache: QueryEmbeddingCache | None = None,
        native_search: Any | None = None,
        pipeline: MemoryRecordSearchPipeline | None = None,
        pipeline_factory: Callable[[bool], MemoryRecordSearchPipeline] | None = None,
    ) -> None:
        self._native_search = native_search
        self._provided_pipeline = pipeline
        self._pipeline = pipeline
        self._query_embedding_cache = (
            query_embedding_cache
            if query_embedding_cache is not None
            else QueryEmbeddingCache()
        )
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
                    query_embedding_cache=self._query_embedding_cache,
                    adaptive_enabled=adaptive_enabled,
                    config=resolved_config,
                )

            factory = default_pipeline_factory
        assert factory is not None
        self._pipeline_factory = factory
        self._adaptive_pipeline: MemoryRecordSearchPipeline | None = None

    async def retrieve(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RetrievalResult[RecordSearchOutcome]:
        """Return the shared typed envelope without changing legacy search results."""
        normalized = RetrievalRequest.normalize(request)
        return await self._retrieve_typed(
            normalized,
            filters=filters,
            side_effect_free=side_effect_free,
        )

    def retrieve_sync(
        self,
        request: RetrievalRequest | str | Mapping[str, object],
        *,
        filters: dict[str, object] | None = None,
        side_effect_free: bool = False,
    ) -> RetrievalResult[RecordSearchOutcome]:
        """Use the same typed retrieval envelope at synchronous process edges."""
        return _run_async_safely(
            lambda: self.retrieve(
                request,
                filters=filters,
                side_effect_free=side_effect_free,
            )
        )

    async def search(
        self,
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        typed_result = await self._retrieve_legacy_request(
            normalized_request,
            filters=filters,
            side_effect_free=side_effect_free,
        )
        return typed_result.items[0]

    async def _search_outcome(
        self,
        request: MemorySearchRequest,
        *,
        filters: dict[str, object] | None,
        side_effect_free: bool,
    ) -> RecordSearchOutcome:
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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        typed_result = await self._retrieve_legacy_request(
            normalized_request,
            filters=filters,
            side_effect_free=side_effect_free,
        )
        outcome = typed_result.items[0]
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
                failure_count=len(typed_result.diagnostics.failures),
                missing_record_count=len(typed_result.diagnostics.missing_ids),
                degraded=typed_result.diagnostics.degraded,
            )
        return outcome, diagnostics

    def search_sync(
        self,
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
                self._adaptive_pipeline = self._build_pipeline(True)
            return self._adaptive_pipeline
        if self._pipeline is None:
            self._pipeline = self._build_pipeline(False)
        return self._pipeline

    def _build_pipeline(self, adaptive_enabled: bool) -> MemoryRecordSearchPipeline:
        token = _ACTIVE_QUERY_EMBEDDING_CACHE.set(self._query_embedding_cache)
        try:
            return self._pipeline_factory(adaptive_enabled)
        finally:
            _ACTIVE_QUERY_EMBEDDING_CACHE.reset(token)

    def _normalize_request(
        self,
        request: MemorySearchRequest | RetrievalRequest | str | Mapping[str, object],
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
        if isinstance(request, RetrievalRequest):
            return MemorySearchRequest(
                query=request.query,
                limit=request.limit,
                adaptive_limit=request.adaptive_limit,
                workspace_id=request.workspace_id,
                memory_type=request.memory_type,
                status=request.status,
                tags=request.tags,
                include_superseded=request.include_superseded,
            )
        if isinstance(request, Mapping):
            request = RetrievalRequest.normalize(request)
            return self._normalize_request(
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

    async def _retrieve_legacy_request(
        self,
        request: MemorySearchRequest,
        *,
        filters: dict[str, object] | None,
        side_effect_free: bool,
    ) -> RetrievalResult[RecordSearchOutcome]:
        typed_request = RetrievalRequest(
            query=request.query,
            limit=request.limit,
            workspace_id=request.workspace_id,
            memory_type=request.memory_type,
            status=request.status,
            tags=request.tags,
            include_superseded=request.include_superseded,
            adaptive_limit=request.adaptive_limit,
        )
        active_filters = dict(filters or {})
        if request.ranking_workspace_id is not None:
            active_filters["_ranking_workspace_id"] = request.ranking_workspace_id
        return await self._retrieve_typed(
            typed_request,
            filters=active_filters,
            side_effect_free=side_effect_free,
        )

    async def _retrieve_typed(
        self,
        request: RetrievalRequest,
        *,
        filters: dict[str, object] | None,
        side_effect_free: bool,
    ) -> RetrievalResult[RecordSearchOutcome]:
        started_at = perf_counter()
        legacy_request = MemorySearchRequest(
            query=request.query,
            limit=request.limit,
            adaptive_limit=request.adaptive_limit,
            workspace_id=request.workspace_id,
            memory_type=request.memory_type,
            status=request.status,
            tags=request.tags,
            include_superseded=request.include_superseded,
        )
        outcome = await self._search_outcome(
            legacy_request,
            filters=filters,
            side_effect_free=side_effect_free,
        )
        outcome_failures = getattr(outcome, "failures", ())
        outcome_missing_ids = getattr(outcome, "missing_record_ids", ())
        failures = tuple(
            RetrievalFailure(
                stage=getattr(failure, "stage", "unknown"),
                message=getattr(failure, "message", str(failure)),
                exception_type=getattr(
                    failure, "exception_type", type(failure).__name__
                ),
            )
            for failure in outcome_failures
        )
        diagnostics = RetrievalDiagnostics(
            elapsed_ms=max((perf_counter() - started_at) * 1000.0, 0.0),
            candidate_count=max(int(getattr(outcome, "candidate_count", 0)), 0),
            returned_count=1,
            degraded=bool(getattr(outcome, "degraded", False)),
            failures=failures,
            missing_ids=tuple(outcome_missing_ids),
            details={"result_count": str(len(getattr(outcome, "results", ())))},
        )
        return RetrievalResult(items=(outcome,), diagnostics=diagnostics)

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

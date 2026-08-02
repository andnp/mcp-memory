from __future__ import annotations

import logging
from datetime import UTC, datetime
from time import perf_counter

from mcp_memory.application.ports import (
    MemoryMutationDependencies,
    MemoryReadDependencies,
    RetrievalTelemetryPort,
)
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.integrations.memory_retrieval import build_memory_retrieval_facade
from mcp_memory.integrations.searchkernel_record_pipeline import (
    build_memory_record_pipeline,
)
from mcp_memory.application.cache_policy import (
    _CACHE_VALIDATION_TOKENS_FIELD,
    _begin_inflight_search_coalescing,
    _compact_cached_read_payload,
    _compact_cached_search_payload,
    _finish_inflight_search_coalescing,
    _increment_shared_read_cache_metric,
    _load_cached_read_fallback,
    _load_cached_search_fallback,
    _load_fresh_cached_search_hit,
    _load_projection_search_fallback,
    _load_validated_cached_read_hit,
    _resolve_read_cache_validation_token,
    _resolve_read_cache_validation_tokens,
    _shared_read_cache_enabled,
    _store_cached_read_response,
    _store_cached_search_response,
    _warm_cached_search_projections,
    build_search_cache_request,
)
from mcp_memory.application.payloads import (
    build_read_payload,
    build_search_result_payloads,
)
from mcp_memory.relational.operations import (
    ReadMemoryRecordOperation,
    SearchMemoryRecordsOperation,
)
from mcp_memory.relational.search import _to_relational_search_result

logger = logging.getLogger(__name__)


def _record_thought(
    ctx: MemoryMutationDependencies,
    content: str,
    *,
    writeback_cache=None,
    max_outbox_entries: int | None = None,
) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    suppression_config = None if ctx.config is None else ctx.config.ingest_suppression
    operation = RecordThoughtOperation(
        ctx.journal,
        ctx.task_queue,
        ctx.workspace_id,
        suppression_config,
        writeback_cache=writeback_cache,
        max_outbox_entries=max_outbox_entries,
    )
    operation.execute(content)
    return {"status": "recorded"}


def _search_memory_records(
    ctx: MemoryReadDependencies,
    arguments: dict,
    *,
    caller_kind: str = "external",
    telemetry: RetrievalTelemetryPort,
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    query = arguments["query"]
    retrieval = ctx.memory_retrieval
    if retrieval is None and ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    if retrieval is None:
        retrieval = build_memory_retrieval_facade(
            ctx.repository,
            config=ctx.config,
            vector_store=ctx.vector_store,
            embedder=ctx.embedder,
            embedding_maintenance=ctx.embedding_maintenance,
            native_search=ctx.relational_search,
        )
    operation = SearchMemoryRecordsOperation(retrieval)
    started_at = perf_counter()
    debug_enabled = arguments["debug"]
    execution_arguments = {
        "query": query,
        "workspace_id": ctx.workspace_id,
        "limit": arguments["limit"],
        "adaptive_limit": arguments.get("adaptive_limit", "limit" not in arguments),
        "memory_type": arguments["memory_type"],
        "status": arguments["status"],
        "include_superseded": arguments["include_superseded"],
        "debug": debug_enabled,
    }
    cache_request = build_search_cache_request(ctx, execution_arguments)
    _increment_shared_read_cache_metric(
        ctx,
        "search_requests",
        caller_kind=caller_kind,
        debug_enabled=debug_enabled,
    )
    cached_payload = _load_fresh_cached_search_hit(
        ctx,
        cache_request,
        caller_kind=caller_kind,
        debug_enabled=debug_enabled,
    )
    if cached_payload is not None:
        _increment_shared_read_cache_metric(
            ctx,
            "fresh_exact_search_hits",
            caller_kind=caller_kind,
            debug_enabled=debug_enabled,
        )
        return cached_payload
    inflight_search = _begin_inflight_search_coalescing(
        ctx,
        cache_request,
        caller_kind=caller_kind,
        debug_enabled=debug_enabled,
    )
    if inflight_search is not None and not inflight_search.is_leader:
        try:
            read_cache = ctx.read_cache
            if read_cache is None:
                raise RuntimeError("shared read cache disappeared during coalescing")
            coalesced_payload = read_cache.wait_for_inflight_search(inflight_search)
        except Exception as error:
            if _shared_read_cache_enabled(
                ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
            ):
                cached_payload = _load_cached_search_fallback(
                    ctx, cache_request, error=error
                )
                if cached_payload is not None:
                    _increment_shared_read_cache_metric(
                        ctx,
                        "stale_exact_search_fallbacks",
                        caller_kind=caller_kind,
                        debug_enabled=debug_enabled,
                    )
                    return cached_payload
                projection_payload = _load_projection_search_fallback(
                    ctx, cache_request, error=error
                )
                if projection_payload is not None:
                    _increment_shared_read_cache_metric(
                        ctx,
                        "projection_fallbacks",
                        caller_kind=caller_kind,
                        debug_enabled=debug_enabled,
                    )
                    return projection_payload
            raise
        duration_ms = (perf_counter() - started_at) * 1000.0
        surfaced_memory_ids = [
            str(result["memory_id"])
            for result in coalesced_payload.get("results", [])
            if isinstance(result, dict) and isinstance(result.get("memory_id"), str)
        ]
        telemetry.record_search(
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
            duration_ms=duration_ms,
        )
        return _compact_cached_search_payload(coalesced_payload)
    diagnostics = None
    try:
        if debug_enabled:
            results, diagnostics = operation.execute_with_diagnostics(
                **execution_arguments
            )
        else:
            results = operation.execute(**execution_arguments)
    except Exception as error:
        _finish_inflight_search_coalescing(ctx, inflight_search, error=error)
        if _shared_read_cache_enabled(
            ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
        ):
            cached_payload = _load_cached_search_fallback(
                ctx, cache_request, error=error
            )
            if cached_payload is not None:
                _increment_shared_read_cache_metric(
                    ctx,
                    "stale_exact_search_fallbacks",
                    caller_kind=caller_kind,
                    debug_enabled=debug_enabled,
                )
                return cached_payload
            projection_payload = _load_projection_search_fallback(
                ctx, cache_request, error=error
            )
            if projection_payload is not None:
                _increment_shared_read_cache_metric(
                    ctx,
                    "projection_fallbacks",
                    caller_kind=caller_kind,
                    debug_enabled=debug_enabled,
                )
                return projection_payload
        raise
    duration_ms = (perf_counter() - started_at) * 1000.0
    surfaced_memory_ids = [result.memory_id for result in results]
    surface_tracker = ctx.surface_tracker or ctx.repository
    touch_last_surfaced = getattr(surface_tracker, "touch_last_surfaced", None)
    if surfaced_memory_ids and callable(touch_last_surfaced):
        touch_last_surfaced(
            surfaced_memory_ids,
            datetime.now(UTC).isoformat(),
            best_effort=True,
        )
    telemetry.record_search(
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        duration_ms=duration_ms,
    )
    result_payloads = build_search_result_payloads(
        results, debug_enabled=debug_enabled
    )
    payload: dict[str, object] = {
        "status": "ok",
        "results": result_payloads,
    }
    if debug_enabled:
        timing_ms = {"total": round(duration_ms, 3)}
        if diagnostics is not None:
            timing_ms |= {
                key: value
                for key, value in diagnostics.timing_ms.items()
                if key != "total"
            }
            payload["search_diagnostics"] = diagnostics.to_payload()
        payload["timing_ms"] = timing_ms
        payload["adaptive_limit_enabled"] = execution_arguments["adaptive_limit"]
        payload["requested_limit"] = execution_arguments["limit"]
        payload["expanded_result_window"] = len(results) > execution_arguments["limit"]
        payload["returned_result_count"] = len(results)
    validation_tokens: dict[str, str] = {}
    if _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        try:
            validation_tokens = _resolve_read_cache_validation_tokens(
                ctx, surfaced_memory_ids
            )
        except Exception:
            logger.warning(
                "Search cache validation token capture failed after authoritative search",
                exc_info=True,
            )
    warmed_projection_rows = _warm_cached_search_projections(
        ctx,
        result_payloads,
        caller_kind=caller_kind,
        debug_enabled=debug_enabled,
        validation_tokens=validation_tokens,
    )
    if warmed_projection_rows > 0:
        _increment_shared_read_cache_metric(
            ctx,
            "warmed_projection_rows",
            amount=warmed_projection_rows,
            caller_kind=caller_kind,
            debug_enabled=debug_enabled,
        )
    if _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        _store_cached_search_response(
            ctx,
            cache_request,
            payload,
            validation_tokens=validation_tokens,
        )
    _finish_inflight_search_coalescing(
        ctx,
        inflight_search,
        payload=payload | {_CACHE_VALIDATION_TOKENS_FIELD: validation_tokens},
    )
    return payload


async def _search_memory_records_async(
    ctx: MemoryReadDependencies,
    arguments: dict,
    *,
    caller_kind: str = "external",
    telemetry: RetrievalTelemetryPort,
) -> dict:
    """Execute the record pipeline without nesting an event loop."""
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    query = arguments["query"]
    limit = arguments["limit"]
    started_at = perf_counter()
    pipeline = build_memory_record_pipeline(
        ctx.repository,
        vector_store=ctx.vector_store,
        embedder=ctx.embedder,
        embedding_maintenance=ctx.embedding_maintenance,
        adaptive_enabled=arguments.get("adaptive_limit", "limit" not in arguments),
        config=ctx.config,
    )
    outcome = await pipeline.search(
        query,
        limit=limit,
        filters={
            "workspace_id": ctx.workspace_id,
            "memory_type": arguments["memory_type"],
            "status": arguments["status"],
            "include_superseded": arguments["include_superseded"],
        },
    )
    results = [
        _to_relational_search_result(result)
        for result in outcome.results
    ]
    surfaced_memory_ids = [result.memory_id for result in results]
    duration_ms = (perf_counter() - started_at) * 1000.0
    telemetry.record_search(
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        duration_ms=duration_ms,
    )
    payload: dict[str, object] = {
        "status": "ok",
        "results": build_search_result_payloads(
            results,
            debug_enabled=arguments["debug"],
        ),
    }
    if arguments["debug"]:
        payload["search_diagnostics"] = {
            "kernel_failures": [
                {
                    "stage": failure.stage,
                    "message": failure.message,
                    "exception_type": failure.exception_type,
                }
                for failure in outcome.failures
            ],
            "missing_record_ids": list(outcome.missing_record_ids),
            "degraded": outcome.degraded,
            "pipeline_diagnostics": list(pipeline.diagnostics.reasons),
            "timing_ms": {"total": round(duration_ms, 3)},
        }
        payload["timing_ms"] = {"total": round(duration_ms, 3)}
        payload["adaptive_limit_enabled"] = arguments.get(
            "adaptive_limit",
            "limit" not in arguments,
        )
        payload["requested_limit"] = limit
        payload["expanded_result_window"] = len(results) > limit
        payload["returned_result_count"] = len(results)
    return payload


def _read_memory_record(
    ctx: MemoryReadDependencies,
    arguments: dict,
    *,
    caller_kind: str = "external",
    telemetry: RetrievalTelemetryPort,
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    operation = ReadMemoryRecordOperation(ctx.relational_search)
    memory_id = arguments["memory_id"]
    include_relationships = arguments["include_relationships"]
    include_superseded = arguments["include_superseded"]
    include_metadata = arguments["include_metadata"]
    resolve_memory_id = getattr(ctx.relational_search, "resolve_memory_id", None)
    telemetry_memory_id = memory_id
    if callable(resolve_memory_id):
        resolved_memory_id = resolve_memory_id(memory_id)
        if isinstance(resolved_memory_id, str) and resolved_memory_id:
            telemetry_memory_id = resolved_memory_id
    default_external_read_shape = (
        caller_kind == "external"
        and not include_relationships
        and not include_superseded
        and not include_metadata
    )
    started_at = perf_counter()
    _increment_shared_read_cache_metric(
        ctx,
        "read_requests",
        caller_kind=caller_kind,
    )
    cached_payload, authoritative_validation_token = (
        _load_validated_cached_read_hit(
            ctx,
            memory_id,
            caller_kind=caller_kind,
        )
        if default_external_read_shape
        else (None, None)
    )
    if cached_payload is not None:
        cached_payload = _compact_cached_read_payload(
            cached_payload,
            include_relationships=include_relationships,
            include_superseded=include_superseded,
            include_metadata=include_metadata,
        )
        telemetry.record_read(
            caller_kind=caller_kind,
            memory_id=telemetry_memory_id,
            duration_ms=(perf_counter() - started_at) * 1000.0,
        )
        return cached_payload
    try:
        result = operation.execute(memory_id)
    except Exception as error:
        if _shared_read_cache_enabled(ctx, caller_kind=caller_kind):
            cached_payload = _load_cached_read_fallback(ctx, memory_id, error=error)
            if cached_payload is not None:
                return _compact_cached_read_payload(
                    cached_payload,
                    include_relationships=include_relationships,
                    include_superseded=include_superseded,
                    include_metadata=include_metadata,
                )
        raise
    duration_ms = (perf_counter() - started_at) * 1000.0
    if result is None:
        return {"status": "error", "error": "memory_not_found"}
    telemetry.record_read(
        caller_kind=caller_kind,
        memory_id=telemetry_memory_id,
        duration_ms=duration_ms,
    )

    payload = build_read_payload(
        result,
        include_relationships=include_relationships,
        include_superseded=include_superseded,
        include_metadata=include_metadata,
    )
    if default_external_read_shape and _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind
    ):
        if authoritative_validation_token is None:
            try:
                authoritative_validation_token = _resolve_read_cache_validation_token(
                    ctx, memory_id
                )
            except Exception:
                logger.warning(
                    "Read cache validation token refresh failed after authoritative read",
                    exc_info=True,
                )
        _store_cached_read_response(
            ctx,
            memory_id,
            payload,
            validation_token=authoritative_validation_token,
        )
    return payload

class RecordThoughtUseCase:
    def __init__(
        self,
        ctx: MemoryMutationDependencies,
        *,
        writeback_cache=None,
        max_outbox_entries: int | None = None,
    ) -> None:
        self._ctx = ctx
        self._writeback_cache = writeback_cache
        self._max_outbox_entries = max_outbox_entries

    def execute(self, content: str) -> dict:
        return _record_thought(
            self._ctx,
            content,
            writeback_cache=self._writeback_cache,
            max_outbox_entries=self._max_outbox_entries,
        )


class SearchMemoryRecordsUseCase:
    def __init__(
        self, ctx: MemoryReadDependencies, telemetry: RetrievalTelemetryPort
    ) -> None:
        self._ctx = ctx
        self._telemetry = telemetry

    def execute(self, arguments: dict, *, caller_kind: str = "external") -> dict:
        return _search_memory_records(
            self._ctx,
            arguments,
            caller_kind=caller_kind,
            telemetry=self._telemetry,
        )

    async def execute_async(
        self,
        arguments: dict,
        *,
        caller_kind: str = "external",
    ) -> dict:
        return await _search_memory_records_async(
            self._ctx,
            arguments,
            caller_kind=caller_kind,
            telemetry=self._telemetry,
        )


class ReadMemoryRecordUseCase:
    def __init__(
        self, ctx: MemoryReadDependencies, telemetry: RetrievalTelemetryPort
    ) -> None:
        self._ctx = ctx
        self._telemetry = telemetry

    def execute(self, arguments: dict, *, caller_kind: str = "external") -> dict:
        return _read_memory_record(
            self._ctx,
            arguments,
            caller_kind=caller_kind,
            telemetry=self._telemetry,
        )

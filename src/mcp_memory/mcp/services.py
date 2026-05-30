from __future__ import annotations

from collections.abc import Sequence
import logging
from time import perf_counter, time
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal_operations import (
    RecordThoughtOperation,
)
from mcp_memory.mcp.validation import (
    optional_bool,
    optional_positive_int,
    optional_string,
    require_string,
)
from mcp_memory.relational.operations import (
    ReadMemoryRecordOperation,
    SearchMemoryRecordsOperation,
)
from mcp_memory.relational.search import RelationalSearchResult
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository
from mcp_memory.serialization import (
    agent_link_payload,
    agent_memory_record_payload,
    search_result_payload,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheInFlightSearch,
    SharedReadCacheProjectionEntry,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheSearchRequest,
)
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


SEARCH_READ_GUIDANCE = (
    "Use these summary-first results to identify the most promising memories, then call read_memory_record for full context on the specific memory_id values you want to inspect."
)
_FRESH_SEARCH_CACHE_HIT_TTL_SECONDS = 5.0
_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0


logger = logging.getLogger(__name__)


def _shared_read_cache_enabled(ctx: ApplicationContext, *, caller_kind: str, debug_enabled: bool = False) -> bool:
    return caller_kind == "external" and not debug_enabled and getattr(ctx, "read_cache", None) is not None


def _increment_shared_read_cache_metric(
    ctx: ApplicationContext,
    metric_name: str,
    *,
    amount: int = 1,
    caller_kind: str,
    debug_enabled: bool = False,
) -> None:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        return
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return
    cache.increment_metric(metric_name, amount=amount)


def _annotate_cached_fallback(
    payload: dict[str, object],
    *,
    cache_status: str,
) -> dict[str, object]:
    if payload.get("cache_status") == cache_status and payload.get("degraded") is True:
        return payload
    return payload | {"cache_status": cache_status, "degraded": True}


def _load_cached_search_fallback(
    ctx: ApplicationContext,
    request: SharedReadCacheSearchRequest,
    *,
    error: Exception,
) -> dict | None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    payload = cache.load_search_response(request)
    if payload is None:
        return None
    logger.warning("Serving cached stale search response after authoritative failure", exc_info=error)
    return _annotate_cached_fallback(payload, cache_status="stale_fallback")


def _load_projection_search_fallback(
    ctx: ApplicationContext,
    request: SharedReadCacheSearchRequest,
    *,
    error: Exception,
) -> dict | None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    result_payloads = cache.search_projection_payloads(request)
    if not result_payloads:
        return None
    logger.warning("Serving projection-backed degraded search response after authoritative failure", exc_info=error)
    return _annotate_cached_fallback(
        {
            "status": "ok",
            "recommended_follow_up_tool": "read_memory_record",
            "results": result_payloads,
            "guidance": SEARCH_READ_GUIDANCE,
        },
        cache_status="projection_fallback",
    )


def _load_fresh_cached_search_hit(
    ctx: ApplicationContext,
    request: SharedReadCacheSearchRequest,
    *,
    caller_kind: str,
    debug_enabled: bool,
) -> dict | None:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        return None
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    return cache.load_fresh_search_response(request, ttl_seconds=_FRESH_SEARCH_CACHE_HIT_TTL_SECONDS)


def _load_cached_read_fallback(ctx: ApplicationContext, memory_id: str, *, error: Exception) -> dict | None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    payload = cache.load_read_response(memory_id)
    if payload is None:
        return None
    logger.warning("Serving cached stale read response after authoritative failure", exc_info=error)
    return _annotate_cached_fallback(payload, cache_status="stale_fallback")


def _store_cached_search_response(
    ctx: ApplicationContext,
    request: SharedReadCacheSearchRequest,
    payload: dict[str, object],
) -> None:
    cache = getattr(ctx, "read_cache", None)
    if cache is not None:
        cache.store_search_response(request, payload)


def _begin_inflight_search_coalescing(
    ctx: ApplicationContext,
    request: SharedReadCacheSearchRequest,
    *,
    caller_kind: str,
    debug_enabled: bool,
) -> SharedReadCacheInFlightSearch | None:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        return None
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    return cache.begin_inflight_search(request)


def _finish_inflight_search_coalescing(
    ctx: ApplicationContext,
    entry: SharedReadCacheInFlightSearch | None,
    *,
    payload: dict[str, object] | None = None,
    error: Exception | None = None,
) -> None:
    if entry is None or not entry.is_leader:
        return
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return
    cache.finish_inflight_search(entry, payload=payload, error=error)


def _resolve_read_cache_validation_tokens(
    ctx: ApplicationContext,
    memory_ids: list[str],
) -> dict[str, str]:
    resolver = getattr(ctx.relational_search, "get_read_cache_validation_tokens", None)
    if not callable(resolver) or not memory_ids:
        return {}
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for memory_id in memory_ids:
        if not isinstance(memory_id, str) or not memory_id or memory_id in seen_ids:
            continue
        normalized_ids.append(memory_id)
        seen_ids.add(memory_id)
    if not normalized_ids:
        return {}
    tokens = resolver(normalized_ids)
    if not isinstance(tokens, dict):
        return {}
    return {
        memory_id: token
        for memory_id, token in tokens.items()
        if isinstance(memory_id, str) and isinstance(token, str) and token
    }


def _resolve_read_cache_validation_token(ctx: ApplicationContext, memory_id: str) -> str | None:
    return _resolve_read_cache_validation_tokens(ctx, [memory_id]).get(memory_id)


def _build_search_result_payloads(
    results: Sequence[RelationalSearchResult],
    *,
    debug_enabled: bool,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for result in results:
        ranking_debug = getattr(result, "ranking_debug", None)
        payloads.append(
            search_result_payload(result)
            | ({"ranking_debug": ranking_debug} if debug_enabled and ranking_debug is not None else {})
        )
    return payloads


def _store_cached_projection_entries(
    ctx: ApplicationContext,
    payloads: list[dict[str, object]],
    *,
    validation_tokens: dict[str, str],
) -> None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None or not payloads:
        return
    cache.store_projection_entries(
        [
            SharedReadCacheProjectionUpsert(
                memory_id=str(payload["memory_id"]),
                payload=payload,
                validation_token=validation_tokens.get(str(payload["memory_id"])),
            )
            for payload in payloads
            if isinstance(payload.get("memory_id"), str) and payload["memory_id"]
        ]
    )


def _warm_cached_search_projections(
    ctx: ApplicationContext,
    result_payloads: list[dict[str, object]],
    *,
    caller_kind: str,
    debug_enabled: bool,
) -> int:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        return 0
    warmed_payloads = [
        payload
        for payload in result_payloads
        if isinstance(payload.get("memory_id"), str) and payload["memory_id"]
    ]
    if not warmed_payloads:
        return 0
    memory_ids = [str(payload["memory_id"]) for payload in warmed_payloads]
    validation_tokens: dict[str, str] = {}
    if memory_ids:
        try:
            validation_tokens = _resolve_read_cache_validation_tokens(ctx, memory_ids)
        except Exception:
            logger.warning(
                "Projection cache validation token refresh failed after authoritative search",
                exc_info=True,
            )
    try:
        _store_cached_projection_entries(ctx, warmed_payloads, validation_tokens=validation_tokens)
    except Exception:
        logger.warning("Projection cache warming failed after authoritative search", exc_info=True)
        return 0
    return len(warmed_payloads)


def _load_validated_cached_projection_entries(
    ctx: ApplicationContext,
    memory_ids: list[str],
    *,
    caller_kind: str,
    debug_enabled: bool = False,
) -> list[SharedReadCacheProjectionEntry]:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        return []
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return []
    entries = cache.load_projection_entries(memory_ids)
    if not entries:
        return []
    entries_with_tokens = [entry for entry in entries if entry.validation_token is not None]
    if not entries_with_tokens:
        cache.delete_projection_entries([entry.memory_id for entry in entries])
        return []
    try:
        authoritative_tokens = _resolve_read_cache_validation_tokens(
            ctx,
            [entry.memory_id for entry in entries_with_tokens],
        )
    except Exception:
        logger.warning(
            "Projection cache validation failed; ignoring cached projections",
            exc_info=True,
        )
        return []
    valid_entries: list[SharedReadCacheProjectionEntry] = []
    stale_ids: list[str] = []
    for entry in entries:
        authoritative_token = authoritative_tokens.get(entry.memory_id)
        if authoritative_token is not None and authoritative_token == entry.validation_token:
            valid_entries.append(entry)
            continue
        stale_ids.append(entry.memory_id)
    if stale_ids:
        cache.delete_projection_entries(stale_ids)
    return valid_entries


def _load_validated_cached_read_hit(
    ctx: ApplicationContext,
    memory_id: str,
    *,
    caller_kind: str,
) -> tuple[dict[str, object] | None, str | None]:
    if not _shared_read_cache_enabled(ctx, caller_kind=caller_kind):
        return (None, None)
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return (None, None)
    entry = cache.load_read_entry(memory_id)
    if entry is None or entry.validation_token is None:
        return (None, None)
    try:
        authoritative_token = _resolve_read_cache_validation_token(ctx, memory_id)
    except Exception:
        _increment_shared_read_cache_metric(
            ctx,
            "read_validation_failures",
            caller_kind=caller_kind,
        )
        logger.warning("Read cache validation failed; falling back to authoritative read", exc_info=True)
        return (None, None)
    if authoritative_token is None or authoritative_token != entry.validation_token:
        _increment_shared_read_cache_metric(
            ctx,
            "read_validation_mismatches",
            caller_kind=caller_kind,
        )
        return (None, authoritative_token)
    _increment_shared_read_cache_metric(
        ctx,
        "validated_read_hits",
        caller_kind=caller_kind,
    )
    return (entry.payload, authoritative_token)


def _store_cached_read_response(
    ctx: ApplicationContext,
    memory_id: str,
    payload: dict[str, object],
    *,
    validation_token: str | None,
) -> None:
    cache = getattr(ctx, "read_cache", None)
    if cache is not None:
        cache.store_read_response(memory_id, payload, validation_token=validation_token)


def record_thought_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    suppression_config = None if ctx.config is None else ctx.config.ingest_suppression
    cache_state = resolve_shared_mode_cache_state(
        ctx.config,
        storage_backend=ctx.storage_backend,
        read_cache=getattr(ctx, "read_cache", None),
    )
    operation = RecordThoughtOperation(
        ctx.journal,
        ctx.task_queue,
        ctx.workspace_id,
        suppression_config,
        writeback_cache=cache_state.writeback_cache,
        max_outbox_entries=cache_state.max_outbox_entries,
    )
    return operation.execute(require_string(arguments, "content"))


def _retrieval_telemetry_repository(ctx: ApplicationContext) -> RetrievalTelemetryRepository:
    repository = ctx.retrieval_telemetry
    if repository is None:
        repository = RetrievalTelemetryRepository(ctx.db_manager, workspace_id=ctx.workspace_id)
        ctx.retrieval_telemetry = repository
    return repository


def _log_slow_memory_tool_operation(
    ctx: ApplicationContext,
    *,
    tool_name: str,
    duration_ms: float,
    data: dict[str, object],
) -> None:
    if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
        return
    runtime_logs = getattr(ctx, "runtime_logs", None)
    if runtime_logs is not None:
        try:
            runtime_logs.write_log(
                source="memory-tool",
                logger_name=__name__,
                level="WARNING",
                message=f"Slow {tool_name} operation",
                created_at=time(),
                data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)} | data,
            )
            return
        except Exception:  # pragma: no cover - defensive fallback for locked telemetry/log stores
            logger.warning("Failed to persist slow %s log; falling back to process logger", tool_name, exc_info=True)
    logger.warning("Slow %s operation: %.3fms %s", tool_name, duration_ms, data)


def _record_search_invocation(
    ctx: ApplicationContext,
    *,
    caller_kind: str,
    query: str,
    surfaced_memory_ids: list[str],
    duration_ms: float,
) -> None:
    _retrieval_telemetry_repository(ctx).record_search(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        duration_ms=duration_ms,
    )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="search_memory_records",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "query": query,
            "result_count": len(surfaced_memory_ids),
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )


def _record_read_invocation(
    ctx: ApplicationContext,
    *,
    caller_kind: str,
    memory_id: str,
    duration_ms: float,
) -> None:
    repository = _retrieval_telemetry_repository(ctx)
    repository.record_read(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        memory_id=memory_id,
        duration_ms=duration_ms,
    )
    try:
        repository.flush()
    except Exception:
        logger.debug("Read telemetry flush failed; continuing without blocking the read response", exc_info=True)
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="read_memory_record",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "memory_id": memory_id,
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )


def search_memory_records_service(
    ctx: ApplicationContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    query = require_string(arguments, "query")
    operation = SearchMemoryRecordsOperation(ctx.relational_search)
    started_at = perf_counter()
    debug_enabled = optional_bool(arguments, "debug", False)
    execution_arguments = {
        "query": query,
        "workspace_id": ctx.workspace_id,
        "limit": optional_positive_int(arguments, "limit", 5),
        "adaptive_limit": "limit" not in arguments,
        "memory_type": optional_string(arguments, "memory_type"),
        "status": optional_string(arguments, "status"),
        "include_superseded": optional_bool(arguments, "include_superseded", False),
        "debug": debug_enabled,
    }
    cache_request = SharedReadCacheSearchRequest(
        query=query,
        workspace_id=ctx.workspace_id,
        limit=execution_arguments["limit"],
        adaptive_limit=execution_arguments["adaptive_limit"],
        memory_type=execution_arguments["memory_type"],
        status=execution_arguments["status"],
        include_superseded=execution_arguments["include_superseded"],
    )
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
            payload = ctx.read_cache.wait_for_inflight_search(inflight_search)
        except Exception as error:
            if _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
                cached_payload = _load_cached_search_fallback(ctx, cache_request, error=error)
                if cached_payload is not None:
                    _increment_shared_read_cache_metric(
                        ctx,
                        "stale_exact_search_fallbacks",
                        caller_kind=caller_kind,
                        debug_enabled=debug_enabled,
                    )
                    return cached_payload
                projection_payload = _load_projection_search_fallback(ctx, cache_request, error=error)
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
            for result in payload.get("results", [])
            if isinstance(result, dict) and isinstance(result.get("memory_id"), str)
        ]
        _record_search_invocation(
            ctx,
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
            duration_ms=duration_ms,
        )
        return payload
    diagnostics = None
    try:
        if debug_enabled:
            results, diagnostics = operation.execute_with_diagnostics(**execution_arguments)
        else:
            results = operation.execute(**execution_arguments)
    except Exception as error:
        _finish_inflight_search_coalescing(ctx, inflight_search, error=error)
        if _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
            cached_payload = _load_cached_search_fallback(ctx, cache_request, error=error)
            if cached_payload is not None:
                _increment_shared_read_cache_metric(
                    ctx,
                    "stale_exact_search_fallbacks",
                    caller_kind=caller_kind,
                    debug_enabled=debug_enabled,
                )
                return cached_payload
            projection_payload = _load_projection_search_fallback(ctx, cache_request, error=error)
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
    _record_search_invocation(
        ctx,
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        duration_ms=duration_ms,
    )
    result_payloads = _build_search_result_payloads(results, debug_enabled=debug_enabled)
    payload = {
        "status": "ok",
        "recommended_follow_up_tool": "read_memory_record",
        "results": result_payloads,
        "guidance": SEARCH_READ_GUIDANCE,
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
    warmed_projection_rows = _warm_cached_search_projections(
        ctx,
        result_payloads,
        caller_kind=caller_kind,
        debug_enabled=debug_enabled,
    )
    if warmed_projection_rows > 0:
        _increment_shared_read_cache_metric(
            ctx,
            "warmed_projection_rows",
            amount=warmed_projection_rows,
            caller_kind=caller_kind,
            debug_enabled=debug_enabled,
        )
    if _shared_read_cache_enabled(ctx, caller_kind=caller_kind, debug_enabled=debug_enabled):
        _store_cached_search_response(ctx, cache_request, payload)
    _finish_inflight_search_coalescing(ctx, inflight_search, payload=payload)
    return payload


def read_memory_record_service(
    ctx: ApplicationContext,
    arguments: dict,
    *,
    caller_kind: str = "external",
) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    operation = ReadMemoryRecordOperation(ctx.relational_search)
    memory_id = require_string(arguments, "memory_id")
    started_at = perf_counter()
    _increment_shared_read_cache_metric(
        ctx,
        "read_requests",
        caller_kind=caller_kind,
    )
    cached_payload, authoritative_validation_token = _load_validated_cached_read_hit(
        ctx,
        memory_id,
        caller_kind=caller_kind,
    )
    if cached_payload is not None:
        _record_read_invocation(
            ctx,
            caller_kind=caller_kind,
            memory_id=memory_id,
            duration_ms=(perf_counter() - started_at) * 1000.0,
        )
        return cached_payload
    try:
        result = operation.execute(memory_id)
    except Exception as error:
        if _shared_read_cache_enabled(ctx, caller_kind=caller_kind):
            cached_payload = _load_cached_read_fallback(ctx, memory_id, error=error)
            if cached_payload is not None:
                return cached_payload
        raise
    duration_ms = (perf_counter() - started_at) * 1000.0
    if result is None:
        return {"status": "error", "error": "memory_not_found"}
    _record_read_invocation(
        ctx,
        caller_kind=caller_kind,
        memory_id=memory_id,
        duration_ms=duration_ms,
    )

    payload = {
        "status": "ok",
        "record": agent_memory_record_payload(result.record),
        "relationships": {
            direction: [agent_link_payload(link) for link in links]
            for direction, links in result.relationships.items()
        },
        "superseded": [agent_memory_record_payload(record) for record in result.superseded],
    }
    if _shared_read_cache_enabled(ctx, caller_kind=caller_kind):
        if authoritative_validation_token is None:
            try:
                authoritative_validation_token = _resolve_read_cache_validation_token(ctx, memory_id)
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

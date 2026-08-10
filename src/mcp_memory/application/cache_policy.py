from __future__ import annotations

from collections.abc import Sequence
from contextvars import ContextVar
import logging
from typing import Any

from mcp_memory.application.ports import (
    DEFAULT_SEARCH_POLICY_VERSION,
    MemoryReadPort,
    SharedReadCacheInFlightSearch,
    SharedReadCacheProjectionEntry,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheSearchRequest,
)
from mcp_memory.core.ports.memory import parse_memory_ref
from mcp_memory.relational.search import RelationalSearchResult
from mcp_memory.serialization import (
    search_result_payload_compact,
    search_result_payload_with_debug_fields,
)


SEARCH_READ_GUIDANCE = (
    "Read promising memory_ref values with read_memory_record or read_memory_records."
)
_FRESH_SEARCH_CACHE_HIT_TTL_SECONDS = 5.0
_CACHE_VALIDATION_TOKENS_FIELD = "_cache_validation_tokens"
logger = logging.getLogger(__name__)
_ACTIVE_INFLIGHT_SEARCH: ContextVar[SharedReadCacheInFlightSearch | None] = ContextVar(
    "active_inflight_search", default=None
)


def build_search_cache_request(
    ctx: MemoryReadPort,
    arguments: dict[str, Any],
    *,
    policy_version: str = DEFAULT_SEARCH_POLICY_VERSION,
    feature_fingerprint: str | None = None,
) -> SharedReadCacheSearchRequest:
    return SharedReadCacheSearchRequest(
        query=str(arguments["query"]),
        workspace_id=(
            arguments["workspace_id"]
            if arguments.get("workspace_id") is not None
            else ctx.workspace_id
        ),
        limit=int(arguments["limit"]),
        adaptive_limit=bool(arguments["adaptive_limit"]),
        memory_type=arguments["memory_type"],
        status=arguments["status"],
        include_superseded=bool(arguments["include_superseded"]),
        ranking_workspace_id=(
            ctx.workspace_id
            if arguments.get("workspace_id") is not None
            else None
        ),
        policy_version=policy_version,
        feature_fingerprint=feature_fingerprint,
    )


def _shared_read_cache_enabled(
    ctx: MemoryReadPort, *, caller_kind: str, debug_enabled: bool = False
) -> bool:
    return (
        caller_kind == "external"
        and not debug_enabled
        and getattr(ctx, "read_cache", None) is not None
    )


def _increment_shared_read_cache_metric(
    ctx: MemoryReadPort,
    metric_name: str,
    *,
    amount: int = 1,
    caller_kind: str,
    debug_enabled: bool = False,
) -> None:
    if not _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
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


def _compact_cached_search_payload(payload: dict[str, object]) -> dict[str, object]:
    compact_payload = dict(payload)
    compact_payload.pop(_CACHE_VALIDATION_TOKENS_FIELD, None)
    results = compact_payload.get("results")
    if isinstance(results, list):
        compact_payload["results"] = [
            _compact_cached_search_result(result)
            for result in results
            if isinstance(result, dict)
        ]
    return compact_payload


def _compact_cached_search_result(result: dict[str, object]) -> dict[str, object]:
    identifier_key = (
        "memory_ref" if isinstance(result.get("memory_ref"), str) else "memory_id"
    )
    return {
        key: result[key]
        for key in (
            identifier_key,
            "title",
            "summary",
            "status",
            "created_at",
            "updated_at",
        )
        if key in result
    }


def _cached_result_memory_id(
    ctx: MemoryReadPort, result: dict[str, object]
) -> str | None:
    memory_id = result.get("memory_id")
    if isinstance(memory_id, str) and memory_id:
        return memory_id
    memory_ref = result.get("memory_ref")
    resolver = getattr(getattr(ctx, "relational_search", None), "resolve_memory_id", None)
    if not isinstance(memory_ref, str):
        return None
    if parse_memory_ref(memory_ref) is None:
        return memory_ref
    if not callable(resolver):
        return None
    resolved_id = resolver(memory_ref)
    return resolved_id if isinstance(resolved_id, str) and resolved_id else None


def _compact_cached_read_payload(
    payload: dict[str, object],
    *,
    include_relationships: bool,
    include_superseded: bool,
    include_metadata: bool,
    summary_only: bool = False,
    content_offset: int = 0,
    content_limit: int | None = None,
) -> dict[str, object]:
    compact_payload = dict(payload)
    compact_payload.pop("related_counts", None)
    if not include_relationships:
        compact_payload.pop("relationships", None)
    if not include_superseded:
        compact_payload.pop("superseded", None)
    if not include_metadata:
        _strip_cached_record_noise(compact_payload.get("record"))
    record = compact_payload.get("record")
    if isinstance(record, dict):
        if summary_only:
            content = str(record.pop("content", ""))
            record["summary"] = record.get("summary") or content[:220]
        elif content_limit is not None and isinstance(record.get("content"), str):
            content = record["content"]
            chunk = content[content_offset : content_offset + content_limit]
            record["content"] = chunk
            record["content_offset"] = content_offset
            record["content_total_chars"] = len(content)
            record["content_has_more"] = content_offset + len(chunk) < len(content)
    compact_superseded = compact_payload.get("superseded")
    if not include_metadata and isinstance(compact_superseded, list):
        for record in compact_superseded:
            _strip_cached_record_noise(record)
    return compact_payload


def _strip_cached_record_noise(record: object) -> None:
    if not isinstance(record, dict):
        return
    for key in (
        "read_count",
        "access_score",
        "last_accessed_at",
        "last_surfaced_at",
        "metadata",
        "workspace_ids",
        "summary",
        "type",
        "status",
        "created_at",
        "updated_at",
        "tags",
    ):
        record.pop(key, None)


def _load_cached_search_fallback(
    ctx: MemoryReadPort,
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
    logger.warning(
        "Serving cached stale search response after authoritative failure",
        exc_info=error,
    )
    return _annotate_cached_fallback(
        _compact_cached_search_payload(payload), cache_status="stale_fallback"
    )


def _load_projection_search_fallback(
    ctx: MemoryReadPort,
    request: SharedReadCacheSearchRequest,
    *,
    error: Exception,
) -> dict | None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    entries = cache.search_projection_entries(request, limit=None)
    if not entries:
        return None
    resolver = getattr(ctx.relational_search, "get_read_cache_validation_tokens", None)
    if callable(resolver):
        entries = _validate_cached_projection_entries(ctx, entries)
    else:
        entries = [entry for entry in entries if entry.validation_token is not None]
    if not entries:
        return None
    entries = entries[: request.limit]
    logger.warning(
        "Serving projection-backed degraded search response after authoritative failure",
        exc_info=error,
    )
    return _annotate_cached_fallback(
        {
            "status": "ok",
            "results": [
                _compact_cached_search_result(entry.payload) for entry in entries
            ],
        },
        cache_status="projection_fallback",
    )


def _load_fresh_cached_search_hit(
    ctx: MemoryReadPort,
    request: SharedReadCacheSearchRequest,
    *,
    caller_kind: str,
    debug_enabled: bool,
) -> dict | None:
    if not _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        return None
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    payload = cache.load_fresh_search_response(
        request, ttl_seconds=_FRESH_SEARCH_CACHE_HIT_TTL_SECONDS
    )
    if payload is None:
        return None
    if not _cached_search_payload_has_current_records(ctx, payload):
        return None
    return _compact_cached_search_payload(payload)


def _load_cached_read_fallback(
    ctx: MemoryReadPort, memory_id: str, *, error: Exception
) -> dict | None:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    payload = cache.load_read_response(memory_id)
    if payload is None:
        return None
    logger.warning(
        "Serving cached stale read response after authoritative failure", exc_info=error
    )
    return _annotate_cached_fallback(payload, cache_status="stale_fallback")


def _store_cached_search_response(
    ctx: MemoryReadPort,
    request: SharedReadCacheSearchRequest,
    payload: dict[str, object],
    *,
    validation_tokens: dict[str, str],
) -> None:
    cache = getattr(ctx, "read_cache", None)
    if cache is not None:
        cache.store_search_response(
            request,
            payload | {_CACHE_VALIDATION_TOKENS_FIELD: validation_tokens},
            inflight_search=_ACTIVE_INFLIGHT_SEARCH.get(),
        )


def _begin_inflight_search_coalescing(
    ctx: MemoryReadPort,
    request: SharedReadCacheSearchRequest,
    *,
    caller_kind: str,
    debug_enabled: bool,
) -> SharedReadCacheInFlightSearch | None:
    if not _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        return None
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return None
    entry = cache.begin_inflight_search(request)
    if entry.is_leader:
        _ACTIVE_INFLIGHT_SEARCH.set(entry)
    return entry


def _finish_inflight_search_coalescing(
    ctx: MemoryReadPort,
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
    try:
        cache.finish_inflight_search(entry, payload=payload, error=error)
    finally:
        if _ACTIVE_INFLIGHT_SEARCH.get() is entry:
            _ACTIVE_INFLIGHT_SEARCH.set(None)


def _resolve_read_cache_validation_tokens(
    ctx: MemoryReadPort,
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


def _resolve_read_cache_validation_token(
    ctx: MemoryReadPort, memory_id: str
) -> str | None:
    return _resolve_read_cache_validation_tokens(ctx, [memory_id]).get(memory_id)


def _cached_search_payload_has_current_records(
    ctx: MemoryReadPort, payload: dict[str, object]
) -> bool:
    resolver = getattr(ctx.relational_search, "get_read_cache_validation_tokens", None)
    if not callable(resolver):
        return True
    results = payload.get("results")
    if not isinstance(results, list):
        return True
    memory_ids: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        memory_id = _cached_result_memory_id(ctx, result)
        if memory_id is None:
            return False
        memory_ids.append(memory_id)
    if not memory_ids:
        return True
    cached_tokens = payload.get(_CACHE_VALIDATION_TOKENS_FIELD)
    if not isinstance(cached_tokens, dict):
        return False
    try:
        validation_tokens = _resolve_read_cache_validation_tokens(ctx, memory_ids)
    except Exception:
        logger.warning(
            "Fresh search cache validation failed; falling back to authoritative search",
            exc_info=True,
        )
        return False
    mismatched_ids = {
        memory_id
        for memory_id in memory_ids
        if not isinstance(cached_tokens.get(memory_id), str)
        or validation_tokens.get(memory_id) != cached_tokens.get(memory_id)
    }
    if mismatched_ids:
        logger.info(
            "Discarding fresh search cache response with stale validation tokens",
            extra={"memory_ids": sorted(mismatched_ids)},
        )
        return False
    return True


def _build_search_result_payloads(
    results: Sequence[RelationalSearchResult],
    *,
    debug_enabled: bool,
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for result in results:
        ranking_debug = getattr(result, "ranking_debug", None)
        base_payload = (
            search_result_payload_with_debug_fields(result)
            if debug_enabled
            else search_result_payload_compact(result)
        )
        payloads.append(
            base_payload
            | (
                {"ranking_debug": ranking_debug}
                if debug_enabled and ranking_debug is not None
                else {}
            )
        )
    return payloads


def _store_cached_projection_entries(
    ctx: MemoryReadPort,
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
                memory_id=memory_id,
                payload=payload,
                validation_token=validation_tokens.get(memory_id),
            )
            for payload in payloads
            if (memory_id := _cached_result_memory_id(ctx, payload)) is not None
        ]
    )


def _warm_cached_search_projections(
    ctx: MemoryReadPort,
    result_payloads: list[dict[str, object]],
    *,
    caller_kind: str,
    debug_enabled: bool,
    validation_tokens: dict[str, str] | None = None,
) -> int:
    if not _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        return 0
    warmed_payloads = [
        payload
        for payload in result_payloads
        if _cached_result_memory_id(ctx, payload) is not None
    ]
    if not warmed_payloads:
        return 0
    memory_ids = [
        memory_id
        for payload in warmed_payloads
        if (memory_id := _cached_result_memory_id(ctx, payload)) is not None
    ]
    resolved_tokens = validation_tokens or {}
    if memory_ids and validation_tokens is None:
        try:
            resolved_tokens = _resolve_read_cache_validation_tokens(ctx, memory_ids)
        except Exception:
            logger.warning(
                "Projection cache validation token refresh failed after authoritative search",
                exc_info=True,
            )
    try:
        _store_cached_projection_entries(
            ctx, warmed_payloads, validation_tokens=resolved_tokens
        )
    except Exception:
        logger.warning(
            "Projection cache warming failed after authoritative search", exc_info=True
        )
        return 0
    return len(warmed_payloads)


def _load_validated_cached_projection_entries(
    ctx: MemoryReadPort,
    memory_ids: list[str],
    *,
    caller_kind: str,
    debug_enabled: bool = False,
) -> list[SharedReadCacheProjectionEntry]:
    if not _shared_read_cache_enabled(
        ctx, caller_kind=caller_kind, debug_enabled=debug_enabled
    ):
        return []
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return []
    entries = cache.load_projection_entries(memory_ids)
    if not entries:
        return []
    return _validate_cached_projection_entries(ctx, entries)


def _validate_cached_projection_entries(
    ctx: MemoryReadPort,
    entries: list[SharedReadCacheProjectionEntry],
) -> list[SharedReadCacheProjectionEntry]:
    cache = getattr(ctx, "read_cache", None)
    if cache is None:
        return []
    entries_with_tokens = [
        entry for entry in entries if entry.validation_token is not None
    ]
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
        if (
            authoritative_token is not None
            and authoritative_token == entry.validation_token
        ):
            valid_entries.append(entry)
            continue
        stale_ids.append(entry.memory_id)
    if stale_ids:
        cache.delete_projection_entries(stale_ids)
    return valid_entries


def _load_validated_cached_read_hit(
    ctx: MemoryReadPort,
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
        logger.warning(
            "Read cache validation failed; falling back to authoritative read",
            exc_info=True,
        )
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
    ctx: MemoryReadPort,
    memory_id: str,
    payload: dict[str, object],
    *,
    validation_token: str | None,
) -> None:
    cache = getattr(ctx, "read_cache", None)
    if cache is not None:
        cache.store_read_response(memory_id, payload, validation_token=validation_token)

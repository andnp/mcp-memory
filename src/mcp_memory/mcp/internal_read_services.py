from __future__ import annotations

from time import perf_counter
from typing import Any

from mcp_memory.application.ports import MemoryReadContext, MemorySearchPort
from mcp_memory.mcp.services import (
    _record_read_invocation,
    _record_search_invocation,
    read_memory_record_service,
    read_memory_records_service,
    search_memory_records_service,
    search_memory_records_async_service,
)
from mcp_memory.mcp.validation import optional_bool, optional_positive_int, optional_string, require_string
from mcp_memory.serialization import (
    agent_link_payload,
    agent_memory_record_payload,
    agent_memory_record_payload_with_metadata,
)


_DEFAULT_MAINTENANCE_SEARCH_LIMIT = 10
_DEFAULT_MAINTENANCE_RELATIONSHIP_LIMIT = 20
_DEFAULT_MAINTENANCE_ADJACENCY_LIMIT = 10
_MAX_MAINTENANCE_READ_LIMIT = 50


def _bounded_limit(arguments: dict, field_name: str, default: int) -> tuple[int, int]:
    requested = optional_positive_int(arguments, field_name, default)
    return requested, min(requested, _MAX_MAINTENANCE_READ_LIMIT)


def _record_payload(record: Any, *, include_metadata: bool) -> dict:
    if include_metadata:
        return agent_memory_record_payload_with_metadata(record)
    return agent_memory_record_payload(record)


def _maintenance_context(ctx: MemoryReadContext) -> MemorySearchPort | None:
    if ctx.relational_search is None:
        return None
    return ctx.relational_search


def _not_initialized() -> dict[str, str]:
    return {"status": "error", "error": "relational_search_not_initialized"}


def internal_search_memory_records_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    return search_memory_records_service(ctx, arguments, caller_kind="internal")


async def internal_search_memory_records_async_service(
    ctx: MemoryReadContext, arguments: dict
) -> dict:
    return await search_memory_records_async_service(
        ctx,
        arguments,
        caller_kind="internal",
    )


def internal_read_memory_record_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    return read_memory_record_service(ctx, arguments, caller_kind="internal")


def internal_read_memory_records_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    return read_memory_records_service(ctx, arguments, caller_kind="internal")


def internal_peek_record_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    """Read one authoritative record without the public read cache or access updates."""
    search = _maintenance_context(ctx)
    if search is None:
        return _not_initialized()

    memory_id = require_string(arguments, "memory_id")
    include_metadata = optional_bool(arguments, "include_metadata", False)
    started_at = perf_counter()
    result = search.peek_memory(memory_id)
    if result is None:
        return {"status": "error", "error": "memory_not_found"}

    _record_read_invocation(
        ctx,
        caller_kind="internal",
        memory_id=memory_id,
        duration_ms=(perf_counter() - started_at) * 1000.0,
    )
    return {
        "status": "ok",
        "record": _record_payload(result.record, include_metadata=include_metadata),
    }


def internal_maintenance_search_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    """Search authoritative records without surfacing, accessing, or caching them."""
    search = _maintenance_context(ctx)
    if search is None:
        return _not_initialized()

    query = require_string(arguments, "query")
    requested_limit, limit = _bounded_limit(
        arguments, "limit", _DEFAULT_MAINTENANCE_SEARCH_LIMIT
    )
    include_metadata = optional_bool(arguments, "include_metadata", False)
    started_at = perf_counter()
    results = search.search_memories_for_maintenance(
        query,
        workspace_id=ctx.workspace_id,
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        include_superseded=optional_bool(arguments, "include_superseded", False),
        limit=limit,
    )
    _record_search_invocation(
        ctx,
        caller_kind="internal",
        query=query,
        surfaced_memory_ids=[result.record.id for result in results],
        duration_ms=(perf_counter() - started_at) * 1000.0,
    )
    payload: dict[str, object] = {
        "status": "ok",
        "results": [
            {
                "record": _record_payload(result.record, include_metadata=include_metadata),
            }
            for result in results
        ],
    }
    if len(results) >= limit and requested_limit > limit:
        payload["truncated"] = True
    return payload


def internal_list_relationships_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    """List a bounded relationship slice from an authoritative maintenance peek."""
    search = _maintenance_context(ctx)
    if search is None:
        return _not_initialized()

    memory_id = require_string(arguments, "memory_id")
    direction = optional_string(arguments, "direction") or "both"
    if direction not in {"incoming", "outgoing", "both"}:
        raise ValueError("direction must be incoming, outgoing, or both")
    _, limit = _bounded_limit(
        arguments, "limit", _DEFAULT_MAINTENANCE_RELATIONSHIP_LIMIT
    )
    started_at = perf_counter()
    result = search.peek_memory(memory_id)
    if result is None:
        return {"status": "error", "error": "memory_not_found"}

    all_relationships = []
    directions = (direction,) if direction != "both" else ("incoming", "outgoing")
    for relationship_direction in directions:
        all_relationships.extend(
            {
                "direction": relationship_direction,
                "link": agent_link_payload(link),
            }
            for link in result.relationships.get(relationship_direction, [])
        )
    relationships = all_relationships[:limit]
    _record_read_invocation(
        ctx,
        caller_kind="internal",
        memory_id=memory_id,
        duration_ms=(perf_counter() - started_at) * 1000.0,
    )
    payload: dict[str, object] = {
        "status": "ok",
        "memory_id": memory_id,
        "relationships": relationships,
    }
    if len(all_relationships) > limit:
        payload["truncated"] = True
    return payload


def internal_bounded_adjacency_service(ctx: MemoryReadContext, arguments: dict) -> dict:
    """Return one-hop neighboring records with a hard output/read bound."""
    search = _maintenance_context(ctx)
    if search is None:
        return _not_initialized()

    memory_id = require_string(arguments, "memory_id")
    direction = optional_string(arguments, "direction") or "both"
    if direction not in {"incoming", "outgoing", "both"}:
        raise ValueError("direction must be incoming, outgoing, or both")
    _, limit = _bounded_limit(
        arguments, "limit", _DEFAULT_MAINTENANCE_ADJACENCY_LIMIT
    )
    include_metadata = optional_bool(arguments, "include_metadata", False)
    started_at = perf_counter()
    root = search.peek_memory(memory_id)
    if root is None:
        return {"status": "error", "error": "memory_not_found"}

    directions = (direction,) if direction != "both" else ("incoming", "outgoing")
    links_by_neighbor: dict[str, list[dict[str, object]]] = {}
    for relationship_direction in directions:
        for link in root.relationships.get(relationship_direction, []):
            neighbor_id = link.source_id if relationship_direction == "incoming" else link.target_id
            links_by_neighbor.setdefault(neighbor_id, []).append(
                {
                    "direction": relationship_direction,
                    "link": agent_link_payload(link),
                }
            )

    neighbor_ids = list(links_by_neighbor)[:limit]
    neighbors: list[dict[str, object]] = []
    telemetry_ids = [memory_id]
    for neighbor_id in neighbor_ids:
        neighbor = search.peek_memory(neighbor_id)
        if neighbor is None:
            continue
        telemetry_ids.append(neighbor_id)
        neighbors.append(
            {
                "record": _record_payload(neighbor.record, include_metadata=include_metadata),
                "relationships": links_by_neighbor[neighbor_id],
            }
        )
    duration_ms = (perf_counter() - started_at) * 1000.0
    for telemetry_id in telemetry_ids:
        _record_read_invocation(
            ctx,
            caller_kind="internal",
            memory_id=telemetry_id,
            duration_ms=duration_ms,
        )
    payload: dict[str, object] = {
        "status": "ok",
        "memory_id": memory_id,
        "neighbors": neighbors,
    }
    if len(links_by_neighbor) > limit:
        payload["truncated"] = True
    return payload

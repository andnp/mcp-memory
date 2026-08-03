from __future__ import annotations

from collections.abc import Sequence

from mcp_memory.relational.search import RelationalSearchResult
from mcp_memory.serialization import (
    agent_link_payload,
    agent_memory_record_payload,
    agent_memory_record_payload_with_metadata,
    search_result_payload_compact,
    search_result_payload_with_debug_fields,
)


def build_search_result_payloads(
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


def build_read_payload(
    result,
    *,
    include_relationships: bool,
    include_superseded: bool,
    include_metadata: bool,
) -> dict[str, object]:
    relationships_payload = {
        direction: [agent_link_payload(link) for link in links]
        for direction, links in result.relationships.items()
    }
    superseded_payload = [
        (
            agent_memory_record_payload_with_metadata(record)
            if include_metadata
            else agent_memory_record_payload(record)
        )
        for record in result.superseded
    ]
    payload: dict[str, object] = {
        "status": "ok",
        "record": (
            agent_memory_record_payload_with_metadata(result.record)
            if include_metadata
            else agent_memory_record_payload(result.record)
        ),
    }
    if include_relationships:
        payload["relationships"] = relationships_payload
    if include_superseded:
        payload["superseded"] = superseded_payload
    return payload

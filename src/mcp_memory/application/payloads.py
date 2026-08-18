from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite

from mcp_memory.relational.search import RelationalSearchResult
from mcp_memory.serialization import (
    agent_link_payload,
    agent_memory_record_payload,
    agent_memory_record_payload_with_metadata,
    search_result_payload_compact,
    search_result_payload_with_debug_fields,
)

MAX_SEARCH_EVIDENCE_EXCERPTS = 3
MAX_SEARCH_EVIDENCE_EXCERPT_CHARS = 240
MAX_SEARCH_EVIDENCE_SCORE = 100.0

_SEARCH_EVIDENCE_LANES = {
    "keyword": "keyword",
    "vector": "semantic",
    "semantic": "semantic",
    "graph": "graph",
}


def _bounded_search_evidence_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if not isfinite(numeric_value) or abs(numeric_value) > MAX_SEARCH_EVIDENCE_SCORE:
        return None
    return round(numeric_value, 6)


def _raw_result_memory_id(raw_result: object) -> str | None:
    record = getattr(raw_result, "record", None)
    memory_id = getattr(record, "source_id", None)
    return memory_id if isinstance(memory_id, str) else None


def _raw_value(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _search_evidence_for_result(raw_result: object) -> dict[str, object]:
    evidence: dict[str, object] = {}
    provenance = getattr(raw_result, "provenance", None)
    strategies = _raw_value(provenance, "strategies")
    if isinstance(strategies, (list, tuple, set, frozenset)):
        lanes = sorted(
            {
                _SEARCH_EVIDENCE_LANES[strategy]
                for strategy in strategies
                if isinstance(strategy, str) and strategy in _SEARCH_EVIDENCE_LANES
            }
        )
        if lanes:
            evidence["lanes"] = lanes

    normalized_score = _bounded_search_evidence_number(
        getattr(raw_result, "normalized_score", None)
    )
    if normalized_score is not None:
        evidence["normalized_score"] = normalized_score

    chunks = _raw_value(raw_result, "chunk_matches")
    if chunks is None:
        chunks = _raw_value(raw_result, "excerpts")
    if isinstance(chunks, (list, tuple)):
        excerpt_candidates: list[tuple[float, str]] = []
        for chunk in chunks:
            content = _raw_value(chunk, "content")
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content:
                continue
            score = _bounded_search_evidence_number(_raw_value(chunk, "score"))
            excerpt_candidates.append((score if score is not None else float("-inf"), content))
        excerpt_candidates.sort(key=lambda item: (-item[0], item[1]))
        evidence["excerpts"] = [
            {
                "content": content[:MAX_SEARCH_EVIDENCE_EXCERPT_CHARS],
                **({"score": score} if isfinite(score) else {}),
            }
            for score, content in excerpt_candidates[:MAX_SEARCH_EVIDENCE_EXCERPTS]
        ]
    return evidence


def build_search_result_payloads(
    results: Sequence[RelationalSearchResult],
    *,
    debug_enabled: bool,
    raw_results: Sequence[object] | None = None,
) -> list[dict[str, object]]:
    raw_results_by_id = {
        memory_id: raw_result
        for raw_result in raw_results or ()
        if (memory_id := _raw_result_memory_id(raw_result)) is not None
    }
    payloads: list[dict[str, object]] = []
    for result in results:
        ranking_debug = getattr(result, "ranking_debug", None)
        if debug_enabled and isinstance(ranking_debug, Mapping):
            raw_result = raw_results_by_id.get(result.memory_id)
            if raw_result is not None:
                evidence = _search_evidence_for_result(raw_result)
                if evidence:
                    ranking_debug = dict(ranking_debug) | {"evidence": evidence}
        elif debug_enabled and ranking_debug is None:
            raw_result = raw_results_by_id.get(result.memory_id)
            if raw_result is not None:
                evidence = _search_evidence_for_result(raw_result)
                if evidence:
                    ranking_debug = {"evidence": evidence}
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
    summary_only: bool = False,
    content_offset: int = 0,
    content_limit: int | None = None,
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
    record_payload = payload["record"]
    if isinstance(record_payload, dict):
        if summary_only:
            record_payload.pop("content", None)
            record_payload["summary"] = result.record.summary or result.record.content[:220]
        elif content_limit is not None:
            content = result.record.content
            chunk = content[content_offset : content_offset + content_limit]
            record_payload["content"] = chunk
            record_payload["content_offset"] = content_offset
            record_payload["content_total_chars"] = len(content)
            record_payload["content_has_more"] = content_offset + len(chunk) < len(content)
    if include_relationships:
        payload["relationships"] = relationships_payload
    if include_superseded:
        payload["superseded"] = superseded_payload
    return payload

from __future__ import annotations

from collections.abc import Iterable
import json
import re
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.ports.search import INTERNAL_SEARCH_TOOL_NAME

# ---------------------------------------------------------------------------
# relationship_proposal_support: prompt helpers and proposal normalization
# ---------------------------------------------------------------------------

def build_candidate_prompt_entries(candidates: list, *, limit: int = 20) -> str:
    return json.dumps(
        [
            {
                "id": record.id,
                "type": record.type,
                "status": record.status,
                "title": record.title,
                "summary": truncate_text(record.summary or record.content, 180),
                "tags": record.tags,
            }
            for record in candidates[:limit]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )


def normalize_graph_link_proposals(proposals: object) -> list[tuple[str, str, str, str]]:
    if not isinstance(proposals, list):
        return []
    normalized: list[tuple[str, str, str, str]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id", "")).strip()
        target_id = str(item.get("target_id", "")).strip()
        link_type = str(item.get("link_type", "")).strip() or "DEPENDS_ON"
        context = str(item.get("context", "")).strip() or "Auto-linked by graph linker."
        if source_id and target_id:
            normalized.append((source_id, target_id, link_type, context))
    return normalized


def normalize_conflict_proposals(proposals: object) -> list[tuple[str, str, str]]:
    if not isinstance(proposals, list):
        return []
    normalized: list[tuple[str, str, str]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        left_id = str(item.get("left_id", "")).strip()
        right_id = str(item.get("right_id", "")).strip()
        context = str(item.get("context", "")).strip() or "Potential contradiction detected."
        if left_id and right_id:
            normalized.append((left_id, right_id, context))
    return normalized


def truncate_text(value: str | None, limit: int) -> str:
    text = "" if value is None else " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"

# ---------------------------------------------------------------------------
# Proposal generators
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
GRAPH_LINKER_AI_MIN_CANDIDATES = 12
GRAPH_LINKER_FALLBACK_LINK_TARGET = 2
CONFLICT_DETECTOR_AI_MIN_CANDIDATES = 15
FALLBACK_GRAPH_LINK_MIN_TOKEN_OVERLAP = 0.5
FALLBACK_GRAPH_LINK_MIN_BROAD_TAG_TOKEN_OVERLAP = 0.7
FALLBACK_GRAPH_LINK_MAX_RELATIONSHIP_COUNT = 8
_AUTO_LINK_BROAD_TAGS = frozenset(
    {
        "agentic-mcp",
        "architecture",
        "backlog",
        "cleanup",
        "maintenance",
        "meta",
        "mcp-memory",
        "process",
        "refactor",
        "refactoring",
        "testing",
        "tooling",
        "workflow",
    }
)


async def propose_graph_links(
    ctx: ApplicationContext,
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str, str]]:
    fallback = _fallback_graph_links(ctx, candidates)
    if (
        provider is None
        or len(fallback) >= GRAPH_LINKER_FALLBACK_LINK_TARGET
        or len(candidates) <= GRAPH_LINKER_AI_MIN_CANDIDATES
    ):
        return fallback

    response = await run_internal_tool_loop(
        ctx,
        provider=provider,
        prompt=_build_linker_prompt(candidates),
        allowed_tool_names=[
            INTERNAL_SEARCH_TOOL_NAME,
            "internal_read_memory_record",
            "internal_list_memory_records",
        ],
        max_rounds=3,
    )
    normalized = normalize_graph_link_proposals(response.response.get("links", []))
    if normalized:
        return normalized

    return fallback


async def propose_conflicts(
    ctx: ApplicationContext,
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str]]:
    if provider is None or len(candidates) <= CONFLICT_DETECTOR_AI_MIN_CANDIDATES:
        return []

    response = await run_internal_tool_loop(
        ctx,
        provider=provider,
        prompt=_build_conflict_prompt(candidates),
        allowed_tool_names=[
            INTERNAL_SEARCH_TOOL_NAME,
            "internal_read_memory_record",
            "internal_list_memory_records",
        ],
        max_rounds=3,
    )
    normalized = normalize_conflict_proposals(response.response.get("conflicts", []))
    if normalized:
        return normalized

    return []


def _fallback_graph_links(ctx: ApplicationContext, candidates: list) -> list[tuple[str, str, str, str]]:
    proposals: list[tuple[str, str, str, str]] = []
    relationship_counts: dict[str, int] = {}
    for source, target in _iter_candidate_pairs(candidates):
        shared_tags = sorted(set(source.tags) & set(target.tags))
        token_overlap = _token_overlap(source.title, target.title)
        if _relationship_count(ctx, source, relationship_counts) > FALLBACK_GRAPH_LINK_MAX_RELATIONSHIP_COUNT:
            continue
        if _relationship_count(ctx, target, relationship_counts) > FALLBACK_GRAPH_LINK_MAX_RELATIONSHIP_COUNT:
            continue
        if not _should_auto_link_candidate_pair(shared_tags, token_overlap):
            continue
        informative_shared_tags = [tag for tag in shared_tags if not _is_broad_auto_link_tag(tag)]
        newer, older = _sort_newer_first(source, target)
        link_type = "AMENDS" if newer.type == older.type else "DEPENDS_ON"
        context = (
            f"Auto-linked from shared tags ({', '.join(informative_shared_tags)})"
            if informative_shared_tags
            else "Auto-linked from title similarity."
        )
        proposals.append((newer.id, older.id, link_type, context))
    return proposals[:10]


def _should_auto_link_candidate_pair(shared_tags: list[str], token_overlap: float) -> bool:
    if not shared_tags:
        return token_overlap >= FALLBACK_GRAPH_LINK_MIN_TOKEN_OVERLAP
    informative_shared_tags = [tag for tag in shared_tags if not _is_broad_auto_link_tag(tag)]
    if informative_shared_tags:
        return True
    return token_overlap >= FALLBACK_GRAPH_LINK_MIN_BROAD_TAG_TOKEN_OVERLAP


def _is_broad_auto_link_tag(tag: str) -> bool:
    normalized = tag.strip().lower()
    if not normalized:
        return True
    return normalized in _AUTO_LINK_BROAD_TAGS


def _relationship_count(
    ctx: ApplicationContext,
    record: Any,
    cache: dict[str, int],
) -> int:
    cached_count = cache.get(record.id)
    if cached_count is not None:
        return cached_count
    repository = getattr(ctx, "repository", None)
    if repository is None:
        cache[record.id] = 0
        return 0
    outgoing = repository.get_links(record.id, direction="outgoing")
    incoming = repository.get_links(record.id, direction="incoming")
    count = len(outgoing) + len(incoming)
    cache[record.id] = count
    return count


def _fallback_conflicts(candidates: list) -> list[tuple[str, str, str]]:
    proposals: list[tuple[str, str, str]] = []
    for left, right in _iter_candidate_pairs(candidates):
        if left.type != right.type:
            continue
        if left.content.strip() == right.content.strip():
            continue
        shared_tags = set(left.tags) & set(right.tags)
        title_overlap = _token_overlap(left.title, right.title)
        if title_overlap < 0.5 and not shared_tags:
            continue
        proposals.append((left.id, right.id, "Potential contradiction detected from overlapping titles/tags."))
    return proposals[:10]


def _build_linker_prompt(candidates: list) -> str:
    entries = build_candidate_prompt_entries(candidates)
    return (
        "Review these active memories and propose only high-confidence typed links.\n"
        "Use DEPENDS_ON when one memory relies on, implements, or is downstream of another.\n"
        "Use AMENDS when a newer memory updates, refines, or corrects an older memory on the same topic.\n"
        "Do not propose weak title-only links, duplicate existing relationships, or symmetric duplicates.\n"
        "If the local candidate list is insufficient, use the internal tools to search and read for better context before proposing links.\n"
        'Return JSON: {"links": [{"source_id": "...", "target_id": "...", "link_type": "DEPENDS_ON|AMENDS", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _build_conflict_prompt(candidates: list) -> str:
    entries = build_candidate_prompt_entries(candidates)
    return (
        "Review these active memories and propose contradictions only when two records make materially incompatible claims.\n"
        "Do not flag mere topic overlap, phrasing differences, or newer refinements of older memories as contradictions.\n"
        "Prefer concrete evidence in the titles, summaries, and any extra context you retrieve with the internal tools.\n"
        "If you need broader context, use the internal tools to search and read before proposing a contradiction.\n"
        'Return JSON: {"conflicts": [{"left_id": "...", "right_id": "...", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _iter_candidate_pairs(candidates: Iterable) -> Iterable[tuple[Any, Any]]:
    candidate_list = list(candidates)
    for index, left in enumerate(candidate_list):
        for right in candidate_list[index + 1 :]:
            yield left, right


def _sort_newer_first(left, right):
    return (left, right) if left.updated_at >= right.updated_at else (right, left)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in TOKEN_PATTERN.findall(left)}
    right_tokens = {token.lower() for token in TOKEN_PATTERN.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _truncate_text(value: str | None, limit: int) -> str:
    return truncate_text(value, limit)

from __future__ import annotations

from inspect import isawaitable
import re
from typing import Any

from mcp_memory.core.sampling import COLD_STORAGE_STRATEGY, ORPHAN_LOW_SUPPORT_STRATEGY, SEMANTIC_STRATEGY
from mcp_memory.core.task_handlers.agentic_guardrails import build_reflection_synthesis_guardrails

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
DEFRAGMENTER_GROUP_SIMILARITY_THRESHOLD = 0.45
DEFRAGMENTER_AI_MIN_GROUP_SIZE = 3
DEFRAGMENTER_AI_MIN_SOURCE_LINES = 200
LOW_SIGNAL_GROUPING_TAGS = {
    "anomaly-sample",
    "architecture",
    "audit",
    "auto-defragmented",
    "auto-ingested",
    "background-agent",
    "deduplicator",
    "maintenance",
    "memory-curator",
    "system1",
    "system1-appended",
    "system-design",
    "task-complete",
}
LOW_SIGNAL_TOPIC_TOKENS = {
    "agent",
    "agents",
    "architecture",
    "background",
    "cli",
    "complete",
    "completed",
    "daemon",
    "design",
    "implementation",
    "implementations",
    "infra",
    "infrastructure",
    "lifecycle",
    "maintenance",
    "migration",
    "migrations",
    "observability",
    "overview",
    "project",
    "projects",
    "repo",
    "repository",
    "roadmap",
    "runtime",
    "service",
    "services",
    "system",
    "systems",
    "task",
    "tasks",
    "testing",
    "tool",
    "tools",
    "transport",
}
DEFRAGMENTER_ALLOWED_STRATEGIES = (
    COLD_STORAGE_STRATEGY,
    SEMANTIC_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
)


def collect_defragment_groups(candidates: list) -> list[list]:
    groups: list[list] = []
    used: set[str] = set()
    for record in candidates:
        if record.id in used:
            continue
        related = [record]
        used.add(record.id)
        for other in candidates:
            if other.id in used or other.id == record.id:
                continue
            shared_tags = shared_meaningful_tags(record.tags, other.tags)
            similarity = topic_token_overlap(record.title + " " + record.content, other.title + " " + other.content)
            if shared_tags or similarity >= DEFRAGMENTER_GROUP_SIMILARITY_THRESHOLD:
                related.append(other)
                used.add(other.id)
        if len(related) >= 2:
            groups.append(related[:5])
    return groups[:3]


async def build_defragmented_memory(group: list, provider: Any = None) -> tuple[str, str]:
    if provider is not None:
        prompt = (
            "Summarize these memories into one reflection. Return JSON with title and content.\n"
            f"{build_reflection_synthesis_guardrails()}\n\n"
            + "\n".join(f"- {item.title}: {item.content}" for item in group)
        )
        response = provider.ask(prompt)
        if isawaitable(response):
            response = await response
        title = str(response.get("title", "")).strip()
        content = str(response.get("content", "")).strip()
        if title and content:
            return title, content

    title = f"Reflection: {group[0].title}"
    content_lines = ["Consolidated observations:"]
    for item in group:
        snippet = item.summary or item.content.strip().splitlines()[0]
        content_lines.append(f"- {item.title}: {snippet}")
    return title, "\n".join(content_lines)


def should_use_provider_for_defragment_group(group: list, source_lines: int) -> bool:
    return len(group) >= DEFRAGMENTER_AI_MIN_GROUP_SIZE and source_lines >= DEFRAGMENTER_AI_MIN_SOURCE_LINES


def resolve_group_workspace_ids(group: list, fallback_workspace_id: str | None) -> list[str]:
    workspace_ids = sorted(
        {
            workspace_id.strip()
            for item in group
            for workspace_id in getattr(item, "workspace_ids", [])
            if isinstance(workspace_id, str) and workspace_id.strip()
        }
    )
    if workspace_ids:
        return workspace_ids
    if isinstance(fallback_workspace_id, str) and fallback_workspace_id.strip():
        return [fallback_workspace_id.strip()]
    return ["workspace-unknown"]


def topic_token_overlap(left: str, right: str) -> float:
    left_tokens = _topic_tokens(left)
    right_tokens = _topic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def shared_meaningful_tags(left_tags: list[str], right_tags: list[str]) -> set[str]:
    return _meaningful_tag_set(left_tags) & _meaningful_tag_set(right_tags)


def count_text_lines(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return text.count("\n") + 1


def _topic_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value)
        if token.lower() not in LOW_SIGNAL_TOPIC_TOKENS
    }


def _meaningful_tag_set(tags: list[str]) -> set[str]:
    meaningful: set[str] = set()
    for tag in tags:
        normalized = tag.strip().lower()
        if (
            not normalized
            or normalized in LOW_SIGNAL_GROUPING_TAGS
            or normalized.endswith("-task")
            or "-task-" in normalized
        ):
            continue
        meaningful.add(normalized)
    return meaningful

from __future__ import annotations

import re

from mcp_memory.context import ApplicationContext
from mcp_memory.embeddings import cosine_similarity
from mcp_memory.core.tasks import TaskRecord

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
FACT_DEDUPLICATION_THRESHOLD = 0.72
OBSERVATION_ABSORPTION_THRESHOLD = 0.62
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


async def run_deterministic_deduplicator_pass(
    ctx: ApplicationContext,
    task: TaskRecord,
    candidates: list,
    seed_records: list,
) -> dict[str, int]:
    facts = [record for record in seed_records if record.type == "fact"]
    observations = [record for record in seed_records if record.type == "observation"]
    if not facts:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}

    active_facts = [record for record in candidates if record.type == "fact"]
    embedding_by_id = _embed_records(ctx, [*active_facts, *observations])
    merged = 0
    archived = 0
    absorbed_observations = 0
    claimed_sources: set[str] = set()

    for group in _collect_similar_fact_groups(facts, embedding_by_id):
        canonical = _choose_canonical_fact(ctx, group)
        for source in group:
            if source.id == canonical.id or source.id in claimed_sources:
                continue
            canonical = await _merge_into_canonical_fact(ctx, canonical, source, task)
            claimed_sources.add(source.id)
            merged += 1
            archived += 1

    active_facts_by_id = {record.id: record for record in active_facts}
    for observation in observations:
        if observation.id in claimed_sources:
            continue
        target = _find_best_fact_target(observation, active_facts, embedding_by_id)
        if target is None:
            continue
        refreshed = active_facts_by_id.get(target.id, target)
        merged_target = await _merge_into_canonical_fact(ctx, refreshed, observation, task)
        active_facts_by_id[merged_target.id] = merged_target
        active_facts = list(active_facts_by_id.values())
        claimed_sources.add(observation.id)
        absorbed_observations += 1
        archived += 1

    return {
        "merged": merged,
        "archived": archived,
        "absorbed_observations": absorbed_observations,
    }


def _embed_records(ctx: ApplicationContext, records: list) -> dict[str, list[float]]:
    embedder = getattr(ctx, "embedder", None)
    if embedder is None:
        return {}
    payloads = [_record_embedding_text(record) for record in records]
    embeddings = embedder.embed(payloads)
    return {
        record.id: embedding
        for record, embedding in zip(records, embeddings, strict=False)
    }


def _collect_similar_fact_groups(facts: list, embedding_by_id: dict[str, list[float]]) -> list[list]:
    groups: list[list] = []
    used: set[str] = set()
    for fact in facts:
        if fact.id in used:
            continue
        group = [fact]
        used.add(fact.id)
        for other in facts:
            if other.id in used or other.id == fact.id:
                continue
            if _memory_similarity(fact, other, embedding_by_id) >= FACT_DEDUPLICATION_THRESHOLD:
                group.append(other)
                used.add(other.id)
        if len(group) >= 2:
            groups.append(group)
    return groups


def _choose_canonical_fact(ctx: ApplicationContext, facts: list):
    assert ctx.repository is not None
    repository = ctx.repository
    return max(
        facts,
        key=lambda record: (
            repository.count_incoming_links(record.id),
            record.access_score,
            record.updated_at,
            record.created_at,
        ),
    )


def _find_best_fact_target(observation, facts: list, embedding_by_id: dict[str, list[float]]):
    best_score = 0.0
    best_target = None
    for fact in facts:
        score = _memory_similarity(observation, fact, embedding_by_id)
        if score >= OBSERVATION_ABSORPTION_THRESHOLD and score > best_score:
            best_score = score
            best_target = fact
    return best_target


async def _merge_into_canonical_fact(
    ctx: ApplicationContext,
    canonical,
    source,
    task: TaskRecord,
):
    assert ctx.repository is not None
    merged_title, merged_content = await _build_merged_fact_content(canonical, source)
    merged_tags = _normalize_tag_values([*canonical.tags, *source.tags])
    merged_metadata = dict(canonical.metadata)
    merged_source_ids = merged_metadata.get("merged_source_ids", [])
    if not isinstance(merged_source_ids, list):
        merged_source_ids = []
    merged_metadata["merged_source_ids"] = sorted({*map(str, merged_source_ids), source.id})
    merged_metadata["deduplicator_task_id"] = task.id
    updated = ctx.repository.update_memory(
        canonical.id,
        title=merged_title,
        content=merged_content,
        tags=merged_tags,
        metadata=merged_metadata,
        workspace_ids=sorted({*canonical.workspace_ids, *source.workspace_ids}),
    )
    if updated is None:
        return canonical
    ctx.repository.add_link(updated.id, source.id, "SUPERSEDES", "Auto-merged into canonical fact memory.")
    ctx.repository.update_memory(source.id, status="archived")
    return updated


async def _build_merged_fact_content(canonical, source) -> tuple[str, str]:
    if source.content.strip() in canonical.content:
        return canonical.title, canonical.content

    merged_lines = [canonical.content.strip()]
    addition = source.summary or source.content.strip()
    if addition and addition not in canonical.content:
        merged_lines.append(f"Merged from {source.title}:\n{addition}")
    return canonical.title, "\n\n".join(part for part in merged_lines if part)


def _memory_similarity(left, right, embedding_by_id: dict[str, list[float]]) -> float:
    shared_tags = _shared_meaningful_tags(left.tags, right.tags)
    lexical_similarity = _topic_token_overlap(left.title + " " + left.content, right.title + " " + right.content)
    semantic_similarity = 0.0
    if left.id in embedding_by_id and right.id in embedding_by_id:
        semantic_similarity = cosine_similarity(embedding_by_id[left.id], embedding_by_id[right.id])
    if not shared_tags and lexical_similarity < 0.12:
        semantic_similarity = 0.0
    tag_bonus = 0.15 if shared_tags else 0.0
    return max(lexical_similarity, semantic_similarity + tag_bonus)


def _record_embedding_text(record) -> str:
    parts = [record.title, record.summary or "", record.content]
    if record.tags:
        parts.append("tags: " + ", ".join(record.tags))
    parts.append(f"type: {record.type}")
    return "\n".join(part for part in parts if part)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in TOKEN_PATTERN.findall(left)}
    right_tokens = {token.lower() for token in TOKEN_PATTERN.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _topic_token_overlap(left: str, right: str) -> float:
    left_tokens = _topic_tokens(left)
    right_tokens = _topic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _topic_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value)
        if token.lower() not in LOW_SIGNAL_TOPIC_TOKENS
    }


def _shared_meaningful_tags(left_tags: list[str], right_tags: list[str]) -> set[str]:
    return _meaningful_tag_set(left_tags) & _meaningful_tag_set(right_tags)


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


def _normalize_tag_values(tags: list[str]) -> list[str]:
    aliases = {
        "unit-test": "testing",
        "unit_tests": "testing",
        "tests": "testing",
        "test": "testing",
        "authn": "auth",
        "authz": "auth",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized_tag = tag.strip().lower().replace("_", "-")
        normalized_tag = aliases.get(normalized_tag, normalized_tag)
        if not normalized_tag or normalized_tag in seen:
            continue
        seen.add(normalized_tag)
        normalized.append(normalized_tag)
    return sorted(normalized)

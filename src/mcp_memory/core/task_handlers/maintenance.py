from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from pathlib import Path
import re
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.embeddings import cosine_similarity
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_AGENT_SCAN_LIMIT,
    DEFAULT_STALE_PLAN_DAYS,
    DEFAULT_SWEEP_RETENTION_DAYS,
)
from mcp_memory.core.tasks import TaskRecord


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
FACT_DEDUPLICATION_THRESHOLD = 0.72
OBSERVATION_ABSORPTION_THRESHOLD = 0.62


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}

    cutoff = (datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)).isoformat()
    workspace_id = _resolve_workspace_id(ctx, task)
    conn = ctx.db_manager.get_connection()
    if workspace_id is None:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE type = 'plan' AND status = 'active' AND updated_at < ?
            """,
            (cutoff,),
        )
    else:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE id IN (
                SELECT memories.id
                FROM memories
                JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                WHERE memories.type = 'plan'
                  AND memories.status = 'active'
                  AND memories.updated_at < ?
                  AND memory_workspaces.workspace_id = ?
            )
            """,
            (cutoff, workspace_id),
        )
    conn.commit()
    return {"updated": cursor.rowcount}


def handle_fact_checker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"degraded": 0, "restored": 0}

    conn = ctx.db_manager.get_connection()
    rows = conn.execute(
        """
        SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
        FROM memories
        JOIN links ON links.source_id = memories.id
        LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
        WHERE links.target_id LIKE 'ext:%'
        """
    ).fetchall()

    degraded_ids: set[str] = set()
    healthy_ids: set[str] = set()
    workspace_root_override = task.data.get("workspace_root")
    for row in rows:
        workspace_root = _resolve_workspace_root(
            ctx,
            row["workspace_id"],
            workspace_root_override,
        )
        target_path = row["target_id"][4:]
        file_path = Path(target_path)
        if not file_path.is_absolute() and workspace_root is not None:
            file_path = workspace_root / target_path

        if file_path.exists():
            healthy_ids.add(row["id"])
        else:
            degraded_ids.add(row["id"])

    restored_ids = healthy_ids - degraded_ids
    if degraded_ids:
        placeholders = ",".join("?" for _ in degraded_ids)
        conn.execute(
            f"UPDATE memories SET status = 'degraded' WHERE id IN ({placeholders})",
            list(degraded_ids),
        )
    if restored_ids:
        placeholders = ",".join("?" for _ in restored_ids)
        conn.execute(
            f"UPDATE memories SET status = 'active' WHERE id IN ({placeholders}) AND status = 'degraded'",
            list(restored_ids),
        )
    conn.commit()
    return {"degraded": len(degraded_ids), "restored": len(restored_ids)}


def handle_sweeper_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"deleted_tasks": 0, "deleted_journal_entries": 0}

    cutoff = datetime.now(UTC) - timedelta(days=DEFAULT_SWEEP_RETENTION_DAYS)
    cutoff_timestamp = cutoff.timestamp()
    conn = ctx.db_manager.get_connection()
    deleted_tasks = conn.execute(
        "DELETE FROM tasks WHERE status = 'completed' AND updated_at < ?",
        (cutoff_timestamp,),
    ).rowcount
    deleted_journal_entries = conn.execute(
        "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < ?",
        (cutoff_timestamp,),
    ).rowcount
    conn.commit()
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_journal_entries": deleted_journal_entries,
    }


async def handle_graph_linker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

    candidates = ctx.repository.list_memories(
        workspace_id=_resolve_workspace_id(ctx, task),
        status="active",
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    if len(candidates) < 2:
        return {"created": 0}

    proposed_pairs = await _propose_graph_links(candidates, provider)
    created = 0
    for source_id, target_id, link_type, context in proposed_pairs:
        if source_id == target_id or _has_link(ctx, source_id, target_id, link_type):
            continue
        ctx.repository.add_link(source_id, target_id, link_type, context)
        created += 1

    return {"created": created}


async def handle_conflict_detector_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.type in {"fact", "plan"}
    ]
    if len(candidates) < 2:
        return {"created": 0}

    proposed_pairs = await _propose_conflicts(candidates, provider)
    created = 0
    for left_id, right_id, context in proposed_pairs:
        if left_id == right_id:
            continue
        if not _has_link(ctx, left_id, right_id, "CONTRADICTS"):
            ctx.repository.add_link(left_id, right_id, "CONTRADICTS", context)
            created += 1
        if not _has_link(ctx, right_id, left_id, "CONTRADICTS"):
            ctx.repository.add_link(right_id, left_id, "CONTRADICTS", context)
            created += 1

    return {"created": created}


async def handle_defragmenter_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "archived": 0}

    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.type in {"journal", "observation"}
        and not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    groups = _collect_defragment_groups(candidates)
    if not groups:
        return {"created": 0, "archived": 0}

    created = 0
    archived = 0
    lines_compressed = 0
    workspace_id = _resolve_workspace_id(ctx, task) or "workspace-unknown"
    for group in groups:
        source_lines = sum(_count_text_lines(item.content) for item in group)
        title, content = await _build_defragmented_memory(group, provider)
        reflection_lines = _count_text_lines(content)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            workspace_ids=[workspace_id],
            tags=_normalize_tag_values([tag for item in group for tag in item.tags] + ["auto-defragmented"]),
            memory_type="reflection",
            metadata={"source_memory_ids": [item.id for item in group], "defragmenter_task_id": task.id},
        )
        assert record is not None
        created += 1
        lines_compressed += max(source_lines - reflection_lines, 0)
        for item in group:
            ctx.repository.add_link(record.id, item.id, "SUPERSEDES", "Auto-defragmented into a reflection.")
            updated = ctx.repository.update_memory(item.id, status="archived")
            if updated is not None:
                archived += 1

    return {"created": created, "archived": archived, "lines_compressed": lines_compressed}


async def handle_deduplicator_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    facts = [record for record in candidates if record.type == "fact"]
    observations = [record for record in candidates if record.type == "observation"]
    if not facts:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}

    embedding_by_id = _embed_records(ctx, [*facts, *observations])
    merged = 0
    archived = 0
    absorbed_observations = 0
    claimed_sources: set[str] = set()

    for group in _collect_similar_fact_groups(facts, embedding_by_id):
        canonical = _choose_canonical_fact(ctx, group)
        for source in group:
            if source.id == canonical.id or source.id in claimed_sources:
                continue
            canonical = await _merge_into_canonical_fact(ctx, canonical, source, task, provider)
            claimed_sources.add(source.id)
            merged += 1
            archived += 1

    active_facts = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            memory_type="fact",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
    ]
    active_facts_by_id = {record.id: record for record in active_facts}
    for observation in observations:
        if observation.id in claimed_sources:
            continue
        target = _find_best_fact_target(observation, active_facts, embedding_by_id)
        if target is None:
            continue
        refreshed = active_facts_by_id.get(target.id, target)
        merged_target = await _merge_into_canonical_fact(ctx, refreshed, observation, task, provider)
        active_facts_by_id[merged_target.id] = merged_target
        active_facts = list(active_facts_by_id.values())
        claimed_sources.add(observation.id)
        absorbed_observations += 1
        archived += 1

    return {"merged": merged, "archived": archived, "absorbed_observations": absorbed_observations}


async def handle_taxonomist_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"updated": 0}

    candidates = ctx.repository.list_memories(
        workspace_id=_resolve_workspace_id(ctx, task),
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    updated = 0
    for record in candidates:
        normalized_tags = _normalize_tag_values(record.tags)
        if provider is not None:
            normalized_tags = await _provider_normalize_tags(provider, record, normalized_tags)
        if normalized_tags == record.tags:
            continue
        refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
        if refreshed is not None:
            updated += 1
    return {"updated": updated}


async def _propose_graph_links(
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str, str]]:
    if provider is not None:
        prompt = _build_linker_prompt(candidates)
        response = provider.ask(prompt)
        if isawaitable(response):
            response = await response
        proposals = response.get("links", [])
        if isinstance(proposals, list):
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
            if normalized:
                return normalized

    return _fallback_graph_links(candidates)


async def _propose_conflicts(
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str]]:
    if provider is not None:
        prompt = _build_conflict_prompt(candidates)
        response = provider.ask(prompt)
        if isawaitable(response):
            response = await response
        proposals = response.get("conflicts", [])
        if isinstance(proposals, list):
            normalized: list[tuple[str, str, str]] = []
            for item in proposals:
                if not isinstance(item, dict):
                    continue
                left_id = str(item.get("left_id", "")).strip()
                right_id = str(item.get("right_id", "")).strip()
                context = str(item.get("context", "")).strip() or "Potential contradiction detected."
                if left_id and right_id:
                    normalized.append((left_id, right_id, context))
            if normalized:
                return normalized

    return _fallback_conflicts(candidates)


def _fallback_graph_links(candidates: list) -> list[tuple[str, str, str, str]]:
    proposals: list[tuple[str, str, str, str]] = []
    for source, target in _iter_candidate_pairs(candidates):
        shared_tags = sorted(set(source.tags) & set(target.tags))
        token_overlap = _token_overlap(source.title, target.title)
        if not shared_tags and token_overlap < 0.34:
            continue
        newer, older = _sort_newer_first(source, target)
        link_type = "AMENDS" if newer.type == older.type else "DEPENDS_ON"
        context = (
            f"Auto-linked from shared tags ({', '.join(shared_tags)})"
            if shared_tags
            else "Auto-linked from title similarity."
        )
        proposals.append((newer.id, older.id, link_type, context))
    return proposals[:10]


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


def _collect_defragment_groups(candidates: list) -> list[list]:
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
            shared_tags = set(record.tags) & set(other.tags)
            similarity = _token_overlap(record.title + " " + record.content, other.title + " " + other.content)
            if shared_tags or similarity >= 0.3:
                related.append(other)
                used.add(other.id)
        if len(related) >= 2:
            groups.append(related[:5])
    return groups[:3]


async def _build_defragmented_memory(group: list, provider: Any = None) -> tuple[str, str]:
    if provider is not None:
        prompt = (
            "Summarize these memories into one reflection. Return JSON with title and content.\n\n"
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


async def _provider_normalize_tags(provider: Any, record, normalized_tags: list[str]) -> list[str]:
    prompt = (
        "Normalize these tags to a concise canonical set. Return JSON with {\"tags\": [...]} only.\n"
        f"Title: {record.title}\nTags: {normalized_tags}"
    )
    response = provider.ask(prompt)
    if isawaitable(response):
        response = await response
    provider_tags = response.get("tags", [])
    if not isinstance(provider_tags, list):
        return normalized_tags
    return _normalize_tag_values([str(tag) for tag in provider_tags]) or normalized_tags


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
    provider: Any = None,
):
    assert ctx.repository is not None
    merged_title, merged_content = await _build_merged_fact_content(canonical, source, provider)
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


async def _build_merged_fact_content(canonical, source, provider: Any = None) -> tuple[str, str]:
    if provider is not None:
        prompt = (
            "Merge these two memories into one canonical fact. Return JSON with title and content.\n"
            'Return only: {"title": "...", "content": "..."}\n\n'
            f"Canonical title: {canonical.title}\nCanonical content:\n{canonical.content}\n\n"
            f"Incoming title: {source.title}\nIncoming content:\n{source.content}"
        )
        response = provider.ask(prompt)
        if isawaitable(response):
            response = await response
        title = str(response.get("title", "")).strip()
        content = str(response.get("content", "")).strip()
        if title and content:
            return title, content

    if source.content.strip() in canonical.content:
        return canonical.title, canonical.content

    merged_lines = [canonical.content.strip()]
    addition = source.summary or source.content.strip()
    if addition and addition not in canonical.content:
        merged_lines.append(f"Merged from {source.title}:\n{addition}")
    return canonical.title, "\n\n".join(part for part in merged_lines if part)


def _memory_similarity(left, right, embedding_by_id: dict[str, list[float]]) -> float:
    shared_tags = set(left.tags) & set(right.tags)
    lexical_similarity = _token_overlap(left.title + " " + left.content, right.title + " " + right.content)
    semantic_similarity = 0.0
    if left.id in embedding_by_id and right.id in embedding_by_id:
        semantic_similarity = cosine_similarity(embedding_by_id[left.id], embedding_by_id[right.id])
    tag_bonus = 0.15 if shared_tags else 0.0
    return max(lexical_similarity, semantic_similarity + tag_bonus)


def _record_embedding_text(record) -> str:
    parts = [record.title, record.summary or "", record.content]
    if record.tags:
        parts.append("tags: " + ", ".join(record.tags))
    parts.append(f"type: {record.type}")
    return "\n".join(part for part in parts if part)


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


def _count_text_lines(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return text.count("\n") + 1


def _has_link(ctx: ApplicationContext, source_id: str, target_id: str, link_type: str) -> bool:
    assert ctx.repository is not None
    return any(
        link.target_id == target_id and link.link_type == link_type
        for link in ctx.repository.get_links(source_id, direction="outgoing")
    )


def _build_linker_prompt(candidates: list) -> str:
    entries = "\n".join(
        f"- id={record.id} type={record.type} title={record.title!r} tags={record.tags}"
        for record in candidates[:20]
    )
    return (
        "Review these memories and propose useful typed links.\n"
        'Return JSON: {"links": [{"source_id": "...", "target_id": "...", "link_type": "DEPENDS_ON|AMENDS", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _build_conflict_prompt(candidates: list) -> str:
    entries = "\n".join(
        f"- id={record.id} type={record.type} title={record.title!r} tags={record.tags} content={record.content[:120]!r}"
        for record in candidates[:20]
    )
    return (
        "Review these memories and propose contradictions when two active records appear to disagree.\n"
        'Return JSON: {"conflicts": [{"left_id": "...", "right_id": "...", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    return ctx.workspace_id


def _resolve_workspace_root(
    ctx: ApplicationContext,
    workspace_id: str | None,
    workspace_root_override: object | None = None,
) -> Path | None:
    if isinstance(workspace_root_override, str) and workspace_root_override.strip():
        override_path = Path(workspace_root_override).expanduser()
        if override_path.exists():
            return override_path
    if workspace_id is None:
        return None
    if ctx.workspace_root is not None and ctx.workspace_id == workspace_id:
        return ctx.workspace_root
    path = Path(workspace_id).expanduser()
    if path.exists():
        return path
    return None
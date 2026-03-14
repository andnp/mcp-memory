from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.core.storage import list_memory_files


FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass(slots=True)
class NormalizedMarkdownMemory:
    title: str
    content: str
    summary: str | None
    memory_type: str
    status: str
    tags: list[str]
    created_at: str
    metadata: dict[str, object]


def parse_markdown_memory(file_path: Path):
    raw_content = file_path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw_content)
    body = raw_content
    frontmatter: dict[str, object] = {}

    if match:
        frontmatter = yaml.safe_load(match.group(1)) or {}
        body = raw_content[match.end() :]

    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
    elif not isinstance(tags, list):
        tags = []

    created_at = _parse_created_at(frontmatter.get("created_at"))
    title = str(frontmatter.get("title") or _derive_title(file_path))
    memory_type = str(frontmatter.get("type") or "journal")
    status = str(frontmatter.get("status") or "active")

    return NormalizedMarkdownMemory(
        title=title,
        content=body,
        summary=None,
        memory_type=memory_type,
        status=status,
        tags=[str(tag).strip() for tag in tags if str(tag).strip()],
        created_at=created_at,
        metadata={
            "legacy_memory_name": file_path.stem,
            "legacy_source_path": str(file_path),
        },
    )


def import_markdown_memory(
    repository: RelationalMemoryRepository,
    file_path: Path,
    workspace_ids: list[str] | None = None,
):
    normalized = parse_markdown_memory(file_path)
    return repository.create_memory(
        title=normalized.title,
        content=normalized.content,
        summary=normalized.summary,
        memory_type=normalized.memory_type,
        status=normalized.status,
        tags=normalized.tags,
        workspace_ids=workspace_ids or [],
        created_at=normalized.created_at,
        updated_at=normalized.created_at,
        metadata=normalized.metadata,
    )


def import_markdown_memories(
    repository: RelationalMemoryRepository,
    memory_path: Path,
    workspace_ids: list[str] | None = None,
):
    imported = []
    for file_path in list_memory_files(memory_path):
        imported.append(import_markdown_memory(repository, file_path, workspace_ids))
    return imported


def _derive_title(file_path: Path):
    return file_path.stem.replace("-", " ").replace("_", " ").strip().title()


def _parse_created_at(value: object):
    if isinstance(value, datetime):
        return _ensure_timezone(value).isoformat()
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.now(timezone.utc).isoformat()
        return _ensure_timezone(parsed).isoformat()
    return datetime.now(timezone.utc).isoformat()


def _ensure_timezone(value: datetime):
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value

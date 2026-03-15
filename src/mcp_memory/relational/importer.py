from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mcp_memory.core.journal import System1Journal
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
            "imported_source_name": file_path.stem,
            "imported_source_path": str(file_path),
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


def resolve_markdown_import_paths(paths_or_globs: list[str | Path]) -> list[Path]:
    resolved_paths: list[Path] = []
    seen: set[Path] = set()

    for raw_path in paths_or_globs:
        pattern = str(raw_path)
        direct_path = Path(pattern).expanduser()
        matches: list[Path] = []

        if direct_path.exists() and direct_path.is_file():
            matches = [direct_path.resolve()]
        else:
            matches = sorted(
                (Path(match).expanduser().resolve() for match in glob.glob(pattern, recursive=True)),
                key=lambda path: str(path),
            )
            matches = [match for match in matches if match.is_file()]

        if not matches:
            raise FileNotFoundError(f"No files matched import path: {pattern}")

        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            resolved_paths.append(match)

    return resolved_paths


def import_markdown_memory_paths(
    repository: RelationalMemoryRepository,
    paths_or_globs: list[str | Path],
    workspace_ids: list[str] | None = None,
):
    imported = []
    for file_path in resolve_markdown_import_paths(paths_or_globs):
        imported.append(import_markdown_memory(repository, file_path, workspace_ids))
    return imported


def record_markdown_memory_as_thought(
    journal: System1Journal,
    file_path: Path,
    workspace_id: str | None = None,
) -> tuple[int, str]:
    """Record a markdown file as a thought in the journal buffer.
    
    Returns a tuple of (entry_id, file_name).
    """
    normalized = parse_markdown_memory(file_path)
    entry = journal.record(content=normalized.content, workspace_id=workspace_id)
    return entry.id, file_path.stem


def record_markdown_memory_paths_as_thoughts(
    journal: System1Journal,
    paths_or_globs: list[str | Path],
    workspace_id: str | None = None,
) -> list[tuple[int, str]]:
    """Record multiple markdown files as thoughts in the journal buffer.
    
    Returns a list of tuples (entry_id, file_name).
    """
    recorded = []
    for file_path in resolve_markdown_import_paths(paths_or_globs):
        recorded.append(record_markdown_memory_as_thought(journal, file_path, workspace_id))
    return recorded


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

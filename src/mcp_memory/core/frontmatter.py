from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

from mcp_memory.core.link_parser import extract_links
from mcp_memory.core.models import MemoryDocument, MemoryFrontmatter
from mcp_memory.core.storage import compute_memory_id


logger = logging.getLogger(__name__)

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(content: str) -> tuple[MemoryFrontmatter, str]:
    match = FRONTMATTER_PATTERN.match(content)
    if not match:
        return MemoryFrontmatter(), content

    try:
        yaml_content = match.group(1)
        data = yaml.safe_load(yaml_content) or {}

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None
        elif not isinstance(created_at, datetime):
            created_at = None

        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        frontmatter = MemoryFrontmatter(
            type=data.get("type", "journal"),
            status=data.get("status", "active"),
            tags=tags,
            created_at=created_at,
        )
        return frontmatter, content[match.end() :]
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse frontmatter: %s", exc)
        return MemoryFrontmatter(), content


def parse_memory_file(memory_path: Path, file_path: Path) -> MemoryDocument:
    content = file_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)
    stat = file_path.stat()
    return MemoryDocument(
        id=compute_memory_id(memory_path, file_path),
        content=body,
        frontmatter=frontmatter,
        links=extract_links(body),
        file_path=str(file_path),
        modified_time=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
    )
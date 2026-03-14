from datetime import datetime, timezone

import pytest

from mcp_memory.core.models import MemoryDocument, MemoryFrontmatter


pytestmark = pytest.mark.small


def test_memory_frontmatter_normalizes_unknown_values() -> None:
    frontmatter = MemoryFrontmatter(type="mystery", status="stale")

    assert frontmatter.type == "journal"
    assert frontmatter.status == "active"


def test_memory_document_preserves_frontmatter_and_content() -> None:
    now = datetime.now(timezone.utc)
    frontmatter = MemoryFrontmatter(type="fact", tags=["testing"], created_at=now)

    document = MemoryDocument(
        id="memory:test-doc",
        content="use real sqlite in tests",
        frontmatter=frontmatter,
        links=[],
        file_path="/tmp/test-doc.md",
        modified_time=now,
    )

    assert document.frontmatter.type == "fact"
    assert document.frontmatter.tags == ["testing"]
    assert document.content == "use real sqlite in tests"
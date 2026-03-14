from pathlib import Path

import pytest

from mcp_memory.core.frontmatter import parse_frontmatter, parse_memory_file


pytestmark = pytest.mark.small


def test_parse_frontmatter_extracts_metadata_and_body() -> None:
    content = (
        '---\n'
        'type: fact\n'
        'status: active\n'
        'tags: [alpha, beta]\n'
        'created_at: "2026-03-14T12:00:00+00:00"\n'
        '---\n\n'
        'Body content here.'
    )

    frontmatter, body = parse_frontmatter(content)

    assert frontmatter.type == "fact"
    assert frontmatter.status == "active"
    assert frontmatter.tags == ["alpha", "beta"]
    assert body == "Body content here."


def test_parse_memory_file_builds_memory_document(tmp_path: Path) -> None:
    memory_root = tmp_path / ".memories"
    file_path = memory_root / "example.md"
    file_path.parent.mkdir(parents=True)
    file_path.write_text('---\ntype: journal\n---\n\nSee [[target]].', encoding="utf-8")

    document = parse_memory_file(memory_root, file_path)

    assert document.id == "memory:example"
    assert document.frontmatter.type == "journal"
    assert document.links[0].target == "target"
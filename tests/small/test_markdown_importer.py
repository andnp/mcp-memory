from pathlib import Path

import pytest

from mcp_memory.relational.importer import (
    import_markdown_memory,
    import_markdown_memory_paths,
    parse_markdown_memory,
    resolve_markdown_import_paths,
)
from mcp_memory.relational.repository import RelationalMemoryRepository

pytestmark = pytest.mark.small


def test_parse_markdown_memory_normalizes_frontmatter(tmp_path: Path) -> None:
    file_path = tmp_path / "epic-01.md"
    file_path.write_text(
        "---\n"
        "type: plan\n"
        "status: active\n"
        "tags: [sqlite, planning]\n"
        "created_at: '2026-03-01T10:30:00+00:00'\n"
        "---\n"
        "\n"
        "Ship the relational bootstrap.\n",
        encoding="utf-8",
    )

    parsed = parse_markdown_memory(file_path)

    assert parsed.title == "Epic 01"
    assert parsed.memory_type == "plan"
    assert parsed.status == "active"
    assert parsed.tags == ["sqlite", "planning"]
    assert parsed.created_at == "2026-03-01T10:30:00+00:00"
    assert parsed.content.strip() == "Ship the relational bootstrap."


def test_import_markdown_memory_creates_relational_record(db_manager, tmp_path: Path) -> None:
    file_path = tmp_path / "search-ranking.md"
    file_path.write_text(
        "---\n"
        "type: fact\n"
        "status: active\n"
        "tags: search, ranking\n"
        "created_at: '2026-03-02T09:00:00+00:00'\n"
        "---\n"
        "\n"
        "Search should become summary-first.\n",
        encoding="utf-8",
    )

    repository = RelationalMemoryRepository(db_manager)
    imported = import_markdown_memory(repository, file_path, ["workspace-a"])

    assert imported is not None
    assert imported.type == "fact"
    assert imported.tags == ["ranking", "search"]
    assert imported.workspace_ids == ["workspace-a"]
    assert imported.created_at == "2026-03-02T09:00:00+00:00"
    assert imported.metadata["imported_source_name"] == "search-ranking"
    assert imported.content.strip() == "Search should become summary-first."


def test_resolve_markdown_import_paths_supports_lists_and_globs(tmp_path: Path) -> None:
    first = tmp_path / "alpha.md"
    second = tmp_path / "beta.md"
    first.write_text("# alpha\n", encoding="utf-8")
    second.write_text("# beta\n", encoding="utf-8")

    resolved = resolve_markdown_import_paths([str(first), str(tmp_path / "*.md")])

    assert resolved == [first.resolve(), second.resolve()]


def test_import_markdown_memory_paths_imports_multiple_files(db_manager, tmp_path: Path) -> None:
    first = tmp_path / "alpha.md"
    second = tmp_path / "beta.md"
    first.write_text("---\ntype: fact\n---\n\nAlpha memory\n", encoding="utf-8")
    second.write_text("---\ntype: plan\n---\n\nBeta memory\n", encoding="utf-8")

    repository = RelationalMemoryRepository(db_manager)
    imported = import_markdown_memory_paths(repository, [str(tmp_path / "*.md")], ["workspace-a"])

    assert [record.title for record in imported if record is not None] == ["Alpha", "Beta"]

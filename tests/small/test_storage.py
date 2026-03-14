from pathlib import Path

import pytest

from mcp_memory.core.storage import (
    compute_memory_id,
    ensure_memory_dirs,
    get_memory_file_path,
    list_memory_files,
)


pytestmark = pytest.mark.small


def test_ensure_memory_dirs_creates_expected_structure(memory_path: Path) -> None:
    ensure_memory_dirs(memory_path)

    assert memory_path.is_dir()
    assert (memory_path / "indices").is_dir()
    assert (memory_path / ".trash").is_dir()


def test_get_memory_file_path_adds_markdown_extension(memory_path: Path) -> None:
    assert get_memory_file_path(memory_path, "idea").name == "idea.md"


def test_list_memory_files_ignores_hidden_entries(memory_path: Path) -> None:
    ensure_memory_dirs(memory_path)
    (memory_path / "visible.md").write_text("hello", encoding="utf-8")
    (memory_path / ".hidden.md").write_text("secret", encoding="utf-8")

    files = list_memory_files(memory_path)

    assert [path.name for path in files] == ["visible.md"]


def test_compute_memory_id_prefers_relative_memory_path(memory_path: Path) -> None:
    file_path = memory_path / "notes.md"

    assert compute_memory_id(memory_path, file_path) == "memory:notes"
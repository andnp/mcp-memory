from pathlib import Path

import pytest

from mcp_memory.search.path_utils import matches_any_excluded, normalize_path


pytestmark = pytest.mark.small


def test_normalize_path_returns_relative_path_under_docs_root(tmp_path: Path) -> None:
    docs_root = tmp_path / "workspace"
    target = docs_root / "notes" / "memory.md"
    target.parent.mkdir(parents=True)
    target.write_text("memory", encoding="utf-8")

    assert normalize_path(str(target), docs_root) == "notes/memory.md"


def test_matches_any_excluded_matches_relative_and_filename(tmp_path: Path) -> None:
    docs_root = tmp_path / "workspace"
    target = docs_root / "notes" / "memory.md"
    target.parent.mkdir(parents=True)
    target.write_text("memory", encoding="utf-8")

    assert matches_any_excluded(str(target), {"notes/memory.md"}, docs_root) is True
    assert matches_any_excluded(str(target), {"memory.md"}, docs_root) is True
    assert matches_any_excluded(str(target), {"other.md"}, docs_root) is False
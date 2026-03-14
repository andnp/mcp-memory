from pathlib import Path

import pytest

from mcp_memory.core.memory_paths import normalize_memory_name, resolve_memory_path


pytestmark = pytest.mark.small


def test_normalize_memory_name_strips_memory_prefix() -> None:
    assert normalize_memory_name("memory:alpha") == "alpha"
    assert normalize_memory_name("beta") == "beta"


def test_resolve_memory_path_returns_existing_memory_file(tmp_path: Path) -> None:
    memory_root = tmp_path / ".memories"
    target = memory_root / "alpha.md"
    target.parent.mkdir(parents=True)
    target.write_text("body", encoding="utf-8")

    assert resolve_memory_path(memory_root, "memory:alpha") == target
    assert resolve_memory_path(memory_root, "missing") is None
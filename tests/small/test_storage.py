from pathlib import Path

import pytest

from mcp_memory.core.storage import ensure_memory_dirs

pytestmark = pytest.mark.small


def test_ensure_memory_dirs_creates_expected_structure(memory_path: Path) -> None:
    ensure_memory_dirs(memory_path)

    assert memory_path.is_dir()
    assert (memory_path / "indices").is_dir()
    assert (memory_path / ".trash").is_dir()


from pathlib import Path

import pytest

from mcp_memory.utils.atomic_io import atomic_write_json, atomic_write_text


pytestmark = pytest.mark.small


def test_atomic_write_text_creates_parent_dirs_and_writes_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "memory.txt"

    atomic_write_text(target, "hello memory")

    assert target.read_text(encoding="utf-8") == "hello memory"


def test_atomic_write_json_writes_sorted_pretty_json(tmp_path: Path) -> None:
    target = tmp_path / "payload.json"

    atomic_write_json(target, {"b": 2, "a": 1})

    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}'
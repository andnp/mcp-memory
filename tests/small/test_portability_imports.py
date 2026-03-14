from pathlib import Path

import pytest


pytestmark = pytest.mark.small


def test_source_tree_contains_no_legacy_src_imports() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory"
    python_files = sorted(root.rglob("*.py"))

    legacy_references: list[str] = []
    for file_path in python_files:
        content = file_path.read_text(encoding="utf-8")
        if "from src." in content or "import src." in content:
            legacy_references.append(str(file_path.relative_to(root.parent)))

    assert legacy_references == []

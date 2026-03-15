from pathlib import Path
import importlib

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


def test_file_backed_markdown_helpers_are_quarantined_out_of_core() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory"
    core_root = package_root / "core"
    storage_source = (core_root / "storage.py").read_text(encoding="utf-8")
    importer_source = (package_root / "relational" / "importer.py").read_text(encoding="utf-8")
    legacy_root = package_root / "legacy"

    assert not (core_root / "memory_paths.py").exists()
    assert list(legacy_root.rglob("*.py")) == []
    assert "get_memory_file_path" not in storage_source
    assert "list_memory_files" not in storage_source
    assert "compute_memory_id" not in storage_source
    assert "def _list_markdown_files" in importer_source


def test_runtime_module_imports_without_core_package_cycle() -> None:
    runtime_module = importlib.import_module("mcp_memory.mcp.runtime")

    assert hasattr(runtime_module, "create_runtime")

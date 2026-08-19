import ast
import importlib
from pathlib import Path

import pytest

from mcp_memory.application.ports import (
    MemorySearchPort,
    SharedReadCacheInFlightSearch,
    SharedReadCacheProjectionEntry,
    SharedReadCacheProjectionUpsert,
    SharedReadCacheReadEntry,
    SharedReadCacheSearchRequest,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheInFlightSearch as StorageSharedReadCacheInFlightSearch,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheProjectionEntry as StorageSharedReadCacheProjectionEntry,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheProjectionUpsert as StorageSharedReadCacheProjectionUpsert,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheReadEntry as StorageSharedReadCacheReadEntry,
)
from mcp_memory.storage.shared_read_cache import (
    SharedReadCacheSearchRequest as StorageSharedReadCacheSearchRequest,
)

pytestmark = pytest.mark.small


class _MemorySearchAdapter:
    def get_health(self) -> object:
        return object()

    def read_memory(self, memory_id: str) -> object:
        return memory_id

    def peek_memory(self, memory_id: str) -> object:
        return memory_id

    def search_memories_for_maintenance(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 50,
        *,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> object:
        return query, workspace_id, limit, memory_type, status, include_superseded

    def resolve_memory_id(self, memory_id: str) -> str:
        return memory_id


def test_application_memory_search_port_accepts_native_adapter_shape() -> None:
    adapter: MemorySearchPort = _MemorySearchAdapter()

    assert adapter.resolve_memory_id("mem-1") == "mem-1"


def test_shared_read_cache_dtos_are_application_owned_with_storage_aliases() -> None:
    assert StorageSharedReadCacheInFlightSearch is SharedReadCacheInFlightSearch
    assert StorageSharedReadCacheProjectionEntry is SharedReadCacheProjectionEntry
    assert StorageSharedReadCacheProjectionUpsert is SharedReadCacheProjectionUpsert
    assert StorageSharedReadCacheReadEntry is SharedReadCacheReadEntry
    assert StorageSharedReadCacheSearchRequest is SharedReadCacheSearchRequest


def test_search_cache_request_normalizes_ranking_context_without_timing_fields() -> None:
    request = SharedReadCacheSearchRequest(
        query="ports",
        workspace_id="workspace-a",
        limit=10,
        adaptive_limit=False,
        memory_type=None,
        status=None,
        include_superseded=False,
        ranking_workspace_id="workspace-b",
    )

    params = request.normalized_params()

    assert params == {
        "cache_schema_version": 1,
        "query": "ports",
        "workspace_id": "workspace-a",
        "limit": 10,
        "adaptive_limit": False,
        "memory_type": None,
        "status": None,
        "include_superseded": False,
        "ranking_workspace_id": "workspace-b",
    }
    assert not any(key.endswith("_ms") or "time" in key for key in params)


FORBIDDEN_CORE_IMPORT_ROOTS = (
    "mcp_memory.application",
    "mcp_memory.integrations",
    "mcp_memory.management",
    "mcp_memory.mcp",
    "mcp_memory.relational",
    "mcp_memory.storage",
)

# These are narrow migration exceptions for known transitional edges. Keep
# them file-and-module specific: they are debt to remove, not package-wide
# exemptions for future core imports.
ALLOWED_TRANSITIONAL_CORE_IMPORTS = {
    ("ingest_claim_lifecycle.py", "mcp_memory.mcp.internal_ingest_keys"),
    (
        "task_handlers/embedding_repair.py",
        "mcp_memory.application.memory_embedding_maintenance",
    ),
    ("task_handlers/ingest.py", "mcp_memory.mcp.validation"),
    ("task_handlers/maintenance_framework.py", "mcp_memory.management.agent_run_reporting"),
    ("task_handlers/maintenance_framework.py", "mcp_memory.management.task_sampling_summary"),
    ("task_handlers/maintenance_work_items.py", "mcp_memory.management.models"),
    ("task_handlers/tool_loop.py", "mcp_memory.mcp.internal_tools"),
    ("task_handlers/tool_loop.py", "mcp_memory.mcp.transport"),
    ("tasks.py", "mcp_memory.storage.sqlite_task_queue"),
    ("task_handlers/curator_support.py", "mcp_memory.integrations.memory_retrieval"),
}


def test_core_import_direction_allows_only_documented_transitional_edges() -> None:
    core_root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory" / "core"
    violations = []
    for file_path in sorted(core_root.rglob("*.py")):
        violations.extend(_find_core_import_violations(file_path, ast.parse(file_path.read_text())))

    assert violations == []


def test_core_import_transitional_debt_matches_documented_allowlist() -> None:
    """Keep the observed transitional edges exactly aligned with the allowlist."""
    core_root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory" / "core"
    transitional_imports = set()
    for file_path in sorted(core_root.rglob("*.py")):
        relative_path = file_path.relative_to(core_root).as_posix()
        for node in ast.walk(ast.parse(file_path.read_text())):
            for imported_module in _imported_modules(node):
                if _is_forbidden_core_import(imported_module):
                    transitional_imports.add((relative_path, imported_module))

    assert transitional_imports == set(ALLOWED_TRANSITIONAL_CORE_IMPORTS)


def test_core_import_direction_detects_unallowlisted_imports() -> None:
    core_root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory" / "core"
    path = core_root / "new_module.py"
    tree = ast.parse("from mcp_memory.management import models\n")

    assert _find_core_import_violations(path, tree) == [
        "new_module.py: mcp_memory.management",
    ]


def _find_core_import_violations(path: Path, tree: ast.AST) -> list[str]:
    core_root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory" / "core"
    relative_path = path.relative_to(core_root).as_posix()
    violations = []
    for node in ast.walk(tree):
        imported_modules = _imported_modules(node)
        for imported_module in imported_modules:
            if not _is_forbidden_core_import(imported_module):
                continue
            if (relative_path, imported_module) not in ALLOWED_TRANSITIONAL_CORE_IMPORTS:
                violations.append(f"{relative_path}: {imported_module}")
    return violations


def _imported_modules(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom) or node.module is None:
        return []

    if _is_forbidden_core_import(node.module):
        return [node.module]
    return [
        f"{node.module}.{alias.name}"
        for alias in node.names
        if _is_forbidden_core_import(f"{node.module}.{alias.name}")
    ]


def _is_forbidden_core_import(module: str) -> bool:
    return any(
        module == root or module.startswith(f"{root}.")
        for root in FORBIDDEN_CORE_IMPORT_ROOTS
    )


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


def test_core_provider_code_does_not_import_persistence_adapters() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "mcp_memory" / "core"
    checked_files = [
        root / "provider_admission.py",
        root / "provider_policy.py",
        root / "task_worker.py",
        root / "providers" / "interfaces.py",
        root / "providers" / "instrumented.py",
    ]
    forbidden_imports = (
        "mcp_memory.provider_usage_store",
        "mcp_memory.task_execution_store",
        "mcp_memory.storage.postgres_provider_usage_store",
        "mcp_memory.storage.postgres_task_execution_store",
    )

    for file_path in checked_files:
        source = file_path.read_text(encoding="utf-8")
        assert not any(import_path in source for import_path in forbidden_imports), file_path

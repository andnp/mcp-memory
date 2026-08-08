"""Keep curator application code dependent on ports, not adapters."""
import ast
import subprocess
import sys
from pathlib import Path

FORBIDDEN = {
    "mcp_memory.curation_store",
    "mcp_memory.curation_action_store",
    "mcp_memory.relational",
    "mcp_memory.storage",
    "mcp_memory.runtime",
    "mcp_memory.runtime_facades",
    "mcp_memory.work_item_store",
    "mcp_memory.provider_usage_store",
    "mcp_memory.task_execution_store",
}
ROOT = Path(__file__).parents[2] / "src/mcp_memory/core"

# These are compatibility/composition roots by design.  Their imports are
# checked explicitly rather than hidden by exempting the whole application.
COMPATIBILITY_ROOTS = {
    ROOT / "ports" / "curation.py",
    ROOT / "ports" / "maintenance.py",
    ROOT / "ports" / "work_items.py",
}

# The verifier needs these persistence-owned read-only types until their
# provider-neutral read port is extracted.
DEFERRED_IMPORTS = {
    ("curation_verifier.py", "mcp_memory.relational.repository", "MemoryLink"),
    ("curation_verifier.py", "mcp_memory.relational.repository", "RelationalMemoryReadContext"),
}
ALLOWED_COMPATIBILITY_IMPORTS = {
    ("curation.py", "mcp_memory.curation_store", "CandidateDisposition"),
    ("curation.py", "mcp_memory.curation_store", "CurationActionReceipt"),
    ("curation.py", "mcp_memory.curation_store", "CurationCandidateState"),
    ("curation.py", "mcp_memory.curation_store", "CurationReceiptHydrationError"),
    ("curation.py", "mcp_memory.curation_store", "CurationReceiptState"),
    ("curation.py", "mcp_memory.curation_store", "CurationRepository"),
    ("curation.py", "mcp_memory.curation_store", "CurationRun"),
    ("curation.py", "mcp_memory.curation_store", "CurationRunOutcome"),
    ("curation.py", "mcp_memory.curation_store", "CurationRunState"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationActionContractError"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationActionFatalError"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationActionStore"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationActionStaleError"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationTransaction"),
    ("curation.py", "mcp_memory.curation_action_store", "CurationActionTransientError"),
    ("curation.py", "mcp_memory.curation_action_store", "MutationResult"),
    ("maintenance.py", "mcp_memory.relational.search", "MaintenanceReadRepositoryLike"),
    ("work_items.py", "mcp_memory.work_item_store", "COMPATIBILITY_GROUP_STRUCTURAL_REVIEW"),
    ("work_items.py", "mcp_memory.work_item_store", "EXECUTION_LANE_AGENTIC"),
    ("work_items.py", "mcp_memory.work_item_store", "WORK_FAMILY_MEMORY_CURATION_REVIEW"),
    ("work_items.py", "mcp_memory.work_item_store", "compatibility_group_families"),
}


def test_curation_application_imports_use_ports():
    violations = []
    paths = (
        list(ROOT.glob("curation*.py"))
        + list((ROOT / "task_handlers").glob("curator*.py"))
        + [
            ROOT / "task_handlers" / "campaigns.py",
            ROOT / "task_handlers" / "deduplicator_handlers.py",
            ROOT / "task_handlers" / "maintenance_work_items.py",
            ROOT / "task_handlers" / "relationship_review_handlers.py",
            ROOT / "task_handlers" / "taxonomist_support.py",
        ]
        + list((ROOT / "ports").glob("*.py"))
    )
    for path in paths:
        violations.extend(_find_violations(path, ast.parse(path.read_text())))
    assert not violations, "direct persistence/runtime imports: " + ", ".join(violations)


def test_provider_application_imports_use_provider_ports():
    violations = []
    paths = [
        ROOT / "provider_admission.py",
        ROOT / "providers" / "instrumented.py",
        ROOT / "ports" / "providers.py",
    ]
    for path in paths:
        violations.extend(_find_violations(path, ast.parse(path.read_text())))
    assert not violations, "direct provider adapter imports: " + ", ".join(violations)


def test_compatibility_roots_reject_unlisted_forbidden_imports():
    path = ROOT / "ports" / "curation.py"
    tree = ast.parse("from mcp_memory.storage import DatabaseManager\n")
    assert _find_violations(path, tree) == ["ports/curation.py: mcp_memory.storage.DatabaseManager"]


def test_curator_task_handler_scope_detects_forbidden_imports():
    path = ROOT / "task_handlers" / "curator_support.py"
    tree = ast.parse("from mcp_memory.curation_store import CurationRun\n")
    assert _find_violations(path, tree) == [
        "task_handlers/curator_support.py: mcp_memory.curation_store.CurationRun"
    ]


def test_curator_task_handler_scope_detects_root_level_forbidden_imports():
    path = ROOT / "task_handlers" / "curator_support.py"
    tree = ast.parse("from mcp_memory import curation_store\n")
    assert _find_violations(path, tree) == [
        "task_handlers/curator_support.py: mcp_memory.curation_store"
    ]


def test_compatibility_roots_allow_documented_legacy_aliases():
    path = ROOT / "ports" / "curation.py"
    tree = ast.parse("from mcp_memory.curation_store import CurationRun\n")
    assert _find_violations(path, tree) == []


def _find_violations(path: Path, tree: ast.AST) -> list[str]:
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append(f"{path.relative_to(ROOT)}: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and _is_forbidden(node.module):
            for alias in node.names:
                allowed = (
                    (path.name, node.module, alias.name) in DEFERRED_IMPORTS
                    or (
                        path in COMPATIBILITY_ROOTS
                        and (path.name, node.module, alias.name) in ALLOWED_COMPATIBILITY_IMPORTS
                    )
                )
                if not allowed:
                    violations.append(f"{path.relative_to(ROOT)}: {node.module}.{alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_module = f"{node.module}.{alias.name}"
                if _is_forbidden(imported_module):
                    violations.append(f"{path.relative_to(ROOT)}: {imported_module}")
    return violations


def _is_forbidden(name: str) -> bool:
    return name in FORBIDDEN or any(name.startswith(prefix + ".") for prefix in FORBIDDEN)


def test_ports_package_import_is_dependency_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import mcp_memory.core.ports as ports

assert "mcp_memory.core.ports.work_items" not in sys.modules
assert ports.WorkItemRepository is not None
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

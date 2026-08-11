"""Medium coverage for typed runtime capabilities and search parity."""

from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    SYSTEM1_INGEST_TASK_NAME,
    bootstrap_background_tasks,
    build_runtime_task_worker,
)
from mcp_memory.mcp.runtime import create_runtime_composition, resolve_workspace_runtime_spec


pytestmark = pytest.mark.medium


def test_runtime_composition_routes_typed_capabilities_to_consumers(tmp_path: Path) -> None:
    """Verify typed bundles preserve shared resources across runtime consumers.

    Background scheduling and task-worker construction must accept composed views.
    """
    composition = create_runtime_composition(resolve_workspace_runtime_spec(cwd=tmp_path))
    try:
        capabilities = composition.capabilities

        assert capabilities.memory.repository is composition.context.repository
        assert capabilities.mutation.repository is composition.context.repository
        assert capabilities.provider.ai_provider_registry is composition.context.ai_provider_registry
        assert capabilities.task.as_context().repository is composition.context.repository
        assert capabilities.management.storage_backend == composition.context.storage_backend

        bootstrap_background_tasks(capabilities.background)
        worker = build_runtime_task_worker(capabilities.task)

        assert SYSTEM1_INGEST_TASK_NAME in worker._handlers  # noqa: SLF001
        assert composition.context.task_queue.count_by_status()["pending"] >= 1
    finally:
        composition.close()

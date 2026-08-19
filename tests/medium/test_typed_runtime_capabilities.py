"""Medium coverage for typed runtime capabilities and search parity."""

import json
from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    SYSTEM1_INGEST_TASK_NAME,
    bootstrap_background_tasks,
    build_runtime_task_worker,
)
from mcp_memory.mcp.runtime import (
    DaemonCapabilityBundle,
    create_runtime_composition,
    resolve_workspace_runtime_spec,
)
from mcp_memory.mcp.services import (
    search_memory_records_async_service,
    search_memory_records_service,
)

pytestmark = pytest.mark.medium


def test_runtime_composition_routes_typed_capabilities_to_consumers(tmp_path: Path) -> None:
    """Verify typed bundles preserve shared resources across runtime consumers.

    Background scheduling and task-worker construction must accept composed views.
    """
    composition = create_runtime_composition(resolve_workspace_runtime_spec(cwd=tmp_path))
    try:
        capabilities = composition.capabilities

        assert isinstance(composition.daemon, DaemonCapabilityBundle)
        assert composition.daemon.memory is capabilities.memory
        assert composition.daemon.background is capabilities.background
        assert composition.daemon.task is capabilities.task
        assert composition.daemon.resources is composition.resources
        assert composition.daemon.writeback is None
        assert capabilities.memory.repository is composition.context.repository
        assert capabilities.mutation.repository is composition.context.repository
        assert capabilities.provider.ai_provider_registry is composition.context.ai_provider_registry
        assert capabilities.task.as_context().repository is composition.context.repository
        assert capabilities.management.storage_backend == composition.context.storage_backend

        bootstrap_background_tasks(capabilities.background)
        worker = build_runtime_task_worker(capabilities.task)

        assert SYSTEM1_INGEST_TASK_NAME in worker._handlers
        assert composition.context.task_queue.count_by_status()["pending"] >= 1
    finally:
        composition.close()


@pytest.mark.asyncio
async def test_search_service_sync_and_async_paths_return_matching_payloads(tmp_path: Path) -> None:
    """Verify sync and async search services share the same result contract.

    The comparison intentionally uses normal output so timing diagnostics cannot vary.
    """
    composition = create_runtime_composition(resolve_workspace_runtime_spec(cwd=tmp_path))
    try:
        repository = composition.context.repository
        assert repository is not None
        record = repository.create_memory(
            title="Typed capability parity",
            content="Search parity should preserve the same record payload.",
            summary="Sync and async search parity.",
            workspace_ids=[composition.context.workspace_id or "workspace"],
            memory_type="fact",
            tags=["parity"],
        )
        assert record is not None
        arguments = {"query": "search parity", "limit": 5}

        sync_payload = search_memory_records_service(composition.context, arguments)
        async_payload = await search_memory_records_async_service(composition.context, arguments)

        assert json.loads(json.dumps(sync_payload)) == json.loads(json.dumps(async_payload))
    finally:
        composition.close()

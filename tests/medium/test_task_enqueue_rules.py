import json
from pathlib import Path

import pytest

from mcp_memory.core.task_handlers.constants import SYSTEM1_INGEST_THRESHOLD
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_record_thought_enqueues_single_ingest_task_at_threshold(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)

    try:
        payloads = []
        for index in range(SYSTEM1_INGEST_THRESHOLD):
            response = await call_memory_tool(runtime, "record_thought", {"content": f"note {index}"})
            payloads.append(json.loads(response[0].text))

        next_response = await call_memory_tool(
            runtime,
            "record_thought",
            {"content": f"note {SYSTEM1_INGEST_THRESHOLD}"},
        )
        next_payload = json.loads(next_response[0].text)

        for payload in payloads:
            assert payload == {"status": "recorded"}
        assert next_payload == {"status": "recorded"}

        assert runtime.task_queue is not None
        counts = runtime.task_queue.count_by_status()
        assert counts == {"pending": 1}
        pending_tasks = runtime.task_queue.list_tasks(status="pending", limit=5)
        assert len(pending_tasks) == 1
        assert pending_tasks[0].workspace_id is None
        assert pending_tasks[0].data["workspace_id"] == runtime.workspace_id
        assert pending_tasks[0].available_at <= pending_tasks[0].updated_at
    finally:
        runtime.close()

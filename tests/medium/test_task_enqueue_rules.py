import json
from pathlib import Path

import pytest

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
        first = await call_memory_tool(runtime, "record_thought", {"content": "first note"})
        second = await call_memory_tool(runtime, "record_thought", {"content": "second note"})
        third = await call_memory_tool(runtime, "record_thought", {"content": "third note"})
        fourth = await call_memory_tool(runtime, "record_thought", {"content": "fourth note"})

        first_payload = json.loads(first[0].text)
        second_payload = json.loads(second[0].text)
        third_payload = json.loads(third[0].text)
        fourth_payload = json.loads(fourth[0].text)

        assert "ingest_task" not in first_payload
        assert "ingest_task" not in second_payload
        assert third_payload["ingest_task"]["task_name"] == "ingest-system1"
        assert third_payload["ingest_task"]["created"] is True
        assert fourth_payload["ingest_task"]["id"] == third_payload["ingest_task"]["id"]
        assert fourth_payload["ingest_task"]["created"] is False

        assert runtime.task_queue is not None
        counts = runtime.task_queue.count_by_status()
        assert counts == {"pending": 1}
    finally:
        runtime.close()
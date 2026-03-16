from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time

import pytest

from mcp_memory.core.agent_runtime import build_runtime_task_worker
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.large


async def _wait_until(predicate, timeout_seconds: float = 2.0, interval_seconds: float = 0.02) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        if await predicate():
            return
        await asyncio.sleep(interval_seconds)
    raise AssertionError("timed out waiting for expected end-to-end state")


def _payload_text(response) -> dict:
    return json.loads(response[0].text)


@pytest.mark.asyncio
async def test_e2e_record_thought_ingests_and_becomes_searchable(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.task_queue is not None
    assert runtime.repository is not None
    assert runtime.workspace_id is not None
    try:
        first = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite journal progress"}))
        second = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite worker retries"}))
        third = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite search flow"}))
        ingest_task = runtime.task_queue.find_open_task("ingest-system1", runtime.workspace_id)
        assert ingest_task is not None
        runtime.task_queue.update_pending_task(ingest_task.id, available_at=time.time())

        worker = build_runtime_task_worker(runtime)
        await worker.start()
        try:
            async def memory_created() -> bool:
                assert runtime.repository is not None
                return bool(runtime.repository.list_memories(workspace_id=runtime.workspace_id))

            await _wait_until(memory_created, timeout_seconds=5.0)
        finally:
            await worker.stop(0.1)

        assert "ingest_task" not in first
        assert "ingest_task" not in second
        assert ingest_task.task_name == "ingest-system1"

        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)
        assert len(records) == 1
        created_record = records[0]
        assert created_record.type == "observation"

        search_payload = _payload_text(
            await call_memory_tool(
                runtime,
                "search_memory_records",
                {"query": "sqlite worker search"},
            )
        )
        assert [result["memory_id"] for result in search_payload["results"]] == [created_record.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_e2e_search_and_read_persist_across_runtime_recreation(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))

    runtime_one = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime_one.repository is not None
        assert runtime_one.workspace_id is not None
        record = runtime_one.repository.create_memory(
            title="Shared fact",
            content="Shared runtimes should see one relational store.",
            workspace_ids=[runtime_one.workspace_id],
            memory_type="fact",
        )
        assert record is not None
    finally:
        runtime_one.close()

    runtime_two = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime_two.workspace_id is not None
        search_payload = _payload_text(
            await call_memory_tool(
                runtime_two,
                "search_memory_records",
                {"query": "shared fact"},
            )
        )
        read_payload = _payload_text(
            await call_memory_tool(runtime_two, "read_memory_record", {"memory_id": record.id})
        )
        assert [result["memory_id"] for result in search_payload["results"]] == [record.id]
        assert read_payload["record"]["id"] == record.id
    finally:
        runtime_two.close()

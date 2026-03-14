from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
    try:
        first = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite journal progress"}))
        second = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite worker retries"}))
        third = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite search flow"}))

        worker = build_runtime_task_worker(runtime)
        await worker.start()
        try:
            async def memory_created() -> bool:
                assert runtime.repository is not None
                return bool(runtime.repository.list_memories(workspace_id=runtime.workspace_id))

            await _wait_until(memory_created)
        finally:
            await worker.stop(0.1)

        assert "ingest_task" not in first
        assert "ingest_task" not in second
        assert third["ingest_task"]["task_name"] == "ingest-system1"

        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)
        assert len(records) == 1
        created_record = records[0]
        assert created_record.type == "observation"

        search_payload = _payload_text(
            await call_memory_tool(
                runtime,
                "search_memory_records",
                {"query": "sqlite worker search", "workspace_id": runtime.workspace_id},
            )
        )
        assert [result["memory_id"] for result in search_payload["results"]] == [created_record.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_e2e_shared_store_persists_records_across_runtime_recreation(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))

    runtime_one = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        created_payload = _payload_text(
            await call_memory_tool(
                runtime_one,
                "create_memory_record",
                {
                    "title": "Shared fact",
                    "content": "Shared runtimes should see one relational store.",
                    "workspace_ids": [runtime_one.workspace_id],
                    "memory_type": "fact",
                },
            )
        )
        created_id = created_payload["record"]["id"]
    finally:
        runtime_one.close()

    runtime_two = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        persisted_list = _payload_text(
            await call_memory_tool(
                runtime_two,
                "list_memory_records",
                {"workspace_id": runtime_two.workspace_id, "limit": 10},
            )
        )
        assert {record["id"] for record in persisted_list["records"]} == {created_id}
    finally:
        runtime_two.close()

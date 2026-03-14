from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mcp_memory.daemon import DaemonLifecycleController
from mcp_memory.mcp.handlers import call_memory_tool


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

    controller = DaemonLifecycleController(shutdown_grace_seconds=0.05)
    runtime = await controller.acquire_runtime(project_override=None, cwd=workspace)
    workspace_id = runtime.project_name or "global"

    try:
        first = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite journal progress"}))
        second = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite worker retries"}))
        third = _payload_text(await call_memory_tool(runtime, "record_thought", {"content": "Capture sqlite search flow"}))

        assert "ingest_task" not in first
        assert "ingest_task" not in second
        assert third["ingest_task"]["task_name"] == "ingest-system1"
        assert third["ingest_task"]["created"] is True

        assert runtime.repository is not None
        assert runtime.journal is not None

        async def memory_created() -> bool:
            return bool(runtime.repository.list_memories(workspace_id=workspace_id))

        await _wait_until(memory_created)

        records = runtime.repository.list_memories(workspace_id=workspace_id)
        assert len(records) == 1
        created_record = records[0]
        assert created_record.type == "observation"
        assert created_record.metadata["ingest_task_id"] == third["ingest_task"]["id"]
        assert runtime.journal.count_by_status().get("processed", 0) == 3

        search_payload = _payload_text(
            await call_memory_tool(
                runtime,
                "search_memory_records",
                {"query": "sqlite worker search", "workspace_id": workspace_id},
            )
        )
        assert [result["memory_id"] for result in search_payload["results"]] == [created_record.id]
        assert search_payload["results"][0]["summary"]

        read_payload = _payload_text(
            await call_memory_tool(runtime, "read_memory_record", {"memory_id": created_record.id})
        )
        assert read_payload["record"]["id"] == created_record.id
        assert read_payload["record"]["access_score"] >= 1.0
    finally:
        shutdown_task = await controller.release_runtime(runtime)
        if shutdown_task is not None:
            await asyncio.shield(shutdown_task)
        await controller.shutdown_now()


@pytest.mark.asyncio
async def test_e2e_shared_runtime_persists_records_across_shutdown(monkeypatch, tmp_path: Path) -> None:
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home_path))

    controller = DaemonLifecycleController(shutdown_grace_seconds=0.05)
    runtime_one = await controller.acquire_runtime(project_override=None, cwd=workspace)
    runtime_two = await controller.acquire_runtime(project_override=None, cwd=workspace)
    workspace_id = runtime_one.project_name or "global"

    try:
        assert runtime_one is runtime_two
        assert controller.client_count == 2

        created_payload = _payload_text(
            await call_memory_tool(
                runtime_one,
                "create_memory_record",
                {
                    "title": "Daemon shared fact",
                    "content": "Shared runtimes should see one relational store.",
                    "workspace_ids": [workspace_id],
                    "memory_type": "fact",
                    "tags": ["daemon", "shared"],
                },
            )
        )
        created_id = created_payload["record"]["id"]

        markdown_file = workspace / "epic-06.md"
        markdown_file.write_text(
            "---\n"
            "type: plan\n"
            "status: active\n"
            "tags: [dashboard, api]\n"
            "created_at: '2026-03-14T10:00:00+00:00'\n"
            "---\n\n"
            "Ship the management API dashboard.\n",
            encoding="utf-8",
        )
        imported_payload = _payload_text(
            await call_memory_tool(
                runtime_two,
                "import_markdown_memory_file",
                {"file_path": str(markdown_file), "workspace_ids": [workspace_id]},
            )
        )
        imported_id = imported_payload["record"]["id"]

        list_payload = _payload_text(
            await call_memory_tool(
                runtime_two,
                "list_memory_records",
                {"workspace_id": workspace_id, "limit": 10},
            )
        )
        assert {record["id"] for record in list_payload["records"]} == {created_id, imported_id}

        stats_payload = _payload_text(await call_memory_tool(runtime_two, "get_memory_stats", {}))
        assert stats_payload["relational_memories"] == 2

        await controller.release_runtime(runtime_one)
        await asyncio.sleep(0.08)
        assert controller.has_runtime is True

        search_payload = _payload_text(
            await call_memory_tool(
                runtime_two,
                "search_memory_records",
                {"query": "management dashboard", "workspace_id": workspace_id},
            )
        )
        assert [result["memory_id"] for result in search_payload["results"]] == [imported_id]

        shutdown_task = await controller.release_runtime(runtime_two)
        if shutdown_task is not None:
            await asyncio.shield(shutdown_task)
        assert controller.has_runtime is False

        runtime_three = await controller.acquire_runtime(project_override=None, cwd=workspace)
        try:
            assert runtime_three is not runtime_one

            persisted_list = _payload_text(
                await call_memory_tool(
                    runtime_three,
                    "list_memory_records",
                    {"workspace_id": workspace_id, "limit": 10},
                )
            )
            assert {record["id"] for record in persisted_list["records"]} == {created_id, imported_id}

            read_payload = _payload_text(
                await call_memory_tool(runtime_three, "read_memory_record", {"memory_id": created_id})
            )
            assert read_payload["record"]["title"] == "Daemon shared fact"
        finally:
            final_shutdown = await controller.release_runtime(runtime_three)
            if final_shutdown is not None:
                await asyncio.shield(final_shutdown)
    finally:
        await controller.shutdown_now()
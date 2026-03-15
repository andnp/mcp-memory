import json
from pathlib import Path

import pytest

from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_search_and_read_tools_work_with_seeded_repository_records(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")

    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Relational bootstrap",
            content="Wire repository-backed memory tools.",
            workspace_ids=["workspace-a"],
            tags=["testing", "sqlite"],
            summary="Repository-backed tool smoke test.",
            memory_type="plan",
            metadata={"phase": 1},
        )
        assert record is not None

        read_result = await call_memory_tool(
            runtime,
            "read_memory_record",
            {"memory_id": record.id},
        )
        search_result = await call_memory_tool(
            runtime,
            "search_memory_records",
            {"workspace_id": "workspace-a", "query": "relational bootstrap"},
        )
        read_payload = json.loads(read_result[0].text)
        search_payload = json.loads(search_result[0].text)

        assert read_payload["record"]["title"] == "Relational bootstrap"
        assert read_payload["record"]["metadata"] == {"phase": 1}
        assert [result["memory_id"] for result in search_payload["results"]] == [record.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_search_and_read_tools_persist_across_runtime_recreation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    first_runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert first_runtime.repository is not None
        record = first_runtime.repository.create_memory(
            title="Persistent fact",
            content="Repository records should survive runtime recreation.",
            workspace_ids=[first_runtime.workspace_id],
            memory_type="fact",
        )
        assert record is not None
    finally:
        first_runtime.close()

    second_runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        read_result = await call_memory_tool(
            second_runtime,
            "read_memory_record",
            {"memory_id": record.id},
        )
        search_result = await call_memory_tool(
            second_runtime,
            "search_memory_records",
            {"query": "persistent fact", "workspace_id": second_runtime.workspace_id},
        )

        read_payload = json.loads(read_result[0].text)
        search_payload = json.loads(search_result[0].text)

        assert read_payload["record"]["title"] == "Persistent fact"
        assert [result["memory_id"] for result in search_payload["results"]] == [record.id]
    finally:
        second_runtime.close()
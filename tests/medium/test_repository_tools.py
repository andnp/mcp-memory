import json
from pathlib import Path

import pytest

from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_repository_tools_create_get_and_list_records(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(project_override=None, cwd=tmp_path / "workspace")

    try:
        create_result = await call_memory_tool(
            runtime,
            "create_memory_record",
            {
                "title": "Relational bootstrap",
                "content": "Wire repository-backed memory tools.",
                "workspace_ids": ["workspace-a"],
                "tags": ["testing", "sqlite"],
                "summary": "Repository-backed tool smoke test.",
                "memory_type": "plan",
                "metadata": {"phase": 1},
            },
        )
        created_payload = json.loads(create_result[0].text)
        memory_id = created_payload["record"]["id"]

        get_result = await call_memory_tool(
            runtime,
            "get_memory_record",
            {"memory_id": memory_id},
        )
        get_payload = json.loads(get_result[0].text)

        list_result = await call_memory_tool(
            runtime,
            "list_memory_records",
            {"workspace_id": "workspace-a", "memory_type": "plan"},
        )
        list_payload = json.loads(list_result[0].text)

        assert created_payload["status"] == "created"
        assert get_payload["record"]["title"] == "Relational bootstrap"
        assert get_payload["record"]["metadata"] == {"phase": 1}
        assert [record["id"] for record in list_payload["records"]] == [memory_id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_repository_tools_persist_across_runtime_recreation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    first_runtime = create_runtime(project_override=None, cwd=tmp_path / "workspace")
    try:
        create_result = await call_memory_tool(
            first_runtime,
            "create_memory_record",
            {
                "title": "Persistent fact",
                "content": "Repository records should survive runtime recreation.",
                "memory_type": "fact",
            },
        )
        memory_id = json.loads(create_result[0].text)["record"]["id"]
    finally:
        first_runtime.close()

    second_runtime = create_runtime(project_override=None, cwd=tmp_path / "workspace")
    try:
        get_result = await call_memory_tool(
            second_runtime,
            "get_memory_record",
            {"memory_id": memory_id},
        )
        stats_result = await call_memory_tool(second_runtime, "get_memory_stats", {})

        get_payload = json.loads(get_result[0].text)
        stats_payload = json.loads(stats_result[0].text)

        assert get_payload["record"]["title"] == "Persistent fact"
        assert stats_payload["relational_memories"] == 1
        assert stats_payload["workspace_id"] == second_runtime.workspace_id
    finally:
        second_runtime.close()
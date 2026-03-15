import json
from pathlib import Path

import pytest

from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_relational_runtime_search_and_read_tools(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        first = runtime.repository.create_memory(
            title="Auth search plan",
            content="Search should return a summary before a full read.",
            summary="Summary-first auth search plan.",
            workspace_ids=["workspace-a"],
            memory_type="plan",
            tags=["auth", "search"],
        )
        second = runtime.repository.create_memory(
            title="Auth legacy plan",
            content="This older auth plan has been replaced.",
            summary="Older auth plan.",
            workspace_ids=["workspace-a"],
            memory_type="plan",
            tags=["auth"],
        )
        assert first is not None and second is not None

        runtime.repository.add_link(first.id, second.id, "SUPERSEDES", "Superseded by new plan")

        search_result = await call_memory_tool(
            runtime,
            "search_memory_records",
            {"query": "auth search", "workspace_id": "workspace-a", "limit": 5},
        )
        search_payload = json.loads(search_result[0].text)

        read_result = await call_memory_tool(
            runtime,
            "read_memory_record",
            {"memory_id": first.id},
        )
        read_payload = json.loads(read_result[0].text)

        assert [result["memory_id"] for result in search_payload["results"]] == [first.id]
        assert search_payload["results"][0]["summary"] == "Summary-first auth search plan."
        assert read_payload["record"]["id"] == first.id
        assert [record["id"] for record in read_payload["superseded"]] == [second.id]
        assert read_payload["relationships"]["outgoing"][0]["link_type"] == "SUPERSEDES"
    finally:
        runtime.close()
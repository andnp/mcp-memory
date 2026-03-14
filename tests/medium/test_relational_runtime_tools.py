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

    runtime = create_runtime(project_override=None, cwd=tmp_path / "workspace")
    try:
        first = await call_memory_tool(
            runtime,
            "create_memory_record",
            {
                "title": "Auth search plan",
                "content": "Search should return a summary before a full read.",
                "summary": "Summary-first auth search plan.",
                "workspace_ids": ["workspace-a"],
                "memory_type": "plan",
                "tags": ["auth", "search"],
            },
        )
        second = await call_memory_tool(
            runtime,
            "create_memory_record",
            {
                "title": "Auth legacy plan",
                "content": "This older auth plan has been replaced.",
                "summary": "Older auth plan.",
                "workspace_ids": ["workspace-a"],
                "memory_type": "plan",
                "tags": ["auth"],
            },
        )

        first_id = json.loads(first[0].text)["record"]["id"]
        second_id = json.loads(second[0].text)["record"]["id"]
        assert runtime.repository is not None
        runtime.repository.add_link(first_id, second_id, "SUPERSEDES", "Superseded by new plan")

        search_result = await call_memory_tool(
            runtime,
            "search_memory_records",
            {"query": "auth search", "workspace_id": "workspace-a", "limit": 5},
        )
        search_payload = json.loads(search_result[0].text)

        read_result = await call_memory_tool(
            runtime,
            "read_memory_record",
            {"memory_id": first_id},
        )
        read_payload = json.loads(read_result[0].text)

        assert [result["memory_id"] for result in search_payload["results"]] == [first_id]
        assert search_payload["results"][0]["summary"] == "Summary-first auth search plan."
        assert read_payload["record"]["id"] == first_id
        assert [record["id"] for record in read_payload["superseded"]] == [second_id]
        assert read_payload["relationships"]["outgoing"][0]["link_type"] == "SUPERSEDES"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_import_markdown_memory_file_uses_existing_importer(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    runtime = create_runtime(project_override=None, cwd=workspace)
    try:
        markdown_file = workspace / "epic-01.md"
        markdown_file.write_text(
            "---\n"
            "type: plan\n"
            "status: active\n"
            "tags: [sqlite, migration]\n"
            "created_at: '2026-03-01T10:30:00+00:00'\n"
            "---\n\n"
            "Ship the relational bootstrap.\n",
            encoding="utf-8",
        )

        import_result = await call_memory_tool(
            runtime,
            "import_markdown_memory_file",
            {"file_path": str(markdown_file), "workspace_ids": ["workspace-a"]},
        )
        import_payload = json.loads(import_result[0].text)

        assert import_payload["status"] == "imported"
        assert import_payload["record"]["title"] == "Epic 01"
        assert import_payload["record"]["metadata"]["legacy_memory_name"] == "epic-01"

        search_result = await call_memory_tool(
            runtime,
            "search_memory_records",
            {"query": "relational bootstrap", "workspace_id": "workspace-a"},
        )
        search_payload = json.loads(search_result[0].text)

        assert [result["title"] for result in search_payload["results"]] == ["Epic 01"]
    finally:
        runtime.close()
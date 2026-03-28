import json
from pathlib import Path

import pytest

from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


class _RuntimeFusionFakeEmbedder:
    model_name = "fusion-fake-mini"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["permission", "security", "identity", "credential"]):
                vectors.append([1.0, 0.0])
            elif any(token in lowered for token in ["sqlite", "database", "wal"]):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


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
            {"query": "auth search", "limit": 5},
        )
        search_payload = json.loads(search_result[0].text)

        read_result = await call_memory_tool(
            runtime,
            "read_memory_record",
            {"memory_id": first.id},
        )
        read_payload = json.loads(read_result[0].text)

        assert [result["memory_id"] for result in search_payload["results"]] == [first.id]
        assert search_payload["recommended_follow_up_tool"] == "read_memory_record"
        assert search_payload["guidance"] == (
            "Use these summary-first results to identify the most promising memories, then call read_memory_record for full context on the specific memory_id values you want to inspect."
        )
        assert search_payload["results"][0]["summary"] == "Summary-first auth search plan."
        assert read_payload["record"]["id"] == first.id
        assert read_payload["record"]["read_count"] == 1
        assert [record["id"] for record in read_payload["superseded"]] == [second.id]
        assert read_payload["relationships"]["outgoing"][0]["link_type"] == "SUPERSEDES"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_relational_runtime_search_debug_reports_total_timing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Timing debug note",
            content="Search timing should be visible in debug mode.",
            summary="Timing summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["timing"],
        )
        assert record is not None

        payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "timing debug", "limit": 5, "debug": True},
                )
            )[0].text
        )

        assert payload["status"] == "ok"
        assert payload["timing_ms"]["total"] >= 0.0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_search_memory_tool_hides_archived_by_default_but_can_request_them(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        active = runtime.repository.create_memory(
            title="Architecture active",
            content="Focused active architecture note.",
            summary="Active architecture summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            status="active",
            tags=["architecture"],
        )
        archived = runtime.repository.create_memory(
            title="Architecture archived",
            content="Archived architecture blob.",
            summary="Archived architecture summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            status="archived",
            tags=["architecture"],
        )
        assert active is not None and archived is not None

        default_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "architecture", "limit": 10},
                )
            )[0].text
        )
        archived_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "architecture", "limit": 10, "status": "archived"},
                )
            )[0].text
        )

        assert [result["memory_id"] for result in default_payload["results"]] == [active.id]
        assert [result["memory_id"] for result in archived_payload["results"]] == [archived.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_search_memory_tool_uses_active_workspace_context(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        local = runtime.repository.create_memory(
            title="Workspace scoped auth note",
            content="Local workspace auth search result.",
            summary="Local auth summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["auth"],
        )
        other = runtime.repository.create_memory(
            title="Cross workspace auth note",
            content="Other workspace auth search result.",
            summary="Cross auth summary.",
            workspace_ids=["workspace-other"],
            memory_type="fact",
            tags=["auth"],
        )
        assert local is not None and other is not None

        implicit_result = await call_memory_tool(
            runtime,
            "search_memory_records",
            {"query": "auth note", "limit": 5},
        )

        implicit_payload = json.loads(implicit_result[0].text)

        assert implicit_payload["results"][0]["memory_id"] == local.id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_relational_runtime_search_combines_keyword_and_semantic_candidates(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.db_manager is not None
        runtime.embedder = _RuntimeFusionFakeEmbedder()
        runtime.vector_store = SQLiteVectorStore(runtime.db_manager)
        assert runtime.relational_search is not None
        runtime.relational_search._embedder = runtime.embedder
        runtime.relational_search._vector_store = runtime.vector_store

        lexical = runtime.repository.create_memory(
            title="Permission checklist",
            content="Permission checklist for rollout and approvals.",
            summary="Lexical permissions summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["security"],
        )
        semantic = runtime.repository.create_memory(
            title="Identity policy",
            content="Credential rotation and authentication policy for all services.",
            summary="Semantic identity summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["auth"],
        )
        assert lexical is not None and semantic is not None

        payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "permissions security", "limit": 5},
                )
            )[0].text
        )

        returned_ids = [result["memory_id"] for result in payload["results"]]
        assert lexical.id in returned_ids
        assert semantic.id in returned_ids
        assert payload["recommended_follow_up_tool"] == "read_memory_record"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_relational_runtime_search_debug_explains_workspace_and_degradation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        local_active = runtime.repository.create_memory(
            title="Search quality active",
            content="Search quality notes for the active workspace.",
            summary="Active quality summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["search"],
        )
        cross_workspace = runtime.repository.create_memory(
            title="Search quality cross workspace",
            content="Search quality notes from another workspace.",
            summary="Cross quality summary.",
            workspace_ids=["workspace-other"],
            memory_type="fact",
            tags=["search"],
        )
        local_stale = runtime.repository.create_memory(
            title="Search quality stale",
            content="Search quality notes that are stale.",
            summary="Stale quality summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            status="stale",
            tags=["search"],
        )
        assert local_active is not None and cross_workspace is not None and local_stale is not None

        payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {
                        "query": "search quality summary",
                        "limit": 5,
                        "debug": True,
                    },
                )
            )[0].text
        )

        returned_ids = [result["memory_id"] for result in payload["results"]]
        debug_by_id = {result["memory_id"]: result["ranking_debug"] for result in payload["results"]}

        assert returned_ids == [local_active.id, cross_workspace.id, local_stale.id]
        assert debug_by_id[local_active.id]["workspace_match"] is True
        assert debug_by_id[local_active.id]["workspace_multiplier"] == 1.2
        assert debug_by_id[cross_workspace.id]["workspace_match"] is False
        assert debug_by_id[cross_workspace.id]["workspace_multiplier"] == 1.0
        assert debug_by_id[local_stale.id]["degradation_multiplier"] == 0.3
        assert debug_by_id[local_active.id]["matched_by_keyword"] is True
    finally:
        runtime.close()

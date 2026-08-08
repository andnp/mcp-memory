import json
from pathlib import Path
from typing import cast

import pytest

from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.relational.search import RelationalMemorySearchService

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
async def test_skill_observation_lifecycle_and_tag_filter(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(cwd=tmp_path / "workspace")
    try:
        recorded = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "record_skill_observation",
                    {
                        "title": "Observer should search open findings",
                        "summary": "Use the observation tag when reviewing skill improvements.",
                        "content": "The daily reviewer needs a deterministic tag filter for open findings.",
                        "skill": "task-observer",
                        "observation_kind": "improvement",
                        "privacy_classification": "internal",
                    },
                )
            )[0].text
        )
        memory_id = recorded["record"]["memory_id"]
        assert recorded["record"]["status"] == "active"
        assert "skill:task-observer" in recorded["record"]["tags"]

        search_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {
                        "query": "deterministic tag filter",
                        "tags": ["skill-observation", "skill:task-observer"],
                        "limit": 5,
                    },
                )
            )[0].text
        )
        assert [result["memory_ref"] for result in search_payload["results"]] == [
            f"mem-{recorded['record']['memory_ref']}"
        ]

        resolved = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "resolve_skill_observation",
                    {
                        "memory_id": memory_id,
                        "resolution": "actioned",
                        "note": "Implemented the tag-constrained review query.",
                    },
                )
            )[0].text
        )
        assert resolved["record"]["status"] == "archived"

        archived_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {
                        "query": "deterministic tag filter",
                        "tags": ["skill-observation"],
                        "status": "archived",
                        "limit": 5,
                    },
                )
            )[0].text
        )
        assert [result["memory_ref"] for result in archived_payload["results"]] == [
            f"mem-{recorded['record']['memory_ref']}"
        ]
    finally:
        runtime.close()


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
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="plan",
            tags=["auth", "search"],
        )
        second = runtime.repository.create_memory(
            title="Auth legacy plan",
            content="This older auth plan has been replaced.",
            summary="Older auth plan.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
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

        assert search_payload["results"][0]["memory_ref"] == f"mem-{first.memory_ref}"
        pass  # removed: recommended_follow_up_tool stripped from response
        assert search_payload["results"][0]["summary"] == "Summary-first auth search plan."
        assert "workspace_ids" not in search_payload["results"][0]
        assert "score" not in search_payload["results"][0]
        assert read_payload["record"]["memory_ref"] == f"mem-{first.memory_ref}"
        assert "read_count" not in read_payload["record"]
        assert "related_counts" not in read_payload
        assert "superseded" not in read_payload
        assert "relationships" not in read_payload

        expanded_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "read_memory_record",
                    {
                        "memory_id": first.id,
                        "include_relationships": True,
                        "include_superseded": True,
                    },
                )
            )[0].text
        )

        assert [
            record["memory_ref"] for record in expanded_payload["superseded"]
        ] == [f"mem-{second.memory_ref}"]
        assert expanded_payload["relationships"]["outgoing"][0]["link_type"] == "SUPERSEDES"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_relational_runtime_search_debug_reports_stage_timing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.db_manager is not None
        runtime.embedder = _RuntimeFusionFakeEmbedder()
        runtime.vector_store = SQLiteVectorStore(runtime.db_manager)
        assert runtime.relational_search is not None
        search = cast(RelationalMemorySearchService, runtime.relational_search)
        search._embedder = runtime.embedder
        search._vector_store = runtime.vector_store
        record = runtime.repository.create_memory(
            title="Timing debug note",
            content="Credential security search timing should be visible in debug mode.",
            summary="Timing summary.",
            workspace_ids=[runtime.workspace_id or "workspace-local"],
            memory_type="fact",
            tags=["timing", "security"],
        )
        assert record is not None

        payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "credential security", "limit": 5, "debug": True},
                )
            )[0].text
        )

        assert "status" not in payload
        assert payload["timing_ms"]["total"] >= 0.0
        assert payload["results"][0]["workspace_ids"] == [runtime.workspace_id or "workspace-local"]
        assert payload["results"][0]["score"] >= 0.0
        assert payload["search_diagnostics"]["timing_ms"] == payload["timing_ms"]
        assert payload["timing_ms"]["search"] >= 0.0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_relational_runtime_search_debug_reports_kernel_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.db_manager is not None
        runtime.embedder = _RuntimeFusionFakeEmbedder()
        runtime.vector_store = SQLiteVectorStore(runtime.db_manager)
        monkeypatch.setattr(runtime.vector_store, "supports_candidate_filtering", True, raising=False)
        assert runtime.relational_search is not None
        search = cast(RelationalMemorySearchService, runtime.relational_search)
        search._embedder = runtime.embedder
        search._vector_store = runtime.vector_store

        for index in range(5):
            record = runtime.repository.create_memory(
                title=f"Operational incident note {index:02d}",
                content="Observed daemon_request_timed_out while servicing the live memory search request.",
                summary=f"Operational incident summary {index:02d}.",
                workspace_ids=[runtime.workspace_id or "workspace-local"],
                memory_type="fact",
                tags=["incident"],
            )
            assert record is not None

        payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "daemon_request_timed_out", "limit": 5, "debug": True},
                )
            )[0].text
        )

        assert "status" not in payload
        assert len(payload["results"]) == 5
        assert payload["search_diagnostics"]["timing_ms"] == payload["timing_ms"]
        assert payload["timing_ms"]["search"] >= 0.0
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

        assert default_payload["results"][0]["memory_ref"] == f"mem-{active.memory_ref}"
        assert archived_payload["results"][0]["memory_ref"] == f"mem-{archived.memory_ref}"
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

        assert implicit_payload["results"][0]["memory_ref"] == f"mem-{local.memory_ref}"
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
        search = cast(RelationalMemorySearchService, runtime.relational_search)
        search._embedder = runtime.embedder
        search._vector_store = runtime.vector_store

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

        returned_refs = [result["memory_ref"] for result in payload["results"]]
        assert f"mem-{lexical.memory_ref}" in returned_refs
        assert f"mem-{semantic.memory_ref}" not in returned_refs
        pass  # removed: recommended_follow_up_tool stripped from response
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
        debug_by_id = {
            result["memory_id"]: result["ranking_debug"]
            for result in payload["results"]
        }

        assert returned_ids == [local_active.id, cross_workspace.id, local_stale.id]
        assert cross_workspace.id in returned_ids
        assert "provenance" in debug_by_id[local_active.id]
        assert "canonical_id" in debug_by_id[local_active.id]
    finally:
        runtime.close()

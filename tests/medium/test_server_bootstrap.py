import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from mcp.types import TextContent

from mcp_memory.cli import main
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools
from mcp_memory.server import MCPServer
from tests.sdk.mcp import FakeAsyncContextManager


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_call_memory_tool_returns_placeholder_payload() -> None:
    result = await call_memory_tool(None, "record_thought", {"content": "auth"})

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    payload = json.loads(result[0].text)
    assert payload["status"] == "error"
    assert payload["error"] == "runtime_not_initialized"


def test_get_memory_tools_returns_expected_names() -> None:
    tools = get_memory_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "record_thought",
        "search_memory_records",
        "read_memory_record",
    ]
    search_tool = next(tool for tool in tools if tool.name == "search_memory_records")
    assert search_tool.description is not None
    assert "follow up with read_memory_record" in search_tool.description


def test_get_internal_maintenance_tools_returns_expected_names() -> None:
    tools = get_internal_maintenance_tools()
    names = [tool.name for tool in tools]

    assert names == [
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_list_memory_records",
        "task_complete",
        "internal_task_complete",
        "internal_get_next_dedup_batch",
        "internal_get_next_curator_batch",
        "internal_get_next_ingest_batch",
        "internal_get_work_batch",
        "internal_get_compatible_work_batch",
        "internal_heartbeat_work_item",
        "internal_complete_work_item",
        "internal_defer_work_item",
        "internal_release_work_item",
        "internal_ingest_append_memory",
        "internal_ingest_create_memory",
        "internal_append_memory_content",
        "internal_archive_memory_record",
        "internal_merge_memory_into_canonical",
        "internal_split_memory_record",
        "internal_create_memory_record",
        "internal_update_memory_record",
        "internal_delete_memory_record",
        "internal_create_memory_link",
        "internal_delete_memory_link",
    ]
    search_tool = next(tool for tool in tools if tool.name == "internal_search_memory_records")
    assert search_tool.description is not None
    assert "follow up with internal_read_memory_record" in search_tool.description
    completion_tool = next(tool for tool in tools if tool.name == "task_complete")
    assert completion_tool.description is not None
    assert "instead of creating journal or memory records" in completion_tool.description


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_claim_compatible_work_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    assert runtime.work_items is not None

    try:
        lightweight_specs = [
            ("memory_tagging", "tag-memory"),
            ("graph_link_review", "graph-memory"),
            ("conflict_review", "conflict-memory"),
        ]
        structural_specs = [
            ("memory_curation_review", ["curator-a", "curator-b"]),
            ("memory_dedup_review", ["dedup-a", "dedup-b"]),
        ]
        for family_key, memory_id in lightweight_specs:
            runtime.work_items.enqueue_unique(
                family_key=family_key,
                execution_lane="agentic",
                workspace_id=runtime.workspace_id,
                payload={"memory_id": memory_id, "workspace_id": runtime.workspace_id},
            )
        for family_key, seed_memory_ids in structural_specs:
            runtime.work_items.enqueue_unique(
                family_key=family_key,
                execution_lane="agentic",
                workspace_id=runtime.workspace_id,
                payload={"seed_memory_ids": seed_memory_ids, "workspace_id": runtime.workspace_id},
            )

        lightweight_result = await call_internal_memory_tool(
            runtime,
            "internal_get_compatible_work_batch",
            {
                "task_id": "compat-lightweight-task",
                "compatibility_group": "lightweight_review",
                "execution_lane": "agentic",
                "workspace_id": runtime.workspace_id,
                "limit": 5,
            },
        )
        lightweight_payload = json.loads(lightweight_result[0].text)

        assert lightweight_payload["status"] == "ok"
        assert lightweight_payload["compatibility_group"] == "lightweight_review"
        assert lightweight_payload["family_keys"] == ["memory_tagging", "graph_link_review", "conflict_review"]
        assert lightweight_payload["claimed_count"] == 3
        assert {record["family_key"] for record in lightweight_payload["records"]} == {
            "memory_tagging",
            "graph_link_review",
            "conflict_review",
        }

        structural_result = await call_internal_memory_tool(
            runtime,
            "internal_get_compatible_work_batch",
            {
                "task_id": "compat-structural-task",
                "compatibility_group": "structural_review",
                "execution_lane": "agentic",
                "workspace_id": runtime.workspace_id,
                "allowed_families": ["memory_curation_review"],
                "limit": 5,
            },
        )
        structural_payload = json.loads(structural_result[0].text)

        assert structural_payload["status"] == "ok"
        assert structural_payload["compatibility_group"] == "structural_review"
        assert structural_payload["family_keys"] == ["memory_curation_review"]
        assert structural_payload["claimed_count"] == 1
        assert [record["family_key"] for record in structural_payload["records"]] == ["memory_curation_review"]

        pending_items = runtime.work_items.list_items(status="pending", limit=10)
        assert {item.family_key for item in pending_items} == {"memory_dedup_review"}
    finally:
        runtime.close()


def test_mcp_server_initializes_with_workspace_root() -> None:
    server = MCPServer(workspace_root="demo-workspace")

    assert server.workspace_root == "demo-workspace"


@pytest.mark.asyncio
async def test_call_memory_tool_records_real_journal_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        result = await call_memory_tool(runtime, "record_thought", {"content": "wire up mcp handlers"})
        payload = json.loads(result[0].text)

        assert payload["status"] == "recorded"
        assert payload["entry"]["content"] == "wire up mcp handlers"
        assert payload["entry"]["workspace_id"] == runtime.workspace_id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_append_and_archive(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert record is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_memory_content",
            {"memory_id": record.id, "content": "Prefer deterministic fixtures.", "tags": ["preferences"]},
        )
        archive_result = await call_internal_memory_tool(
            runtime,
            "internal_archive_memory_record",
            {"memory_id": record.id},
        )

        appended_payload = json.loads(append_result[0].text)
        archived_payload = json.loads(archive_result[0].text)
        assert appended_payload["status"] == "ok"
        assert "deterministic fixtures" in appended_payload["record"]["content"]
        assert archived_payload["record"]["status"] == "archived"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_rejects_invalid_ingest_mutations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        create_result = await call_internal_memory_tool(
            runtime,
            "internal_ingest_create_memory",
            {
                "task_id": "ingest-invalid",
                "entry_ids": [],
                "title": "Marker",
                "content": "marker",
            },
        )

        create_payload = json.loads(create_result[0].text)
        assert create_payload["status"] == "error"
        assert create_payload["error"] == "invalid_ingest_payload"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_append_workspace_ids_and_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        record = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
            metadata={"appended_entry_ids": [1]},
        )
        assert record is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_memory_content",
            {
                "memory_id": record.id,
                "content": "Prefer deterministic fixtures.",
                "workspace_ids": ["workspace-b"],
                "metadata": {"appended_entry_ids": [2], "ingest_task_id": "ingest-agentic-task"},
            },
        )

        payload = json.loads(append_result[0].text)

        assert payload["status"] == "ok"
        assert "deterministic fixtures" in payload["record"]["content"]
        assert set(payload["record"]["workspace_ids"]) == {runtime.workspace_id, "workspace-b"}
        assert payload["record"]["metadata"]["appended_entry_ids"] == [1, 2]
        assert payload["record"]["metadata"]["ingest_task_id"] == "ingest-agentic-task"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_fetch_dedup_curator_and_ingest_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.journal is not None
        fact = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        observation = runtime.repository.create_memory(
            title="Auth observation",
            content="Observed another note about JWT enforcement.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        runtime.journal.record("JWT rollout note one", workspace_id=runtime.workspace_id)
        runtime.journal.record("JWT rollout note two", workspace_id=runtime.workspace_id)
        assert fact is not None and observation is not None

        dedup_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_dedup_batch",
            {"task_id": "dedup-batch-test", "strategy": "anomaly", "limit": 20},
        )
        curator_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_curator_batch",
            {"task_id": "curator-batch-test", "strategy": "anomaly", "limit": 5, "exclude_memory_ids": [fact.id]},
        )
        ingest_result = await call_internal_memory_tool(
            runtime,
            "internal_get_next_ingest_batch",
            {"task_id": "agentic-ingest-batch", "batch_size": 10, "grouping_strategy": "fifo"},
        )

        dedup_payload = json.loads(dedup_result[0].text)
        curator_payload = json.loads(curator_result[0].text)
        ingest_payload = json.loads(ingest_result[0].text)

        assert dedup_payload["status"] == "ok"
        assert dedup_payload["records"]
        assert {record["id"] for record in dedup_payload["records"]} >= {fact.id, observation.id}
        assert dedup_payload["requested_strategy"] == "anomaly"
        assert dedup_payload["strategy"] == "anomaly"
        assert dedup_payload["strategy_used"] == "anomaly"
        assert dedup_payload["strategy_fallback_reason"] is None
        assert dedup_payload["candidate_count"] >= len(dedup_payload["records"])
        assert dedup_payload["sampled_memory_ids"]

        assert curator_payload["status"] == "ok"
        assert curator_payload["requested_strategy"] == "anomaly"
        assert curator_payload["strategy"] == "anomaly"
        assert curator_payload["strategy_used"] == "anomaly"
        assert curator_payload["strategy_fallback_reason"] is None
        assert curator_payload["excluded_memory_ids"] == [fact.id]
        assert curator_payload["records"]
        assert all(record["id"] != fact.id for record in curator_payload["records"])
        assert curator_payload["sampled_memory_ids"]

        assert ingest_payload["status"] == "ok"
        assert ingest_payload["task_id"] == "agentic-ingest-batch"
        assert ingest_payload["requested_grouping_strategy"] == "fifo"
        assert ingest_payload["grouping_strategy_used"] == "fifo"
        assert ingest_payload["grouping_fallback_reason"] is None
        assert len(ingest_payload["claimed_entry_ids"]) == 2
        assert ingest_payload["group_count"] >= 1
        assert ingest_payload["groups"]
        assert ingest_payload["groups"][0]["entries"]
        assert ingest_payload["groups"][0]["entries"][0]["status"] == "claimed"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_ingest_tools_preserve_ingest_invariants(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        existing = runtime.repository.create_memory(
            title="Testing preferences",
            content="Prefer pytest.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
            metadata={"appended_entry_ids": [1]},
        )
        assert existing is not None

        append_result = await call_internal_memory_tool(
            runtime,
            "internal_append_to_existing_memory_for_ingest",
            {
                "memory_id": existing.id,
                "content": "Prefer deterministic fixtures.",
                "task_id": "ingest-maintenance-task",
                "entry_ids": [2],
                "workspace_ids": ["workspace-b"],
                "tags": ["testing"],
            },
        )
        create_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record_for_ingest",
            {
                "title": "Pytest fixture policy",
                "content": "- [2026-03-16 12:00] Prefer deterministic fixtures.",
                "task_id": "ingest-maintenance-task",
                "entry_ids": [3, "4"],
                "workspace_ids": [runtime.workspace_id, "workspace-b"],
                "tags": ["pytest"],
            },
        )

        append_payload = json.loads(append_result[0].text)
        create_payload = json.loads(create_result[0].text)
        assert append_payload["status"] == "ok"
        assert "deterministic fixtures" in append_payload["record"]["content"]
        assert set(append_payload["record"]["workspace_ids"]) == {runtime.workspace_id, "workspace-b"}
        assert append_payload["record"]["metadata"]["appended_entry_ids"] == [1, 2]
        assert append_payload["record"]["metadata"]["appended_via_ingest"] is True
        assert append_payload["record"]["metadata"]["ingest_task_id"] == "ingest-maintenance-task"
        assert append_payload["record"]["tags"] == ["testing"]

        assert create_payload["status"] == "ok"
        assert create_payload["record"]["metadata"]["created_via_ingest"] is True
        assert create_payload["record"]["metadata"]["source_entry_ids"] == [3, 4]
        assert create_payload["record"]["metadata"]["ingest_task_id"] == "ingest-maintenance-task"
        assert create_payload["record"]["tags"] == ["pytest"]
        assert create_payload["record"]["summary"]
        assert runtime.task_queue.find_open_task("summarize-memory", runtime.workspace_id or "global") is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_create_update_link_and_delete(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        created_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record",
            {
                "title": "Canonical testing preference",
                "content": "Prefer deterministic fixtures.",
                "memory_type": "fact",
                "tags": ["testing"],
            },
        )
        created_payload = json.loads(created_result[0].text)
        created_id = created_payload["record"]["id"]

        related = runtime.repository.create_memory(
            title="Related testing note",
            content="Pytest remains the standard.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
        )
        assert related is not None

        update_result = await call_internal_memory_tool(
            runtime,
            "internal_update_memory_record",
            {"memory_id": created_id, "content": "Prefer deterministic pytest fixtures.", "tags": ["testing", "pytest"]},
        )
        create_link_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_link",
            {"source_id": created_id, "target_id": related.id, "link_type": "AMENDS", "context": "Curator linked related note."},
        )
        archive_result = await call_internal_memory_tool(
            runtime,
            "internal_archive_memory_record",
            {"memory_id": created_id},
        )
        delete_link_result = await call_internal_memory_tool(
            runtime,
            "internal_delete_memory_link",
            {"source_id": created_id, "target_id": related.id, "link_type": "AMENDS"},
        )
        delete_result = await call_internal_memory_tool(
            runtime,
            "internal_delete_memory_record",
            {"memory_id": created_id, "confirm": True},
        )

        update_payload = json.loads(update_result[0].text)
        create_link_payload = json.loads(create_link_result[0].text)
        archive_payload = json.loads(archive_result[0].text)
        delete_link_payload = json.loads(delete_link_result[0].text)
        delete_payload = json.loads(delete_result[0].text)

        assert created_payload["status"] == "ok"
        assert update_payload["record"]["content"] == "Prefer deterministic pytest fixtures."
        assert update_payload["record"]["tags"] == ["pytest", "testing"]
        assert create_link_payload["status"] == "ok"
        assert archive_payload["record"]["status"] == "archived"
        assert delete_link_payload["status"] == "ok"
        assert delete_payload["status"] == "ok"
        assert runtime.repository.get_memory(created_id) is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_create_record_and_enqueue_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.task_queue is not None
        created_result = await call_internal_memory_tool(
            runtime,
            "internal_create_memory_record",
            {
                "title": "Canonical testing preference",
                "content": "Prefer deterministic fixtures.",
                "memory_type": "observation",
                "tags": ["testing"],
                "enqueue_summary_task": True,
            },
        )
        created_payload = json.loads(created_result[0].text)
        created_id = created_payload["record"]["id"]
        summary_task = runtime.task_queue.find_open_task_with_data_any_workspace(
            "summarize-memory",
            data_fields={"memory_id": created_id},
        )

        assert created_payload["status"] == "ok"
        assert summary_task is not None
        assert summary_task.data["memory_id"] == created_id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_merge_can_override_canonical_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
            metadata={"merged_source_ids": ["older-source"]},
        )
        source = runtime.repository.create_memory(
            title="Auth rollout detail",
            content="JWT rollout is now required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth", "rollout"],
        )
        assert canonical is not None and source is not None

        merge_result = await call_internal_memory_tool(
            runtime,
            "internal_merge_memory_into_canonical",
            {
                "canonical_memory_id": canonical.id,
                "source_memory_id": source.id,
                "title": "Canonical auth rollout fact",
                "content": "JWT rollout is required for all clients.",
                "summary": "Unified rollout requirement.",
                "tags": ["auth", "rollout"],
                "metadata": {"deduplicator_task_id": "dedup-live-task"},
                "link_context": "Merged rollout observation into canonical auth fact.",
            },
        )

        payload = json.loads(merge_result[0].text)
        updated = runtime.repository.get_memory(canonical.id)
        archived = runtime.repository.get_memory(source.id)
        links = runtime.repository.get_links(canonical.id)

        assert payload["status"] == "ok"
        assert payload["canonical"]["title"] == "Canonical auth rollout fact"
        assert payload["canonical"]["summary"] == "Unified rollout requirement."
        assert payload["canonical"]["tags"] == ["auth", "rollout"]
        assert updated is not None and updated.metadata["deduplicator_task_id"] == "dedup-live-task"
        assert updated.metadata["merged_source_ids"] == sorted(["older-source", source.id])
        assert archived is not None and archived.status == "archived"
        assert any(link.target_id == source.id and link.context == "Merged rollout observation into canonical auth fact." for link in links)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_call_internal_memory_tool_can_split_memory_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.repository is not None
        original = runtime.repository.create_memory(
            title="Oversized auth rollout",
            content="Part one. Part two. Part three.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "rollout"],
        )
        assert original is not None

        split_result = await call_internal_memory_tool(
            runtime,
            "internal_split_memory_record",
            {
                "memory_id": original.id,
                "archive_original": True,
                "parts": [
                    {"title": "Auth rollout prerequisites", "content": "Part one.", "tags": ["prereq"]},
                    {"title": "Auth rollout execution", "content": "Part two. Part three.", "tags": ["execution"]},
                ],
            },
        )

        payload = json.loads(split_result[0].text)
        first_child_id = payload["created"][0]["id"]
        second_child_id = payload["created"][1]["id"]
        first_links = runtime.repository.get_links(first_child_id)
        second_links = runtime.repository.get_links(second_child_id)
        archived_original = runtime.repository.get_memory(original.id)
        first_metadata = payload["created"][0]["metadata"]
        second_metadata = payload["created"][1]["metadata"]
        original_metadata = payload["original"]["metadata"]

        assert payload["status"] == "ok"
        assert len(payload["created"]) == 2
        assert payload["archived"]["status"] == "archived"
        assert archived_original is not None and archived_original.status == "archived"
        assert any(link.target_id == original.id and link.link_type == "DEPENDS_ON" for link in first_links)
        assert any(link.target_id == original.id and link.link_type == "DEPENDS_ON" for link in second_links)
        assert first_metadata["split_from_memory_id"] == original.id
        assert second_metadata["split_from_memory_id"] == original.id
        assert first_metadata["split_group_id"] == second_metadata["split_group_id"] == original_metadata["split_group_id"]
        assert first_metadata["split_part_index"] == 1
        assert second_metadata["split_part_index"] == 2
        assert first_metadata["split_part_count"] == 2
        assert second_metadata["split_part_count"] == 2
        assert first_metadata["split_sibling_memory_ids"] == [second_child_id]
        assert second_metadata["split_sibling_memory_ids"] == [first_child_id]
        assert original_metadata["split_child_memory_ids"] == [first_child_id, second_child_id]
        assert original_metadata["split_child_count"] == 2
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_mcp_server_run_autostarts_daemon_and_invokes_stdio(monkeypatch) -> None:
    read_stream = object()
    write_stream = object()
    captured: dict[str, object] = {}
    hook_calls: list[tuple[str, dict[str, object]]] = []

    server = MCPServer(workspace_root="demo")

    async def fake_run(read_arg, write_arg, init_options) -> None:
        captured["read_stream"] = read_arg
        captured["write_stream"] = write_arg
        captured["init_options"] = init_options

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: hook_calls.append((path, payload)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((read_stream, write_stream)),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    await server.run()

    assert captured["read_stream"] is read_stream
    assert captured["write_stream"] is write_stream
    assert captured["init_options"] is not None
    assert [path for path, _ in hook_calls] == ["/api/hooks/session-start", "/api/hooks/session-end"]
    start_payload = hook_calls[0][1]
    end_payload = hook_calls[1][1]
    assert start_payload["session_id"] == end_payload["session_id"]
    assert start_payload["source"] == "mcp-stdio"


@pytest.mark.asyncio
async def test_mcp_server_run_still_sends_session_end_hook_on_failure(monkeypatch) -> None:
    hook_calls: list[tuple[str, dict[str, object]]] = []
    server = MCPServer(workspace_root="demo")

    class FakeMetadata:
        base_url = "http://127.0.0.1:8123"

    async def fake_run(*_args) -> None:
        raise RuntimeError("stdio disconnected")

    monkeypatch.setattr("mcp_memory.server.ensure_daemon_started", lambda workspace_root, cwd=None: FakeMetadata())
    monkeypatch.setattr(
        server,
        "_request_json",
        lambda path, payload: hook_calls.append((path, payload)) or {"status": "ok"},
    )
    monkeypatch.setattr(
        "mcp_memory.server.stdio_server",
        lambda: FakeAsyncContextManager((object(), object())),
    )
    monkeypatch.setattr(server.server, "run", fake_run)

    with pytest.raises(RuntimeError, match="stdio disconnected"):
        await server.run()

    assert [path for path, _ in hook_calls] == ["/api/hooks/session-start", "/api/hooks/session-end"]
    assert hook_calls[0][1]["session_id"] == hook_calls[1][1]["session_id"]


def test_cli_help_lists_grouped_public_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])
    help_lines = result.output.splitlines()

    assert result.exit_code == 0
    assert "run" in result.output
    assert "daemon" in result.output
    assert "log" in result.output
    assert "install" in result.output
    assert "agents" in result.output
    assert "stats" in result.output
    assert "stash" in result.output
    assert "import-markdown" in result.output
    assert "daemon-status" not in result.output
    assert "daemon-stop" not in result.output
    assert "daemon-restart" not in result.output
    assert "dashboard" not in result.output
    assert not any(line.strip().startswith("logs") for line in help_lines)
    assert not any(line.strip().startswith("log-summary") for line in help_lines)
    assert not any(line.strip().startswith("log-prune") for line in help_lines)


def test_daemon_help_lists_nested_management_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["daemon", "--help"])

    assert result.exit_code == 0
    assert "status" in result.output
    assert "stop" in result.output
    assert "restart" in result.output
    assert "dashboard" in result.output


def test_log_help_lists_nested_log_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["log", "--help"])

    assert result.exit_code == 0
    assert "show" in result.output
    assert "summary" in result.output
    assert "prune" in result.output

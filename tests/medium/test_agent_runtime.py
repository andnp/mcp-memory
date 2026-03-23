import json
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mcp_memory.core.agent_runtime import (
    AGENTIC_TASK_NAMES,
    CONFLICT_DETECTOR_TASK_NAME,
    CONFLICT_SCREENING_TASK_NAME,
    CURATOR_FRONTIER_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUP_PREP_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINK_DISCOVERY_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
    SUMMARIZE_MEMORY_TASK_NAME,
    SWEEPER_TASK_NAME,
    SYSTEM1_INGEST_TASK_NAME,
    TAG_NORMALIZER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    bootstrap_background_tasks,
    build_runtime_task_worker,
    handle_defragmenter_task,
    handle_conflict_detector_task,
    handle_conflict_screening_task,
    handle_curator_frontier_task,
    handle_dedup_prep_task,
    handle_memory_curator_task,
    handle_deduplicator_task,
    handle_fact_checker_task,
    handle_graph_link_discovery_task,
    handle_graph_linker_task,
    handle_ingest_system1_task,
    handle_project_manager_task,
    handle_tag_normalizer_task,
    handle_summarize_memory_task,
    handle_sweeper_task,
    handle_taxonomist_task,
    _provider_for_task,
)
from mcp_memory.core.providers.interfaces import ProviderRateLimitExceeded
from mcp_memory.context import ApplicationContext
from mcp_memory.core.providers import AgenticRunResult, CopilotCLIAgenticProvider
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.task_handlers.maintenance import (
    CURATOR_MAX_MEMORY_CHARS,
    CURATOR_MAX_SEED_RECORDS,
    DEDUPLICATOR_OBSERVATION_SEED_RECORDS,
    _select_curator_seed_records,
)
from mcp_memory.core.task_handlers.deduplicator_support import build_deduplicator_agent_prompt
from mcp_memory.core.task_handlers.ingest import (
    INGEST_APPEND_TOOL_NAME,
    INGEST_CREATE_TOOL_NAME,
    _build_ingest_agent_prompt,
    _normalize_ingest_agentic_result,
)
from mcp_memory.core.task_handlers import SYSTEM1_INGEST_PRIORITY, SUMMARIZE_MEMORY_PRIORITY, task_priority
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue, TaskRecord
from mcp_memory.mcp.handlers import call_internal_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp import runtime as runtime_module
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
)
from tests.sdk.providers import FakeAIProvider


pytestmark = pytest.mark.medium


def test_normalize_ingest_agentic_result_preserves_rich_final_json_contract() -> None:
    result = _normalize_ingest_agentic_result(
        AgenticRunResult(
            status="success",
            summary="Provider summary",
            parsed={
                "response": json.dumps(
                    {
                        "summary": "Provider summary",
                        "created_memory_ids": ["memory-created"],
                        "touched_memory_ids": ["memory-created", "memory-existing"],
                        "matched_memory_ids": ["memory-existing"],
                        "meaningful_actions": 2,
                        "entry_outcomes": [
                            {
                                "entry_id": 101,
                                "disposition": "created",
                                "memory_id": "memory-created",
                                "reason": "Captured the explicit config key timeout_ms=5000.",
                            },
                            {
                                "entry_id": 102,
                                "disposition": "matched_existing",
                                "memory_id": "memory-existing",
                                "reason": "Exact match for error string E_CONNRESET in existing canonical memory.",
                            },
                        ],
                    }
                ),
                "stats": {
                    "tools": {
                        "totalCalls": 3,
                        "byName": {
                            "mcp_mcp-memory-internal_internal_get_next_ingest_batch": {"count": 1},
                            f"mcp_mcp-memory-internal_{INGEST_CREATE_TOOL_NAME}": {"count": 1},
                            f"mcp_mcp-memory-internal_{INGEST_APPEND_TOOL_NAME}": {"count": 1},
                        },
                    }
                },
            },
        )
    )

    assert result["summary"] == "Provider summary"
    assert result["created_memory_ids"] == ["memory-created"]
    assert result["touched_memory_ids"] == ["memory-created", "memory-existing"]
    assert result["matched_memory_ids"] == ["memory-existing"]
    assert result["meaningful_actions"] == 2
    assert result["entry_outcomes"] == [
        {
            "entry_id": 101,
            "disposition": "created",
            "memory_id": "memory-created",
            "reason": "Captured the explicit config key timeout_ms=5000.",
        },
        {
            "entry_id": 102,
            "disposition": "matched_existing",
            "memory_id": "memory-existing",
            "reason": "Exact match for error string E_CONNRESET in existing canonical memory.",
        },
    ]
    assert result["tool_calls_executed"] == 3
    assert result["mutations"] == 2


def test_normalize_ingest_agentic_result_remains_backward_compatible_with_thin_json() -> None:
    result = _normalize_ingest_agentic_result(
        AgenticRunResult(
            status="success",
            summary="Thin summary",
            parsed={
                "response": json.dumps(
                    {
                        "summary": "Thin summary",
                        "created_memory_ids": ["memory-created"],
                        "meaningful_actions": 1,
                    }
                ),
                "stats": {
                    "tools": {
                        "totalCalls": 1,
                        "byName": {
                            f"mcp_mcp-memory-internal_{INGEST_CREATE_TOOL_NAME}": {"count": 1},
                        },
                    }
                },
            },
        )
    )

    assert result["summary"] == "Thin summary"
    assert result["created_memory_ids"] == ["memory-created"]
    assert result["touched_memory_ids"] == ["memory-created"]
    assert result["matched_memory_ids"] == []
    assert result["entry_outcomes"] == []
    assert result["meaningful_actions"] == 1
    assert result["tool_calls_executed"] == 1
    assert result["mutations"] == 1


def test_build_ingest_agent_prompt_requests_concrete_auditable_entry_outcomes() -> None:
    prompt = _build_ingest_agent_prompt(
        TaskRecord(
            id="ingest-prompt-test",
            task_name=SYSTEM1_INGEST_TASK_NAME,
            data={"workspace_id": "workspace-test"},
            workspace_id="workspace-test",
            status="pending",
            priority=100,
            retries_count=0,
            max_retries=3,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=None,
            started_at=None,
            completed_at=None,
            last_error=None,
        ),
        workspace_id="workspace-test",
        batch_size=10,
        grouping_strategy="fifo",
        max_batches_per_run=8,
    )

    assert "Preserve concrete symbols, file paths, thresholds, IDs, error strings, config keys, and commit refs" in prompt
    assert "When uncertain, prefer narrow concrete observations over broad abstraction." in prompt
    assert "Aim to drain the queue for this task in one run" in prompt
    assert "Include explicit per-entry outcomes for every claimed entry" in prompt
    assert '"entry_outcomes": [{"entry_id": 123, "disposition": "created"|"appended"|"matched_existing"|"ignored"|"no_mutation"' in prompt
    assert "Do not make vague claims like 'matched existing canonical memories'" in prompt
    assert "Provide a reason whenever an entry outcome is ignored, no_mutation, or matched_existing." in prompt
    assert "Do not create memories that only log task completion, queue progress, tool usage" in prompt


def test_build_deduplicator_agent_prompt_routes_closeout_to_task_complete() -> None:
    prompt = build_deduplicator_agent_prompt(
        TaskRecord(
            id="dedup-prompt-test",
            task_name=DEDUPLICATOR_TASK_NAME,
            data={},
            workspace_id="workspace-test",
            status="pending",
            priority=100,
            retries_count=0,
            max_retries=3,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=None,
            started_at=None,
            completed_at=None,
            last_error=None,
        ),
        [],
        strategy_used="semantic",
    )

    assert "use task_complete for operational closeout only" in prompt
    assert "Do not call record_thought or create journal/observation memories" in prompt


@pytest.mark.asyncio
async def test_ingest_handler_creates_relational_memories_and_summary_tasks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("track sqlite queue status updates")
        runtime.journal.record("track sqlite queue worker retries")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-test",
        )

        result = await handle_ingest_system1_task(runtime, task)
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert len(result["created_memory_ids"]) == 1
        assert len(result["claimed_entry_ids"]) == 2
        assert len(result["deleted_entry_ids"]) == 0
        assert len(result["recoverable_entry_ids"]) == 2
        assert result["released_entry_ids"] == []
        assert result["meaningful_actions"] == 1
        assert result["processed_entry_ids"] == result["recoverable_entry_ids"]
        assert result["entry_dispositions"] == [
            {
                "entry_id": result["claimed_entry_ids"][0],
                "disposition": "created",
                "finalization_status": "recoverable",
                "memory_id": records[0].id,
                "memory_title": records[0].title,
            },
            {
                "entry_id": result["claimed_entry_ids"][1],
                "disposition": "created",
                "finalization_status": "recoverable",
                "memory_id": records[0].id,
                "memory_title": records[0].title,
            },
        ]
        assert result["touched_memory_ids"] == [records[0].id]
        assert result["appended_memory_ids"] == []
        assert result["matched_memory_ids"] == []
        assert runtime.journal.count_by_status() == {"recoverable": 2}
        assert len(records) == 1
        assert records[0].type == "observation"
        assert records[0].tags == []
        assert records[0].metadata["created_via_ingest"] is True
        assert records[0].metadata["source_entry_ids"] == result["claimed_entry_ids"]
        assert records[0].metadata["ingest_task_id"] == "ingest-test"
        summary_task = runtime.task_queue.find_open_task(
            SUMMARIZE_MEMORY_TASK_NAME,
            runtime.workspace_id or "global",
        )
        assert summary_task is not None
        assert summary_task.data["memory_id"] == records[0].id
        assert summary_task.priority == SUMMARIZE_MEMORY_PRIORITY
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_can_append_directly_into_existing_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        runtime.journal.record("The user also prefers deterministic fixtures for pytest.")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-append-test",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "results": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer deterministic fixtures for pytest.",
                        }
                    ]
                }
            ]
        )
        result = await handle_ingest_system1_task(runtime, task, provider)
        updated = runtime.repository.get_memory(target.id)
        memories = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert result["created_memory_ids"] == [target.id]
        assert result["deleted_entry_ids"] == []
        assert len(result["recoverable_entry_ids"]) == 1
        assert result["released_entry_ids"] == []
        assert updated is not None
        assert "deterministic fixtures" in updated.content
        assert updated.tags == ["testing"]
        assert updated.metadata["appended_via_ingest"] is True
        assert updated.metadata["appended_entry_ids"] == [result["claimed_entry_ids"][0]]
        assert len(memories) == 1
        assert runtime.journal.count_by_status() == {"recoverable": 1}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_can_use_agentic_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        entry = runtime.journal.record(
            "The user also prefers deterministic fixtures for pytest.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-task",
        )

        class _AgenticProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def run_agent(self, prompt: str) -> AgenticRunResult:
                self.prompts.append(prompt)
                batch_result = await call_internal_memory_tool(
                    runtime,
                    "internal_get_next_ingest_batch",
                    {"task_id": task.id, "batch_size": 10},
                )
                batch_payload = json.loads(batch_result[0].text)
                claimed_entry_ids = batch_payload["claimed_entry_ids"]
                append_result = await call_internal_memory_tool(
                    runtime,
                    INGEST_APPEND_TOOL_NAME,
                    {
                        "memory_id": target.id,
                        "content": "Prefer deterministic fixtures for pytest.",
                        "task_id": task.id,
                        "entry_ids": claimed_entry_ids,
                        "workspace_ids": [runtime.workspace_id],
                        "tags": ["testing"],
                    },
                )
                append_payload = json.loads(append_result[0].text)
                return AgenticRunResult(
                    status="success",
                    summary="Agentic ingest appended the new preference into the canonical testing memory.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Agentic ingest appended the new preference into the canonical testing memory.",
                                "created_memory_ids": [append_payload["record"]["id"]],
                                "meaningful_actions": 1,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 2,
                                "byName": {
                                    "mcp_mcp-memory-internal_internal_get_next_ingest_batch": {"count": 1},
                                    f"mcp_mcp-memory-internal_{INGEST_APPEND_TOOL_NAME}": {"count": 1},
                                },
                            }
                        },
                    },
                )

        provider = _AgenticProvider()
        result = await handle_ingest_system1_task(runtime, task, provider)
        updated = runtime.repository.get_memory(target.id)

        assert result["created_memory_ids"] == [target.id]
        assert result["claimed_entry_ids"] == [entry.id]
        assert result["deleted_entry_ids"] == []
        assert result["recoverable_entry_ids"] == [entry.id]
        assert result["released_entry_ids"] == []
        assert result["meaningful_actions"] == 1
        assert result["execution_mode"] == "agentic_mcp"
        assert result["tool_calls_executed"] == 2
        assert result["mutations"] == 1
        assert result["tool_names_used"] == [
            "internal_get_next_ingest_batch",
            INGEST_APPEND_TOOL_NAME,
        ]
        assert result["provider_reported_tool_calls"] == 2
        assert result["provider_reported_mutations"] == 1
        assert result["provider_reported_tool_names_used"] == [
            "mcp_mcp-memory-internal_internal_get_next_ingest_batch",
            f"mcp_mcp-memory-internal_{INGEST_APPEND_TOOL_NAME}",
        ]
        assert result["provider_reported_entry_outcomes"] == []
        assert result["provider_reported_touched_memory_ids"] == [target.id]
        assert result["provider_reported_matched_memory_ids"] == []
        assert result["entry_dispositions"] == [
            {
                "entry_id": entry.id,
                "disposition": "appended",
                "finalization_status": "recoverable",
                "memory_id": target.id,
                "memory_title": target.title,
            }
        ]
        assert result["touched_memory_ids"] == [target.id]
        assert result["appended_memory_ids"] == [target.id]
        assert result["matched_memory_ids"] == [target.id]
        assert updated is not None
        assert "deterministic fixtures" in updated.content
        assert updated.tags == ["testing"]
        assert updated.metadata["appended_via_ingest"] is True
        assert updated.metadata["appended_entry_ids"] == [entry.id]
        assert updated.metadata["ingest_task_id"] == task.id
        assert runtime.journal.count_by_status() == {"recoverable": 1}
        assert provider.prompts
        assert "internal_get_next_ingest_batch" in provider.prompts[0]
        assert INGEST_APPEND_TOOL_NAME in provider.prompts[0]
        assert INGEST_CREATE_TOOL_NAME in provider.prompts[0]
        assert "Small-to-medium records beat large mixed-topic blobs." in provider.prompts[0]
        assert "Do not merge, append, or rewrite across different projects, products, or repositories" in provider.prompts[0]
        assert "Do not delete or release journal claims yourself" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_agentic_provider_can_drain_multiple_batches_in_one_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        first = runtime.journal.record("First queue thought", workspace_id=runtime.workspace_id)
        second = runtime.journal.record("Second queue thought", workspace_id=runtime.workspace_id)
        third = runtime.journal.record("Third queue thought", workspace_id=runtime.workspace_id)
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id, "batch_size": 2, "max_batches_per_run": 4},
            available_at=0.0,
            task_id="ingest-agentic-multi-batch",
        )

        class _MultiBatchAgenticProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            async def run_agent(self, prompt: str) -> AgenticRunResult:
                self.prompts.append(prompt)
                created_ids: list[str] = []
                for expected_count, title in ((2, "Batch one memory"), (1, "Batch two memory")):
                    batch_result = await call_internal_memory_tool(
                        runtime,
                        "internal_get_next_ingest_batch",
                        {"task_id": task.id, "batch_size": 2, "grouping_strategy": "fifo"},
                    )
                    batch_payload = json.loads(batch_result[0].text)
                    claimed_entry_ids = batch_payload["claimed_entry_ids"]
                    assert len(claimed_entry_ids) == expected_count
                    create_result = await call_internal_memory_tool(
                        runtime,
                        INGEST_CREATE_TOOL_NAME,
                        {
                            "task_id": task.id,
                            "entry_ids": claimed_entry_ids,
                            "title": title,
                            "content": "\n".join(
                                f"- handled {entry_id}" for entry_id in claimed_entry_ids
                            ),
                            "workspace_ids": [runtime.workspace_id],
                            "tags": ["testing"],
                        },
                    )
                    create_payload = json.loads(create_result[0].text)
                    created_ids.append(create_payload["record"]["id"])
                return AgenticRunResult(
                    status="success",
                    summary="Agentic ingest drained multiple batches in one run.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Agentic ingest drained multiple batches in one run.",
                                "created_memory_ids": created_ids,
                                "meaningful_actions": 2,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 4,
                                "byName": {
                                    "mcp_mcp-memory-internal_internal_get_next_ingest_batch": {"count": 2},
                                    f"mcp_mcp-memory-internal_{INGEST_CREATE_TOOL_NAME}": {"count": 2},
                                },
                            }
                        },
                    },
                )

        provider = _MultiBatchAgenticProvider()
        result = await handle_ingest_system1_task(runtime, task, provider)

        assert result["claimed_entry_ids"] == [first.id, second.id, third.id]
        assert sorted(result["recoverable_entry_ids"]) == [first.id, second.id, third.id]
        assert result["released_entry_ids"] == []
        assert result["meaningful_actions"] == 2
        assert result["tool_calls_executed"] == 4
        assert result["mutations"] == 2
        assert result["provider_reported_tool_calls"] == 4
        assert result["provider_reported_mutations"] == 2
        assert result["tool_names_used"] == ["internal_get_next_ingest_batch", INGEST_CREATE_TOOL_NAME]
        assert len(result["created_memory_ids"]) == 2
        assert runtime.journal.count_by_status() == {"recoverable": 3}
        assert provider.prompts
        assert "Aim to drain the queue for this task in one run" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_agentic_append_uses_task_attributed_counts_when_provider_reports_zero_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        entry = runtime.journal.record(
            "The user also prefers deterministic fixtures for pytest.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-zero-stats-append",
        )

        class _ZeroStatsAppendProvider:
            async def run_agent(self, prompt: str) -> AgenticRunResult:
                batch_result = await call_internal_memory_tool(
                    runtime,
                    "internal_get_next_ingest_batch",
                    {"task_id": task.id, "batch_size": 10},
                )
                batch_payload = json.loads(batch_result[0].text)
                claimed_entry_ids = batch_payload["claimed_entry_ids"]
                append_result = await call_internal_memory_tool(
                    runtime,
                    INGEST_APPEND_TOOL_NAME,
                    {
                        "memory_id": target.id,
                        "content": "Prefer deterministic fixtures for pytest.",
                        "task_id": task.id,
                        "entry_ids": claimed_entry_ids,
                        "workspace_ids": [runtime.workspace_id],
                        "tags": ["testing"],
                    },
                )
                append_payload = json.loads(append_result[0].text)
                return AgenticRunResult(
                    status="success",
                    summary="Agentic ingest appended the new preference into the canonical testing memory.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Agentic ingest appended the new preference into the canonical testing memory.",
                                "created_memory_ids": [append_payload["record"]["id"]],
                                "meaningful_actions": 1,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 0,
                                "byName": {},
                            }
                        },
                    },
                )

        result = await handle_ingest_system1_task(runtime, task, _ZeroStatsAppendProvider())

        assert result["claimed_entry_ids"] == [entry.id]
        assert result["recoverable_entry_ids"] == [entry.id]
        assert result["tool_calls_executed"] == 2
        assert result["mutations"] == 1
        assert result["tool_names_used"] == [
            "internal_get_next_ingest_batch",
            INGEST_APPEND_TOOL_NAME,
        ]
        assert result["provider_reported_tool_calls"] == 0
        assert result["provider_reported_mutations"] == 0
        assert result["provider_reported_tool_names_used"] == []
        assert result["appended_memory_ids"] == [target.id]
        assert result["matched_memory_ids"] == [target.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_agentic_create_uses_task_attributed_counts_when_provider_reports_zero_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        entry = runtime.journal.record(
            "The user wants truthful ingest telemetry for create paths too.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-zero-stats-create",
        )

        class _ZeroStatsCreateProvider:
            async def run_agent(self, prompt: str) -> AgenticRunResult:
                batch_result = await call_internal_memory_tool(
                    runtime,
                    "internal_get_next_ingest_batch",
                    {"task_id": task.id, "batch_size": 10},
                )
                batch_payload = json.loads(batch_result[0].text)
                claimed_entry_ids = batch_payload["claimed_entry_ids"]
                create_result = await call_internal_memory_tool(
                    runtime,
                    INGEST_CREATE_TOOL_NAME,
                    {
                        "task_id": task.id,
                        "entry_ids": claimed_entry_ids,
                        "title": "Truthful ingest telemetry",
                        "content": "- [2026-03-19 00:00] The user wants truthful ingest telemetry for create paths too.",
                        "workspace_ids": [runtime.workspace_id],
                        "tags": ["testing"],
                    },
                )
                create_payload = json.loads(create_result[0].text)
                return AgenticRunResult(
                    status="success",
                    summary="Agentic ingest created a new memory.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Agentic ingest created a new memory.",
                                "created_memory_ids": [create_payload["record"]["id"]],
                                "meaningful_actions": 1,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 0,
                                "byName": {},
                            }
                        },
                    },
                )

        result = await handle_ingest_system1_task(runtime, task, _ZeroStatsCreateProvider())

        assert result["claimed_entry_ids"] == [entry.id]
        assert result["recoverable_entry_ids"] == [entry.id]
        assert result["tool_calls_executed"] == 2
        assert result["mutations"] == 1
        assert result["tool_names_used"] == [
            "internal_get_next_ingest_batch",
            INGEST_CREATE_TOOL_NAME,
        ]
        assert result["provider_reported_tool_calls"] == 0
        assert result["provider_reported_mutations"] == 0
        assert result["provider_reported_tool_names_used"] == []
        assert len(result["created_memory_ids"]) == 1
        assert result["touched_memory_ids"] == result["created_memory_ids"]
        assert result["appended_memory_ids"] == []
        assert result["matched_memory_ids"] == []
        assert result["entry_dispositions"] == [
            {
                "entry_id": entry.id,
                "disposition": "created",
                "finalization_status": "recoverable",
                "memory_id": result["created_memory_ids"][0],
                "memory_title": "Truthful ingest telemetry",
            }
        ]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_agentic_releases_unhandled_claimed_entries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        first_entry = runtime.journal.record(
            "The user also prefers deterministic fixtures for pytest.",
            workspace_id=runtime.workspace_id,
        )
        second_entry = runtime.journal.record(
            "The user wants me to make the change easy, then make the easy change.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-partial-task",
        )

        class _PartialAgenticProvider:
            async def run_agent(self, prompt: str) -> AgenticRunResult:
                batch_result = await call_internal_memory_tool(
                    runtime,
                    "internal_get_next_ingest_batch",
                    {"task_id": task.id, "batch_size": 10, "grouping_strategy": "fifo"},
                )
                batch_payload = json.loads(batch_result[0].text)
                claimed_entry_ids = batch_payload["claimed_entry_ids"]
                append_result = await call_internal_memory_tool(
                    runtime,
                    INGEST_APPEND_TOOL_NAME,
                    {
                        "memory_id": target.id,
                        "content": "Prefer deterministic fixtures for pytest.",
                        "task_id": task.id,
                        "entry_ids": [claimed_entry_ids[0]],
                        "workspace_ids": [runtime.workspace_id],
                        "tags": ["testing"],
                    },
                )
                append_payload = json.loads(append_result[0].text)
                return AgenticRunResult(
                    status="success",
                    summary="Agentic ingest appended only the handled claimed entry.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Agentic ingest appended only the handled claimed entry.",
                                "created_memory_ids": [append_payload["record"]["id"]],
                                "meaningful_actions": 1,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 2,
                                "byName": {
                                    "mcp_mcp-memory-internal_internal_get_next_ingest_batch": {"count": 1},
                                    f"mcp_mcp-memory-internal_{INGEST_APPEND_TOOL_NAME}": {"count": 1},
                                },
                            }
                        },
                    },
                )

        result = await handle_ingest_system1_task(runtime, task, _PartialAgenticProvider())
        pending_entries = runtime.journal.get_pending(workspace_id=runtime.workspace_id)

        assert result["claimed_entry_ids"] == [first_entry.id, second_entry.id]
        assert result["deleted_entry_ids"] == []
        assert result["recoverable_entry_ids"] == [first_entry.id]
        assert result["released_entry_ids"] == [second_entry.id]
        assert result["processed_entry_ids"] == [first_entry.id]
        assert result["entry_dispositions"] == [
            {
                "entry_id": first_entry.id,
                "disposition": "appended",
                "finalization_status": "recoverable",
                "memory_id": target.id,
                "memory_title": target.title,
            },
            {
                "entry_id": second_entry.id,
                "disposition": "released_unhandled",
                "finalization_status": "released",
            },
        ]
        assert result["touched_memory_ids"] == [target.id]
        assert result["appended_memory_ids"] == [target.id]
        assert result["matched_memory_ids"] == [target.id]
        assert [entry.id for entry in pending_entries] == [second_entry.id]
        assert runtime.journal.count_by_status() == {"pending": 1, "recoverable": 1}
        assert runtime.repository.get_memory(target.id) is not None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_skips_agentic_provider_for_low_novelty_routed_batches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None
    assert runtime.config is not None

    try:
        runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        runtime.journal.record(
            "Prefer pytest-based integration coverage.",
            workspace_id=runtime.workspace_id,
        )
        runtime.config.ingest_escalation.agentic_pending_count_threshold = 10
        runtime.config.ingest_escalation.novelty_threshold = 0.9

        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-json-fallback",
        )

        class _AgenticProvider:
            def __init__(self) -> None:
                self.run_agent_calls = 0
                self._budget_key = "copilot-mini"

            def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
                return self

            def supports_agentic(self) -> bool:
                return True

            async def run_agent(self, prompt: str) -> AgenticRunResult:
                self.run_agent_calls += 1
                return AgenticRunResult(status="success", summary="should not run")

        agentic_provider = _AgenticProvider()
        runtime.ai_provider_registry = {
            "copilot-mini": {
                "agentic": agentic_provider,
            }
        }

        result = await handle_ingest_system1_task(runtime, task, agentic_provider)

        assert result["meaningful_actions"] == 1
        assert agentic_provider.run_agent_calls == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_falls_back_when_agentic_provider_uses_no_tools_while_entries_are_pending(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        entry = runtime.journal.record(
            "This pending thought should not be silently ignored.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-no-tools",
        )

        class _NoToolAgenticProvider:
            async def run_agent(self, prompt: str) -> AgenticRunResult:
                return AgenticRunResult(
                    status="success",
                    summary="No action taken.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "No action taken.",
                                "created_memory_ids": [],
                                "meaningful_actions": 0,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 0,
                                "byName": {},
                            }
                        },
                    },
                )

        result = await handle_ingest_system1_task(runtime, task, _NoToolAgenticProvider())

        pending_entries = runtime.journal.get_pending(workspace_id=runtime.workspace_id)
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert result["created_memory_ids"]
        assert result["claimed_entry_ids"] == [entry.id]
        assert result["deleted_entry_ids"] == []
        assert result["recoverable_entry_ids"] == [entry.id]
        assert result["released_entry_ids"] == []
        assert result["meaningful_actions"] == 1
        assert result.get("execution_mode") is None
        assert pending_entries == []
        assert len(records) == 1
        assert records[0].metadata["ingest_task_id"] == task.id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_releases_claims_when_agentic_provider_raises_after_claiming_entries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None

    try:
        entry = runtime.journal.record(
            "This claimed thought should be released if the agentic path crashes.",
            workspace_id=runtime.workspace_id,
        )
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-agentic-exception-release",
        )

        class _ExplodingAgenticProvider:
            async def run_agent(self, prompt: str) -> AgenticRunResult:
                await call_internal_memory_tool(
                    runtime,
                    "internal_get_next_ingest_batch",
                    {"task_id": task.id, "batch_size": 10},
                )
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            await handle_ingest_system1_task(runtime, task, _ExplodingAgenticProvider())

        assert [pending.id for pending in runtime.journal.get_pending(workspace_id=runtime.workspace_id)] == [entry.id]
        assert runtime.journal.count_by_status() == {"pending": 1}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_skips_provider_when_no_pending_entries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None

    try:
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-no-pending",
        )

        provider = FakeAIProvider(responses=[{"summary": "should not be used"}])
        result = await handle_ingest_system1_task(runtime, task, provider)

        assert result["created_memory_ids"] == []
        assert result["claimed_entry_ids"] == []
        assert result["reason"] == "no_pending_entries"
        assert provider.call_count == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_append_accumulates_workspace_ids_across_runtimes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True, exist_ok=True)
    workspace_b.mkdir(parents=True, exist_ok=True)

    runtime_a = create_runtime(workspace_root_override=None, cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=None, cwd=workspace_b)
    assert runtime_a.repository is not None
    assert runtime_b.repository is not None
    assert runtime_b.journal is not None
    assert runtime_b.task_queue is not None

    try:
        target = runtime_a.repository.create_memory(
            title="Shared testing preferences",
            content="Prefer pytest integration coverage.",
            workspace_ids=[runtime_a.workspace_id or "workspace-a"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None

        runtime_b.journal.record(
            "The user applies the same testing preference in this workspace.",
            workspace_id=runtime_b.workspace_id,
        )
        task = runtime_b.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime_b.workspace_id,
            data={"workspace_id": runtime_b.workspace_id},
            available_at=0.0,
            task_id="cross-workspace-append",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer pytest integration coverage.",
                        }
                    ]
                }
            ]
        )

        result = await handle_ingest_system1_task(runtime_b, task, provider)
        updated = runtime_b.repository.get_memory(target.id)

        assert result["created_memory_ids"] == [target.id]
        assert updated is not None
        assert set(updated.workspace_ids) == {runtime_a.workspace_id, runtime_b.workspace_id}
    finally:
        runtime_a.close()
        runtime_b.close()


@pytest.mark.asyncio
async def test_ingest_handler_can_use_internal_tools_before_appending(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        target = runtime.repository.create_memory(
            title="User testing preferences",
            content="Prefer pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        assert target is not None
        runtime.journal.record("The user also prefers deterministic fixtures for pytest.")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-tool-loop-test",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_search_memory_records",
                            "arguments": {
                                "query": "pytest deterministic fixtures",
                                "workspace_id": runtime.workspace_id,
                                "limit": 5,
                            },
                        }
                    ]
                },
                {
                    "actions": [
                        {
                            "type": "append",
                            "entry_indices": [0],
                            "target_memory_id": target.id,
                            "content": "Prefer deterministic fixtures for pytest.",
                        }
                    ]
                },
            ]
        )

        result = await handle_ingest_system1_task(runtime, task, provider)
        updated = runtime.repository.get_memory(target.id)

        assert result["created_memory_ids"] == [target.id]
        assert result["deleted_entry_ids"] == []
        assert len(result["recoverable_entry_ids"]) == 1
        assert updated is not None
        assert "deterministic fixtures" in updated.content
        assert provider.call_count == 2
        assert "internal_search_memory_records" in provider.prompts[1]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ingest_handler_releases_claimed_entries_when_actions_are_ignore_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        entry = runtime.journal.record("ignore-only thought", workspace_id=runtime.workspace_id)
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="ingest-ignore-only",
        )

        provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "ignore",
                            "entry_indices": [0],
                        }
                    ]
                }
            ]
        )

        result = await handle_ingest_system1_task(runtime, task, provider)

        assert result["created_memory_ids"] == []
        assert result["deleted_entry_ids"] == []
        assert result["released_entry_ids"] == [entry.id]
        assert result["meaningful_actions"] == 0
        assert [pending.id for pending in runtime.journal.get_pending(workspace_id=runtime.workspace_id)] == [entry.id]
        assert runtime.repository.list_memories(workspace_id=runtime.workspace_id) == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_summarize_handler_uses_provider_and_falls_back(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None

    try:
        record = runtime.repository.create_memory(
            title="Queue summary",
            content="First sentence. Second sentence. Third sentence.",
            workspace_ids=[runtime.workspace_id or "global"],
            summary="stale",
        )
        assert record is not None
        task = runtime.task_queue.enqueue(
            SUMMARIZE_MEMORY_TASK_NAME,
            data={"memory_id": record.id},
            available_at=0.0,
            task_id=f"summary:{record.id}",
        )

        provider = FakeAIProvider(responses=[{"summary": "Provider-generated summary."}])
        provider_result = await handle_summarize_memory_task(runtime, task, provider)
        updated = runtime.repository.get_memory(record.id)
        assert provider_result["summary"] == "First sentence. Second sentence."
        assert updated is not None
        assert updated.summary == "First sentence. Second sentence."

        failing_provider = FakeAIProvider(error=RuntimeError("offline"))
        fallback_result = await handle_summarize_memory_task(runtime, task, failing_provider)
        assert fallback_result["summary"] == "First sentence. Second sentence."
    finally:
        runtime.close()


def test_bootstrap_background_tasks_is_idempotent(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)

    bootstrap_background_tasks(ctx)
    bootstrap_background_tasks(ctx)

    assert queue.count_by_status() == {"pending": 14}
    assert queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None) is not None
    assert queue.find_open_task(FACT_CHECKER_TASK_NAME, None) is not None
    assert queue.find_open_task(CURATOR_FRONTIER_TASK_NAME, None) is not None
    assert queue.find_open_task(GRAPH_LINK_DISCOVERY_TASK_NAME, None) is not None
    assert queue.find_open_task(GRAPH_LINKER_TASK_NAME, None) is not None
    assert queue.find_open_task(CONFLICT_SCREENING_TASK_NAME, None) is not None
    assert queue.find_open_task(CONFLICT_DETECTOR_TASK_NAME, None) is not None
    assert queue.find_open_task(DEDUP_PREP_TASK_NAME, None) is not None
    assert queue.find_open_task(TAG_NORMALIZER_TASK_NAME, None) is not None
    assert queue.find_open_task(DEFRAGMENTER_TASK_NAME, None) is not None
    assert queue.find_open_task(DEDUPLICATOR_TASK_NAME, None) is not None
    assert queue.find_open_task(TAXONOMIST_TASK_NAME, None) is not None
    assert queue.find_open_task(SWEEPER_TASK_NAME, None) is not None
    assert queue.find_open_task(CURATOR_TASK_NAME, None) is not None
    project_manager_task = queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None)
    deduplicator_task = queue.find_open_task(DEDUPLICATOR_TASK_NAME, None)
    assert project_manager_task is not None
    assert deduplicator_task is not None
    curator_task = queue.find_open_task(CURATOR_TASK_NAME, None)
    assert project_manager_task.priority == task_priority(PROJECT_MANAGER_TASK_NAME)
    assert deduplicator_task.priority == task_priority(DEDUPLICATOR_TASK_NAME)
    assert curator_task is not None
    assert curator_task.data["interval_seconds"] == RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME]
    assert RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME] == 3600.0


def test_bootstrap_background_tasks_enqueues_ingest_when_pending_thoughts_exist(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    journal = System1Journal(db_manager)
    journal.record("capture pending context", workspace_id="workspace-a")
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        task_queue=queue,
        journal=journal,
    )

    bootstrap_background_tasks(ctx)

    ingest_task = queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, None)
    assert ingest_task is not None
    assert ingest_task.data["trigger"] == "system1_debounce"
    assert ingest_task.available_at > ingest_task.created_at
    assert ingest_task.priority == SYSTEM1_INGEST_PRIORITY


def test_bootstrap_background_tasks_pulls_ingest_forward_at_threshold(db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    journal = System1Journal(db_manager)
    for index in range(40):
        journal.record(f"capture pending context {index}", workspace_id="workspace-a")
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        db_manager=db_manager,
        task_queue=queue,
        journal=journal,
    )

    bootstrap_background_tasks(ctx)

    ingest_task = queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, None)
    assert ingest_task is not None
    assert ingest_task.data["trigger"] == "system1_threshold"
    assert ingest_task.available_at <= ingest_task.updated_at
    assert ingest_task.priority == SYSTEM1_INGEST_PRIORITY


def test_bootstrap_background_tasks_respects_persistent_task_cadence(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 200.0)
    monkeypatch.setattr("mcp_memory.core.agent_runtime.compute_recurring_jitter_seconds", lambda interval_seconds: 0.0)

    task = queue.enqueue(
        PROJECT_MANAGER_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="project-manager-seed",
    )
    claimed = queue.claim_next(now=100.0)
    assert claimed is not None
    queue.complete(task.id, completed_at=120.0, run_result={"updated": 1})

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    bootstrap_background_tasks(ctx)

    scheduled = queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None)
    assert scheduled is not None
    assert scheduled.available_at == pytest.approx(120.0 + RECURRING_TASK_INTERVAL_SECONDS[PROJECT_MANAGER_TASK_NAME])


def test_bootstrap_background_tasks_uses_one_hour_curator_cadence(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 20000.0)
    monkeypatch.setattr("mcp_memory.core.agent_runtime.compute_recurring_jitter_seconds", lambda interval_seconds: 0.0)

    task = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="curator-seed",
    )
    claimed = queue.claim_next(now=100.0)
    assert claimed is not None
    queue.complete(task.id, completed_at=1200.0, run_result={"mutations": 1})

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    bootstrap_background_tasks(ctx)

    scheduled = queue.find_open_task(CURATOR_TASK_NAME, None)
    assert scheduled is not None
    assert scheduled.available_at == pytest.approx(max(1200.0 + RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME], 20000.0))
    assert scheduled.data["interval_seconds"] == 3600.0


def test_bootstrap_background_tasks_refreshes_stale_recurring_cadence(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 1500.0)
    monkeypatch.setattr("mcp_memory.core.agent_runtime.compute_recurring_jitter_seconds", lambda interval_seconds: 0.0)

    completed = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="curator-completed-seed",
    )
    claimed = queue.claim_next(now=100.0)
    assert claimed is not None
    queue.complete(completed.id, completed_at=1200.0, run_result={"mutations": 1})

    stale = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        available_at=22800.0,
        data={
            "workspace_id": None,
            "trigger": "recurring_follow_up",
            "interval_seconds": 21600.0,
        },
        task_id="curator-stale-recurring",
    )

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    bootstrap_background_tasks(ctx)

    refreshed = queue.get_task(stale.id)
    assert refreshed.status == "pending"
    assert refreshed.data["interval_seconds"] == RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME]
    assert refreshed.available_at == pytest.approx(1200.0 + RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME])


def test_bootstrap_background_tasks_applies_jitter_to_new_recurring_tasks(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 200.0)
    monkeypatch.setattr("mcp_memory.core.agent_runtime.compute_recurring_jitter_seconds", lambda interval_seconds: 30.0)

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue)
    bootstrap_background_tasks(ctx)

    scheduled = queue.find_open_task(PROJECT_MANAGER_TASK_NAME, None)
    assert scheduled is not None
    assert scheduled.available_at == pytest.approx(230.0)
    assert scheduled.data["trigger"] == "recurring_schedule"
    assert scheduled.data["jitter_seconds"] == pytest.approx(30.0)


def test_bootstrap_background_tasks_preserves_idle_paused_recurring_maintenance(monkeypatch, db_manager) -> None:
    queue = SQLiteTaskQueue(db_manager)
    from mcp_memory.context import ApplicationContext

    journal = System1Journal(db_manager)
    monkeypatch.setattr("mcp_memory.core.journal.time.time", lambda: 100.0)
    journal.record("old thought", workspace_id="workspace-a")

    paused = queue.enqueue(
        CURATOR_TASK_NAME,
        workspace_id=None,
        available_at=0.0,
        data={
            "workspace_id": None,
            "trigger": "recurring_follow_up",
            "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME],
        },
        task_id="paused-curator-bootstrap",
    )
    assert queue.claim_next(now=200.0) is not None
    queue.complete(
        paused.id,
        completed_at=201.0,
        run_result={
            "paused_for_idle": True,
            "idle_seconds": 4000.0,
            "last_thought_at": 100.0,
        },
    )

    monkeypatch.setattr("mcp_memory.core.agent_runtime.time.time", lambda: 5000.0)
    monkeypatch.setattr("mcp_memory.core.maintenance_idle.time.time", lambda: 5000.0)
    monkeypatch.setattr("mcp_memory.core.agent_runtime.compute_recurring_jitter_seconds", lambda interval_seconds: 0.0)

    ctx = ApplicationContext(workspace_id="workspace-a", db_manager=db_manager, task_queue=queue, journal=journal)

    bootstrap_background_tasks(ctx)

    assert queue.find_open_task(CURATOR_TASK_NAME, None) is None


def test_project_manager_only_stales_old_active_plans_in_task_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None

    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        current_workspace_plan = runtime.repository.create_memory(
            title="Old workspace plan",
            content="This plan should go stale.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        other_workspace_plan = runtime.repository.create_memory(
            title="Old other workspace plan",
            content="This plan belongs elsewhere.",
            workspace_ids=["workspace-other"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        fresh_workspace_plan = runtime.repository.create_memory(
            title="Fresh workspace plan",
            content="This plan should stay active.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
        )
        assert current_workspace_plan is not None
        assert other_workspace_plan is not None
        assert fresh_workspace_plan is not None

        result = handle_project_manager_task(
            runtime,
            TaskRecord(
                id="project-manager-workspace-scope",
                task_name=PROJECT_MANAGER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        current_workspace_plan_record = runtime.repository.get_memory(current_workspace_plan.id)
        other_workspace_plan_record = runtime.repository.get_memory(other_workspace_plan.id)
        fresh_workspace_plan_record = runtime.repository.get_memory(fresh_workspace_plan.id)

        assert result == {"updated": 1}
        assert current_workspace_plan_record is not None
        assert other_workspace_plan_record is not None
        assert fresh_workspace_plan_record is not None
        assert current_workspace_plan_record.status == "stale"
        assert other_workspace_plan_record.status == "active"
        assert fresh_workspace_plan_record.status == "active"
    finally:
        runtime.close()


def test_fact_checker_restores_degraded_links_when_file_reappears(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None

    try:
        memory = runtime.repository.create_memory(
            title="Repairable ext link",
            content="Tracks a file that will return.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        assert memory is not None
        runtime.repository.add_link(memory.id, "ext:docs/plan.md", "REFERENCES")

        first = handle_fact_checker_task(
            runtime,
            TaskRecord(
                id="fact-checker-first-pass",
                task_name=FACT_CHECKER_TASK_NAME,
                data={
                    "workspace_id": runtime.workspace_id,
                    "workspace_root": str(workspace),
                },
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        repaired_file = workspace / "docs" / "plan.md"
        repaired_file.parent.mkdir(parents=True, exist_ok=True)
        repaired_file.write_text("restored", encoding="utf-8")

        second = handle_fact_checker_task(
            runtime,
            TaskRecord(
                id="fact-checker-second-pass",
                task_name=FACT_CHECKER_TASK_NAME,
                data={
                    "workspace_id": runtime.workspace_id,
                    "workspace_root": str(workspace),
                },
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        refreshed_memory = runtime.repository.get_memory(memory.id)

        assert first == {"degraded": 1, "restored": 0}
        assert second == {"degraded": 0, "restored": 1}
        assert refreshed_memory is not None
        assert refreshed_memory.status == "active"
    finally:
        runtime.close()


def test_sweeper_preserves_unexpired_recoverable_entries_and_is_idempotent(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.db_manager is not None
    assert runtime.journal is not None

    class _VectorStoreSpy:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str, str | None]] = []

        def delete(self, *, source_kind: str, source_id: str, model_name: str | None) -> None:
            self.deleted.append((source_kind, source_id, model_name))

    runtime.vector_store = _VectorStoreSpy()
    runtime.embedder = None

    try:
        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=8)).timestamp()
        future_recoverable_until = (datetime.now(UTC) + timedelta(days=1)).timestamp()
        conn = runtime.db_manager.get_connection()
        conn.execute(
            "INSERT INTO tasks (id, task_name, workspace_id, data, status, priority, retries_count, max_retries, created_at, updated_at, available_at, completed_at, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-sweeper-completed-task",
                "sweeper",
                runtime.workspace_id,
                "{}",
                "completed",
                100,
                0,
                3,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                None,
            ),
        )
        conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (?, ?, ?, ?)",
            ("processed note", runtime.workspace_id, cutoff_timestamp, "processed"),
        )
        expired_id = conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status, recoverable_until) VALUES (?, ?, ?, ?, ?)",
            ("expired recoverable note", runtime.workspace_id, cutoff_timestamp, "recoverable", cutoff_timestamp),
        ).lastrowid
        preserved_id = conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status, recoverable_until) VALUES (?, ?, ?, ?, ?)",
            (
                "still recoverable note",
                runtime.workspace_id,
                cutoff_timestamp,
                "recoverable",
                future_recoverable_until,
            ),
        ).lastrowid
        conn.commit()

        task = TaskRecord(
            id="sweeper-idempotence-task",
            task_name=SWEEPER_TASK_NAME,
            data={"workspace_id": runtime.workspace_id},
            workspace_id=runtime.workspace_id,
            status="running",
            priority=100,
            retries_count=0,
            max_retries=3,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=0.0,
            started_at=0.0,
            completed_at=None,
            last_error=None,
        )

        first = handle_sweeper_task(runtime, task)
        second = handle_sweeper_task(runtime, task)

        remaining_recoverable = conn.execute(
            "SELECT id FROM system1_journal WHERE status = 'recoverable' ORDER BY id ASC"
        ).fetchall()

        assert first == {"deleted_tasks": 1, "deleted_journal_entries": 2}
        assert second == {"deleted_tasks": 0, "deleted_journal_entries": 0}
        assert [row[0] for row in remaining_recoverable] == [preserved_id]
        assert runtime.journal.count_by_status() == {"recoverable": 1}
        assert runtime.vector_store.deleted == [("thought", str(expired_id), None)]
    finally:
        runtime.close()


def test_project_manager_fact_checker_and_sweeper_tasks_update_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None
    assert runtime.task_queue is not None
    assert runtime.db_manager is not None

    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        stale_plan = runtime.repository.create_memory(
            title="Old plan",
            content="This plan is stale.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        healthy_memory = runtime.repository.create_memory(
            title="Healthy ext link",
            content="Tracks a valid file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        broken_memory = runtime.repository.create_memory(
            title="Broken ext link",
            content="Tracks a missing file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        assert stale_plan is not None
        assert healthy_memory is not None
        assert broken_memory is not None
        valid_file = workspace / "README.md"
        valid_file.write_text("ok", encoding="utf-8")
        runtime.repository.add_link(healthy_memory.id, f"ext:{valid_file.name}", "REFERENCES")
        runtime.repository.add_link(broken_memory.id, "ext:missing.txt", "REFERENCES")

        conn = runtime.db_manager.get_connection()
        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=8)).timestamp()
        conn.execute(
            "INSERT INTO tasks (id, task_name, workspace_id, data, status, priority, retries_count, max_retries, created_at, updated_at, available_at, completed_at, last_error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-completed-task",
                "sweeper",
                runtime.workspace_id,
                "{}",
                "completed",
                100,
                0,
                3,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                cutoff_timestamp,
                None,
            ),
        )
        conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (?, ?, ?, ?)",
            ("processed note", runtime.workspace_id, cutoff_timestamp, "processed"),
        )
        conn.execute(
            "INSERT INTO system1_journal (content, workspace_id, timestamp, status, recoverable_until) VALUES (?, ?, ?, ?, ?)",
            ("recoverable note", runtime.workspace_id, cutoff_timestamp, "recoverable", cutoff_timestamp),
        )
        conn.commit()

        project_result = handle_project_manager_task(
            runtime,
            TaskRecord(
                id="project-manager-task",
                task_name=PROJECT_MANAGER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        fact_result = handle_fact_checker_task(
            runtime,
            TaskRecord(
                id="fact-checker-task",
                task_name=FACT_CHECKER_TASK_NAME,
                data={
                    "workspace_id": runtime.workspace_id,
                    "workspace_root": str(workspace),
                },
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        sweep_result = handle_sweeper_task(
            runtime,
            TaskRecord(
                id="sweeper-task",
                task_name=SWEEPER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        assert project_result["updated"] == 1
        assert fact_result["degraded"] == 1
        assert sweep_result["deleted_tasks"] == 1
        assert sweep_result["deleted_journal_entries"] == 2
        refreshed_stale = runtime.repository.get_memory(stale_plan.id)
        refreshed_healthy = runtime.repository.get_memory(healthy_memory.id)
        refreshed_broken = runtime.repository.get_memory(broken_memory.id)
        assert refreshed_stale is not None and refreshed_stale.status == "stale"
        assert refreshed_healthy is not None and refreshed_healthy.status == "active"
        assert refreshed_broken is not None and refreshed_broken.status == "degraded"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_task_worker_periodically_recovers_dead_subprocess_tasks(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    task = queue.enqueue(
        SYSTEM1_INGEST_TASK_NAME,
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="midrun-dead-provider-task",
    )
    assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None
    queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-midrun-dead", updated_at=2.0)
    repository.record_conversation(
        request_id="req-midrun-dead",
        attempt=1,
        task_name=SYSTEM1_INGEST_TASK_NAME,
        task_id=task.id,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=9999,
        prompt_text="ingest pending thoughts",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=1.0,
        completed_at=1.0,
    )

    calls = 0
    original = SQLiteTaskQueue.recover_abandoned_running_tasks

    def recover(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        kwargs.pop("stale_after_seconds", None)
        kwargs.pop("now", None)
        return original(self, stale_after_seconds=0.0, now=5.0)

    monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)
    monkeypatch.setattr(SQLiteTaskQueue, "recover_abandoned_running_tasks", recover)
    worker = RuntimeTaskWorker(
        ctx,
        handlers={SYSTEM1_INGEST_TASK_NAME: lambda context, queued_task: None},
        poll_interval_seconds=0.01,
        abandoned_recovery_interval_seconds=0.01,
        abandoned_task_stale_after_seconds=0.0,
    )

    await worker.start()
    try:
        for _ in range(30):
            if queue.get_task(task.id).status == "failed":
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.stop(0.05)

    failed = queue.get_task(task.id)
    conversation = repository.get_conversation("req-midrun-dead")[0]

    assert failed.status == "failed"
    assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"
    assert conversation.status == "error"
    assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"


@pytest.mark.asyncio
async def test_runtime_task_worker_survives_unexpected_post_claim_errors(db_manager, monkeypatch) -> None:
    queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(db_manager=db_manager, task_queue=queue, workspace_id="workspace-a")
    first_task = queue.enqueue(
        "test-runtime-worker-resilience",
        workspace_id="workspace-a",
        max_retries=1,
        available_at=0.0,
        task_id="runtime-worker-resilience-1",
    )
    second_task = queue.enqueue(
        "test-runtime-worker-resilience",
        workspace_id="workspace-a",
        available_at=0.0,
        task_id="runtime-worker-resilience-2",
    )

    original_complete = queue.complete
    calls = 0

    def flaky_complete(task_id: str, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic completion failure")
        return original_complete(task_id, *args, **kwargs)

    monkeypatch.setattr(queue, "complete", flaky_complete)

    worker = RuntimeTaskWorker(
        ctx,
        handlers={"test-runtime-worker-resilience": lambda context, queued_task: {"task_id": queued_task.id}},
        poll_interval_seconds=0.01,
        abandoned_recovery_interval_seconds=60.0,
    )

    await worker.start()
    try:
        for _ in range(100):
            first = queue.get_task(first_task.id)
            second = queue.get_task(second_task.id)
            if first.status == "failed" and second.status == "completed":
                break
            await asyncio.sleep(0.01)
    finally:
        await worker.stop(0.05)

    refreshed_first = queue.get_task(first_task.id)
    refreshed_second = queue.get_task(second_task.id)

    assert refreshed_first.status == "failed"
    assert refreshed_first.last_error == "Unhandled runtime task worker error: synthetic completion failure"
    assert refreshed_second.status == "completed"


@pytest.mark.asyncio
async def test_runtime_task_worker_releases_claimed_work_items_when_task_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.task_queue is not None
    assert runtime.work_items is not None

    try:
        work_item, created = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": ["seed-memory"], "workspace_id": runtime.workspace_id},
            priority=100,
            idempotency_key="test:runtime-worker-release-work-item",
        )
        assert created is True

        task = runtime.task_queue.enqueue(
            "test-work-item-release",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
            task_id="test-work-item-release-task",
        )

        def failing_handler(context, queued_task):
            claimed = context.work_items.claim_batch(
                family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
                execution_lane=EXECUTION_LANE_AGENTIC,
                lease_owner=queued_task.id,
                limit=1,
                workspace_id=queued_task.workspace_id,
            )
            assert len(claimed) == 1
            raise RuntimeError("synthetic handler crash")

        worker = RuntimeTaskWorker(
            runtime,
            handlers={"test-work-item-release": failing_handler},
            poll_interval_seconds=0.01,
            abandoned_recovery_interval_seconds=60.0,
        )

        await worker.start()
        try:
            for _ in range(100):
                refreshed_task = runtime.task_queue.get_task(task.id)
                refreshed_item = runtime.work_items.get_item(work_item.id)
                if refreshed_task.status == "failed" and refreshed_item.status == "pending":
                    break
                await asyncio.sleep(0.01)
        finally:
            await worker.stop(0.05)

        refreshed_task = runtime.task_queue.get_task(task.id)
        refreshed_item = runtime.work_items.get_item(work_item.id)

        assert refreshed_task.status == "failed"
        assert refreshed_task.last_error == "synthetic handler crash"
        assert refreshed_item.status == "pending"
        assert refreshed_item.lease_owner is None
        assert refreshed_item.attempt_count >= 1
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_task_worker_recovers_work_items_owned_by_terminal_tasks_on_startup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.task_queue is not None
    assert runtime.work_items is not None

    try:
        task = runtime.task_queue.enqueue(
            "test-startup-work-item-recovery",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
            task_id="test-startup-work-item-recovery-task",
        )
        claimed_task = runtime.task_queue.claim_next(now=1.0)
        assert claimed_task is not None

        work_item, created = runtime.work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            workspace_id=runtime.workspace_id,
            payload={"seed_memory_ids": ["seed-memory"], "workspace_id": runtime.workspace_id},
            priority=100,
            idempotency_key="test:runtime-worker-startup-recover-work-item",
        )
        assert created is True

        claimed_items = runtime.work_items.claim_batch(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            lease_owner=task.id,
            limit=1,
            workspace_id=runtime.workspace_id,
        )
        assert len(claimed_items) == 1

        runtime.task_queue.fail_permanently(task.id, "synthetic terminal task")

        worker = RuntimeTaskWorker(
            runtime,
            handlers={"test-startup-work-item-recovery": lambda context, queued_task: None},
            poll_interval_seconds=0.01,
            abandoned_recovery_interval_seconds=60.0,
        )

        await worker.start()
        try:
            for _ in range(100):
                refreshed_item = runtime.work_items.get_item(work_item.id)
                if refreshed_item.status == "pending":
                    break
                await asyncio.sleep(0.01)
        finally:
            await worker.stop(0.05)

        refreshed_item = runtime.work_items.get_item(work_item.id)
        assert refreshed_item.status == "pending"
        assert refreshed_item.lease_owner is None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_and_conflict_detector_create_links(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        authority_fact = runtime.repository.create_memory(
            title="Auth contract",
            content="Bearer tokens are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        related_plan = runtime.repository.create_memory(
            title="Auth rollout",
            content="Ship the auth contract to all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "rollout"],
        )
        conflicting_fact = runtime.repository.create_memory(
            title="Auth contract",
            content="Bearer tokens are optional.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        assert authority_fact is not None and related_plan is not None and conflicting_fact is not None

        graph_result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        conflict_result = await handle_conflict_detector_task(
            runtime,
            TaskRecord(
                id="conflict-detector-task",
                task_name=CONFLICT_DETECTOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        outgoing_from_plan = runtime.repository.get_links(related_plan.id, direction="outgoing")
        outgoing_from_fact = runtime.repository.get_links(authority_fact.id, direction="outgoing")
        incoming_to_fact = runtime.repository.get_links(authority_fact.id, direction="incoming")

        assert graph_result["created"] >= 1
        assert any(link.target_id == authority_fact.id for link in outgoing_from_plan)
        assert conflict_result["created"] == 2
        assert any(link.link_type == "CONTRADICTS" for link in outgoing_from_fact)
        assert any(link.link_type == "CONTRADICTS" for link in incoming_to_fact)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_skips_provider_when_fallback_is_sufficient(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Auth rollout plan",
            content="Roll out auth across services.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "rollout"],
        )
        second = runtime.repository.create_memory(
            title="Auth api contract",
            content="Document the auth api contract.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "api"],
        )
        third = runtime.repository.create_memory(
            title="Auth frontend tasks",
            content="Update the auth frontend screens.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            tags=["auth", "ui"],
        )
        assert first is not None and second is not None and third is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": first.id,
                            "target_id": third.id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider result",
                        }
                    ]
                }
            ]
        )

        result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-fallback-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["created"] >= 2
        assert provider.call_count == 0
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_link_discovery_seeds_agentic_review_when_fallback_is_sparse(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        for index in range(13):
            record = runtime.repository.create_memory(
                title=f"isolated-topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None

        result = await handle_graph_link_discovery_task(
            runtime,
            TaskRecord(
                id="graph-link-discovery-task",
                task_name=GRAPH_LINK_DISCOVERY_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        review_items = runtime.work_items.list_items(family_key="graph_link_review", limit=5)

        assert result["created"] == 0
        assert result["seeded_work_item_count"] == 1
        assert result["work_item_family"] == "graph_link_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "frontier_seed"
        assert result["seed_record_count"] == 13
        assert [item.status for item in review_items] == ["pending"]
        assert result["created_work_item_id"] == review_items[0].id
        assert len(review_items[0].payload["candidate_memory_ids"]) == 13
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_consumes_seeded_review_work_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        records = []
        for index in range(13):
            record = runtime.repository.create_memory(
                title=f"isolated-topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None
            records.append(record)

        discovery_result = await handle_graph_link_discovery_task(
            runtime,
            TaskRecord(
                id="graph-link-discovery-review-seed",
                task_name=GRAPH_LINK_DISCOVERY_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        assert discovery_result["seeded_work_item_count"] == 1

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": records[0].id,
                            "target_id": records[1].id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider detected dependency",
                        }
                    ]
                }
            ]
        )

        result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-review-consumer",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        links = runtime.repository.get_links(records[0].id, direction="outgoing")
        review_items = runtime.work_items.list_items(family_key="graph_link_review", limit=5)

        assert result["created"] == 1
        assert result["claimed_work_item_count"] == 1
        assert result["execution_mode"] == "agentic_review"
        assert result["work_item_family"] == "graph_link_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "claimed_review_work_item"
        assert result["seed_record_count"] == 13
        assert result["claimed_work_item_id"] == review_items[0].id
        assert provider.call_count == 1
        assert len(links) == 1
        assert links[0].target_id == records[1].id
        assert [item.status for item in review_items] == ["completed"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_escalates_to_provider_with_internal_tool_prompt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        records = []
        for index in range(13):
            record = runtime.repository.create_memory(
                title=f"topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None
            records.append(record)

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": records[0].id,
                            "target_id": records[1].id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider detected dependency",
                        }
                    ]
                }
            ]
        )

        result = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-ai-tool-loop-task",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["created"] == 1
        assert provider.call_count == 1
        assert provider.prompts
        assert "Available internal tools:" in provider.prompts[0]
        assert "internal_search_memory_records" in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert "Use DEPENDS_ON when one memory relies on" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_graph_linker_treats_variant_link_type_spellings_as_existing_links(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        records = []
        for index in range(13):
            record = runtime.repository.create_memory(
                title=f"topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None
            records.append(record)

        provider = FakeAIProvider(
            responses=[
                {
                    "links": [
                        {
                            "source_id": records[0].id,
                            "target_id": records[1].id,
                            "link_type": "DEPENDS_ON",
                            "context": "provider detected dependency",
                        }
                    ]
                },
                {
                    "links": [
                        {
                            "source_id": records[0].id,
                            "target_id": records[1].id,
                            "link_type": "depends-on",
                            "context": "provider restated dependency with different casing",
                        }
                    ]
                },
            ]
        )

        first = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-variant-link-type-task-1",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )
        second = await handle_graph_linker_task(
            runtime,
            TaskRecord(
                id="graph-linker-variant-link-type-task-2",
                task_name=GRAPH_LINKER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        links = runtime.repository.get_links(records[0].id, direction="outgoing")

        assert first["created"] == 1
        assert second["created"] == 0
        assert len(links) == 1
        assert links[0].link_type == "DEPENDS_ON"
        assert links[0].context == "provider detected dependency"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_conflict_screening_seeds_agentic_review_when_fallback_is_sparse(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        for index in range(16):
            record = runtime.repository.create_memory(
                title=f"isolated-conflict-topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None

        result = await handle_conflict_screening_task(
            runtime,
            TaskRecord(
                id="conflict-screening-task",
                task_name=CONFLICT_SCREENING_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        review_items = runtime.work_items.list_items(family_key="conflict_review", limit=5)

        assert result["created"] == 0
        assert result["seeded_work_item_count"] == 1
        assert result["work_item_family"] == "conflict_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "frontier_seed"
        assert result["seed_record_count"] == 16
        assert [item.status for item in review_items] == ["pending"]
        assert result["created_work_item_id"] == review_items[0].id
        assert len(review_items[0].payload["candidate_memory_ids"]) == 16
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_conflict_detector_consumes_seeded_review_work_items(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        records = []
        for index in range(16):
            record = runtime.repository.create_memory(
                title=f"isolated-conflict-topic-{index}",
                content=f"body-{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=[f"tag-{index}"],
            )
            assert record is not None
            records.append(record)

        screening_result = await handle_conflict_screening_task(
            runtime,
            TaskRecord(
                id="conflict-screening-review-seed",
                task_name=CONFLICT_SCREENING_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        assert screening_result["seeded_work_item_count"] == 1

        provider = FakeAIProvider(
            responses=[
                {
                    "conflicts": [
                        {
                            "left_id": records[0].id,
                            "right_id": records[1].id,
                            "context": "provider detected semantic conflict",
                        }
                    ]
                }
            ]
        )

        result = await handle_conflict_detector_task(
            runtime,
            TaskRecord(
                id="conflict-detector-review-consumer",
                task_name=CONFLICT_DETECTOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        outgoing = runtime.repository.get_links(records[0].id, direction="outgoing")
        incoming = runtime.repository.get_links(records[1].id, direction="incoming")
        review_items = runtime.work_items.list_items(family_key="conflict_review", limit=5)

        assert result["created"] == 2
        assert result["claimed_work_item_count"] == 1
        assert result["execution_mode"] == "agentic_review"
        assert result["work_item_family"] == "conflict_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "claimed_review_work_item"
        assert result["seed_record_count"] == 16
        assert result["claimed_work_item_id"] == review_items[0].id
        assert provider.call_count == 1
        assert any(link.target_id == records[1].id and link.link_type == "CONTRADICTS" for link in outgoing)
        assert any(link.source_id == records[0].id and link.link_type == "CONTRADICTS" for link in incoming)
        assert [item.status for item in review_items] == ["completed"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_conflict_detector_escalates_to_provider_when_fallback_is_sparse(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        records = []
        for index in range(16):
            record = runtime.repository.create_memory(
                    title=f"topic{index}",
                    content=f"body{index}",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                    tags=[f"tag{index}"],
            )
            assert record is not None
            records.append(record)

        provider = FakeAIProvider(
            responses=[
                {
                    "conflicts": [
                        {
                            "left_id": records[0].id,
                            "right_id": records[1].id,
                            "context": "provider detected semantic conflict",
                        }
                    ]
                }
            ]
        )

        result = await handle_conflict_detector_task(
            runtime,
            TaskRecord(
                id="conflict-detector-ai-task",
                task_name=CONFLICT_DETECTOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "conflict-frontier"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["created"] == 2
        assert result["strategy_used"] == "conflict-frontier"
        assert provider.call_count == 1
        assert provider.prompts
        assert "Available internal tools:" in provider.prompts[0]
        assert "internal_search_memory_records" in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert "materially incompatible claims" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_defragmenter_and_tag_normalizer_update_memory_state(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Auth journal 1",
            content="Captured auth rollout note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["unit_test", "auth"],
        )
        second = runtime.repository.create_memory(
            title="Auth journal 2",
            content="Captured auth rollout follow-up.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["tests", "auth"],
        )
        stable = runtime.repository.create_memory(
            title="Stable fact",
            content="Keep this as-is.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["Auth"],
        )
        assert first is not None and second is not None and stable is not None

        defrag_result = await handle_defragmenter_task(
            runtime,
            TaskRecord(
                id="defragmenter-task",
                task_name=DEFRAGMENTER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "cold-storage"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        tax_result = await handle_tag_normalizer_task(
            runtime,
            TaskRecord(
            id="tag-normalizer-task",
            task_name=TAG_NORMALIZER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "never-surfaced"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        reflections = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="reflection",
            limit=10,
        )
        updated_first = runtime.repository.get_memory(first.id)
        updated_second = runtime.repository.get_memory(second.id)
        updated_stable = runtime.repository.get_memory(stable.id)

        assert defrag_result["created"] == 1
        assert defrag_result["archived"] == 2
        assert defrag_result["strategy_used"] == "cold-storage"
        assert reflections
        source_memory_ids = reflections[0].metadata["source_memory_ids"]
        assert isinstance(source_memory_ids, list)
        assert set(source_memory_ids) == {first.id, second.id}
        assert updated_first is not None and updated_first.status == "archived"
        assert updated_second is not None and updated_second.status == "archived"
        assert tax_result["updated"] >= 1
        assert tax_result["strategy_used"] == "never-surfaced"
        assert updated_stable is not None and updated_stable.tags == ["auth"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_defragmenter_skips_provider_for_small_groups(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="Short note one",
            content="Brief auth note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        second = runtime.repository.create_memory(
            title="Short note two",
            content="Another brief auth note.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth"],
        )
        assert first is not None and second is not None

        provider = FakeAIProvider(responses=[{"title": "Provider title", "content": "Provider content"}])
        result = await handle_defragmenter_task(
            runtime,
            TaskRecord(
                id="defragmenter-small-task",
                task_name=DEFRAGMENTER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        reflections = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="reflection",
            limit=10,
        )
        assert result["created"] == 1
        assert provider.call_count == 0
        assert reflections[0].title in {"Reflection: Short note one", "Reflection: Short note two"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_defragmenter_does_not_group_records_from_generic_system_tags_alone(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="CLI cleanup note",
            content="Add a visible log command group for the CLI.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auto-ingested", "system1"],
        )
        second = runtime.repository.create_memory(
            title="Task queue note",
            content="SQLiteTaskQueue claim_next is currently global and not workspace-scoped.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auto-ingested", "system1"],
        )
        assert first is not None and second is not None

        result = await handle_defragmenter_task(
            runtime,
            TaskRecord(
                id="defragmenter-generic-tags-task",
                task_name=DEFRAGMENTER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        reflections = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="reflection",
            limit=10,
        )

        assert result["created"] == 0
        assert result["archived"] == 0
        assert reflections == []
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_tag_normalizer_skips_provider_and_seeds_enrichment_for_untagged_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        record = runtime.repository.create_memory(
            title="Tagged fact",
            content="A fact with messy tags.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["Tests", "Authn"],
        )
        untagged = runtime.repository.create_memory(
            title="Untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert record is not None and untagged is not None

        result = await handle_tag_normalizer_task(
            runtime,
            TaskRecord(
                id="tag-normalizer-deterministic-task",
                task_name=TAG_NORMALIZER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        updated = runtime.repository.get_memory(record.id)
        assert result["updated"] == 1
        assert updated is not None and updated.tags == ["auth", "testing"]
        assert result["seeded_enrichment_count"] == 1
        normalization_items = runtime.work_items.list_items(family_key="memory_tag_normalization", limit=5)
        enrichment_items = runtime.work_items.list_items(family_key="memory_tagging", limit=5)
        assert [item.status for item in normalization_items] == ["completed"]
        assert [item.status for item in enrichment_items] == ["pending"]
        assert [item.payload for item in enrichment_items] == [
            {"memory_id": untagged.id, "workspace_id": runtime.workspace_id}
        ]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_uses_provider_for_untagged_records(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"tags": ["Authn", "Tests"]}])
        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-provider-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated = runtime.repository.get_memory(record.id)
        assert result["updated"] == 1
        assert provider.call_count == 1
        assert result["claimed_work_item_count"] == 1
        assert updated is not None and updated.tags == ["auth", "testing"]
        assert runtime.work_items is not None
        work_items = runtime.work_items.list_items(family_key="memory_tagging", limit=5)
        assert [item.status for item in work_items] == ["completed"]
        assert work_items[0].payload == {"memory_id": record.id, "workspace_id": runtime.workspace_id}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_agentic_provider_uses_work_item_lifecycle_tools(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Agentic untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert record is not None

        class _AgenticProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def supports_agentic(self) -> bool:
                return True

            async def run_agent(self, prompt: str) -> AgenticRunResult:
                self.prompts.append(prompt)
                batch_result = await call_internal_memory_tool(
                    runtime,
                    "internal_get_work_batch",
                    {
                        "task_id": "taxonomist-agentic-task",
                        "family_key": "memory_tagging",
                        "execution_lane": "agentic",
                        "workspace_id": runtime.workspace_id,
                        "limit": 1,
                    },
                )
                batch_payload = json.loads(batch_result[0].text)
                work_item = batch_payload["records"][0]
                await call_internal_memory_tool(
                    runtime,
                    "internal_read_memory_record",
                    {"memory_id": record.id},
                )
                await call_internal_memory_tool(
                    runtime,
                    "internal_update_memory_record",
                    {
                        "memory_id": record.id,
                        "tags": ["auth", "testing"],
                    },
                )
                await call_internal_memory_tool(
                    runtime,
                    "internal_complete_work_item",
                    {"work_item_id": work_item["id"]},
                )
                return AgenticRunResult(
                    status="success",
                    summary="Tagged one record through the work-item lifecycle tools.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Tagged one record through the work-item lifecycle tools.",
                                "updated_memory_ids": [record.id],
                            }
                        )
                    },
                )

        provider = _AgenticProvider()
        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-agentic-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated = runtime.repository.get_memory(record.id)
        assert result["updated"] == 1
        assert result["execution_mode"] == "agentic_mcp"
        assert result["provider_calls_used"] == 1
        assert result["claimed_work_item_count"] == 1
        assert result["summary"] == "Tagged one record through the work-item lifecycle tools."
        assert updated is not None and updated.tags == ["auth", "testing"]
        assert runtime.work_items is not None
        work_items = runtime.work_items.list_items(family_key="memory_tagging", limit=5)
        assert [item.status for item in work_items] == ["completed"]
        assert provider.prompts
        assert "internal_get_work_batch" in provider.prompts[0]
        assert "internal_complete_work_item" in provider.prompts[0]
        assert "internal_update_memory_record" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_limits_provider_calls_per_run_to_burst_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="First untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        second = runtime.repository.create_memory(
            title="Second untagged fact",
            content="A second auth fact needing tags.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert first is not None and second is not None

        provider = FakeAIProvider(responses=[{"tags": ["Authn", "Tests"]}, {"tags": ["Architecture"]}])
        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-provider-budget-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated_first = runtime.repository.get_memory(first.id)
        updated_second = runtime.repository.get_memory(second.id)
        assert result["updated"] == 1
        assert result["provider_calls_used"] == 1
        assert result["provider_call_budget"] == 1
        assert result["claimed_work_item_count"] == 1
        assert provider.call_count == 1
        assert updated_first is not None and updated_first.tags == ["auth", "testing"]
        assert updated_second is not None and updated_second.tags == []
        assert runtime.work_items is not None
        work_items = runtime.work_items.list_items(family_key="memory_tagging", limit=5)
        assert len(work_items) == 2
        assert {item.payload["memory_id"] for item in work_items} == {first.id, second.id}
        assert {item.status for item in work_items} == {"completed", "pending"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_taxonomist_soft_fails_provider_rate_limit_and_keeps_task_successful(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    try:
        tagged = runtime.repository.create_memory(
            title="Tagged fact",
            content="A fact with messy tags.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "testing"],
        )
        untagged = runtime.repository.create_memory(
            title="Untagged fact",
            content="JWT auth requirement for tests.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=[],
        )
        assert tagged is not None and untagged is not None

        provider = FakeAIProvider(
            responses=[{"tags": ["ignored"]}],
            error=ProviderRateLimitExceeded(
                "gemini-3-flash-preview",
                calls_in_window=1,
                burst_call_limit=1,
                burst_window_seconds=600.0,
                retry_delay_seconds=600.0,
            ),
        )
        provider._provider_key = "gemini-cli"  # type: ignore[attr-defined]
        provider._provider_name = "Gemini CLI"  # type: ignore[attr-defined]
        provider._model_name = "gemini-3-flash-preview"  # type: ignore[attr-defined]

        result = await handle_taxonomist_task(
            runtime,
            TaskRecord(
                id="taxonomist-provider-backoff-task",
                task_name=TAXONOMIST_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated_tagged = runtime.repository.get_memory(tagged.id)
        updated_untagged = runtime.repository.get_memory(untagged.id)
        event_rows = runtime.db_manager.get_connection().execute(
            "SELECT event_kind, reason_code, retry_delay_seconds FROM provider_policy_events WHERE task_id = ?",
            ("taxonomist-provider-backoff-task",),
        ).fetchall()

        assert result["updated"] == 0
        assert result["provider_calls_used"] == 0
        assert result["provider_deferred_reason_code"] == "model_burst_limit_exceeded"
        assert result["provider_deferred_retry_delay_seconds"] == 600.0
        assert result["claimed_work_item_count"] == 1
        assert provider.call_count == 1
        assert updated_tagged is not None and updated_tagged.tags == ["auth", "testing"]
        assert updated_untagged is not None and updated_untagged.tags == []
        assert [tuple(row) for row in event_rows] == [
            ("provider_deferred", "model_burst_limit_exceeded", 600.0)
        ]
        assert runtime.work_items is not None
        work_items = runtime.work_items.list_items(family_key="memory_tagging", limit=5)
        assert [item.status for item in work_items] == ["deferred"]
        assert work_items[0].last_error == "model_burst_limit_exceeded"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_merges_related_facts_and_absorbs_observations(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical_fact = runtime.repository.create_memory(
            title="User testing preferences",
            content="The user prefers pytest-based integration coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing", "preferences"],
        )
        duplicate_fact = runtime.repository.create_memory(
            title="Testing preferences",
            content="Use pytest for integration tests and avoid brittle mocks.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["testing"],
        )
        new_observation = runtime.repository.create_memory(
            title="Fresh testing preference",
            content="The user likes deterministic pytest fixtures for test coverage.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["testing", "preference"],
        )
        assert canonical_fact is not None and duplicate_fact is not None and new_observation is not None

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )
        refreshed_canonical = runtime.repository.get_memory(canonical_fact.id)
        refreshed_duplicate = runtime.repository.get_memory(duplicate_fact.id)
        refreshed_observation = runtime.repository.get_memory(new_observation.id)
        fact_links = runtime.repository.get_links(active_facts[0].id, direction="outgoing") if active_facts else []

        assert result["merged"] >= 1
        assert result["absorbed_observations"] == 1
        assert len(active_facts) == 1
        assert refreshed_canonical is not None
        assert refreshed_duplicate is not None
        assert {refreshed_canonical.status, refreshed_duplicate.status} == {"active", "archived"}
        assert refreshed_observation is not None and refreshed_observation.status == "archived"
        assert any(
            link.target_id in {canonical_fact.id, duplicate_fact.id} and link.target_id != active_facts[0].id and link.link_type == "SUPERSEDES"
            for link in fact_links
        )
        assert any(link.target_id == new_observation.id and link.link_type == "SUPERSEDES" for link in fact_links)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_dedup_prep_seeds_agentic_review_work(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        first = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        second = runtime.repository.create_memory(
            title="Duplicate auth fact copy",
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
        assert first is not None and second is not None and observation is not None

        result = await handle_dedup_prep_task(
            runtime,
            TaskRecord(
                id="dedup-prep-task",
                task_name=DEDUP_PREP_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "anomaly"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        review_items = runtime.work_items.list_items(family_key="memory_dedup_review", limit=5)

        assert result["seeded_work_item_count"] == 1
        assert result["work_item_family"] == "memory_dedup_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "frontier_seed"
        assert result["seed_record_count"] >= 2
        assert [item.status for item in review_items] == ["pending"]
        assert result["created_work_item_id"] == review_items[0].id
        assert set(review_items[0].payload["seed_memory_ids"]) >= {first.id, second.id}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_consumes_seeded_review_work(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        first = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        second = runtime.repository.create_memory(
            title="Duplicate auth fact copy",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert first is not None and second is not None

        prep_result = await handle_dedup_prep_task(
            runtime,
            TaskRecord(
                id="dedup-prep-review-seed",
                task_name=DEDUP_PREP_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "anomaly"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )
        assert prep_result["seeded_work_item_count"] == 1

        class _AgenticProvider:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def supports_agentic(self) -> bool:
                return True

            async def run_agent(self, prompt: str) -> AgenticRunResult:
                self.prompts.append(prompt)
                await call_internal_memory_tool(
                    runtime,
                    "internal_merge_memory_into_canonical",
                    {
                        "canonical_memory_id": first.id,
                        "source_memory_id": second.id,
                        "metadata": {"deduplicator_task_id": "deduplicator-seeded-review"},
                    },
                )
                return AgenticRunResult(
                    status="success",
                    summary="Deduplicator merged duplicate facts via seeded MCP review.",
                    parsed={
                        "response": json.dumps(
                            {
                                "summary": "Deduplicator merged duplicate facts via seeded MCP review.",
                                "merged": 1,
                                "archived": 1,
                                "absorbed_observations": 0,
                            }
                        ),
                        "stats": {
                            "tools": {
                                "totalCalls": 1,
                                "byName": {
                                    "mcp_mcp-memory-internal_internal_merge_memory_into_canonical": {"count": 1}
                                },
                            }
                        },
                    },
                )

        provider = _AgenticProvider()
        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-seeded-review",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        canonical = runtime.repository.get_memory(first.id)
        archived = runtime.repository.get_memory(second.id)
        review_items = runtime.work_items.list_items(family_key="memory_dedup_review", limit=5)

        assert result["merged"] == 1
        assert result["archived"] == 1
        assert result["claimed_work_item_count"] == 1
        assert result["execution_mode"] == "agentic_mcp"
        assert canonical is not None
        assert archived is not None and archived.status == "archived"
        assert [item.status for item in review_items] == ["completed"]
        assert provider.prompts
        assert "internal_get_next_dedup_batch" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_skips_provider_for_short_non_subset_merges(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical = runtime.repository.create_memory(
            title="JWT auth note",
            content="JWT required. Rotate keys daily.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="JWT auth policy",
            content="Bearer auth required. Log token use.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[{"title": "Provider merged title", "content": "Provider merged content"}]
        )
        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-short-merge-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )

        assert result["merged"] >= 1
        assert provider.call_count == 0
        assert len(active_facts) == 1
        assert active_facts[0].title != "Provider merged title"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_skips_provider_for_subset_merge(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical = runtime.repository.create_memory(
            title="JWT requirement",
            content="JWTs are required for all clients. Bearer tokens are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="JWT requirement summary",
            content="JWTs are required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[{"title": "Provider merged title", "content": "Provider merged content"}]
        )
        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-subset-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated_canonical = runtime.repository.get_memory(canonical.id)
        updated_duplicate = runtime.repository.get_memory(duplicate.id)
        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )

        assert result["merged"] >= 1
        assert provider.call_count == 0
        assert updated_canonical is not None
        assert updated_duplicate is not None
        assert len(active_facts) == 1
        assert active_facts[0].title != "Provider merged title"
        assert {updated_canonical.status, updated_duplicate.status} == {"active", "archived"}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_limits_observation_absorption_to_seed_subset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)

    try:
        canonical = runtime.repository.create_memory(
            title="JWT rollout fact",
            content="\n".join(
                [
                    "JWTs are required for every client rollout.",
                    "Bearer tokens must be enabled.",
                    "Rotate signing keys daily.",
                    "Audit token issuance.",
                    "Document auth setup.",
                    "Track rollout readiness.",
                    "Validate gateway auth.",
                    "Coordinate client teams.",
                    "Keep rollout notes current.",
                    "Monitor auth regressions.",
                    "Publish migration guidance.",
                ]
            ),
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "rollout"],
        )
        assert canonical is not None

        for index in range(10):
            created = runtime.repository.create_memory(
                title=f"JWT rollout observation {index}",
                content="\n".join(
                    [
                        f"JWT rollout observation {index}.",
                        "Bearer tokens are required.",
                        "Rotate signing keys daily.",
                        "Audit token usage in services.",
                        "Document auth setup for teams.",
                        "Track rollout readiness per client.",
                        "Validate gateway auth.",
                        "Coordinate client teams.",
                        "Keep migration notes current.",
                        "Monitor auth regressions.",
                        "Publish migration guidance.",
                    ]
                ),
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["auth", "rollout"],
            )
            assert created is not None

        provider = FakeAIProvider(
            responses=[
                {"title": "Provider merged title", "content": f"Provider merged content {index}"}
                for index in range(DEDUPLICATOR_OBSERVATION_SEED_RECORDS)
            ]
        )

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-seed-subset-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        archived_observations = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="observation",
            status="archived",
            limit=20,
        )
        remaining_observations = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="observation",
            status="active",
            limit=20,
        )

        absorbed_observations = result["absorbed_observations"]

        assert 0 < absorbed_observations <= DEDUPLICATOR_OBSERVATION_SEED_RECORDS
        assert len(result["seed_memory_ids"]) <= 8
        assert provider.call_count == 0
        assert len(archived_observations) == absorbed_observations
        assert len(remaining_observations) == 10 - absorbed_observations
        assert absorbed_observations < 10
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_can_use_agentic_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    class _AgenticProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(
                status="success",
                summary="Deduplicator merged duplicate facts via MCP tools.",
                parsed={
                    "response": json.dumps(
                        {
                            "summary": "Deduplicator merged duplicate facts via MCP tools.",
                            "merged": 1,
                            "archived": 1,
                            "absorbed_observations": 0,
                        }
                    ),
                    "stats": {
                        "tools": {
                            "totalCalls": 4,
                            "byName": {
                                "mcp_mcp-memory-internal_internal_get_next_dedup_batch": {"count": 1},
                                "mcp_mcp-memory-internal_internal_merge_memory_into_canonical": {"count": 2},
                                "mcp_mcp-memory-internal_internal_read_memory_record": {"count": 1},
                            },
                        }
                    },
                },
            )

    try:
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs must be required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = _AgenticProvider()

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-agentic-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["summary"] == "Deduplicator merged duplicate facts via MCP tools."
        assert result["merged"] == 1
        assert result["archived"] == 1
        assert result["absorbed_observations"] == 0
        assert result["execution_mode"] == "agentic_mcp"
        assert result["tool_calls_executed"] == 4
        assert result["mutations"] == 2
        assert result["tool_names_used"] == [
            "mcp_mcp-memory-internal_internal_get_next_dedup_batch",
            "mcp_mcp-memory-internal_internal_merge_memory_into_canonical",
            "mcp_mcp-memory-internal_internal_read_memory_record",
        ]
        assert provider.prompts
        assert "Use the workspace-local internal MCP maintenance tools directly" in provider.prompts[0]
        assert "internal_get_next_dedup_batch" in provider.prompts[0]
        assert "internal_merge_memory_into_canonical" in provider.prompts[0]
        assert "task_complete" in provider.prompts[0]
        assert "Small-to-medium records beat large mixed-topic blobs." in provider.prompts[0]
        assert "Merge only when the records describe the same durable concept" in provider.prompts[0]
        assert "deduplicator_task_id='deduplicator-agentic-task'" in provider.prompts[0]
        assert canonical.id in provider.prompts[0]
        assert duplicate.id in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_does_not_merge_cross_project_generic_architecture_facts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        first = runtime.repository.create_memory(
            title="CoreRL architecture overview",
            content="CoreRL architecture. Daemon lifecycle. CLI infrastructure.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture"],
        )
        second = runtime.repository.create_memory(
            title="mcp-memory architecture overview",
            content="mcp-memory architecture. Daemon lifecycle. CLI infrastructure.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture"],
        )
        assert first is not None and second is not None

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-cross-project-generic-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        active_facts = runtime.repository.list_memories(
            workspace_id=runtime.workspace_id,
            memory_type="fact",
            status="active",
            limit=10,
        )

        assert result["merged"] == 0
        assert result["archived"] == 0
        assert len(active_facts) == 2
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_deduplicator_accepts_copilot_style_agentic_payload_without_tool_stats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    class _CopilotLikeAgenticProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(
                status="success",
                summary="Deduplicator merged duplicate facts via Copilot agentic MCP.",
                parsed={
                    "summary": "Deduplicator merged duplicate facts via Copilot agentic MCP.",
                    "merged": 1,
                    "archived": 1,
                    "absorbed_observations": 0,
                },
            )

    try:
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs must be required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = _CopilotLikeAgenticProvider()

        result = await handle_deduplicator_task(
            runtime,
            TaskRecord(
                id="deduplicator-copilot-agentic-task",
                task_name=DEDUPLICATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["summary"] == "Deduplicator merged duplicate facts via Copilot agentic MCP."
        assert result["merged"] == 1
        assert result["archived"] == 1
        assert result["absorbed_observations"] == 0
        assert result["execution_mode"] == "agentic_mcp"
        assert result["tool_calls_executed"] == 0
        assert result["mutations"] == 0
        assert result["tool_names_used"] == []
        assert provider.prompts
        assert "internal_get_next_dedup_batch" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_can_use_internal_tools_to_merge_memories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs must be required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_merge_memory_into_canonical",
                            "arguments": {
                                "canonical_memory_id": canonical.id,
                                "source_memory_id": duplicate.id,
                            },
                        }
                    ]
                },
                {"summary": "Merged duplicate auth fact into canonical memory.", "actions_taken": 1},
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated_canonical = runtime.repository.get_memory(canonical.id)
        updated_duplicate = runtime.repository.get_memory(duplicate.id)

        assert result["summary"] == "Merged duplicate auth fact into canonical memory."
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert provider.prompts
        assert "Work in high-impact maintenance mode" in provider.prompts[0]
        assert "internal_get_next_curator_batch" in provider.prompts[0]
        assert "task_complete" in provider.prompts[0]
        assert "Treat the seed memories as a starting frontier, not a hard boundary" in provider.prompts[0]
        assert "Seed memories (compact view):" in provider.prompts[0]
        assert "inputSchema" not in provider.prompts[0]
        assert "read_count" not in provider.prompts[0]
        assert "required_fields" in provider.prompts[0]
        assert updated_canonical is not None and "JWTs must be required" in updated_canonical.content
        assert updated_duplicate is not None and updated_duplicate.status == "archived"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_can_use_internal_tools_to_split_oversized_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        oversized = runtime.repository.create_memory(
            title="Oversized rollout memory",
            content="Oversized rollout detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth", "rollout"],
        )
        assert oversized is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "tool_calls": [
                        {
                            "name": "internal_split_memory_record",
                            "arguments": {
                                "memory_id": oversized.id,
                                "archive_original": True,
                                "parts": [
                                    {
                                        "title": "Rollout prerequisites",
                                        "content": "Step one and step two.",
                                        "tags": ["prereq"],
                                    },
                                    {
                                        "title": "Rollout execution",
                                        "content": "Step three and step four.",
                                        "tags": ["execution"],
                                    },
                                ],
                            },
                        }
                    ]
                },
                {"summary": "Split oversized rollout memory into two focused facts.", "actions_taken": 1},
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-split-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        split_children = runtime.repository.list_memories(workspace_id=runtime.workspace_id, limit=10)
        refreshed_original = runtime.repository.get_memory(oversized.id)
        created_split_children = [
            record
            for record in split_children
            if record.title in {"Rollout prerequisites", "Rollout execution"}
        ]

        assert result["summary"] == "Split oversized rollout memory into two focused facts."
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert "internal_split_memory_record" in result["tool_names_used"]
        assert refreshed_original is not None and refreshed_original.status == "archived"
        assert len(created_split_children) == 2
        assert refreshed_original.metadata["split_child_count"] == 2
        original_child_ids = refreshed_original.metadata["split_child_memory_ids"]
        assert isinstance(original_child_ids, list)
        assert set(str(item) for item in original_child_ids) == {record.id for record in created_split_children}
        split_group_ids = {record.metadata["split_group_id"] for record in created_split_children}
        assert split_group_ids == {refreshed_original.metadata["split_group_id"]}
        for child in created_split_children:
            assert child.metadata["split_from_memory_id"] == oversized.id
            assert child.metadata["split_part_count"] == 2
            sibling_ids = child.metadata["split_sibling_memory_ids"]
            assert isinstance(sibling_ids, list)
            assert set(str(item) for item in sibling_ids) == {
                record.id for record in created_split_children if record.id != child.id
            }
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_distrusts_action_claims_without_internal_tool_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "summary": "Split oversized memory and archived stale observations.",
                    "actions_taken": 7,
                }
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-unverified-summary-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        refreshed = runtime.repository.get_memory(record.id)

        assert result["tool_calls_executed"] == 0
        assert result["mutations"] == 0
        assert result["tool_names_used"] == []
        assert result["summary"] == (
            "Provider reported actions_taken=7 without using internal tools; "
            "no curator maintenance actions were executed."
        )
        assert refreshed is not None and refreshed.status == "active"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_retries_when_provider_claims_actions_without_tool_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        canonical = runtime.repository.create_memory(
            title="Canonical auth fact",
            content="JWTs are required.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        duplicate = runtime.repository.create_memory(
            title="Duplicate auth fact",
            content="JWTs must be required for all clients.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["auth"],
        )
        assert canonical is not None and duplicate is not None

        provider = FakeAIProvider(
            responses=[
                {
                    "summary": "Merged duplicate auth fact into canonical memory.",
                    "actions_taken": 1,
                },
                {
                    "tool_calls": [
                        {
                            "name": "internal_merge_memory_into_canonical",
                            "arguments": {
                                "canonical_memory_id": canonical.id,
                                "source_memory_id": duplicate.id,
                            },
                        }
                    ]
                },
                {"summary": "Merged duplicate auth fact into canonical memory.", "actions_taken": 1},
            ]
        )

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-retry-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        updated_canonical = runtime.repository.get_memory(canonical.id)
        updated_duplicate = runtime.repository.get_memory(duplicate.id)

        assert provider.call_count == 3
        assert "validation_error" in provider.prompts[1]
        assert result["summary"] == "Merged duplicate auth fact into canonical memory."
        assert result["tool_calls_executed"] == 1
        assert result["mutations"] == 1
        assert updated_canonical is not None and "JWTs must be required" in updated_canonical.content
        assert updated_duplicate is not None and updated_duplicate.status == "archived"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_can_use_agentic_provider(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    class _AgenticProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(status="success", summary="Curator completed maintenance via MCP tools.")

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        provider = _AgenticProvider()

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-agentic-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["summary"] == "Curator completed maintenance via MCP tools."
        assert result["execution_mode"] == "agentic_mcp"
        assert provider.prompts
        assert "Use the workspace-local internal MCP maintenance tools directly" in provider.prompts[0]
        assert "Aim for multiple coherent, high-value maintenance actions in one run" in provider.prompts[0]
        assert "internal_get_next_curator_batch" in provider.prompts[0]
        assert "Treat the provided seed memories as a starting frontier" in provider.prompts[0]
        assert "Small-to-medium records beat large mixed-topic blobs." in provider.prompts[0]
        assert "Prefer split-and-link over expanding a memory that already spans multiple topics" in provider.prompts[0]
        assert record.id in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_curator_frontier_seeds_agentic_review_work_item(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        result = await handle_curator_frontier_task(
            runtime,
            TaskRecord(
                id="curator-frontier-task",
                task_name=CURATOR_FRONTIER_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=85,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        queued = runtime.work_items.list_items(family_key="memory_curation_review", limit=10)

        assert result["seeded_work_item_count"] == 1
        assert result["work_item_family"] == "memory_curation_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "frontier_seed"
        assert result["seed_record_count"] == 1
        assert queued
        assert result["created_work_item_id"] == queued[0].id
        assert queued[0].family_key == "memory_curation_review"
        assert queued[0].payload["seed_memory_ids"] == [record.id]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_consumes_seeded_review_work_item_first(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.work_items is not None

    class _AgenticProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(status="success", summary="Curator completed seeded maintenance via MCP tools.")

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None
        work_item, created = runtime.work_items.enqueue_unique(
            family_key="memory_curation_review",
            execution_lane="agentic",
            workspace_id=runtime.workspace_id,
            priority=95,
            idempotency_key=f"memory_curation_review:{record.id}",
            payload={
                "workspace_id": runtime.workspace_id,
                "seed_memory_ids": [record.id],
                "strategy_used": "anomaly",
                "candidate_count": 1,
            },
        )
        assert created is True

        provider = _AgenticProvider()

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-seeded-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=95,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        refreshed = runtime.work_items.get_item(work_item.id)

        assert result["summary"] == "Curator completed seeded maintenance via MCP tools."
        assert result["claimed_work_item_count"] == 1
        assert result["execution_mode"] == "agentic_mcp"
        assert result["work_item_family"] == "memory_curation_review"
        assert result["work_item_execution_lane"] == "agentic"
        assert result["seed_source"] == "claimed_review_work_item"
        assert result["seed_record_count"] == 1
        assert result["claimed_work_item_id"] == work_item.id
        assert refreshed.status == "completed"
        assert provider.prompts
        assert f'"{record.id}"' in provider.prompts[0]
        assert "exclude_memory_ids" in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_accepts_copilot_style_agentic_summary_payload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    class _CopilotLikeAgenticProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            return AgenticRunResult(
                status="success",
                summary="Curator completed Copilot MCP maintenance.",
                parsed={"summary": "Curator completed Copilot MCP maintenance."},
            )

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        provider = _CopilotLikeAgenticProvider()

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-copilot-agentic-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        assert result["summary"] == "Curator completed Copilot MCP maintenance."
        assert result["execution_mode"] == "agentic_mcp"
        assert provider.prompts
        assert "output final JSON only in the form {\"summary\": \"...\"}" in provider.prompts[0]
        assert record.id in provider.prompts[0]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_same_run_fails_over_to_next_agentic_route_on_ordinary_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.config is not None

    class _AgenticProvider:
        def __init__(self, *, provider_key: str, error: Exception | None = None, summary: str | None = None) -> None:
            self._provider_key = provider_key
            self._error = error
            self._summary = summary
            self.prompts: list[str] = []
            self.usage_contexts: list[dict[str, str | None]] = []

        def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
            self.usage_contexts.append(
                {
                    "task_name": task_name,
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                }
            )
            return self

        def supports_agentic(self) -> bool:
            return True

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            if self._error is not None:
                raise self._error
            return AgenticRunResult(status="success", summary=self._summary)

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        first_provider = _AgenticProvider(provider_key="copilot-strong", error=RuntimeError("Exit code -15"))
        fallback_provider = _AgenticProvider(
            provider_key="gemini-cheap",
            summary="Curator completed maintenance via fallback route.",
        )
        runtime.config.provider_routing.task_routes[CURATOR_TASK_NAME] = ["copilot-strong", "gemini-cheap"]
        runtime.ai_provider_registry = {
            "copilot-strong": {"agentic": first_provider},
            "gemini-cheap": {"agentic": fallback_provider},
        }
        task = TaskRecord(
            id="memory-curator-failover-task",
            task_name=CURATOR_TASK_NAME,
            data={"workspace_id": runtime.workspace_id},
            workspace_id=runtime.workspace_id,
            status="running",
            priority=100,
            retries_count=0,
            max_retries=3,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=0.0,
            started_at=0.0,
            completed_at=None,
            last_error=None,
        )

        selected = _provider_for_task(runtime, None, None, CURATOR_TASK_NAME, task)
        result = await handle_memory_curator_task(runtime, task, selected)

        assert result["summary"] == "Curator completed maintenance via fallback route."
        assert result["execution_mode"] == "agentic_mcp"
        assert len(first_provider.prompts) == 1
        assert len(fallback_provider.prompts) == 1
        assert first_provider.usage_contexts == [
            {
                "task_name": CURATOR_TASK_NAME,
                "task_id": task.id,
                "workspace_id": runtime.workspace_id,
            }
        ]
        assert fallback_provider.usage_contexts == [
            {
                "task_name": CURATOR_TASK_NAME,
                "task_id": task.id,
                "workspace_id": runtime.workspace_id,
            }
        ]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_does_not_same_run_failover_on_retry_delay_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None
    assert runtime.config is not None

    class _RetryLaterError(RuntimeError):
        def __init__(self, message: str, *, retry_delay_seconds: float) -> None:
            super().__init__(message)
            self.retry_delay_seconds = retry_delay_seconds

    class _AgenticProvider:
        def __init__(self, *, provider_key: str, error: Exception | None = None, summary: str | None = None) -> None:
            self._provider_key = provider_key
            self._error = error
            self._summary = summary
            self.prompts: list[str] = []

        def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
            return self

        def supports_agentic(self) -> bool:
            return True

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            self.prompts.append(prompt)
            if self._error is not None:
                raise self._error
            return AgenticRunResult(status="success", summary=self._summary)

    try:
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content="Oversized architecture detail. " * 220,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None

        first_provider = _AgenticProvider(
            provider_key="copilot-strong",
            error=_RetryLaterError("provider backoff", retry_delay_seconds=45.0),
        )
        fallback_provider = _AgenticProvider(
            provider_key="gemini-cheap",
            summary="Curator completed maintenance via fallback route.",
        )
        runtime.config.provider_routing.task_routes[CURATOR_TASK_NAME] = ["copilot-strong", "gemini-cheap"]
        runtime.ai_provider_registry = {
            "copilot-strong": {"agentic": first_provider},
            "gemini-cheap": {"agentic": fallback_provider},
        }
        task = TaskRecord(
            id="memory-curator-no-failover-task",
            task_name=CURATOR_TASK_NAME,
            data={"workspace_id": runtime.workspace_id},
            workspace_id=runtime.workspace_id,
            status="running",
            priority=100,
            retries_count=0,
            max_retries=3,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=0.0,
            started_at=0.0,
            completed_at=None,
            last_error=None,
        )

        selected = _provider_for_task(runtime, None, None, CURATOR_TASK_NAME, task)

        with pytest.raises(_RetryLaterError, match="provider backoff"):
            await handle_memory_curator_task(runtime, task, selected)

        assert len(first_provider.prompts) == 1
        assert fallback_provider.prompts == []
    finally:
        runtime.close()


def test_agentic_task_names_contains_curator_deduplicator_and_ingest() -> None:
    assert AGENTIC_TASK_NAMES == {CURATOR_TASK_NAME, DEDUPLICATOR_TASK_NAME, SYSTEM1_INGEST_TASK_NAME}


@pytest.mark.asyncio
async def test_memory_curator_prompt_truncates_large_seed_summaries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        huge_summary = "Important architecture note. " * 80
        record = runtime.repository.create_memory(
            title="Very long architecture reflection title that should be shortened before being sent to Gemini for curator work",
            content=huge_summary,
            summary=huge_summary,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="reflection",
            tags=["architecture", "long", "summary", "curator", "debug", "prompt", "extra-tag"],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        result = await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-truncate-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        prompt = provider.prompts[0]
        assert result["summary"] == "No-op"
        assert huge_summary[:400] not in prompt
        assert "Very long architecture reflection title that should be shortened before being s" in prompt
        assert "Very long architecture reflection title that should be shortened before being sent to Gemini for curator work" not in prompt
        assert "…" in prompt
        assert len(prompt) < 7000
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_prompt_flags_oversized_seed_memories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        oversized_content = "Oversized architecture detail. " * 220
        record = runtime.repository.create_memory(
            title="Oversized architecture record",
            content=oversized_content,
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["architecture", "oversized"],
        )
        assert record is not None
        assert len(record.content.strip()) > CURATOR_MAX_MEMORY_CHARS

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-oversized-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        prompt = provider.prompts[0]
        marker = "Seed memories (compact view):\n"
        suffix = "\n\nAvailable internal tools:"
        start = prompt.index(marker) + len(marker)
        end = prompt.index(suffix, start)
        seed_payload = json.loads(prompt[start:end])

        assert f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized." in prompt
        assert "Do not merge, append, or rewrite across different projects, products, or repositories" in prompt
        assert "No-op is acceptable when no change adds clear value." in prompt
        assert seed_payload[0]["id"] == record.id
        assert seed_payload[0]["oversized_for_curator"] is True
        assert seed_payload[0]["content_size_chars"] == len(record.content.strip())
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_memory_curator_prompt_serializes_seed_memories_as_json(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        record = runtime.repository.create_memory(
            title="Auth rollout note",
            content="JWT rollout needs client coordination.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="observation",
            tags=["auth", "rollout"],
        )
        assert record is not None

        provider = FakeAIProvider(responses=[{"summary": "No-op", "actions_taken": 0}])

        await handle_memory_curator_task(
            runtime,
            TaskRecord(
                id="memory-curator-json-prompt-task",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
            provider,
        )

        prompt = provider.prompts[0]
        marker = "Seed memories (compact view):\n"
        suffix = "\n\nAvailable internal tools:"
        start = prompt.index(marker) + len(marker)
        end = prompt.index(suffix, start)
        seed_json = prompt[start:end]
        seed_payload = json.loads(seed_json)

        assert isinstance(seed_payload, list)
        assert seed_payload
        assert seed_payload[0]["id"] == record.id
        assert seed_payload[0]["title"] == "Auth rollout note"
        assert seed_payload[0]["content_size_chars"] == len(record.content.strip())
        assert seed_payload[0]["oversized_for_curator"] is False
    finally:
        runtime.close()


def _curator_task_for_tests(workspace_id: str | None, *, task_id: str = "memory-curator-seed-test") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        task_name=CURATOR_TASK_NAME,
        data={"workspace_id": workspace_id, "strategy": "anomaly"},
        workspace_id=workspace_id,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


def test_memory_curator_recent_records_are_seeded_when_available(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        base_time = datetime(2026, 3, 1, tzinfo=UTC)
        for index in range(8):
            created = runtime.repository.create_memory(
                title=f"Older observation {index}",
                content="short observation",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["routine"],
                created_at=(base_time + timedelta(days=index)).isoformat(),
                updated_at=(base_time + timedelta(days=index)).isoformat(),
            )
            assert created is not None

        recent = runtime.repository.create_memory(
            title="Recent rollout concern",
            content="Short recent fact that should still get curator pressure.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["recent"],
            created_at=(base_time + timedelta(days=40)).isoformat(),
            updated_at=(base_time + timedelta(days=41)).isoformat(),
        )
        assert recent is not None

        seed_records = _select_curator_seed_records(runtime, _curator_task_for_tests(runtime.workspace_id))

        assert len(seed_records) == min(9, CURATOR_MAX_SEED_RECORDS)
        assert recent.id in {record.id for record in seed_records}
    finally:
        runtime.close()


def test_memory_curator_recency_quota_is_capped(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        base_time = datetime(2026, 2, 1, tzinfo=UTC)
        older_ids: list[str] = []
        recent_ids: list[str] = []
        for index in range(12):
            older = runtime.repository.create_memory(
                title=f"Older observation {index}",
                content="short observation",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["older"],
                created_at=(base_time + timedelta(days=index)).isoformat(),
                updated_at=(base_time + timedelta(days=index)).isoformat(),
            )
            assert older is not None
            older_ids.append(older.id)

        for index in range(5):
            recent = runtime.repository.create_memory(
                title=f"Recent fact {index}",
                content="short recent fact",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=["recent"],
                created_at=(base_time + timedelta(days=50 + index)).isoformat(),
                updated_at=(base_time + timedelta(days=50 + index)).isoformat(),
            )
            assert recent is not None
            recent_ids.append(recent.id)

        seed_records = _select_curator_seed_records(runtime, _curator_task_for_tests(runtime.workspace_id))
        seeded_ids = {record.id for record in seed_records}

        assert len(seed_records) == CURATOR_MAX_SEED_RECORDS
        assert len(seeded_ids.intersection(recent_ids)) == 4
        assert seeded_ids.issuperset(older_ids)
    finally:
        runtime.close()


def test_memory_curator_anomaly_seeds_survive_recency_bias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        base_time = datetime(2026, 1, 1, tzinfo=UTC)
        first_anomaly = runtime.repository.create_memory(
            title="Oversized fact one",
            content="A" * (CURATOR_MAX_MEMORY_CHARS + 100),
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["oversized"],
            created_at=base_time.isoformat(),
            updated_at=base_time.isoformat(),
        )
        second_anomaly = runtime.repository.create_memory(
            title="Oversized fact two",
            content="B" * (CURATOR_MAX_MEMORY_CHARS + 200),
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["oversized"],
            created_at=(base_time + timedelta(days=1)).isoformat(),
            updated_at=(base_time + timedelta(days=1)).isoformat(),
        )
        assert first_anomaly is not None and second_anomaly is not None

        for index in range(8):
            created = runtime.repository.create_memory(
                title=f"Recent note {index}",
                content="short recent note",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=["recent"],
                created_at=(base_time + timedelta(days=30 + index)).isoformat(),
                updated_at=(base_time + timedelta(days=30 + index)).isoformat(),
            )
            assert created is not None

        seed_records = _select_curator_seed_records(runtime, _curator_task_for_tests(runtime.workspace_id))
        seeded_ids = {record.id for record in seed_records}

        assert len(seed_records) == min(10, CURATOR_MAX_SEED_RECORDS)
        assert first_anomaly.id in seeded_ids
        assert second_anomaly.id in seeded_ids
    finally:
        runtime.close()


def test_memory_curator_fill_still_includes_older_candidates_after_recency_quota(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        base_time = datetime(2026, 2, 1, tzinfo=UTC)
        older_ids: list[str] = []
        for index in range(4):
            older = runtime.repository.create_memory(
                title=f"Older observation {index}",
                content="short observation",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["older"],
                created_at=(base_time + timedelta(days=index)).isoformat(),
                updated_at=(base_time + timedelta(days=index)).isoformat(),
            )
            assert older is not None
            older_ids.append(older.id)

        recent_ids: list[str] = []
        for index in range(6):
            recent = runtime.repository.create_memory(
                title=f"Recent fact {index}",
                content="short recent fact",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="fact",
                tags=["recent"],
                created_at=(base_time + timedelta(days=40 + index)).isoformat(),
                updated_at=(base_time + timedelta(days=40 + index)).isoformat(),
            )
            assert recent is not None
            recent_ids.append(recent.id)

        seed_records = _select_curator_seed_records(runtime, _curator_task_for_tests(runtime.workspace_id))
        seeded_ids = {record.id for record in seed_records}

        assert len(seed_records) == min(10, CURATOR_MAX_SEED_RECORDS)
        assert len(seeded_ids.intersection(recent_ids)) == 6
        assert seeded_ids.issuperset(older_ids)
    finally:
        runtime.close()


def test_memory_curator_size_anomaly_pass_can_surface_largest_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.repository is not None

    try:
        largest = runtime.repository.create_memory(
            title="Largest fact",
            content="L" * (CURATOR_MAX_MEMORY_CHARS - 100),
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["largest"],
        )
        assert largest is not None
        for index in range(9):
            created = runtime.repository.create_memory(
                title=f"Observation {index}",
                content="short observation",
                workspace_ids=[runtime.workspace_id or "global"],
                memory_type="observation",
                tags=["routine"],
            )
            assert created is not None

        seed_records = _select_curator_seed_records(
            runtime,
            TaskRecord(
                id="aaa",
                task_name=CURATOR_TASK_NAME,
                data={"workspace_id": runtime.workspace_id, "strategy": "anomaly"},
                workspace_id=runtime.workspace_id,
                status="running",
                priority=100,
                retries_count=0,
                max_retries=3,
                created_at=0.0,
                updated_at=0.0,
                available_at=0.0,
                claimed_at=0.0,
                started_at=0.0,
                completed_at=None,
                last_error=None,
            ),
        )

        assert len(seed_records) == min(10, CURATOR_MAX_SEED_RECORDS)
        assert largest.id in {record.id for record in seed_records}
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_processes_enqueued_ingest_task(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.ai_json_provider = None
        runtime.ai_agent_provider = None
        runtime.ai_provider = None
        runtime.ai_provider_registry = {}
        runtime.journal.record("first queue thought")
        runtime.journal.record("second queue thought")
        runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
        )
        worker = build_runtime_task_worker(runtime)
        await worker.start()

        for _ in range(50):
            if runtime.task_queue.count_by_status().get("completed", 0) >= 1:
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        assert runtime.repository.list_memories(workspace_id=runtime.workspace_id)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_uses_configured_provider_for_ingest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None

    try:
        runtime.journal.record("provider-backed ingest thought", workspace_id=runtime.workspace_id)
        fake_provider = FakeAIProvider(
            responses=[
                {
                    "actions": [
                        {
                            "type": "create",
                            "entry_indices": [0],
                            "title": "Provider-backed ingest thought",
                            "content": "Provider-backed ingest content.",
                        }
                    ]
                }
            ]
        )
        runtime.ai_json_provider = fake_provider
        runtime.ai_provider = fake_provider
        runtime.ai_provider_registry = {}
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="provider-backed-ingest",
        )
        worker = build_runtime_task_worker(runtime)

        await worker.start()
        for _ in range(50):
            if runtime.task_queue.get_task(task.id).status == "completed":
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        assert fake_provider.call_count >= 1
        assert fake_provider.prompts
        assert any(
            "Analyze these system1 journal entries" in prompt
            for prompt in fake_provider.prompts
        )
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)
        assert records
        assert records[0].title == "Provider-backed ingest thought"
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task_name",
    [SYSTEM1_INGEST_TASK_NAME, DEDUPLICATOR_TASK_NAME, CURATOR_TASK_NAME],
)
async def test_runtime_agentic_routes_prefer_copilot_agentic_provider(
    monkeypatch,
    tmp_path: Path,
    task_name: str,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(runtime_module, "_provider_command_available", lambda provider: True)

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.task_queue is not None

    try:
        task = runtime.task_queue.enqueue(
            task_name,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id=f"copilot-agentic-route-{task_name}",
        )

        selected = _provider_for_task(
            runtime,
            runtime.ai_json_provider,
            runtime.ai_agent_provider,
            task_name,
            task,
        )

        assert selected is not None
        wrapped_provider = getattr(selected, "_provider", None)
        assert isinstance(wrapped_provider, CopilotCLIAgenticProvider)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_runtime_worker_drains_multiple_ingest_batches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None

    try:
        runtime.ai_json_provider = None
        runtime.ai_agent_provider = None
        runtime.ai_provider = None
        runtime.ai_provider_registry = {}
        for index in range(45):
            runtime.journal.record(f"sqlite queue batch item {index}")

        runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="drain-ingest-test",
        )

        worker = build_runtime_task_worker(runtime)
        await worker.start()
        for _ in range(150):
            if runtime.journal.count_by_status().get("pending", 0) == 0:
                break
            await asyncio.sleep(0.02)
        await worker.stop(0.1)

        pending_counts = runtime.journal.count_by_status()
        assert pending_counts.get("pending", 0) == 0
        assert pending_counts.get("claimed", 0) == 0
        assert pending_counts.get("recoverable", 0) == 45
        assert runtime.task_queue.find_open_task(SYSTEM1_INGEST_TASK_NAME, runtime.workspace_id) is None
    finally:
        runtime.close()


class _SemanticFakeEmbedder:
    model_name = "semantic-fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if any(token in lowered for token in ["jwt", "session", "auth", "cookie"]):
                vectors.append([1.0, 0.0])
            elif any(token in lowered for token in ["sqlite", "wal", "database"]):
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.2, 0.2])
        return vectors


@pytest.mark.asyncio
async def test_ingest_handler_clusters_semantically_related_thoughts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.journal is not None
    assert runtime.task_queue is not None
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    runtime.embedder = _SemanticFakeEmbedder()
    runtime.vector_store = SQLiteVectorStore(runtime.db_manager)
    if runtime.relational_search is not None:
        runtime.relational_search._embedder = runtime.embedder  # noqa: SLF001
        runtime.relational_search._vector_store = runtime.vector_store  # noqa: SLF001

    try:
        first = runtime.journal.record("jwt refresh rollout")
        second = runtime.journal.record("session cookie migration")
        third = runtime.journal.record("sqlite wal tuning")
        task = runtime.task_queue.enqueue(
            SYSTEM1_INGEST_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
            available_at=0.0,
            task_id="semantic-ingest-test",
        )

        result = await handle_ingest_system1_task(runtime, task)
        records = runtime.repository.list_memories(workspace_id=runtime.workspace_id)

        assert len(result["created_memory_ids"]) == 2
        assert len(records) == 2
        source_sets = []
        for record in records:
            source_entry_ids = record.metadata["source_entry_ids"]
            assert isinstance(source_entry_ids, list)
            source_sets.append(set(source_entry_ids))
        assert {first.id, second.id} in source_sets
        assert {third.id} in source_sets
    finally:
        runtime.close()

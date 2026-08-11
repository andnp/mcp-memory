"""Verify the internal search tool's runtime integration contract."""

import json

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.providers.copilot_sdk import CopilotSDKAgenticProvider
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.task_handlers import ingest
from mcp_memory.core.task_handlers.tool_loop import InternalToolLoopResult
from mcp_memory.internal_tool_call_tracking import internal_tool_is_mutating
from mcp_memory.mcp import transport
from mcp_memory.mcp.internal_search_contract import INTERNAL_SEARCH_TOOL_NAME
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools


def _task() -> TaskRecord:
    return TaskRecord(
        id="task-1",
        task_name="ingest",
        data={},
        workspace_id="workspace-1",
        status="running",
        priority=0,
        retries_count=0,
        max_retries=1,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
    )


def test_internal_search_name_is_registered_and_dispatchable() -> None:
    """Registration and transport service lookup use the shared tool name."""
    registered_names = {tool.name for tool in get_internal_maintenance_tools()}

    assert INTERNAL_SEARCH_TOOL_NAME in registered_names
    assert INTERNAL_SEARCH_TOOL_NAME in transport.internal_tool_services()


def test_internal_search_is_read_only_for_tracking() -> None:
    """Search calls do not count as mutating internal tool calls."""
    assert not internal_tool_is_mutating(INTERNAL_SEARCH_TOOL_NAME)


def test_internal_search_is_exposed_by_the_agentic_provider() -> None:
    """The agentic provider allowlist includes the registered search name."""
    assert INTERNAL_SEARCH_TOOL_NAME in CopilotSDKAgenticProvider._DEFAULT_INTERNAL_TOOL_NAMES


@pytest.mark.asyncio
async def test_internal_dispatch_routes_the_registered_search_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal dispatch invokes the service selected by the registered search name."""
    calls: list[dict] = []

    def search_service(_ctx: ApplicationContext, arguments: dict) -> dict:
        calls.append(arguments)
        return {"results": [{"memory_ref": "mem-1"}]}

    monkeypatch.setattr(
        transport,
        "internal_tool_services",
        lambda: {INTERNAL_SEARCH_TOOL_NAME: search_service},
    )

    response = await transport.dispatch_internal_memory_tool(
        ApplicationContext(),
        INTERNAL_SEARCH_TOOL_NAME,
        {"query": "runtime contract"},
    )

    assert json.loads(response[0].text)["results"] == [{"memory_ref": "mem-1"}]
    assert calls == [{"query": "runtime contract"}]


@pytest.mark.asyncio
async def test_ingest_handler_allows_the_shared_search_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ingest handler passes the registered search name to its tool loop."""
    captured: dict[str, object] = {}

    async def fake_tool_loop(*_args, **kwargs) -> InternalToolLoopResult:
        captured.update(kwargs)
        return InternalToolLoopResult(response={"actions": []})

    monkeypatch.setattr(ingest, "run_internal_tool_loop", fake_tool_loop)

    await ingest._analyze_ingest_actions(ApplicationContext(), object(), "workspace-1", [])

    assert captured["allowed_tool_names"] == [
        INTERNAL_SEARCH_TOOL_NAME,
        "internal_read_memory_record",
        "internal_list_memory_records",
    ]


def test_ingest_prompt_mentions_the_shared_search_name() -> None:
    """The ingest instructions name the same search tool exposed at runtime."""
    prompt = ingest.build_ingest_agent_prompt(
        _task(),
        workspace_id="workspace-1",
        batch_size=5,
        grouping_strategy="balanced",
        max_batches_per_run=2,
    )

    assert f"Use {INTERNAL_SEARCH_TOOL_NAME}, internal_read_memory_record" in prompt

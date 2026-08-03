import asyncio
import json
from pathlib import Path
import time

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_call_memory_tool_rejects_uninitialized_context() -> None:
    payload = json.loads((await call_memory_tool(object(), "record_thought", {}))[0].text)

    assert payload == {
        "error": "runtime_not_initialized",
        "status": "error",
        "tool": "record_thought",
    }


@pytest.mark.asyncio
async def test_call_memory_tool_rejects_unknown_tool_name() -> None:
    payload = json.loads((await call_memory_tool(ApplicationContext(), "typo_tool", {}))[0].text)

    assert payload["status"] == "error"
    assert payload["error"] == "unknown_tool"
    assert payload["tool"] == "typo_tool"


@pytest.mark.asyncio
async def test_record_thought_requires_string_content(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    try:
        payload = json.loads((await call_memory_tool(runtime, "record_thought", {"content": ["nope"]}))[0].text)
        assert payload["status"] == "error"
        assert payload["error"] == "invalid_arguments"
        assert payload["detail"] == "content must be a string"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_search_memory_records_rejects_invalid_query_and_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    try:
        query_payload = json.loads((await call_memory_tool(runtime, "search_memory_records", {"query": "   "}))[0].text)
        limit_payload = json.loads((await call_memory_tool(runtime, "search_memory_records", {"query": "auth", "limit": 0}))[0].text)

        assert query_payload["error"] == "invalid_arguments"
        assert query_payload["detail"] == "query is required"
        assert limit_payload["error"] == "invalid_arguments"
        assert limit_payload["detail"] == "limit must be at least 1"
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_removed_tools_are_rejected_as_unknown_tool() -> None:
    for tool_name in [
        "get_pending_thoughts",
        "get_memory_stats",
        "create_memory_record",
        "get_memory_record",
        "list_memory_records",
        "import_markdown_memory_file",
        "create_memory_link",
        "delete_memory_link",
    ]:
        payload = json.loads((await call_memory_tool(ApplicationContext(), tool_name, {}))[0].text)

        assert payload["status"] == "error"
        assert payload["error"] == "unknown_tool"
        assert payload["tool"] == tool_name


@pytest.mark.asyncio
async def test_call_memory_tool_offloads_sync_service_work(monkeypatch) -> None:
    def slow_service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        time.sleep(0.25)
        return {"status": "ok"}

    monkeypatch.setattr(
        "mcp_memory.mcp.transport.tool_services",
        lambda: {"search_memory_records": slow_service},
    )

    ticker = asyncio.create_task(asyncio.sleep(0.05, result="tick"))
    tool_task = asyncio.create_task(
        call_memory_tool(ApplicationContext(), "search_memory_records", {"query": "resilience"})
    )

    assert await asyncio.wait_for(ticker, timeout=0.2) == "tick"
    payload = json.loads((await tool_task)[0].text)
    assert payload == {}

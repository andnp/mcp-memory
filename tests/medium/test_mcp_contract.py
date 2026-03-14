import json
from pathlib import Path

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
async def test_create_memory_record_validates_required_fields_and_types(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    try:
        missing_payload = json.loads((await call_memory_tool(runtime, "create_memory_record", {"content": "x", "workspace_ids": ["a"]}))[0].text)
        type_payload = json.loads((await call_memory_tool(runtime, "create_memory_record", {"title": "ok", "content": "x", "workspace_ids": "a"}))[0].text)

        assert missing_payload["error"] == "invalid_arguments"
        assert missing_payload["detail"] == "title must be a string"
        assert type_payload["error"] == "invalid_arguments"
        assert type_payload["detail"] == "workspace_ids must be a list of strings"
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
async def test_import_markdown_memory_file_rejects_missing_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    runtime = create_runtime(cwd=tmp_path / "workspace")
    try:
        payload = json.loads((await call_memory_tool(runtime, "import_markdown_memory_file", {"file_path": str(tmp_path / "missing.md"), "workspace_ids": ["workspace-a"]}))[0].text)

        assert payload["status"] == "error"
        assert payload["error"] == "file_not_found"
        assert "missing.md" in payload["detail"]
    finally:
        runtime.close()
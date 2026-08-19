"""Focused contracts for the provider-neutral internal tool loop."""

import json

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.tool_dispatch import ToolDefinition, ToolResponsePart
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop

pytestmark = pytest.mark.small


class _FakeToolDispatch:
    def __init__(self, response: tuple[ToolResponsePart, ...]) -> None:
        self.response = response
        self.dispatched: list[tuple[str, dict[str, object]]] = []

    def available_tools(self) -> dict[str, ToolDefinition]:
        return {
            "internal_search": ToolDefinition(
                name="internal_search",
                input_schema={"properties": {"query": {"type": "string"}}},
            ),
            "unavailable_service": ToolDefinition(
                name="unavailable_service",
                input_schema={},
            ),
        }

    async def dispatch(
        self,
        ctx: ApplicationContext,
        name: str,
        arguments: dict[str, object],
    ) -> tuple[ToolResponsePart, ...]:
        self.dispatched.append((name, arguments))
        return self.response


class _FakeProvider:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def ask(self, _prompt: str) -> dict[str, object]:
        self.prompts.append(_prompt)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_tool_loop_dispatches_selected_tools_and_preserves_payload() -> None:
    """The loop selects allowed tools and returns the dispatcher's JSON payload."""
    dispatch = _FakeToolDispatch((ToolResponsePart('{"status":"ok","results":[]}'),))
    ctx = ApplicationContext(tool_dispatch=dispatch)
    provider = _FakeProvider(
        [
            {"tool_calls": [{"name": "internal_search", "arguments": {"query": "ports"}}]},
            {"status": "ok", "results": []},
        ]
    )

    result = await run_internal_tool_loop(
        ctx,
        provider,
        prompt="search",
        allowed_tool_names=["internal_search"],
        max_rounds=2,
    )

    assert result.response == {"status": "ok", "results": []}
    assert result.tool_calls_executed == 1
    assert result.successful_tool_calls == 1
    assert result.tool_names_used == ["internal_search"]
    assert dispatch.dispatched == [("internal_search", {"query": "ports"})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ((), {"status": "error", "error": "empty_tool_response", "tool": "internal_search"}),
        ((ToolResponsePart(None),), {"status": "error", "error": "invalid_tool_response", "tool": "internal_search"}),
    ],
)
async def test_tool_loop_preserves_dispatch_response_errors(
    response: tuple[ToolResponsePart, ...],
    expected: dict[str, str],
) -> None:
    """Empty or non-text dispatch responses retain their stable error payloads."""
    ctx = ApplicationContext(tool_dispatch=_FakeToolDispatch(response))
    provider = _FakeProvider(
        [
            {"tool_calls": [{"name": "internal_search", "arguments": {}}]},
            {"status": "ok"},
        ]
    )

    result = await run_internal_tool_loop(
        ctx,
        provider,
        prompt="search",
        allowed_tool_names=["internal_search"],
        max_rounds=2,
    )

    assert result.response == {"status": "ok"}
    assert json.dumps(expected, sort_keys=True, separators=(",", ":")) in provider.prompts[1]

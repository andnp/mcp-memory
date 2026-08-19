from __future__ import annotations

from typing import Any

import pytest

from mcp_memory.toolkit.bounded_tool_loop import ToolSpec, run_bounded_tool_loop

pytestmark = pytest.mark.small


class _ScriptedProvider:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)

    def ask(self, _prompt: str) -> dict[str, Any]:
        return self._responses.pop(0)


async def _dispatch_ok(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"status": "ok", "tool": name, "arguments": arguments}


@pytest.mark.asyncio
async def test_stops_after_final_response_with_no_tool_calls() -> None:
    provider = _ScriptedProvider([{"status": "ok", "actions_taken": 0}])

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
    )

    assert result.response == {"status": "ok", "actions_taken": 0}
    assert result.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_executes_allowed_tool_call_and_tracks_success() -> None:
    provider = _ScriptedProvider(
        [
            {"tool_calls": [{"name": "internal_do_thing", "arguments": {"x": 1}}]},
            {"status": "ok", "actions_taken": 1},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
    )

    assert result.tool_calls_executed == 1
    assert result.successful_tool_calls == 1
    assert result.tool_names_used == ["internal_do_thing"]
    assert result.response == {"status": "ok", "actions_taken": 1}


@pytest.mark.asyncio
async def test_rejects_tool_call_not_in_allowlist_without_dispatching() -> None:
    calls: list[str] = []

    async def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append(name)
        return {"status": "ok"}

    provider = _ScriptedProvider(
        [
            {"tool_calls": [{"name": "internal_forbidden", "arguments": {}}]},
            {"status": "ok", "actions_taken": 0},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        dispatch,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
    )

    assert calls == []
    assert result.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_anti_hallucination_guard_forces_correction_round() -> None:
    provider = _ScriptedProvider(
        [
            {"actions_taken": 3},
            {"status": "ok", "actions_taken": 0},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
    )

    assert result.response == {"status": "ok", "actions_taken": 0}
    assert result.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_anti_hallucination_guard_returns_invalid_response_when_rounds_exhausted() -> None:
    provider = _ScriptedProvider(
        [
            {"actions_taken": 3},
            {"actions_taken": 5},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
        max_rounds=2,
    )

    assert result.response == {"actions_taken": 5}
    assert result.tool_calls_executed == 0


@pytest.mark.asyncio
async def test_returns_tool_loop_exhausted_when_rounds_run_out_without_final_response() -> None:
    provider = _ScriptedProvider(
        [
            {"tool_calls": [{"name": "internal_do_thing", "arguments": {}}]},
            {"tool_calls": [{"name": "internal_do_thing", "arguments": {}}]},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
        max_rounds=2,
    )

    assert result.response["status"] == "error"
    assert result.response["error"] == "tool_loop_exhausted"
    assert result.tool_calls_executed == 2


@pytest.mark.asyncio
async def test_caps_tool_calls_per_round() -> None:
    executed: list[str] = []

    async def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        executed.append(name)
        return {"status": "ok"}

    provider = _ScriptedProvider(
        [
            {
                "tool_calls": [
                    {"name": "internal_do_thing", "arguments": {"i": i}} for i in range(5)
                ]
            },
            {"status": "ok", "actions_taken": 5},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        dispatch,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
        max_tool_calls_per_round=2,
    )

    assert len(executed) == 2
    assert result.tool_calls_executed == 2


@pytest.mark.asyncio
async def test_on_tool_call_result_observer_receives_each_executed_call() -> None:
    observed: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    provider = _ScriptedProvider(
        [
            {"tool_calls": [{"name": "internal_do_thing", "arguments": {"x": 1}}]},
            {"status": "ok", "actions_taken": 1},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
        on_tool_call_result=lambda name, arguments, tool_result: observed.append(
            (name, arguments, tool_result)
        ),
    )

    assert result.tool_calls_executed == 1
    assert observed == [
        (
            "internal_do_thing",
            {"x": 1},
            {"status": "ok", "tool": "internal_do_thing", "arguments": {"x": 1}},
        )
    ]


@pytest.mark.asyncio
async def test_custom_unsupported_no_tool_response_error_overrides_default_when_default_allows() -> None:
    provider = _ScriptedProvider(
        [
            {"status": "ok", "actions_taken": 0},
            {"status": "ok", "actions_taken": 0, "final": True},
        ]
    )

    result = await run_bounded_tool_loop(
        provider,
        _dispatch_ok,
        prompt="do the thing",
        allowed_tools={"internal_do_thing": ToolSpec(name="internal_do_thing")},
        unsupported_no_tool_response_error=lambda response, executed: (
            "must retry" if not response.get("final") else None
        ),
    )

    assert result.response == {"status": "ok", "actions_taken": 0, "final": True}

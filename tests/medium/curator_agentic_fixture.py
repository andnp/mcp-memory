from __future__ import annotations

import inspect
import json
from typing import Any

from copilot.tools import ToolInvocation

from mcp_memory.core.providers.interfaces import AgenticRunResult


class _AgenticCuratorSession:
    def __init__(self, provider: Any, tools: list[Any]) -> None:
        self._provider = provider
        self._tools = tools
        self._feedback_turns = 0

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        if "curator's exploration phase" in prompt:
            payload = {"rounds": 1, "tool_calls": 1, "record_ids": []}
            return AgenticRunResult(status="success", parsed=payload, raw_text=json.dumps(payload))

        payload_start = prompt.find("{")
        payload, _ = json.JSONDecoder().raw_decode(prompt[payload_start:])
        plan = await self._provider.ask_json(
            "typed curation prompt\n" + json.dumps(payload, sort_keys=True)
        )
        if "Measured quality feedback:" in prompt:
            self._feedback_turns += 1
        if self._feedback_turns > 0:
            plan = {
                **plan,
                "actions": [],
                "retained": [
                    {
                        "memory_id": memory_id,
                        "reason": "already_focused",
                        "rationale": "the scripted provider has no adaptive correction",
                    }
                    for memory_id in plan.get("seed_memory_ids", [])
                ],
            }
        tool = next(tool for tool in self._tools if tool.name == "submit_curation_plan")
        assert tool.handler is not None
        result = tool.handler(
            ToolInvocation(
                tool_name=tool.name,
                arguments={"plan": plan},
            )
        )
        if inspect.isawaitable(result):
            await result
        return AgenticRunResult(status="success", raw_text=json.dumps(plan))

    async def close(self) -> None:
        return None


class AgenticCuratorProvider:
    def __init__(self, planner_provider: Any) -> None:
        self._planner_provider = planner_provider
        self.provider_trust_class = getattr(planner_provider, "provider_trust_class", "local")

    def with_usage_context(self, **context: object) -> AgenticCuratorProvider:
        del context
        return self

    async def open_agent_session(
        self,
        *,
        allowed_tool_names: tuple[str, ...] | None = None,
        tools: list[Any] | None = None,
    ) -> _AgenticCuratorSession:
        del allowed_tool_names
        return _AgenticCuratorSession(self._planner_provider, tools or [])

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        del prompt
        payload = {"rounds": 1, "tool_calls": 1, "record_ids": []}
        return AgenticRunResult(status="success", parsed=payload, raw_text=json.dumps(payload))


def agentic_curator(planner_provider: Any) -> AgenticCuratorProvider:
    return AgenticCuratorProvider(planner_provider)

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mcp_memory.core.curation_investigation import (
    READ_ONLY_CURATOR_INVESTIGATION_TOOLS,
    CurationInvestigationLimits,
    run_curator_investigation,
)


class _Investigator:
    def __init__(self, parsed: dict[str, object]) -> None:
        self.parsed = parsed
        self.allowed_tools: tuple[str, ...] | None = None
        self.prompt = ""

    def with_allowed_tool_names(self, names: tuple[str, ...]) -> _Investigator:
        self.allowed_tools = names
        return self

    async def run_agent(self, prompt: str) -> SimpleNamespace:
        self.prompt = prompt
        return SimpleNamespace(
            status="success",
            parsed=self.parsed,
            raw_text=json.dumps(self.parsed),
        )


@pytest.mark.asyncio
async def test_investigation_scopes_tools_and_bounds_prompt() -> None:
    provider = _Investigator(
        {
            "rounds": 2,
            "tool_calls": 3,
            "record_ids": ["record-a", "record-b"],
        }
    )
    result = await run_curator_investigation(
        provider,
        seed_records=[],
        task_id="task-1",
        limits=CurationInvestigationLimits(
            max_rounds=2,
            max_tool_calls=3,
            max_context_characters=500,
            max_result_characters=500,
            max_records=2,
        ),
    )

    assert result.status == "completed"
    assert result.record_ids == ("record-a", "record-b")
    assert provider.allowed_tools == READ_ONLY_CURATOR_INVESTIGATION_TOOLS
    prompt_payload = json.loads(provider.prompt.split("\n", 1)[1])
    assert prompt_payload["limits"]["max_rounds"] == 2
    assert prompt_payload["limits"]["max_tool_calls"] == 3
    assert prompt_payload["available_tools"] == list(READ_ONLY_CURATOR_INVESTIGATION_TOOLS)
    assert "internal_update_memory_record" not in provider.prompt


@pytest.mark.asyncio
async def test_investigation_rejects_reported_budget_overrun_and_large_result() -> None:
    overrun = _Investigator({"rounds": 3, "tool_calls": 1, "record_ids": ["id"]})
    result = await run_curator_investigation(
        overrun,
        seed_records=[],
        task_id="task-1",
        limits=CurationInvestigationLimits(max_rounds=2, max_result_characters=500),
    )
    assert result.status == "rejected"
    assert result.reason == "investigation_budget_exceeded"

    oversized = _Investigator({"rounds": 1, "tool_calls": 1, "record_ids": ["x" * 100]})
    result = await run_curator_investigation(
        oversized,
        seed_records=[],
        task_id="task-1",
        limits=CurationInvestigationLimits(max_result_characters=20),
    )
    assert result.status == "rejected"
    assert result.reason == "result_too_large"

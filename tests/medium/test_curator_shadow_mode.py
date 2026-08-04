from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from copilot.tools import ToolInvocation

from mcp_memory.config import CurationConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.tasks import TaskRecord


pytestmark = pytest.mark.medium


def test_curator_config_has_no_legacy_execution_flags() -> None:
    config = CurationConfig()

    assert not hasattr(config, "shadow_mode_enabled")
    assert not hasattr(config, "normalize_execution_enabled")
    assert not hasattr(config, "create_link_execution_enabled")


def test_curator_requires_an_agentic_provider() -> None:
    from mcp_memory.core.curation_shadow import _curator_agentic_provider

    task = TaskRecord(
        id="curator-agentic-task",
        task_name=CURATOR_TASK_NAME,
        data={},
        workspace_id="workspace-a",
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
        execution_epoch=7,
    )

    assert (
        _curator_agentic_provider(
            cast(ApplicationContext, SimpleNamespace(ai_agent_provider=None)),
            task,
            object(),
        )
        is None
    )


def _incremental_tools(max_proposed_actions: int | None = None) -> tuple[Any, Any, Any, Any, Any]:
    from mcp_memory.core.curation_models import CurationPlanningRequest
    from mcp_memory.core.curation_planner import IncrementalCurationPlanState
    from mcp_memory.core.curation_shadow import _incremental_curation_tools

    state = IncrementalCurationPlanState(max_proposed_actions=max_proposed_actions)
    seed_id = uuid4()
    state.begin(
        CurationPlanningRequest(
            run_id=uuid4(),
            plan_id=uuid4(),
            frontier_key="frontier",
            context_fingerprint="context",
        ),
        seed_memory_ids=(seed_id,),
    )
    tools = _incremental_curation_tools(state)
    return (state, *tools)


def _normalize_action(seed_id: str, *, summary: str = "Specific summary.") -> dict[str, object]:
    return {
        "operation": "normalize_memory",
        "action_id": str(uuid4()),
        "target_id": seed_id,
        "summary": summary,
        "confidence": 0.9,
        "rationale": "make the memory more specific",
    }


def test_incremental_tools_return_model_feedback_for_invalid_actions() -> None:
    state, propose, *_ = _incremental_tools()

    result = propose.handler(
        ToolInvocation(tool_name=propose.name, arguments={"action": {"operation": "normalize_memory"}})
    )

    assert "structurally invalid" in result.text_result_for_llm
    assert state.require_builder().action_count == 0


def test_incremental_tools_support_replacement_removal_and_budget_feedback() -> None:
    state, propose, replace, remove, _ = _incremental_tools(max_proposed_actions=2)
    seed_id = str(state.require_builder().seed_memory_ids[0])
    action = _normalize_action(seed_id)

    first = propose.handler(ToolInvocation(tool_name=propose.name, arguments={"action": action}))
    duplicate = propose.handler(ToolInvocation(tool_name=propose.name, arguments={"action": action}))
    assert first.text_result_for_llm == "Curation action added to the pending plan."
    assert "already been proposed" in duplicate.text_result_for_llm

    replacement = {**action, "summary": "Revised specific summary."}
    replaced = replace.handler(ToolInvocation(tool_name=replace.name, arguments={"action": replacement}))
    assert replaced.text_result_for_llm == "Curation action replaced in the pending plan."
    assert state.require_builder().build(rationale="test").actions[0].summary == "Revised specific summary."

    second = _normalize_action(seed_id)
    propose.handler(ToolInvocation(tool_name=propose.name, arguments={"action": second}))
    budget = propose.handler(
        ToolInvocation(tool_name=propose.name, arguments={"action": _normalize_action(seed_id)})
    )
    assert "budget is exhausted" in budget.text_result_for_llm

    removed = remove.handler(
        ToolInvocation(
            tool_name=remove.name,
            arguments={"action_id": action["action_id"]},
        )
    )
    assert removed.text_result_for_llm == "Curation action removed from the pending plan."
    assert state.require_builder().action_count == 1


def test_incremental_tools_validate_retention_decisions() -> None:
    state, *_tools, retain = _incremental_tools()
    invalid = retain.handler(
        ToolInvocation(tool_name=retain.name, arguments={"decision": {"memory_id": str(uuid4())}})
    )

    assert "rejected as invalid" in invalid.text_result_for_llm
    seed_id = str(state.require_builder().seed_memory_ids[0])
    valid = retain.handler(
        ToolInvocation(
            tool_name=retain.name,
            arguments={
                "decision": {
                    "memory_id": seed_id,
                    "reason": "already_focused",
                    "rationale": "the memory is already focused",
                }
            },
        )
    )
    assert valid.text_result_for_llm == "Retention decision added to the pending plan."
    assert len(state.require_builder().build(rationale="test").retained) == 1

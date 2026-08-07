from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.config import CurationConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_investigation import CURATOR_AGENT_TOOLS
from mcp_memory.core.curation_validation import CurationMutationBudget
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


def test_curator_prompt_requires_direct_mcp_mutations() -> None:
    from mcp_memory.core.curation_shadow import _direct_curator_prompt

    task = TaskRecord(
        id="direct-mcp-task",
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

    prompt = _direct_curator_prompt(
        task=task,
        seed_records=[],
        campaign_hypothesis=None,
        mutation_budget=CurationMutationBudget(max_accepted_mutations=2),
    )

    assert "no planning or submission stage" in prompt
    assert "submit_curation_plan" not in prompt
    assert all(tool_name.startswith("internal_") for tool_name in CURATOR_AGENT_TOOLS)

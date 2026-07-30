from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.config import CurationConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.core.tasks import TaskRecord


pytestmark = pytest.mark.medium


class _ContextBindingJSONProvider:
    def __init__(self) -> None:
        self.context: dict[str, object] | None = None

    def with_usage_context(self, **context: object) -> "_ContextBindingJSONProvider":
        bound = _ContextBindingJSONProvider()
        bound.context = context
        return bound

    async def ask_json(self, prompt: str) -> dict[str, object]:
        return {}


def test_curator_config_has_no_legacy_execution_flags() -> None:
    config = CurationConfig()

    assert not hasattr(config, "shadow_mode_enabled")
    assert not hasattr(config, "normalize_execution_enabled")
    assert not hasattr(config, "create_link_execution_enabled")


def test_curator_json_provider_binds_registry_fallback_to_task() -> None:
    from mcp_memory.core.curation_shadow import _curator_json_provider

    registry_provider = _ContextBindingJSONProvider()
    task = TaskRecord(
        id="curator-fallback-task",
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

    bound = _curator_json_provider(
        cast(ApplicationContext, SimpleNamespace(ai_json_provider=registry_provider)),
        task,
        object(),
    )

    assert bound is not registry_provider
    assert bound.context == {
        "task_name": CURATOR_TASK_NAME,
        "task_id": "curator-fallback-task",
        "execution_epoch": 7,
        "workspace_id": "workspace-a",
    }

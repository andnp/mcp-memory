from __future__ import annotations

from mcp_memory.config import AIConfig, Config, ProviderRoutingConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.agent_runtime import _provider_for_task
from mcp_memory.core.tasks import TaskRecord


class _FakeProvider:
    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self.available = available

    def budget_available(self) -> bool:
        return self.available

    def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
        return {
            "provider": self.name,
            "task_name": task_name,
            "task_id": task_id,
            "workspace_id": workspace_id,
        }


def test_provider_for_task_uses_fallback_route_when_first_provider_is_over_budget() -> None:
    ctx = ApplicationContext(
        config=Config(
            ai=AIConfig(provider="none"),
            provider_routing=ProviderRoutingConfig(
                task_routes={"summarize-memory": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-cli", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=False)},
            "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=True)},
        },
    )
    task = TaskRecord(
        id="summary-task",
        task_name="summarize-memory",
        data={"memory_id": "abc"},
        workspace_id="workspace-a",
        status="pending",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
    )

    selected = _provider_for_task(ctx, None, None, "summarize-memory", task)

    assert selected == {
        "provider": "gemini-cheap",
        "task_name": "summarize-memory",
        "task_id": "summary-task",
        "workspace_id": "workspace-a",
    }


def test_provider_for_task_falls_back_to_legacy_default_when_routed_profile_is_unavailable() -> None:
    default_provider = _FakeProvider("default-provider", available=True)
    ctx = ApplicationContext(
        config=Config(
            ai=AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
            provider_routing=ProviderRoutingConfig(
                task_routes={"summarize-memory": ["copilot-mini"]},
                profiles={"copilot-mini": AIConfig(provider="copilot-cli", model="gpt-5-mini")},
            ),
        ),
        ai_provider_registry={},
    )
    task = TaskRecord(
        id="summary-task",
        task_name="summarize-memory",
        data={"memory_id": "abc"},
        workspace_id="workspace-a",
        status="pending",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
    )

    selected = _provider_for_task(ctx, default_provider, None, "summarize-memory", task)

    assert selected == {
        "provider": "default-provider",
        "task_name": "summarize-memory",
        "task_id": "summary-task",
        "workspace_id": "workspace-a",
    }


def test_provider_for_agentic_task_uses_next_agentic_route_before_deterministic() -> None:
    ctx = ApplicationContext(
        config=Config(
            ai=AIConfig(provider="none"),
            provider_routing=ProviderRoutingConfig(
                task_routes={"deduplicator": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-cli", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {
                "agentic": _FakeProvider("copilot-mini-agentic", available=False),
            },
            "gemini-cheap": {
                "agentic": _FakeProvider("gemini-cheap-agentic", available=True),
            }
        },
    )
    task = TaskRecord(
        id="dedup-task",
        task_name="deduplicator",
        data={},
        workspace_id="workspace-a",
        status="pending",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
    )

    selected = _provider_for_task(ctx, None, None, "deduplicator", task)

    assert selected == {
        "provider": "gemini-cheap-agentic",
        "task_name": "deduplicator",
        "task_id": "dedup-task",
        "workspace_id": "workspace-a",
    }

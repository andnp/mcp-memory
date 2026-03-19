from __future__ import annotations

import logging
from types import SimpleNamespace

from mcp_memory.config import AIConfig, Config, ProviderRoutingConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.agent_runtime import _provider_for_task
from mcp_memory.core.provider_policy import (
    ProviderSelectionInputs,
    ProviderSelectionRequest,
    select_provider_for_inputs,
    select_provider_for_request,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.management.route_audit import build_task_route_audit


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

    assert selected is None


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

    assert selected is None


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


def test_provider_for_task_returns_none_when_all_routed_providers_are_over_budget(caplog) -> None:
    ctx = ApplicationContext(
        config=Config(
            ai=AIConfig(provider="none"),
            provider_routing=ProviderRoutingConfig(
                    task_routes={"graph-linker": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-cli", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=False)},
            "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=False)},
        },
    )
    task = TaskRecord(
        id="summary-task",
        task_name="graph-linker",
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

    with caplog.at_level(logging.WARNING):
        selected = _provider_for_task(ctx, None, None, "graph-linker", task)

    assert selected is None
    assert any("exhausted all configured routes" in message for message in caplog.messages)


def test_provider_for_deterministic_task_never_selects_provider() -> None:
    default_provider = _FakeProvider("default-provider", available=True)
    ctx = ApplicationContext(
        config=Config(
            ai=AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
            provider_routing=ProviderRoutingConfig(),
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

    assert selected is None


def test_select_provider_for_inputs_matches_context_wrapper_for_route_selection() -> None:
    config = Config(
        ai=AIConfig(provider="none"),
        provider_routing=ProviderRoutingConfig(
            task_routes={"deduplicator": ["gemini-cheap"]},
            profiles={
                "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
            },
        ),
    )
    registry = {
        "gemini-cheap": {"agentic": _FakeProvider("gemini-cheap-agentic", available=True)},
    }
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

    selected = select_provider_for_inputs(
        ProviderSelectionInputs(config=config, ai_provider_registry=registry),
        None,
        None,
        "deduplicator",
        task,
        agentic_task_names={"deduplicator"},
    )

    assert selected == {
        "provider": "gemini-cheap-agentic",
        "task_name": "deduplicator",
        "task_id": "dedup-task",
        "workspace_id": "workspace-a",
    }


def test_select_provider_for_request_matches_task_wrapper_for_route_selection() -> None:
    config = Config(
        ai=AIConfig(provider="none"),
        provider_routing=ProviderRoutingConfig(
            task_routes={"deduplicator": ["gemini-cheap"]},
            profiles={
                "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
            },
        ),
    )
    registry = {
        "gemini-cheap": {"agentic": _FakeProvider("gemini-cheap-agentic", available=True)},
    }
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
    inputs = ProviderSelectionInputs(config=config, ai_provider_registry=registry)

    selected_from_task = select_provider_for_inputs(
        inputs,
        None,
        None,
        "deduplicator",
        task,
        agentic_task_names={"deduplicator"},
    )
    selected_from_request = select_provider_for_request(
        inputs,
        None,
        None,
        ProviderSelectionRequest(task_name="deduplicator", task_id="dedup-task", workspace_id="workspace-a"),
        agentic_task_names={"deduplicator"},
    )

    assert selected_from_request == selected_from_task


def test_build_task_route_audit_selects_provider_without_fabricated_task_record() -> None:
    class _AuditProvider:
        def __init__(self, provider_key: str, model_name: str) -> None:
            self._provider_key = provider_key
            self._model_name = model_name
            self.bound_calls: list[dict[str, str | None]] = []

        def supports_agentic(self) -> bool:
            return True

        def with_usage_context(self, *, task_name: str | None, task_id: str | None = None, workspace_id: str | None = None):
            self.bound_calls.append(
                {
                    "task_name": task_name,
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                }
            )
            return SimpleNamespace(
                _provider_key=self._provider_key,
                _model_name=self._model_name,
                _provider=self,
                supports_agentic=self.supports_agentic,
            )

    provider = _AuditProvider("gemini-cheap", "gemini-3-flash-preview")
    config = Config(
        ai=AIConfig(provider="none"),
        provider_routing=ProviderRoutingConfig(
            task_routes={"deduplicator": ["gemini-cheap"]},
            profiles={
                "gemini-cheap": AIConfig(provider="gemini-cli", model="gemini-3-flash-preview"),
            },
        ),
    )
    provider_usage_repo = SimpleNamespace(
        summarize_usage=lambda workspace_id=None: [],
        list_conversations=lambda task_name, limit=1: [],
    )

    audits = build_task_route_audit(
        config=config,
        workspace_id="workspace-a",
        provider_usage_repo=provider_usage_repo,
        ai_json_provider=None,
        ai_agent_provider=None,
        ai_provider_registry={"gemini-cheap": {"agentic": provider}},
    )

    deduplicator_audit = next(item for item in audits if item.task_name == "deduplicator")

    assert deduplicator_audit.resolved_provider_key == "gemini-cheap"
    assert deduplicator_audit.resolved_model_name == "gemini-3-flash-preview"
    assert deduplicator_audit.resolved_provider_type == "_AuditProvider"
    assert deduplicator_audit.resolved_supports_agentic is True
    assert provider.bound_calls == [
        {
            "task_name": "deduplicator",
            "task_id": "audit:deduplicator",
            "workspace_id": "workspace-a",
        }
    ]

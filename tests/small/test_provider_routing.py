from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import mcp_memory.core.provider_policy as provider_policy_module

from mcp_memory.config import AIConfig, Config, ProviderRoutingConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.core.agent_runtime import _provider_for_task
from mcp_memory.core.provider_policy import (
    AgenticRouteFailoverProvider,
    ProviderSelectionInputs,
    ProviderSelectionRequest,
    select_provider_for_inputs,
    select_provider_for_request,
)
from mcp_memory.core.providers.interfaces import ProviderAuthenticationRequired
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.management.route_audit import build_task_route_audit


class _FakeProvider:
    def __init__(self, name: str, *, available: bool = True) -> None:
        self.name = name
        self.available = available
        self.skipped: list[str | None] = []

    def budget_available(self) -> bool:
        return self.available

    def record_admission_skip(self, decision) -> None:
        self.skipped.append(decision.reason)

    def with_usage_context(
        self,
        *,
        task_name: str | None,
        task_id: str | None = None,
        execution_epoch: int | None = None,
        workspace_id: str | None = None,
    ) -> object:
        return {
            "provider": self.name,
            "task_name": task_name,
            "task_id": task_id,
            "workspace_id": workspace_id,
        }


class _FakePolicyEventRepository:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record_event(self, **kwargs) -> None:
        self.events.append(kwargs)


def test_provider_for_task_uses_fallback_route_when_first_provider_is_over_budget() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={"summarize-memory": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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
            provider_routing=ProviderRoutingConfig(
                task_routes={"summarize-memory": ["copilot-mini"]},
                profiles={"copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini")},
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


def test_provider_for_taxonomist_uses_next_json_route_before_legacy_default() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={"taxonomist": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=False)},
            "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=True)},
        },
    )
    task = TaskRecord(
        id="taxonomist-task",
        task_name="taxonomist",
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

    selected = _provider_for_task(ctx, None, None, "taxonomist", task)

    assert selected == {
        "provider": "gemini-cheap",
        "task_name": "taxonomist",
        "task_id": "taxonomist-task",
        "workspace_id": "workspace-a",
    }


def test_provider_for_task_uses_next_route_when_first_model_is_burst_limited() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={"taxonomist": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
                },
                model_burst_call_limit=1,
                model_burst_window_seconds=600.0,
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=False)},
            "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=True)},
        },
    )
    task = TaskRecord(
        id="taxonomist-task",
        task_name="taxonomist",
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

    selected = _provider_for_task(ctx, None, None, "taxonomist", task)

    assert selected == {
        "provider": "gemini-cheap",
        "task_name": "taxonomist",
        "task_id": "taxonomist-task",
        "workspace_id": "workspace-a",
    }


def test_provider_for_runtime_providerless_structural_seeders_returns_none_even_with_routes() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={
                    "deduplicator": ["gemini-cheap"],
                    "conflict-detector": ["gemini-cheap"],
                },
                profiles={
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "gemini-cheap": {
                "agentic": _FakeProvider("gemini-cheap-agentic", available=True),
                "json": _FakeProvider("gemini-cheap", available=True),
            }
        },
    )

    for task_name in ("deduplicator", "conflict-detector"):
        task = TaskRecord(
            id=f"{task_name}-task",
            task_name=task_name,
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

        assert _provider_for_task(ctx, None, None, task_name, task) is None


def test_provider_for_runtime_providerless_structural_seeders_ignores_task_class_overrides() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_classes={
                    "deduplicator": "cheap_agentic",
                    "conflict-detector": "cheap_json",
                },
                task_routes={
                    "deduplicator": ["gemini-cheap"],
                    "conflict-detector": ["gemini-cheap"],
                },
                profiles={
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "gemini-cheap": {
                "agentic": _FakeProvider("gemini-cheap-agentic", available=True),
                "json": _FakeProvider("gemini-cheap", available=True),
            }
        },
    )

    for task_name in ("deduplicator", "conflict-detector"):
        task = TaskRecord(
            id=f"override:{task_name}",
            task_name=task_name,
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

        assert _provider_for_task(ctx, None, None, task_name, task) is None


def test_provider_for_task_returns_none_when_all_routed_providers_are_over_budget(caplog) -> None:
    provider_policy_module._provider_warning_state.clear()
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                    task_routes={"graph-linker": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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


def test_provider_for_taxonomist_prefers_default_non_agentic_routes() -> None:
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=True)},
        },
    )
    task = TaskRecord(
        id="taxonomist-task",
        task_name="taxonomist",
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

    selected = _provider_for_task(ctx, None, None, "taxonomist", task)

    assert selected == {
        "provider": "copilot-mini",
        "task_name": "taxonomist",
        "task_id": "taxonomist-task",
        "workspace_id": "workspace-a",
    }


def test_provider_for_task_records_first_class_route_events_when_routes_exhausted() -> None:
    provider_policy_module._provider_warning_state.clear()
    event_repo = _FakePolicyEventRepository()
    task = TaskRecord(
        id="graph-task",
        task_name="graph-linker",
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
    inputs = ProviderSelectionInputs(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={"graph-linker": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
                },
            ),
        ),
        ai_provider_registry={
            "copilot-mini": {"json": _FakeProvider("copilot-mini", available=False)},
            "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=False)},
        },
        provider_policy_events=event_repo,
    )

    first = select_provider_for_inputs(inputs, None, None, "graph-linker", task, agentic_task_names=set())
    second = select_provider_for_inputs(inputs, None, None, "graph-linker", task, agentic_task_names=set())

    assert first is None
    assert second is None
    assert [event["event_kind"] for event in event_repo.events] == [
        "route_skipped",
        "route_skipped",
        "route_exhausted",
        "route_skipped",
        "route_skipped",
        "route_exhausted",
    ]
    assert event_repo.events[2]["candidate_routes"] == ["copilot-mini", "gemini-cheap"]
    assert event_repo.events[2]["warning_suppressed"] is False
    assert event_repo.events[-1]["warning_suppressed"] is True


def test_provider_for_task_throttles_duplicate_routed_warning_logs(caplog) -> None:
    provider_policy_module._provider_warning_state.clear()
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(
                task_routes={"graph-linker": ["copilot-mini", "gemini-cheap"]},
                profiles={
                    "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                    "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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
        first = _provider_for_task(ctx, None, None, "graph-linker", task)
        second = _provider_for_task(ctx, None, None, "graph-linker", task)

    assert first is None
    assert second is None
    assert [record.message for record in caplog.records if "exhausted all configured routes" in record.message] == [
        "Provider routing exhausted all configured routes"
    ]


def test_provider_for_task_throttles_duplicate_legacy_fallback_warning_logs(caplog) -> None:
    provider_policy_module._provider_warning_state.clear()
    default_provider = _FakeProvider("default-provider", available=False)
    ctx = ApplicationContext(
        config=Config(
            provider_routing=ProviderRoutingConfig(),
        ),
        ai_provider_registry={},
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
        first = _provider_for_task(ctx, default_provider, None, "graph-linker", task)
        second = _provider_for_task(ctx, default_provider, None, "graph-linker", task)

    assert first is None
    assert second is None
    assert [
        record.message
        for record in caplog.records
        if "Legacy fallback provider is unavailable due to admission control" in record.message
    ] == ["Legacy fallback provider is unavailable due to admission control"]


def test_provider_for_deterministic_task_never_selects_provider() -> None:
    default_provider = _FakeProvider("default-provider", available=True)
    ctx = ApplicationContext(
        config=Config(
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
        provider_routing=ProviderRoutingConfig(
            task_routes={"taxonomist": ["gemini-cheap"]},
            profiles={
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
            },
        ),
    )
    registry = {
        "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=True)},
    }
    task = TaskRecord(
        id="taxonomist-task",
        task_name="taxonomist",
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
        "taxonomist",
        task,
        agentic_task_names=set(),
    )

    assert selected == {
        "provider": "gemini-cheap",
        "task_name": "taxonomist",
        "task_id": "taxonomist-task",
        "workspace_id": "workspace-a",
    }


def test_select_provider_for_request_matches_task_wrapper_for_route_selection() -> None:
    config = Config(
        provider_routing=ProviderRoutingConfig(
            task_routes={"taxonomist": ["gemini-cheap"]},
            profiles={
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
            },
        ),
    )
    registry = {
        "gemini-cheap": {"json": _FakeProvider("gemini-cheap", available=True)},
    }
    task = TaskRecord(
        id="taxonomist-task",
        task_name="taxonomist",
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
        "taxonomist",
        task,
        agentic_task_names=set(),
    )
    selected_from_request = select_provider_for_request(
        inputs,
        None,
        None,
        ProviderSelectionRequest(task_name="taxonomist", task_id="taxonomist-task", workspace_id="workspace-a"),
        agentic_task_names=set(),
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

        def with_usage_context(
            self,
            *,
            task_name: str | None,
            task_id: str | None = None,
            execution_epoch: int | None = None,
            workspace_id: str | None = None,
        ):
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
        provider_routing=ProviderRoutingConfig(
            task_routes={
                "deduplicator": ["gemini-cheap"],
                "taxonomist": ["gemini-cheap"],
            },
            profiles={
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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
        ai_provider_registry={
            "gemini-cheap": {
                "agentic": provider,
                "json": provider,
            }
        },
    )

    deduplicator_audit = next(item for item in audits if item.task_name == "deduplicator")
    taxonomist_audit = next(item for item in audits if item.task_name == "taxonomist")

    assert deduplicator_audit.resolved_provider_key is None
    assert deduplicator_audit.resolved_model_name is None
    assert deduplicator_audit.resolved_provider_type is None
    assert deduplicator_audit.resolved_supports_agentic is None
    assert taxonomist_audit.resolved_provider_key == "gemini-cheap"
    assert taxonomist_audit.resolved_model_name == "gemini-3-flash-preview"
    assert taxonomist_audit.resolved_provider_type == "_AuditProvider"
    assert taxonomist_audit.resolved_supports_agentic is True
    assert {
        (call["task_name"], call["task_id"], call["workspace_id"])
        for call in provider.bound_calls
    } == {
        ("taxonomist", "audit:taxonomist", "workspace-a"),
    }


def test_build_task_route_audit_suppresses_runtime_only_structural_seeder_provider_resolution() -> None:
    class _AuditProvider:
        def __init__(self) -> None:
            self.bound_calls: list[dict[str, str | None]] = []

        def supports_agentic(self) -> bool:
            return True

        def with_usage_context(
            self,
            *,
            task_name: str | None,
            task_id: str | None = None,
            execution_epoch: int | None = None,
            workspace_id: str | None = None,
        ):
            self.bound_calls.append(
                {
                    "task_name": task_name,
                    "task_id": task_id,
                    "workspace_id": workspace_id,
                }
            )
            return SimpleNamespace(
                _provider_key="gemini-cheap",
                _model_name="gemini-3-flash-preview",
                _provider=self,
                supports_agentic=self.supports_agentic,
            )

    provider = _AuditProvider()
    config = Config(
        provider_routing=ProviderRoutingConfig(
            task_routes={
                "deduplicator": ["gemini-cheap"],
                "conflict-detector": ["gemini-cheap"],
            },
            profiles={
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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
        ai_provider_registry={"gemini-cheap": {"agentic": provider, "json": provider}},
    )

    suppressed_task_names = {"deduplicator", "conflict-detector"}
    suppressed_audits = [item for item in audits if item.task_name in suppressed_task_names]

    assert {item.task_name for item in suppressed_audits} == suppressed_task_names
    assert all(item.resolved_provider_key is None for item in suppressed_audits)
    assert all(item.resolved_model_name is None for item in suppressed_audits)
    assert all(item.resolved_provider_type is None for item in suppressed_audits)
    assert all(item.resolved_supports_agentic is None for item in suppressed_audits)
    assert all(call["task_name"] not in suppressed_task_names for call in provider.bound_calls)


def test_build_task_route_audit_keeps_providerless_structural_seeders_deterministic_under_overrides() -> None:
    provider = _FakeProvider("gemini-cheap", available=True)
    config = Config(
        provider_routing=ProviderRoutingConfig(
            task_classes={
                "deduplicator": "cheap_agentic",
                "conflict-detector": "cheap_json",
            },
            task_routes={
                "deduplicator": ["gemini-cheap"],
                "conflict-detector": ["gemini-cheap"],
            },
            profiles={
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
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
        ai_provider_registry={"gemini-cheap": {"agentic": provider, "json": provider}},
    )

    for task_name in ("deduplicator", "conflict-detector"):
        audit = next(item for item in audits if item.task_name == task_name)
        assert audit.task_class == "deterministic"
        assert audit.execution_kind == "deterministic"
        assert audit.configured_primary_route is None
        assert audit.configured_fallback_routes == []
        assert audit.resolved_provider_key is None


def test_select_provider_for_request_records_real_route_skip_but_route_audit_does_not() -> None:
    class _BoundSkippingProvider(SimpleNamespace):
        def record_admission_skip(self, decision) -> None:
            self.skipped.append(decision.reason)

    class _SkippingProvider(_FakeProvider):
        def with_usage_context(
            self,
            *,
            task_name: str | None,
            task_id: str | None = None,
            execution_epoch: int | None = None,
            workspace_id: str | None = None,
        ) -> object:
            return _BoundSkippingProvider(
                provider=self.name,
                task_name=task_name,
                task_id=task_id,
                workspace_id=workspace_id,
                skipped=self.skipped,
            )

    unavailable = _SkippingProvider("copilot-mini", available=False)
    fallback = _FakeProvider("gemini-cheap", available=True)
    config = Config(
        provider_routing=ProviderRoutingConfig(
            task_routes={"graph-linker": ["copilot-mini", "gemini-cheap"]},
            profiles={
                "copilot-mini": AIConfig(provider="copilot-sdk", model="gpt-5-mini"),
                "gemini-cheap": AIConfig(provider="copilot-sdk", model="gemini-3-flash-preview"),
            },
        ),
    )
    inputs = ProviderSelectionInputs(
        config=config,
        ai_provider_registry={
            "copilot-mini": {"json": unavailable},
            "gemini-cheap": {"json": fallback},
        },
    )

    selected = select_provider_for_request(
        inputs,
        None,
        None,
        ProviderSelectionRequest(task_name="graph-linker", task_id="task-1", workspace_id="workspace-a"),
        agentic_task_names=set(),
    )

    assert selected == {
        "provider": "gemini-cheap",
        "task_name": "graph-linker",
        "task_id": "task-1",
        "workspace_id": "workspace-a",
    }
    assert unavailable.skipped == [None]

    select_provider_for_request(
        inputs,
        None,
        None,
        ProviderSelectionRequest(task_name="graph-linker", task_id="audit:graph-linker", workspace_id="workspace-a"),
        agentic_task_names=set(),
        record_admission_skips=False,
    )

    assert unavailable.skipped == [None]


@pytest.mark.asyncio
async def test_agentic_route_failover_provider_skips_to_next_route_on_auth_required() -> None:
    class _AuthRequiredProvider:
        async def run_agent(self, prompt: str):
            raise ProviderAuthenticationRequired("Gemini CLI Agentic", error_text="Interactive authentication required")

    class _HealthyProvider:
        async def run_agent(self, prompt: str):
            return {"provider": "fallback", "prompt": prompt}

    provider = AgenticRouteFailoverProvider(
        [_AuthRequiredProvider(), _HealthyProvider()],
        route_keys=["gemini-cheap", "copilot-mini"],
        task_name="deduplicator",
    )

    result = await provider.run_agent("continue with fallback")

    assert result == {"provider": "fallback", "prompt": "continue with fallback"}

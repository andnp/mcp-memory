from __future__ import annotations

from pathlib import Path

import pytest

from mcp_memory.context import (
    ApplicationContext,
    BackgroundTaskCapabilities,
    ManagementRuntimeCapabilities,
    MemoryReadCapabilities,
    MutationCapabilities,
    ProviderCapabilities,
    TaskRuntimeCapabilities,
)

pytestmark = pytest.mark.small


def test_management_view_exposes_only_management_capabilities() -> None:
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        workspace_root=Path("/tmp/workspace"),
        memory_path=Path("/tmp/memory"),
        db_manager=object(),
        repository=object(),
        internal_tool_call_tracker=object(),
    )

    view = ctx.management_view()

    assert view.workspace_id == "workspace-a"
    assert view.repository is ctx.repository

    view.provider_usage = "usage"
    assert ctx.provider_usage == "usage"

    with pytest.raises(AttributeError):
        getattr(view, "internal_tool_call_tracker")


def test_task_runtime_view_exposes_only_runtime_capabilities() -> None:
    tracker = object()
    action_store = object()
    quality_store = object()
    direct_evidence = object()
    ctx = ApplicationContext(
        session_id="session-a",
        task_queue="queue",
        curation_action_store=action_store,
        curation_quality=quality_store,
        direct_mutation_evidence=direct_evidence,
        ai_json_provider="json",
        ai_agent_provider="agent",
        provider_policy_events="events",
        journal="journal",
        internal_tool_call_tracker=tracker,
    )

    view = ctx.task_runtime_view()

    assert view.task_queue == "queue"
    assert view.ai_json_provider == "json"
    assert view.session_id == "session-a"
    assert view.internal_tool_call_tracker is tracker
    assert view.curation_action_store is action_store
    assert view.curation_quality is quality_store
    assert view.direct_mutation_evidence is direct_evidence

    with pytest.raises(AttributeError):
        getattr(view, "read_cache")


def test_capability_bundles_isolate_concerns() -> None:
    ctx = ApplicationContext(
        repository="repository",
        task_queue="queue",
        read_cache="cache",
        ai_json_provider="json",
        provider_policy_events="policy",
        runtime_logs="logs",
        internal_tool_call_tracker="tracker",
    )

    memory = ctx.memory_capabilities()
    mutation = ctx.mutation_capabilities()
    provider = ctx.provider_capabilities()
    background = ctx.background_task_capabilities()
    task_runtime = ctx.task_runtime_capabilities()
    management = ctx.management_capabilities()

    assert isinstance(memory, MemoryReadCapabilities)
    assert memory.repository == "repository"
    assert not hasattr(memory, "task_queue")
    assert not hasattr(memory, "journal")
    assert isinstance(mutation, MutationCapabilities)
    assert mutation.task_queue == "queue"
    assert not hasattr(mutation, "read_cache")
    assert isinstance(provider, ProviderCapabilities)
    assert provider.ai_json_provider == "json"
    assert not hasattr(provider, "repository")
    assert isinstance(background, BackgroundTaskCapabilities)
    assert background.task_queue == "queue"
    assert isinstance(task_runtime, TaskRuntimeCapabilities)
    assert task_runtime.memory.repository == "repository"
    assert task_runtime.mutation.task_queue == "queue"
    assert isinstance(management, ManagementRuntimeCapabilities)
    assert management.runtime_logs == "logs"
    assert management.memory.read_cache == "cache"


def test_task_runtime_capabilities_adapt_to_legacy_handler_context() -> None:
    action_store = object()
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        session_id="session-a",
        repository="repository",
        task_queue="queue",
        curation_action_store=action_store,
        ai_json_provider="json",
        internal_tool_call_tracker="tracker",
    )

    adapted = ctx.task_runtime_capabilities().as_context()

    assert adapted.workspace_id == "workspace-a"
    assert adapted.session_id == "session-a"
    assert adapted.repository == "repository"
    assert adapted.task_queue == "queue"
    assert adapted.ai_json_provider == "json"
    assert adapted.internal_tool_call_tracker == "tracker"
    assert adapted.curation_action_store is action_store


def test_task_runtime_adapter_preserves_direct_quality_capabilities() -> None:
    """Direct curator handlers retain both durable evidence capabilities."""
    quality_store = object()
    direct_evidence = object()
    ctx = ApplicationContext(
        curation_quality=quality_store,
        direct_mutation_evidence=direct_evidence,
    )

    capabilities = ctx.task_runtime_capabilities()
    adapted = capabilities.as_context()

    assert capabilities.mutation.curation_quality is quality_store
    assert capabilities.mutation.direct_mutation_evidence is direct_evidence
    assert adapted.curation_quality is quality_store
    assert adapted.direct_mutation_evidence is direct_evidence

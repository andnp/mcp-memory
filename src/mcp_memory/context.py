from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.config import Config
from mcp_memory.core.ports import (
    EmbeddingMaintenancePort,
    MemoryIDResolutionPort,
    ReadCacheValidationPort,
    SearchHealthPort,
    StartupHealthPort,
)
from mcp_memory.core.ports.providers import (
    ProviderPolicyEventPort,
    ProviderUsagePort,
    TaskExecutionAttemptPort,
)
from mcp_memory.core.ports.tasks import TaskQueue
from mcp_memory.core.ports.work_items import WorkItemRepository
from mcp_memory.core.providers.interfaces import AgenticTaskProvider, JSONTaskProvider

TaskQueueProtocol = TaskQueue


class TaskQueueContext(Protocol):
    @property
    def task_queue(self) -> TaskQueueProtocol | None: ...


class MemoryPipelineContext(Protocol):
    config: Config | None
    workspace_id: str | None
    workspace_root: Path | None
    memory_path: Path | None
    db_manager: Any
    journal: Any
    repository: Any
    relational_search: MemorySearchPort | None
    task_queue: Any
    search_health: SearchHealthPort | None
    startup_health: StartupHealthPort | None
    read_cache_validation: ReadCacheValidationPort | None
    memory_id_resolution: MemoryIDResolutionPort | None


class ManagementContext(MemoryPipelineContext, Protocol):
    storage_backend: str | None
    read_cache: Any
    ai_json_provider: Any
    ai_agent_provider: Any
    ai_provider_registry: dict[str, Any] | None
    provider_usage: Any
    runtime_logs: Any
    retrieval_telemetry: Any
    embedding_integrity_events: Any
    embedder: Any
    vector_store: Any


class TaskRuntimeContext(MemoryPipelineContext, Protocol):
    session_id: str | None
    curation: Any
    curation_action_store: Any
    curation_quality: Any
    direct_mutation_evidence: Any
    ai_json_provider: Any
    ai_agent_provider: Any
    ai_provider_registry: dict[str, Any] | None
    provider_usage: Any
    provider_policy_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any
    internal_tool_call_tracker: Any


class ProviderSelectionContext(Protocol):
    config: Config | None
    ai_provider_registry: dict[str, Any] | None
    provider_policy_events: Any


class BackgroundTaskBootstrapContext(Protocol):
    config: Config | None
    journal: Any
    task_queue: Any


@dataclass(frozen=True)
class MemoryReadCapabilities:
    config: Config | None = None
    workspace_id: str | None = None
    workspace_root: Path | None = None
    memory_path: Path | None = None
    db_manager: object | None = None
    repository: object | None = None
    relational_search: MemorySearchPort | None = None
    read_cache: object | None = None
    embedder: object | None = None
    vector_store: object | None = None
    embedding_maintenance: EmbeddingMaintenancePort | None = None
    search_health: SearchHealthPort | None = None
    startup_health: StartupHealthPort | None = None
    read_cache_validation: ReadCacheValidationPort | None = None
    memory_id_resolution: MemoryIDResolutionPort | None = None

    @classmethod
    def from_context(cls, ctx: MemoryPipelineContext) -> MemoryReadCapabilities:
        return cls(
            config=ctx.config,
            workspace_id=ctx.workspace_id,
            workspace_root=ctx.workspace_root,
            memory_path=ctx.memory_path,
            db_manager=ctx.db_manager,
            repository=ctx.repository,
            relational_search=ctx.relational_search,
            read_cache=getattr(ctx, "read_cache", None),
            embedder=getattr(ctx, "embedder", None),
            vector_store=getattr(ctx, "vector_store", None),
            embedding_maintenance=getattr(ctx, "embedding_maintenance", None),
            search_health=getattr(ctx, "search_health", None),
            startup_health=getattr(ctx, "startup_health", None),
            read_cache_validation=getattr(ctx, "read_cache_validation", None),
            memory_id_resolution=getattr(ctx, "memory_id_resolution", None),
        )


@dataclass(frozen=True)
class MutationCapabilities:
    workspace_id: str | None = None
    journal: object | None = None
    repository: object | None = None
    relational_search: MemorySearchPort | None = None
    task_queue: object | None = None
    curation: object | None = None
    mutation_history: object | None = None
    curation_action_store: object | None = None
    curation_quality: object | None = None
    direct_mutation_evidence: object | None = None

    @classmethod
    def from_context(cls, ctx: MemoryPipelineContext) -> MutationCapabilities:
        return cls(
            workspace_id=ctx.workspace_id,
            journal=ctx.journal,
            repository=ctx.repository,
            relational_search=ctx.relational_search,
            task_queue=getattr(ctx, "task_queue", None),
            curation=getattr(ctx, "curation", None),
            mutation_history=getattr(ctx, "mutation_history", None),
            curation_action_store=getattr(ctx, "curation_action_store", None),
            curation_quality=getattr(ctx, "curation_quality", None),
            direct_mutation_evidence=getattr(ctx, "direct_mutation_evidence", None),
        )


@dataclass(frozen=True)
class ProviderCapabilities:
    config: Config | None = None
    ai_json_provider: JSONTaskProvider | None = None
    ai_agent_provider: AgenticTaskProvider | None = None
    ai_provider_registry: dict[str, dict[str, object]] | None = None
    provider_usage: ProviderUsagePort | None = None
    provider_policy_events: ProviderPolicyEventPort | None = None
    task_execution_attempts: TaskExecutionAttemptPort | None = None

    @classmethod
    def from_context(
        cls,
        ctx: ProviderSelectionContext | ManagementContext | TaskRuntimeContext,
    ) -> ProviderCapabilities:
        return cls(
            config=ctx.config,
            ai_json_provider=cast(JSONTaskProvider | None, getattr(ctx, "ai_json_provider", None)),
            ai_agent_provider=cast(AgenticTaskProvider | None, getattr(ctx, "ai_agent_provider", None)),
            ai_provider_registry=cast(
                dict[str, dict[str, object]] | None,
                ctx.ai_provider_registry,
            ),
            provider_usage=cast(ProviderUsagePort | None, getattr(ctx, "provider_usage", None)),
            provider_policy_events=getattr(ctx, "provider_policy_events", None),
            task_execution_attempts=cast(
                TaskExecutionAttemptPort | None,
                getattr(ctx, "task_execution_attempts", None),
            ),
        )


@dataclass(frozen=True)
class BackgroundTaskCapabilities:
    config: Config | None = None
    journal: object | None = None
    task_queue: TaskQueueProtocol | None = None

    @classmethod
    def from_context(
        cls,
        ctx: BackgroundTaskBootstrapContext,
    ) -> BackgroundTaskCapabilities:
        return cls(config=ctx.config, journal=ctx.journal, task_queue=ctx.task_queue)


@dataclass(frozen=True)
class TaskRuntimeCapabilities:
    memory: MemoryReadCapabilities
    mutation: MutationCapabilities
    provider: ProviderCapabilities
    session_id: str | None = None
    work_items: WorkItemRepository | None = None
    embedding_repair_queue: object | None = None
    internal_tool_call_tracker: object | None = None

    @classmethod
    def from_context(cls, ctx: TaskRuntimeContext) -> TaskRuntimeCapabilities:
        return cls(
            memory=MemoryReadCapabilities.from_context(ctx),
            mutation=MutationCapabilities.from_context(ctx),
            provider=ProviderCapabilities.from_context(ctx),
            session_id=ctx.session_id,
            work_items=ctx.work_items,
            embedding_repair_queue=getattr(ctx, "embedding_repair_queue", None),
            internal_tool_call_tracker=getattr(ctx, "internal_tool_call_tracker", None),
        )

    def as_context(self) -> TaskRuntimeContext:
        return cast(TaskRuntimeContext, _TaskRuntimeContextAdapter(
            config=self.provider.config,
            workspace_id=self.memory.workspace_id,
            workspace_root=self.memory.workspace_root,
            session_id=self.session_id,
            memory_path=self.memory.memory_path,
            db_manager=self.memory.db_manager,
            journal=self.mutation.journal,
            repository=self.mutation.repository,
            relational_search=self.mutation.relational_search,
            embedding_maintenance=self.memory.embedding_maintenance,
            search_health=self.memory.search_health,
            startup_health=self.memory.startup_health,
            read_cache_validation=self.memory.read_cache_validation,
            memory_id_resolution=self.memory.memory_id_resolution,
            task_queue=self.mutation.task_queue,
            curation=self.mutation.curation,
            curation_action_store=self.mutation.curation_action_store,
            curation_quality=self.mutation.curation_quality,
            direct_mutation_evidence=self.mutation.direct_mutation_evidence,
            mutation_history=self.mutation.mutation_history,
            ai_json_provider=self.provider.ai_json_provider,
            ai_agent_provider=self.provider.ai_agent_provider,
            ai_provider_registry=self.provider.ai_provider_registry,
            provider_usage=self.provider.provider_usage,
            provider_policy_events=self.provider.provider_policy_events,
            task_execution_attempts=self.provider.task_execution_attempts,
            work_items=self.work_items,
            embedding_repair_queue=self.embedding_repair_queue,
            internal_tool_call_tracker=self.internal_tool_call_tracker,
        ))


@dataclass(frozen=True)
class ManagementRuntimeCapabilities:
    memory: MemoryReadCapabilities
    mutation: MutationCapabilities
    provider: ProviderCapabilities
    storage_backend: str | None = None
    runtime_logs: object | None = None
    retrieval_telemetry: object | None = None
    embedding_integrity_events: object | None = None

    @classmethod
    def from_context(cls, ctx: ManagementContext) -> ManagementRuntimeCapabilities:
        return cls(
            memory=MemoryReadCapabilities.from_context(ctx),
            mutation=MutationCapabilities.from_context(ctx),
            provider=ProviderCapabilities.from_context(ctx),
            storage_backend=ctx.storage_backend,
            runtime_logs=ctx.runtime_logs,
            retrieval_telemetry=ctx.retrieval_telemetry,
            embedding_integrity_events=ctx.embedding_integrity_events,
        )


@dataclass(frozen=True)
class _TaskRuntimeContextAdapter:
    config: Config | None
    workspace_id: str | None
    workspace_root: Path | None
    session_id: str | None
    memory_path: Path | None
    db_manager: object | None
    journal: object | None
    repository: object | None
    relational_search: MemorySearchPort | None
    embedding_maintenance: EmbeddingMaintenancePort | None
    search_health: SearchHealthPort | None
    startup_health: StartupHealthPort | None
    read_cache_validation: ReadCacheValidationPort | None
    memory_id_resolution: MemoryIDResolutionPort | None
    task_queue: object | None
    curation: object | None
    curation_action_store: object | None
    curation_quality: object | None
    direct_mutation_evidence: object | None
    mutation_history: object | None
    ai_json_provider: JSONTaskProvider | None
    ai_agent_provider: AgenticTaskProvider | None
    ai_provider_registry: dict[str, dict[str, object]] | None
    provider_usage: ProviderUsagePort | None
    provider_policy_events: ProviderPolicyEventPort | None
    task_execution_attempts: TaskExecutionAttemptPort | None
    work_items: WorkItemRepository | None
    embedding_repair_queue: object | None
    internal_tool_call_tracker: object | None


class _ApplicationContextView:
    def __init__(self, source: ApplicationContext, allowed_fields: frozenset[str]) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_allowed_fields", allowed_fields)

    def __getattr__(self, name: str) -> Any:
        allowed_fields = object.__getattribute__(self, "_allowed_fields")
        if name in allowed_fields:
            source = object.__getattribute__(self, "_source")
            return getattr(source, name)
        raise AttributeError(f"{type(self).__name__} does not expose attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"_source", "_allowed_fields"}:
            object.__setattr__(self, name, value)
            return
        allowed_fields = object.__getattribute__(self, "_allowed_fields")
        if name not in allowed_fields:
            raise AttributeError(f"{type(self).__name__} does not expose attribute {name!r}")
        source = object.__getattribute__(self, "_source")
        setattr(source, name, value)

    def __dir__(self) -> list[str]:
        return sorted(object.__getattribute__(self, "_allowed_fields"))


_MEMORY_PIPELINE_FIELDS = frozenset(
    {
        "config",
        "workspace_id",
        "workspace_root",
        "memory_path",
        "db_manager",
        "journal",
        "repository",
        "relational_search",
        "embedding_maintenance",
        "search_health",
        "startup_health",
        "read_cache_validation",
        "memory_id_resolution",
        "task_queue",
    }
)

_MANAGEMENT_FIELDS = _MEMORY_PIPELINE_FIELDS | frozenset(
    {
        "storage_backend",
        "read_cache",
        "ai_json_provider",
        "ai_agent_provider",
        "ai_provider_registry",
        "provider_usage",
        "runtime_logs",
        "retrieval_telemetry",
        "embedding_integrity_events",
        "embedder",
        "vector_store",
        "mutation_history",
        "curation",
        "curation_action_store",
    }
)

_TASK_RUNTIME_FIELDS = _MEMORY_PIPELINE_FIELDS | frozenset(
    {
        "session_id",
        "curation",
        "curation_action_store",
        "mutation_history",
        "ai_json_provider",
        "ai_agent_provider",
        "ai_provider_registry",
        "provider_usage",
        "provider_policy_events",
        "task_execution_attempts",
        "work_items",
        "embedding_repair_queue",
        "internal_tool_call_tracker",
        "curation_quality",
        "direct_mutation_evidence",
    }
)

_PROVIDER_SELECTION_FIELDS = frozenset(
    {
        "config",
        "ai_provider_registry",
        "provider_policy_events",
    }
)

_BACKGROUND_TASK_BOOTSTRAP_FIELDS = frozenset(
    {
        "config",
        "journal",
        "task_queue",
    }
)


@dataclass
class ApplicationContext:
    config: Config | None = None
    workspace_id: str | None = None
    workspace_root: Path | None = None
    session_id: str | None = None
    memory_path: Path | None = None
    storage_backend: str | None = None
    db_manager: Any = None
    journal: Any = None
    repository: Any = None
    relational_search: MemorySearchPort | None = None
    embedding_maintenance: EmbeddingMaintenancePort | None = None
    search_health: SearchHealthPort | None = None
    startup_health: StartupHealthPort | None = None
    read_cache_validation: ReadCacheValidationPort | None = None
    memory_id_resolution: MemoryIDResolutionPort | None = None
    read_cache: Any = None
    task_queue: Any = None
    curation: Any = None
    mutation_history: Any = None
    ai_json_provider: Any = None
    ai_agent_provider: Any = None
    ai_provider_registry: dict[str, Any] | None = None
    provider_usage: Any = None
    runtime_logs: Any = None
    retrieval_telemetry: Any = None
    provider_policy_events: Any = None
    embedding_integrity_events: Any = None
    task_execution_attempts: Any = None
    work_items: Any = None
    embedding_repair_queue: Any = None
    embedder: Any = None
    vector_store: Any = None
    curation_action_store: Any = None
    curation_quality: Any = None
    direct_mutation_evidence: Any = None
    ingress_batch_evidence: Any = None
    ingress_action_receipts: Any = None
    source_coverage: Any = None
    ingress_quality_evidence: Any = None
    ingress_mutation_transaction: Any = None
    internal_tool_call_tracker: Any = None
    _auxiliary_resources_closed: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.provider_usage is not None:
            return
        if not hasattr(self.db_manager, "get_connection"):
            return
        from mcp_memory.provider_usage_store import ProviderUsageRepository

        self.provider_usage = ProviderUsageRepository(
            self.db_manager,
            workspace_id=self.workspace_id,
        )

    def memory_pipeline_view(self) -> MemoryPipelineContext:
        return cast(MemoryPipelineContext, _ApplicationContextView(self, _MEMORY_PIPELINE_FIELDS))

    def memory_capabilities(self) -> MemoryReadCapabilities:
        return MemoryReadCapabilities.from_context(self)

    def mutation_capabilities(self) -> MutationCapabilities:
        return MutationCapabilities.from_context(self)

    def provider_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities.from_context(self)

    def background_task_capabilities(self) -> BackgroundTaskCapabilities:
        return BackgroundTaskCapabilities.from_context(self.background_task_bootstrap_view())

    def task_runtime_capabilities(self) -> TaskRuntimeCapabilities:
        return TaskRuntimeCapabilities.from_context(self.task_runtime_view())

    def management_capabilities(self) -> ManagementRuntimeCapabilities:
        return ManagementRuntimeCapabilities.from_context(self.management_view())

    def management_view(self) -> ManagementContext:
        return cast(ManagementContext, _ApplicationContextView(self, _MANAGEMENT_FIELDS))

    def task_runtime_view(self) -> TaskRuntimeContext:
        return cast(TaskRuntimeContext, _ApplicationContextView(self, _TASK_RUNTIME_FIELDS))

    def provider_selection_view(self) -> ProviderSelectionContext:
        return cast(ProviderSelectionContext, _ApplicationContextView(self, _PROVIDER_SELECTION_FIELDS))

    def background_task_bootstrap_view(self) -> BackgroundTaskBootstrapContext:
        return cast(BackgroundTaskBootstrapContext, _ApplicationContextView(self, _BACKGROUND_TASK_BOOTSTRAP_FIELDS))

    def close_auxiliary_resources(self, *, excluded_resources: tuple[Any, ...] = ()) -> None:
        """Close context-owned runtime resources not owned by the storage group."""
        if self._auxiliary_resources_closed:
            return
        self._auxiliary_resources_closed = True
        retrieval_telemetry = self.retrieval_telemetry
        if any(retrieval_telemetry is resource for resource in excluded_resources):
            return
        close_method = getattr(retrieval_telemetry, "close", None)
        if callable(close_method):
            close_method()

    def close(self) -> None:
        for resource in (
            self.read_cache,
            self.retrieval_telemetry,
            self.runtime_logs,
            self.embedding_integrity_events,
            self.repository,
        ):
            close_method = getattr(resource, "close", None)
            if callable(close_method):
                close_method()
        if self.db_manager is not None:
            close_method = getattr(self.db_manager, "close", None)
            if callable(close_method):
                close_method()

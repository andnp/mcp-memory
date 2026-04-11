from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from mcp_memory.config import Config

if TYPE_CHECKING:
    from mcp_memory.core.tasks import TaskRecord


class TaskQueueProtocol(Protocol):
    def find_open_task_with_data_any_workspace(
        self,
        task_name: str,
        *,
        data_fields: dict[str, Any],
    ) -> TaskRecord | None: ...

    def enqueue(
        self,
        task_name: str,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        max_retries: int = 3,
        available_at: float | None = None,
        task_id: str | None = None,
    ) -> TaskRecord: ...

    def clear_running_task_data_keys(self, task_id: str, *, field_names: list[str]) -> TaskRecord: ...

    def get_task(self, task_id: str) -> TaskRecord: ...

    def extend_running_task_data_int_list(
        self,
        task_id: str,
        *,
        field_name: str,
        values: list[int],
    ) -> TaskRecord: ...

    def extend_running_task_data_object_list(
        self,
        task_id: str,
        *,
        field_name: str,
        values: list[dict[str, Any]],
    ) -> TaskRecord: ...


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
    relational_search: Any
    task_queue: Any


class ManagementContext(MemoryPipelineContext, Protocol):
    storage_backend: str | None
    read_cache: Any
    ai_json_provider: Any
    ai_agent_provider: Any
    ai_provider: Any
    ai_provider_registry: dict[str, Any] | None
    provider_usage: Any
    runtime_logs: Any
    retrieval_telemetry: Any
    embedding_integrity_events: Any
    embedder: Any
    vector_store: Any


class TaskRuntimeContext(MemoryPipelineContext, Protocol):
    ai_json_provider: Any
    ai_agent_provider: Any
    ai_provider: Any
    ai_provider_registry: dict[str, Any] | None
    provider_usage: Any
    provider_policy_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any


class ProviderSelectionContext(Protocol):
    config: Config | None
    ai_provider_registry: dict[str, Any] | None
    provider_policy_events: Any


class BackgroundTaskBootstrapContext(Protocol):
    config: Config | None
    journal: Any
    task_queue: Any


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
    relational_search: Any = None
    read_cache: Any = None
    task_queue: Any = None
    ai_json_provider: Any = None
    ai_agent_provider: Any = None
    ai_provider: Any = None
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
    search_health: Any = None
    internal_tool_call_tracker: Any = None

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

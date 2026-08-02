from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from mcp_memory.context import ManagementContext, ManagementRuntimeCapabilities
from mcp_memory.embedding_integrity_event_store import EmbeddingIntegrityEventRepository
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.storage.noop import NoopProviderUsageRepository
from mcp_memory.storage.postgres_embedding_integrity_event_store import PostgresEmbeddingIntegrityEventRepository
from mcp_memory.storage.postgres_runtime_log_store import PostgresRuntimeLogRepository


@dataclass(frozen=True)
class ManagementContextResources:
    provider_usage: Any
    runtime_logs: Any
    retrieval_telemetry: Any
    embedding_integrity_events: Any
    mutation_history: Any

    @classmethod
    def from_capabilities(
        cls,
        capabilities: ManagementRuntimeCapabilities,
    ) -> ManagementContextResources:
        """Build only from composition-injected resources.

        Legacy context construction uses ``ensure_management_context_resources`` below.
        Runtime composition must provide these repositories explicitly.
        """
        return cls(
            provider_usage=capabilities.provider.provider_usage,
            runtime_logs=capabilities.runtime_logs,
            retrieval_telemetry=(
                capabilities.retrieval_telemetry
                or _build_default_retrieval_telemetry(capabilities)
            ),
            embedding_integrity_events=capabilities.embedding_integrity_events,
            mutation_history=capabilities.mutation.mutation_history,
        )


def _resource_context_values(ctx):
    memory = getattr(ctx, "memory", ctx)
    return (
        memory,
        getattr(memory, "workspace_id", getattr(ctx, "workspace_id", None)),
        getattr(memory, "db_manager", getattr(ctx, "db_manager", None)),
        getattr(memory, "config", getattr(ctx, "config", None)),
        getattr(ctx, "storage_backend", None),
    )


def _build_default_provider_usage(ctx: ManagementRuntimeCapabilities):
    _, workspace_id, db_manager, _, storage_backend = _resource_context_values(ctx)
    if (storage_backend or "sqlite") == "postgres":
        return NoopProviderUsageRepository(workspace_id=workspace_id)
    if not hasattr(db_manager, "get_connection"):
        return NoopProviderUsageRepository(workspace_id=workspace_id)
    return ProviderUsageRepository(
        cast(Any, db_manager),
        workspace_id=workspace_id,
    )


def _build_default_runtime_logs(ctx: ManagementRuntimeCapabilities):
    _, workspace_id, db_manager, context_config, storage_backend = _resource_context_values(ctx)
    config = None if context_config is None else context_config.logging
    if (storage_backend or "sqlite") == "postgres":
        return PostgresRuntimeLogRepository(
            cast(Any, db_manager),
            workspace_id=workspace_id,
            config=config,
        )
    return RuntimeLogRepository(
        cast(Any, db_manager),
        workspace_id=workspace_id,
        config=config,
    )


def _build_default_embedding_integrity_events(ctx: ManagementRuntimeCapabilities):
    _, _, db_manager, _, storage_backend = _resource_context_values(ctx)
    if db_manager is None:
        return None
    if hasattr(db_manager, "get_connection"):
        return EmbeddingIntegrityEventRepository(
            cast(Any, db_manager),
            workspace_id=None,
        )
    if (storage_backend or "sqlite") == "postgres":
        if not hasattr(db_manager, "open_connection"):
            return None
        return PostgresEmbeddingIntegrityEventRepository(
            cast(Any, db_manager),
            workspace_id=None,
        )
    return None


def _build_default_mutation_history(ctx: ManagementRuntimeCapabilities):
    if ctx.memory.db_manager is None or (ctx.storage_backend or "sqlite") == "postgres":
        return None
    if not hasattr(ctx.memory.db_manager, "get_connection"):
        return None
    return SQLiteMutationHistoryStore(cast(Any, ctx.memory.db_manager))


def _build_default_retrieval_telemetry(ctx: ManagementRuntimeCapabilities):
    if ctx.memory.db_manager is None:
        return None
    return RetrievalTelemetryRepository(
        cast(Any, ctx.memory.db_manager),
        workspace_id=ctx.memory.workspace_id,
        storage_backend=ctx.storage_backend,
    )


def ensure_management_context_resources(
    ctx: ManagementRuntimeCapabilities | ManagementContext,
) -> ManagementContextResources:
    capabilities = (
        ctx
        if isinstance(ctx, ManagementRuntimeCapabilities)
        else ManagementRuntimeCapabilities.from_context(ctx)
    )
    provider_usage = capabilities.provider.provider_usage or _build_default_provider_usage(capabilities)
    runtime_logs = capabilities.runtime_logs or _build_default_runtime_logs(capabilities)
    embedding_integrity_events = capabilities.embedding_integrity_events
    if embedding_integrity_events is None:
        embedding_integrity_events = _build_default_embedding_integrity_events(capabilities)
    retrieval_telemetry = capabilities.retrieval_telemetry
    if retrieval_telemetry is None:
        retrieval_telemetry = RetrievalTelemetryRepository(
            cast(Any, capabilities.memory.db_manager),
            workspace_id=capabilities.memory.workspace_id,
            storage_backend=capabilities.storage_backend,
        )
    mutation_history = capabilities.mutation.mutation_history or _build_default_mutation_history(capabilities)

    return ManagementContextResources(
        provider_usage=provider_usage,
        runtime_logs=runtime_logs,
        retrieval_telemetry=retrieval_telemetry,
        embedding_integrity_events=embedding_integrity_events,
        mutation_history=mutation_history,
    )

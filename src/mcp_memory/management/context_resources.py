from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from mcp_memory.context import ManagementContext, ManagementRuntimeCapabilities
from mcp_memory.embedding_integrity_event_store import EmbeddingIntegrityEventRepository
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


def _build_default_provider_usage(ctx: ManagementRuntimeCapabilities):
    if (ctx.storage_backend or "sqlite") == "postgres":
        return NoopProviderUsageRepository(workspace_id=ctx.memory.workspace_id)
    if not hasattr(ctx.memory.db_manager, "get_connection"):
        return NoopProviderUsageRepository(workspace_id=ctx.memory.workspace_id)
    return ProviderUsageRepository(
        cast(Any, ctx.memory.db_manager),
        workspace_id=ctx.memory.workspace_id,
    )


def _build_default_runtime_logs(ctx: ManagementRuntimeCapabilities):
    config = None if ctx.memory.config is None else ctx.memory.config.logging
    if (ctx.storage_backend or "sqlite") == "postgres":
        return PostgresRuntimeLogRepository(
            cast(Any, ctx.memory.db_manager),
            workspace_id=ctx.memory.workspace_id,
            config=config,
        )
    return RuntimeLogRepository(
        cast(Any, ctx.memory.db_manager),
        workspace_id=ctx.memory.workspace_id,
        config=config,
    )


def _build_default_embedding_integrity_events(ctx: ManagementRuntimeCapabilities):
    if ctx.memory.db_manager is None:
        return None
    if hasattr(ctx.memory.db_manager, "get_connection"):
        return EmbeddingIntegrityEventRepository(
            cast(Any, ctx.memory.db_manager),
            workspace_id=None,
        )
    if (ctx.storage_backend or "sqlite") == "postgres":
        if not hasattr(ctx.memory.db_manager, "open_connection"):
            return None
        return PostgresEmbeddingIntegrityEventRepository(
            cast(Any, ctx.memory.db_manager),
            workspace_id=None,
        )
    return None


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
        )

    return ManagementContextResources(
        provider_usage=provider_usage,
        runtime_logs=runtime_logs,
        retrieval_telemetry=retrieval_telemetry,
        embedding_integrity_events=embedding_integrity_events,
    )
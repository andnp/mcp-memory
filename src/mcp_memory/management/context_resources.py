from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.context import ManagementContext
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


def _build_default_provider_usage(ctx: ManagementContext):
    if (ctx.storage_backend or "sqlite") == "postgres":
        return NoopProviderUsageRepository(workspace_id=ctx.workspace_id)
    if not hasattr(ctx.db_manager, "get_connection"):
        return NoopProviderUsageRepository(workspace_id=ctx.workspace_id)
    return ProviderUsageRepository(ctx.db_manager, workspace_id=ctx.workspace_id)


def _build_default_runtime_logs(ctx: ManagementContext):
    config = None if ctx.config is None else ctx.config.logging
    if (ctx.storage_backend or "sqlite") == "postgres":
        return PostgresRuntimeLogRepository(
            ctx.db_manager,
            workspace_id=ctx.workspace_id,
            config=config,
        )
    return RuntimeLogRepository(
        ctx.db_manager,
        workspace_id=ctx.workspace_id,
        config=config,
    )


def _build_default_embedding_integrity_events(ctx: ManagementContext):
    if ctx.db_manager is None:
        return None
    if hasattr(ctx.db_manager, "get_connection"):
        return EmbeddingIntegrityEventRepository(
            ctx.db_manager,
            workspace_id=None,
        )
    if (ctx.storage_backend or "sqlite") == "postgres":
        if not hasattr(ctx.db_manager, "open_connection"):
            return None
        return PostgresEmbeddingIntegrityEventRepository(
            ctx.db_manager,
            workspace_id=None,
        )
    return None


def ensure_management_context_resources(ctx: ManagementContext) -> ManagementContextResources:
    provider_usage = ctx.provider_usage or _build_default_provider_usage(ctx)
    runtime_logs = ctx.runtime_logs or _build_default_runtime_logs(ctx)
    embedding_integrity_events = ctx.embedding_integrity_events
    if embedding_integrity_events is None:
        embedding_integrity_events = _build_default_embedding_integrity_events(ctx)
    retrieval_telemetry = ctx.retrieval_telemetry
    if retrieval_telemetry is None:
        retrieval_telemetry = RetrievalTelemetryRepository(ctx.db_manager, workspace_id=ctx.workspace_id)

    ctx.provider_usage = provider_usage
    ctx.runtime_logs = runtime_logs
    ctx.embedding_integrity_events = embedding_integrity_events
    ctx.retrieval_telemetry = retrieval_telemetry

    return ManagementContextResources(
        provider_usage=provider_usage,
        runtime_logs=runtime_logs,
        retrieval_telemetry=retrieval_telemetry,
        embedding_integrity_events=embedding_integrity_events,
    )
from __future__ import annotations

from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.models import EmbeddingStatusPayload, SearchHealthPayload


def build_embedding_status(embedder) -> EmbeddingStatusPayload:
    status = describe_embedder(embedder)
    if status is None:
        return EmbeddingStatusPayload()
    return EmbeddingStatusPayload(
        model_name=status.model_name,
        backend=status.backend,
        model_cached=status.model_cached,
    )


def build_search_health(relational_search) -> SearchHealthPayload:
    if relational_search is None:
        return SearchHealthPayload()
    health = relational_search.get_health()
    return SearchHealthPayload(
        semantic_enabled=health.semantic_enabled,
        available=health.available,
        degraded=health.degraded,
        fallback_count=health.fallback_count,
        rebuild_count=health.rebuild_count,
        background_repair_enabled=health.background_repair_enabled,
        background_repair_wait_seconds=health.background_repair_wait_seconds,
        queued_repair_backlog_count=health.queued_repair_backlog_count,
        running_repair_count=health.running_repair_count,
        oldest_queued_repair_age_seconds=health.oldest_queued_repair_age_seconds,
        repair_wait_count=health.repair_wait_count,
        partial_semantic_search_count=health.partial_semantic_search_count,
        last_partial_semantic_at=health.last_partial_semantic_at,
        last_repair_wait_seconds=health.last_repair_wait_seconds,
        last_repair_candidate_count=health.last_repair_candidate_count,
        last_repair_pending_count=health.last_repair_pending_count,
        last_error=health.last_error,
        last_failure_at=health.last_failure_at,
        last_recovery_at=health.last_recovery_at,
        last_integrity_check_at=health.last_integrity_check_at,
        integrity_check_error=health.integrity_check_error,
    )

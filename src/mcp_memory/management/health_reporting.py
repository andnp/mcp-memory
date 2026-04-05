from __future__ import annotations

import mcp_memory.core.tasks as task_queue_module
import time

from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.models import EmbeddingStatusPayload, ExecutionAttemptHealthPayload, SearchHealthPayload
from mcp_memory.management.reporting_queries import fetch_running_task_attempt_rows


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


def _coerce_int(value: object) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


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


def build_execution_attempt_health(
    db_manager,
    *,
    stale_after_seconds: float = 60.0,
    now: float | None = None,
) -> ExecutionAttemptHealthPayload:
    rows = fetch_running_task_attempt_rows(db_manager)
    if not rows:
        return ExecutionAttemptHealthPayload(stale_after_seconds=stale_after_seconds)

    current_time = now if now is not None else time.time()
    running_task_count = len(rows)
    running_attempt_count = 0
    fresh_attempt_count = 0
    stale_attempt_count = 0
    missing_attempt_count = 0
    live_subprocess_count = 0
    dead_subprocess_count = 0

    for row in rows:
        attempt_started_at = row["attempt_started_at"]
        last_heartbeat_at = row["last_heartbeat_at"]
        subprocess_pid = row["attempt_subprocess_pid"]
        if attempt_started_at is None:
            missing_attempt_count += 1
            continue

        running_attempt_count += 1
        recent_activity_at = max(
            _coerce_float(row["updated_at"]),
            _coerce_float(row["started_at"] or row["updated_at"]),
            _coerce_float(row["claimed_at"] or row["updated_at"]),
            _coerce_float(last_heartbeat_at or attempt_started_at),
        )
        age_seconds = max(current_time - recent_activity_at, 0.0)
        if age_seconds >= stale_after_seconds:
            stale_attempt_count += 1
        else:
            fresh_attempt_count += 1

        if subprocess_pid is None:
            continue
        if task_queue_module._is_process_alive(_coerce_int(subprocess_pid)):
            live_subprocess_count += 1
        else:
            dead_subprocess_count += 1

    return ExecutionAttemptHealthPayload(
        stale_after_seconds=stale_after_seconds,
        running_task_count=running_task_count,
        running_attempt_count=running_attempt_count,
        fresh_attempt_count=fresh_attempt_count,
        stale_attempt_count=stale_attempt_count,
        missing_attempt_count=missing_attempt_count,
        live_subprocess_count=live_subprocess_count,
        dead_subprocess_count=dead_subprocess_count,
    )

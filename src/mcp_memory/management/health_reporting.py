from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

from mcp_memory.core.ports import SearchHealthPort
from mcp_memory.core.ports.tasks import is_process_alive
from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.models import (
    EmbeddingIntegrityEventSummaryPayload,
    EmbeddingStatusPayload,
    ExecutionAttemptHealthPayload,
    SearchDiagnosticsPayload,
    SearchHealthPayload,
)
from mcp_memory.management.reporting_rows import RunningTaskAttemptRow, fetch_running_task_attempt_rows

type RunningTaskAttemptRowFetcher = Callable[[object], Sequence[RunningTaskAttemptRow]]
type ProcessAliveChecker = Callable[[int], bool]


class _SearchDiagnosticsValues(TypedDict, total=False):
    available: bool
    sample_rate: float | None
    observed_searches: int
    sampled_searches: int
    serialized_diagnostics: int
    sampled_rate: float | None
    degraded_count: int | None
    degraded_rate: float | None
    planner_decisions: dict[str, int] | None
    cache_status: dict[str, int] | None
    failure_stages: dict[str, int] | None
    diagnostic_serialization_failures: int


def _narrow_search_diagnostics(
    values: dict[str, object] | None,
) -> _SearchDiagnosticsValues:
    if values is None:
        return {}
    narrowed: _SearchDiagnosticsValues = {}
    available = values.get("available")
    if isinstance(available, bool):
        narrowed["available"] = available
    for key in ("sample_rate", "sampled_rate", "degraded_rate"):
        value = values.get(key)
        if value is None or (isinstance(value, (float, int)) and not isinstance(value, bool)):
            narrowed[key] = None if value is None else float(value)
    for key in (
        "observed_searches",
        "sampled_searches",
        "serialized_diagnostics",
        "degraded_count",
        "diagnostic_serialization_failures",
    ):
        value = values.get(key)
        if value is None and key == "degraded_count":
            narrowed[key] = None
        elif isinstance(value, int) and not isinstance(value, bool):
            narrowed[key] = value
    for key in ("planner_decisions", "cache_status", "failure_stages"):
        value = values.get(key)
        if value is None:
            narrowed[key] = None
        elif isinstance(value, Mapping):
            narrowed[key] = {
                label: count
                for label, count in value.items()
                if isinstance(label, str)
                and isinstance(count, int)
                and not isinstance(count, bool)
            }
    return narrowed


def build_embedding_status(
    embedder,
    *,
    storage_backend: str | None = None,
    vector_store=None,
    integrity_event_summary: EmbeddingIntegrityEventSummaryPayload | None = None,
) -> EmbeddingStatusPayload:
    fallback_persistence_policy = "blocked" if (storage_backend or "sqlite") == "postgres" else "allowed"
    status = describe_embedder(embedder)
    if status is None:
        payload = EmbeddingStatusPayload(fallback_persistence_policy=fallback_persistence_policy)
    else:
        payload = EmbeddingStatusPayload(
            model_name=status.model_name,
            configured_model_name=status.configured_model_name,
            backend=status.backend,
            model_cached=status.model_cached,
            fallback_persistence_policy=fallback_persistence_policy,
        )

    get_write_policy_state = getattr(vector_store, "get_write_policy_state", None)
    if callable(get_write_policy_state):
        policy_state = get_write_policy_state()
        payload.fallback_persistence_policy = getattr(
            policy_state,
            "fallback_persistence_policy",
            payload.fallback_persistence_policy,
        )
        payload.blocked_fallback_write_count = int(getattr(policy_state, "blocked_fallback_write_count", 0))
        payload.last_blocked_fallback_model_name = getattr(
            policy_state,
            "last_blocked_fallback_model_name",
            None,
        )
    if integrity_event_summary is not None:
        payload.integrity_events = integrity_event_summary
    return payload


def build_search_health(
    relational_search,
    *,
    search_health: SearchHealthPort | None = None,
    search_diagnostics: dict[str, object] | None = None,
) -> SearchHealthPayload:
    health_provider = search_health if search_health is not None else relational_search
    if health_provider is None:
        return SearchHealthPayload(
            search_diagnostics=SearchDiagnosticsPayload(
                **_narrow_search_diagnostics(search_diagnostics)
            )
        )
    health = health_provider.get_health()
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
        search_diagnostics=SearchDiagnosticsPayload(
            **_narrow_search_diagnostics(search_diagnostics)
        ),
    )


def build_execution_attempt_health(
    db_manager,
    *,
    stale_after_seconds: float = 60.0,
    now: float | None = None,
    fetch_running_attempt_rows: RunningTaskAttemptRowFetcher = fetch_running_task_attempt_rows,
    process_is_alive: ProcessAliveChecker = is_process_alive,
) -> ExecutionAttemptHealthPayload:
    rows = fetch_running_attempt_rows(db_manager)
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
        attempt_started_at = row.attempt_started_at
        last_heartbeat_at = row.last_heartbeat_at
        subprocess_pid = row.attempt_subprocess_pid
        if attempt_started_at is None:
            missing_attempt_count += 1
            continue

        running_attempt_count += 1
        recent_activity_at = max(
            row.updated_at,
            row.started_at or row.updated_at,
            row.claimed_at or row.updated_at,
            last_heartbeat_at or attempt_started_at,
        )
        age_seconds = max(current_time - recent_activity_at, 0.0)
        if age_seconds >= stale_after_seconds:
            stale_attempt_count += 1
        else:
            fresh_attempt_count += 1

        if subprocess_pid is None:
            continue
        if process_is_alive(subprocess_pid):
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

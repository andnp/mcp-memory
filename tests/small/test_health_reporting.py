from __future__ import annotations

from types import SimpleNamespace

import pytest

from mcp_memory.embeddings import HashingEmbedder
from mcp_memory.management.health_reporting import (
    build_embedding_status,
    build_execution_attempt_health,
    build_search_health,
)
from mcp_memory.management.models import EmbeddingIntegrityEventSummaryPayload, SearchHealthPayload
from mcp_memory.management.reporting_rows import RunningTaskAttemptRow


pytestmark = pytest.mark.small


def test_build_execution_attempt_health_returns_defaults_when_no_rows() -> None:
    payload = build_execution_attempt_health(
        None,
        stale_after_seconds=45.0,
        now=123.0,
        fetch_running_attempt_rows=lambda db_manager: [],
    )

    assert payload.stale_after_seconds == 45.0
    assert payload.running_task_count == 0
    assert payload.running_attempt_count == 0
    assert payload.fresh_attempt_count == 0
    assert payload.stale_attempt_count == 0
    assert payload.missing_attempt_count == 0
    assert payload.live_subprocess_count == 0
    assert payload.dead_subprocess_count == 0


def test_build_execution_attempt_health_counts_fresh_stale_missing_and_subprocess_states() -> None:
    observed_pids: list[int] = []
    rows = [
        RunningTaskAttemptRow(
            task_id="task-fresh",
            updated_at=200.0,
            started_at=180.0,
            attempt_started_at=190.0,
            last_heartbeat_at=195.0,
            attempt_subprocess_pid=101,
        ),
        RunningTaskAttemptRow(
            task_id="task-stale",
            updated_at=100.0,
            started_at=90.0,
            claimed_at=95.0,
            attempt_started_at=100.0,
            attempt_subprocess_pid=202,
        ),
        RunningTaskAttemptRow(
            task_id="task-missing",
            updated_at=210.0,
            started_at=205.0,
            claimed_at=206.0,
            attempt_subprocess_pid=303,
        ),
    ]

    payload = build_execution_attempt_health(
        None,
        stale_after_seconds=60.0,
        now=240.0,
        fetch_running_attempt_rows=lambda db_manager: rows,
        process_is_alive=lambda pid: observed_pids.append(pid) or pid == 101,
    )

    assert payload.running_task_count == 3
    assert payload.running_attempt_count == 2
    assert payload.fresh_attempt_count == 1
    assert payload.stale_attempt_count == 1
    assert payload.missing_attempt_count == 1
    assert payload.live_subprocess_count == 1
    assert payload.dead_subprocess_count == 1
    assert observed_pids == [101, 202]


def test_build_execution_attempt_health_skips_subprocess_liveness_without_pid() -> None:
    def unexpected_process_check(pid: int) -> bool:
        raise AssertionError("subprocess liveness should not be checked without a pid")

    payload = build_execution_attempt_health(
        None,
        stale_after_seconds=60.0,
        now=240.0,
        fetch_running_attempt_rows=lambda db_manager: [
            RunningTaskAttemptRow(
                task_id="task-running-no-pid",
                updated_at=200.0,
                started_at=180.0,
                claimed_at=181.0,
                attempt_started_at=190.0,
                last_heartbeat_at=220.0,
            )
        ],
        process_is_alive=unexpected_process_check,
    )

    assert payload.running_task_count == 1
    assert payload.running_attempt_count == 1
    assert payload.fresh_attempt_count == 1
    assert payload.stale_attempt_count == 0
    assert payload.missing_attempt_count == 0
    assert payload.live_subprocess_count == 0
    assert payload.dead_subprocess_count == 0


def test_build_embedding_status_uses_vector_store_override_and_integrity_summary() -> None:
    integrity_summary = EmbeddingIntegrityEventSummaryPayload(
        total=2,
        by_kind={"blocked_fallback_write": 2},
    )

    payload = build_embedding_status(
        None,
        storage_backend="postgres",
        vector_store=SimpleNamespace(
            get_write_policy_state=lambda: SimpleNamespace(
                fallback_persistence_policy="allowed",
                blocked_fallback_write_count=7,
                last_blocked_fallback_model_name="hash:test-model",
            )
        ),
        integrity_event_summary=integrity_summary,
    )

    assert payload.model_name is None
    assert payload.backend is None
    assert payload.fallback_persistence_policy == "allowed"
    assert payload.blocked_fallback_write_count == 7
    assert payload.last_blocked_fallback_model_name == "hash:test-model"
    assert payload.integrity_events == integrity_summary


def test_build_embedding_status_maps_real_embedder_status() -> None:
    payload = build_embedding_status(
        HashingEmbedder(model_name="hash:test-model"),
        storage_backend="postgres",
    )

    assert payload.model_name == "hash:test-model"
    assert payload.configured_model_name is None
    assert payload.backend == "hashing"
    assert payload.model_cached is True
    assert payload.fallback_persistence_policy == "blocked"


def test_build_search_health_returns_defaults_when_search_is_missing() -> None:
    assert build_search_health(None) == SearchHealthPayload()


def test_build_search_health_maps_relational_search_health_fields() -> None:
    payload = build_search_health(
        SimpleNamespace(
            get_health=lambda: SimpleNamespace(
                semantic_enabled=True,
                available=True,
                degraded=True,
                fallback_count=3,
                rebuild_count=4,
                background_repair_enabled=True,
                background_repair_wait_seconds=12.5,
                queued_repair_backlog_count=6,
                running_repair_count=2,
                oldest_queued_repair_age_seconds=33.0,
                repair_wait_count=5,
                partial_semantic_search_count=8,
                last_partial_semantic_at="2026-04-10T00:00:00+00:00",
                last_repair_wait_seconds=1.5,
                last_repair_candidate_count=9,
                last_repair_pending_count=7,
                last_error="degraded-search",
                last_failure_at="2026-04-09T23:00:00+00:00",
                last_recovery_at="2026-04-09T23:05:00+00:00",
                last_integrity_check_at="2026-04-09T23:10:00+00:00",
                integrity_check_error="checksum-mismatch",
            )
        )
    )

    assert payload.semantic_enabled is True
    assert payload.available is True
    assert payload.degraded is True
    assert payload.fallback_count == 3
    assert payload.rebuild_count == 4
    assert payload.background_repair_enabled is True
    assert payload.background_repair_wait_seconds == 12.5
    assert payload.queued_repair_backlog_count == 6
    assert payload.running_repair_count == 2
    assert payload.oldest_queued_repair_age_seconds == 33.0
    assert payload.repair_wait_count == 5
    assert payload.partial_semantic_search_count == 8
    assert payload.last_partial_semantic_at == "2026-04-10T00:00:00+00:00"
    assert payload.last_repair_wait_seconds == 1.5
    assert payload.last_repair_candidate_count == 9
    assert payload.last_repair_pending_count == 7
    assert payload.last_error == "degraded-search"
    assert payload.last_failure_at == "2026-04-09T23:00:00+00:00"
    assert payload.last_recovery_at == "2026-04-09T23:05:00+00:00"
    assert payload.last_integrity_check_at == "2026-04-09T23:10:00+00:00"
    assert payload.integrity_check_error == "checksum-mismatch"
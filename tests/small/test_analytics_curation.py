from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from uuid import uuid4

import pytest

from mcp_memory.management.analytics_curation import build_curation_metrics


pytestmark = pytest.mark.small


def test_curation_metrics_use_persisted_receipts_and_history_not_provider_claims(db_manager) -> None:
    now = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    created_at = now - timedelta(hours=1)
    run_id = str(uuid4())
    action_id = str(uuid4())
    event_id = str(uuid4())
    memory_id = str(uuid4())
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO curation_runs (
            run_id, frontier_key, context_fingerprint, state, outcome,
            rejection_codes_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_id, "frontier", "fingerprint", "terminal", "no_op", json.dumps(["unsafe_operation"]), created_at.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO curation_action_receipts (
            run_id, action_id, operation, status, intent_hash, error_code
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (run_id, action_id, "normalize_memory", "verified", "intent-hash", None),
    )
    connection.execute(
        """
        INSERT INTO memory_mutation_events (
            id, operation, actor_kind, family, curation_run_id, provider_id,
            status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, "normalize_memory", "maintenance", "curator", run_id, "provider-claim-only", "applied", created_at.isoformat()),
    )
    connection.execute(
        """
        INSERT INTO memory_record_revisions (
            event_id, memory_id, role, before_exists, after_exists,
            before_snapshot, after_snapshot
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (event_id, memory_id, "target", 1, 1, "{}", "{}"),
    )
    connection.execute(
        """
        INSERT INTO curation_candidate_state (
            memory_id, disposition, consecutive_no_op_count, cooldown_until
        ) VALUES (?, ?, ?, ?)
        """,
        (memory_id, "cooldown", 2, (now + timedelta(hours=2)).isoformat()),
    )
    connection.execute(
        """
        INSERT INTO work_items (
            id, family_key, execution_lane, payload_json, status,
            available_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid4()),
            "graph_link_review",
            "agentic",
            json.dumps({"route_kind": "specialist_route", "primary_family": "graph_linker", "reason_code": "needs_different_specialist"}),
            "pending",
            now.timestamp(),
            now.timestamp(),
            now.timestamp(),
        ),
    )
    connection.commit()

    metrics = build_curation_metrics(db_manager, window_hours=24, now=now.timestamp())

    assert metrics.run_states == {"terminal": 1}
    assert metrics.operation_counts == {"normalize_memory": 1}
    assert metrics.rejection_reasons == {"unsafe_operation": 1}
    assert metrics.no_op_runs == 1
    assert metrics.candidate.no_op_candidates == 1
    assert metrics.candidate.cooldown_candidates == 1
    assert metrics.specialist_routes.by_family == {"graph_linker": 1}
    assert metrics.provider_disclosure.by_provider == {"provider-claim-only": 1}
    assert metrics.history.restorable_event_count == 1
    assert metrics.history.restore_available is True
    assert metrics.verified_yield == 1.0
    assert metrics.verified_receipt_count == 1


def test_curation_metrics_ignore_provider_reported_mutation_counts(db_manager) -> None:
    now = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)
    connection = db_manager.get_connection()
    connection.execute(
        """
        INSERT INTO task_runs (task_id, task_name, status, started_at, completed_at, duration_seconds, result_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (str(uuid4()), "memory-curator", "completed", now.timestamp(), now.timestamp(), 0.0, json.dumps({"mutations": 99})),
    )
    connection.commit()

    metrics = build_curation_metrics(db_manager, window_hours=24, now=now.timestamp())

    assert metrics.operation_counts == {}
    assert metrics.verified_receipt_count == 0
    assert metrics.verified_yield == 0.0

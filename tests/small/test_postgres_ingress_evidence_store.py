from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    IngressReceiptStatus,
    IngressSourceSnapshot,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.core.ports.ingress import IngressActionReceiptIdentityConflictError
from mcp_memory.storage.postgres_ingress_evidence_store import (
    PostgresIngressActionReceiptRepository,
    PostgresIngressBatchEvidenceRepository,
    PostgresSourceCoverageRepository,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.connection.calls.append((query, params))

    def fetchone(self) -> tuple[object, ...] | None:
        return self.connection.fetchone_result

    def fetchall(self) -> list[tuple[object, ...]]:
        return self.connection.fetchall_result


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.commits = 0
        self.rollbacks = 0
        self.fetchone_result: tuple[object, ...] | None = None
        self.fetchall_result: list[tuple[object, ...]] = []

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _Lease:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Connection:
        return self.connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _Sessions:
    def __init__(self) -> None:
        self.connection = _Connection()

    def open_connection(self) -> _Lease:
        return _Lease(self.connection)


def test_postgres_batch_evidence_serializes_and_hydrates_replay_snapshot() -> None:
    """Batch evidence preserves ordering, nullable fields, and nested snapshot data."""
    evidence = IngressBatchEvidence(
        batch_id="batch-1", task_id="task-1", execution_epoch=3, batch_sequence=2,
        claimed_entry_ids=("entry-2", "entry-1"), source_fingerprint="fingerprint",
        source_entries=(IngressSourceSnapshot(
            entry_id="entry-1", workspace_ids=("workspace-b", "workspace-a"),
            timestamp="2026-08-08T00:00:00Z", content_digest="digest",
            snapshot={"nested": ("b", "a")},
        ),), provider_route="route", execution_mode="apply", policy_version="policy-1",
        schema_version="schema-1", claimed_at="claimed", finalized_at=None,
        grouping_strategy=None, grouping_fallback_reason="fallback",
    )
    sessions = _Sessions()
    repository = PostgresIngressBatchEvidenceRepository(cast(SessionManager[DbConnectionLike], sessions))

    repository.save(evidence)
    params = sessions.connection.calls[0][1]
    assert params is not None
    assert json.loads(cast(str, params[4])) == ["entry-2", "entry-1"]
    assert json.loads(cast(str, params[6]))[0]["snapshot"] == {"nested": ["b", "a"]}
    assert sessions.connection.commits == 1

    sessions.connection.fetchone_result = (
        "batch-1", "task-1", 3, 2, ["entry-2", "entry-1"], "fingerprint",
        [{"entry_id": "entry-1", "workspace_ids": ["workspace-a"], "timestamp": "claimed",
          "content_digest": "digest", "snapshot": {"nullable": None}}],
        "route", "apply", "policy-1", "schema-1", "claimed", None, None, "fallback",
    )
    hydrated = repository.get("batch-1")
    assert hydrated is not None
    assert hydrated.batch_id == evidence.batch_id
    assert hydrated.claimed_entry_ids == evidence.claimed_entry_ids
    assert hydrated.source_entries[0].snapshot == {"nullable": None}
    assert hydrated.finalized_at is None
    assert hydrated.grouping_fallback_reason == evidence.grouping_fallback_reason


def test_postgres_action_receipt_preserves_enum_nulls_and_revision_tokens() -> None:
    """Action receipts retain enum values, nulls, timestamps, and replay tokens."""
    receipt = IngressActionReceipt(
        action_id="action-1", batch_id="batch-1", operation="append", entry_ids=("entry-1",),
        target_ids=(), canonical_payload_digest="payload", status=IngressReceiptStatus.STALE,
        mutation_evidence_id=None, created_at="created", terminalized_at=None, error_code="stale",
        before_revision_tokens={"entry-1": "before"}, after_revision_tokens={"entry-1": "after"},
    )
    sessions = _Sessions()
    repository = PostgresIngressActionReceiptRepository(cast(SessionManager[DbConnectionLike], sessions))

    repository.save(receipt)
    params = sessions.connection.calls[0][1]
    assert params is not None
    assert params[6] == "stale"
    assert params[7] is None
    assert json.loads(cast(str, params[11])) == {"entry-1": "before"}

    sessions.connection.fetchone_result = (
        "action-1", "batch-1", "append", ["entry-1"], [], "payload", "stale", None,
        "created", None, "stale", {"entry-1": "before"}, {"entry-1": "after"},
    )
    assert repository.get("action-1") == receipt


def test_postgres_action_receipt_reservation_replays_and_rejects_collision() -> None:
    """Postgres reservation retrieves the winner before deciding replay safety.

    The fake connection verifies conflict-safe insertion, same-transaction
    retrieval, commit on admission, and rollback on a digest collision.
    """
    receipt = IngressActionReceipt(
        action_id="action-1", batch_id="batch-1", operation="append", entry_ids=("entry-1",),
        target_ids=(), canonical_payload_digest="payload", status=IngressReceiptStatus.STALE,
        mutation_evidence_id=None, created_at="created", terminalized_at=None, error_code="stale",
        before_revision_tokens={"entry-1": "before"}, after_revision_tokens={"entry-1": "after"},
    )
    stored_row = (
        "action-1", "batch-1", "append", ["entry-1"], [], "payload", "stale", None,
        "created", None, "stale", {"entry-1": "before"}, {"entry-1": "after"},
    )
    sessions = _Sessions()
    sessions.connection.fetchone_result = stored_row
    repository = PostgresIngressActionReceiptRepository(cast(SessionManager[DbConnectionLike], sessions))

    assert repository.reserve(receipt) == receipt
    assert "ON CONFLICT (action_id) DO NOTHING" in sessions.connection.calls[0][0]
    assert "FROM ingress_action_receipts WHERE action_id = %s" in sessions.connection.calls[1][0]
    assert sessions.connection.commits == 1

    replay = replace(receipt, status=IngressReceiptStatus.FAILED)
    assert repository.reserve(replay) == receipt
    assert sessions.connection.commits == 2

    sessions.connection.fetchone_result = (*stored_row[:5], "different-payload", *stored_row[6:])
    with pytest.raises(IngressActionReceiptIdentityConflictError):
        repository.reserve(receipt)
    assert sessions.connection.rollbacks == 1


def test_postgres_source_coverage_lists_requested_entries_in_stable_order() -> None:
    """Coverage listing returns hydrated enum outcomes in entry-id order."""
    sessions = _Sessions()
    repository = PostgresSourceCoverageRepository(cast(SessionManager[DbConnectionLike], sessions))
    coverage = SourceCoverage("entry-2", SourceCoverageOutcome.NO_MUTATION, reason=None)

    assert repository.save(coverage) == coverage
    params = sessions.connection.calls[0][1]
    assert params == ("entry-2", None, "no_mutation", None)

    sessions.connection.fetchall_result = [
        ("entry-1", "action-1", "created", None),
        ("entry-2", None, "no_mutation", None),
    ]
    assert repository.list_for_entries(("entry-2", "entry-1")) == [
        SourceCoverage("entry-1", SourceCoverageOutcome.CREATED, action_id="action-1"),
        coverage,
    ]
    assert repository.list_for_entries(()) == []

"""SQLite persistence for ingress replay evidence and source coverage."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    IngressReceiptStatus,
    IngressSourceSnapshot,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.core.ports.ingress import (
    IngressActionReceiptIdentityConflictError,
    SourceCoverageAssignmentConflictError,
)
from mcp_memory.utils.db import DatabaseManager


class SQLiteIngressBatchEvidenceStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def save(self, evidence: IngressBatchEvidence) -> IngressBatchEvidence:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO ingress_batch_evidence (
                    batch_id, task_id, execution_epoch, batch_sequence,
                    claimed_entry_ids_json, source_fingerprint, source_entries_json,
                    grouping_strategy, grouping_fallback_reason, provider_route,
                    execution_mode, policy_version, schema_version, claimed_at, finalized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                    task_id = excluded.task_id,
                    execution_epoch = excluded.execution_epoch,
                    batch_sequence = excluded.batch_sequence,
                    claimed_entry_ids_json = excluded.claimed_entry_ids_json,
                    source_fingerprint = excluded.source_fingerprint,
                    source_entries_json = excluded.source_entries_json,
                    grouping_strategy = excluded.grouping_strategy,
                    grouping_fallback_reason = excluded.grouping_fallback_reason,
                    provider_route = excluded.provider_route,
                    execution_mode = excluded.execution_mode,
                    policy_version = excluded.policy_version,
                    schema_version = excluded.schema_version,
                    claimed_at = excluded.claimed_at,
                    finalized_at = excluded.finalized_at
                """,
                _batch_values(evidence),
            )
        return evidence

    def get(self, batch_id: str) -> IngressBatchEvidence | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM ingress_batch_evidence WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        return None if row is None else _batch_from_row(row)

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[IngressBatchEvidence]:
        rows = self._db.get_connection().execute(
            """
            SELECT * FROM ingress_batch_evidence
            WHERE task_id = ? AND execution_epoch = ?
            ORDER BY batch_sequence, batch_id
            """,
            (task_id, execution_epoch),
        ).fetchall()
        return [_batch_from_row(row) for row in rows]


class SQLiteIngressActionReceiptStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def reserve(self, receipt: IngressActionReceipt) -> IngressActionReceipt:
        conn = self._db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO ingress_action_receipts (
                    action_id, batch_id, operation, entry_ids_json, target_ids_json,
                    canonical_payload_digest, status, mutation_evidence_id,
                    before_revision_tokens_json, after_revision_tokens_json,
                    created_at, terminalized_at, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO NOTHING
                """,
                _receipt_values(receipt),
            )
            row = conn.execute(
                "SELECT * FROM ingress_action_receipts WHERE action_id = ?",
                (receipt.action_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("ingress action receipt insert was not persisted")
            stored = _receipt_from_row(row)
            if stored.canonical_payload_digest != receipt.canonical_payload_digest:
                raise IngressActionReceiptIdentityConflictError(
                    receipt.action_id,
                    stored.canonical_payload_digest,
                    receipt.canonical_payload_digest,
                )
            conn.commit()
            return stored
        except Exception:
            conn.rollback()
            raise

    def save(self, receipt: IngressActionReceipt) -> IngressActionReceipt:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO ingress_action_receipts (
                    action_id, batch_id, operation, entry_ids_json, target_ids_json,
                    canonical_payload_digest, status, mutation_evidence_id,
                    before_revision_tokens_json, after_revision_tokens_json,
                    created_at, terminalized_at, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO UPDATE SET
                    batch_id = excluded.batch_id,
                    operation = excluded.operation,
                    entry_ids_json = excluded.entry_ids_json,
                    target_ids_json = excluded.target_ids_json,
                    canonical_payload_digest = excluded.canonical_payload_digest,
                    status = excluded.status,
                    mutation_evidence_id = excluded.mutation_evidence_id,
                    before_revision_tokens_json = excluded.before_revision_tokens_json,
                    after_revision_tokens_json = excluded.after_revision_tokens_json,
                    created_at = excluded.created_at,
                    terminalized_at = excluded.terminalized_at,
                    error_code = excluded.error_code
                """,
                _receipt_values(receipt),
            )
        return receipt

    def get(self, action_id: str) -> IngressActionReceipt | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM ingress_action_receipts WHERE action_id = ?", (action_id,)
        ).fetchone()
        return None if row is None else _receipt_from_row(row)

    def list_for_batch(self, batch_id: str) -> list[IngressActionReceipt]:
        rows = self._db.get_connection().execute(
            """
            SELECT * FROM ingress_action_receipts
            WHERE batch_id = ?
            ORDER BY created_at, action_id
            """,
            (batch_id,),
        ).fetchall()
        return [_receipt_from_row(row) for row in rows]


class SQLiteSourceCoverageStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def save(self, coverage: SourceCoverage) -> SourceCoverage:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO ingress_source_coverage (entry_id, outcome, action_id, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    outcome = excluded.outcome,
                    action_id = excluded.action_id,
                    reason = excluded.reason
                """,
                (coverage.entry_id, coverage.outcome.value, coverage.action_id, coverage.reason),
            )
        return coverage

    def assign_terminal(self, coverage: SourceCoverage) -> SourceCoverage:
        _validate_terminal_coverage(coverage)
        conn = self._db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO ingress_source_coverage (entry_id, outcome, action_id, reason)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_id) DO NOTHING
                """,
                (coverage.entry_id, coverage.outcome.value, coverage.action_id, coverage.reason),
            )
            row = conn.execute(
                "SELECT * FROM ingress_source_coverage WHERE entry_id = ?",
                (coverage.entry_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError("source coverage assignment was not persisted")
            stored = _coverage_from_row(row)
            if (stored.action_id, stored.outcome) != (coverage.action_id, coverage.outcome):
                raise SourceCoverageAssignmentConflictError(
                    coverage.entry_id,
                    stored.action_id,
                    stored.outcome,
                    coverage.action_id,
                    coverage.outcome,
                )
            conn.commit()
            return stored
        except Exception:
            conn.rollback()
            raise

    def get(self, entry_id: str) -> SourceCoverage | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM ingress_source_coverage WHERE entry_id = ?", (entry_id,)
        ).fetchone()
        return None if row is None else _coverage_from_row(row)

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]:
        if not entry_ids:
            return []
        placeholders = ", ".join("?" for _ in entry_ids)
        ordering = " ".join(f"WHEN ? THEN {index}" for index, _ in enumerate(entry_ids))
        rows = self._db.get_connection().execute(
            f"""
            SELECT * FROM ingress_source_coverage
            WHERE entry_id IN ({placeholders})
            ORDER BY CASE entry_id {ordering} END
            """,
            (*entry_ids, *entry_ids),
        ).fetchall()
        return [_coverage_from_row(row) for row in rows]


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_value(value: object, default: Any) -> Any:
    return default if value is None else json.loads(str(value))


def _batch_values(evidence: IngressBatchEvidence) -> tuple[object, ...]:
    return (
        evidence.batch_id,
        evidence.task_id,
        evidence.execution_epoch,
        evidence.batch_sequence,
        _json_text(evidence.claimed_entry_ids),
        evidence.source_fingerprint,
        _json_text([
            {
                "entry_id": entry.entry_id,
                "workspace_ids": entry.workspace_ids,
                "timestamp": entry.timestamp,
                "content_digest": entry.content_digest,
                "snapshot": entry.snapshot,
            }
            for entry in evidence.source_entries
        ]),
        evidence.grouping_strategy,
        evidence.grouping_fallback_reason,
        evidence.provider_route,
        evidence.execution_mode,
        evidence.policy_version,
        evidence.schema_version,
        evidence.claimed_at,
        evidence.finalized_at,
    )


def _batch_from_row(row: sqlite3.Row) -> IngressBatchEvidence:
    source_entries = _json_value(row["source_entries_json"], default=[])
    return IngressBatchEvidence(
        batch_id=row["batch_id"],
        task_id=row["task_id"],
        execution_epoch=int(row["execution_epoch"]),
        batch_sequence=int(row["batch_sequence"]),
        claimed_entry_ids=tuple(_json_value(row["claimed_entry_ids_json"], default=[])),
        source_fingerprint=row["source_fingerprint"],
        source_entries=tuple(
            IngressSourceSnapshot(
                entry_id=entry["entry_id"],
                workspace_ids=tuple(entry["workspace_ids"]),
                timestamp=entry["timestamp"],
                content_digest=entry["content_digest"],
                snapshot=dict(entry["snapshot"]),
            )
            for entry in source_entries
        ),
        provider_route=row["provider_route"],
        execution_mode=row["execution_mode"],
        policy_version=row["policy_version"],
        schema_version=row["schema_version"],
        claimed_at=row["claimed_at"],
        finalized_at=row["finalized_at"],
        grouping_strategy=row["grouping_strategy"],
        grouping_fallback_reason=row["grouping_fallback_reason"],
    )


def _receipt_values(receipt: IngressActionReceipt) -> tuple[object, ...]:
    return (
        receipt.action_id,
        receipt.batch_id,
        receipt.operation,
        _json_text(receipt.entry_ids),
        _json_text(receipt.target_ids),
        receipt.canonical_payload_digest,
        receipt.status.value,
        receipt.mutation_evidence_id,
        _json_text(receipt.before_revision_tokens),
        _json_text(receipt.after_revision_tokens),
        receipt.created_at,
        receipt.terminalized_at,
        receipt.error_code,
    )


def _receipt_from_row(row: sqlite3.Row) -> IngressActionReceipt:
    return IngressActionReceipt(
        action_id=row["action_id"],
        batch_id=row["batch_id"],
        operation=row["operation"],
        entry_ids=tuple(_json_value(row["entry_ids_json"], default=[])),
        target_ids=tuple(_json_value(row["target_ids_json"], default=[])),
        canonical_payload_digest=row["canonical_payload_digest"],
        status=IngressReceiptStatus(row["status"]),
        mutation_evidence_id=row["mutation_evidence_id"],
        created_at=row["created_at"],
        terminalized_at=row["terminalized_at"],
        error_code=row["error_code"],
        before_revision_tokens=dict(_json_value(row["before_revision_tokens_json"], default={})),
        after_revision_tokens=dict(_json_value(row["after_revision_tokens_json"], default={})),
    )


def _coverage_from_row(row: sqlite3.Row) -> SourceCoverage:
    return SourceCoverage(
        entry_id=row["entry_id"],
        outcome=SourceCoverageOutcome(row["outcome"]),
        action_id=row["action_id"],
        reason=row["reason"],
    )


def _validate_terminal_coverage(coverage: SourceCoverage) -> None:
    if getattr(coverage.outcome, "value", coverage.outcome) == "released_unhandled":
        raise ValueError("released_unhandled is not terminal source coverage")

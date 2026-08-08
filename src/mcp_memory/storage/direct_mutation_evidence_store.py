"""SQLite persistence for direct mutation evidence."""

from __future__ import annotations

import json
import sqlite3

from mcp_memory.core.direct_mutation_evidence import (
    DirectMutationEntityDelta,
    DirectMutationEvidence,
)
from mcp_memory.utils.db import DatabaseManager


class SQLiteDirectMutationEvidenceStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def append(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence:
        existing = self.get_by_idempotency_key(evidence.idempotency_key)
        if existing is not None:
            return existing
        return self.save(evidence)

    def save(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence:
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO direct_mutation_evidence (
                    evidence_id, task_id, execution_epoch, session_id, call_id,
                    sequence, tool_name, operation, idempotency_key, payload_json,
                    ledger_json, outcome, error_code, started_at, completed_at, finalized_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(evidence_id) DO UPDATE SET
                    payload_json = excluded.payload_json, ledger_json = excluded.ledger_json,
                    outcome = excluded.outcome, error_code = excluded.error_code,
                    completed_at = excluded.completed_at, finalized_at = excluded.finalized_at
                """,
                _evidence_values(evidence),
            )
            conn.execute(
                "DELETE FROM direct_mutation_entity_deltas WHERE evidence_id = ?",
                (evidence.evidence_id,),
            )
            conn.executemany(
                """
                INSERT INTO direct_mutation_entity_deltas (
                    evidence_id, ordinal, entity_kind, entity_id, before_revision,
                    after_revision, before_exists, after_exists, transition, snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        evidence.evidence_id,
                        ordinal,
                        delta.kind,
                        delta.entity_id,
                        delta.before_revision,
                        delta.after_revision,
                        int(delta.before_exists),
                        int(delta.after_exists),
                        delta.transition,
                        json.dumps(delta.snapshot, sort_keys=True),
                    )
                    for ordinal, delta in enumerate(evidence.deltas)
                ],
            )
        return evidence

    def get(self, evidence_id: str) -> DirectMutationEvidence | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM direct_mutation_evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return None if row is None else self._from_row(row)

    def get_by_idempotency_key(self, idempotency_key: str) -> DirectMutationEvidence | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM direct_mutation_evidence WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        return None if row is None else self._from_row(row)

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[DirectMutationEvidence]:
        rows = self._db.get_connection().execute(
            "SELECT * FROM direct_mutation_evidence WHERE task_id = ? AND execution_epoch = ? ORDER BY sequence, evidence_id",
            (task_id, execution_epoch),
        ).fetchall()
        return [self._from_row(row) for row in rows]

    def _from_row(self, row: sqlite3.Row) -> DirectMutationEvidence:
        deltas = self._db.get_connection().execute(
            "SELECT * FROM direct_mutation_entity_deltas WHERE evidence_id = ? ORDER BY ordinal",
            (row["evidence_id"],),
        ).fetchall()
        return DirectMutationEvidence(
            evidence_id=row["evidence_id"],
            task_id=row["task_id"],
            execution_epoch=row["execution_epoch"],
            session_id=row["session_id"],
            call_id=row["call_id"],
            sequence=row["sequence"],
            tool_name=row["tool_name"],
            operation=row["operation"],
            idempotency_key=row["idempotency_key"],
            payload=json.loads(row["payload_json"]),
            ledger_entry=json.loads(row["ledger_json"]),
            deltas=tuple(
                DirectMutationEntityDelta(
                    kind=item["entity_kind"], entity_id=item["entity_id"],
                    before_revision=item["before_revision"], after_revision=item["after_revision"],
                    before_exists=bool(item["before_exists"]), after_exists=bool(item["after_exists"]),
                    transition=item["transition"], snapshot=json.loads(item["snapshot_json"]),
                ) for item in deltas
            ),
            outcome=row["outcome"], error_code=row["error_code"],
            started_at=row["started_at"], completed_at=row["completed_at"], finalized_at=row["finalized_at"],
        )


def _evidence_values(evidence: DirectMutationEvidence) -> tuple[object, ...]:
    return (
        evidence.evidence_id, evidence.task_id, evidence.execution_epoch, evidence.session_id,
        evidence.call_id, evidence.sequence, evidence.tool_name, evidence.operation,
        evidence.idempotency_key, json.dumps(evidence.payload, sort_keys=True),
        json.dumps(evidence.ledger_entry, sort_keys=True), evidence.outcome, evidence.error_code,
        evidence.started_at, evidence.completed_at, evidence.finalized_at,
    )

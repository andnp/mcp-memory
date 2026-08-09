"""SQLite persistence for immutable ingress quality evidence."""

from __future__ import annotations

import json
import sqlite3

from mcp_memory.core.curation_quality_policy import QualityOutcome
from mcp_memory.core.ingress_evidence import IngressQualityEvidence
from mcp_memory.core.ports.ingress import IngressQualityEvidenceConflictError
from mcp_memory.utils.db import DatabaseManager


class SQLiteIngressQualityEvidenceStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def save(self, evidence: IngressQualityEvidence) -> IngressQualityEvidence:
        connection = self._db.get_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO ingress_quality_evidence (
                    action_id, disposition, reason, evaluated_at, evaluator, query_provenance_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(action_id) DO NOTHING
                """,
                _values(evidence),
            )
            row = connection.execute(
                "SELECT * FROM ingress_quality_evidence WHERE action_id = ?", (evidence.action_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("ingress quality evidence was not persisted")
            stored = _from_row(row)
            if stored != evidence:
                raise IngressQualityEvidenceConflictError(evidence.action_id, stored, evidence)
            connection.commit()
            return stored
        except Exception:
            connection.rollback()
            raise

    def get(self, action_id: str) -> IngressQualityEvidence | None:
        row = self._db.get_connection().execute(
            "SELECT * FROM ingress_quality_evidence WHERE action_id = ?", (action_id,)
        ).fetchone()
        return None if row is None else _from_row(row)


def _values(evidence: IngressQualityEvidence) -> tuple[object, ...]:
    return (
        evidence.action_id,
        evidence.disposition.value,
        evidence.reason,
        evidence.evaluated_at,
        evidence.evaluator,
        json.dumps(evidence.query_provenance, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _from_row(row: sqlite3.Row) -> IngressQualityEvidence:
    return IngressQualityEvidence(
        action_id=row["action_id"],
        disposition=QualityOutcome(row["disposition"]),
        reason=row["reason"],
        evaluated_at=row["evaluated_at"],
        evaluator=row["evaluator"],
        query_provenance=json.loads(row["query_provenance_json"]),
    )

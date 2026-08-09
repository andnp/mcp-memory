"""Postgres persistence for immutable ingress quality evidence."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Iterator, cast

from mcp_memory.core.curation_quality_policy import QualityOutcome
from mcp_memory.core.ingress_evidence import IngressQualityEvidence
from mcp_memory.core.ports.ingress import IngressQualityEvidenceConflictError
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class PostgresIngressQualityEvidenceStore:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    @contextmanager
    def _open_connection(self) -> Iterator[DbConnectionLike]:
        if self._sessions is None:
            raise RuntimeError("ingress_quality_evidence_unavailable")
        with self._sessions.open_connection() as connection:
            yield connection

    def save(self, evidence: IngressQualityEvidence) -> IngressQualityEvidence:
        with self._open_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO ingress_quality_evidence (
                            action_id, disposition, reason, evaluated_at, evaluator, query_provenance_json
                        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                        ON CONFLICT (action_id) DO NOTHING
                        """,
                        _values(evidence),
                    )
                    cursor.execute(
                        "SELECT action_id, disposition, reason, evaluated_at, evaluator, query_provenance_json "
                        "FROM ingress_quality_evidence WHERE action_id = %s",
                        (evidence.action_id,),
                    )
                    row = cursor.fetchone()
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
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT action_id, disposition, reason, evaluated_at, evaluator, query_provenance_json "
                    "FROM ingress_quality_evidence WHERE action_id = %s",
                    (action_id,),
                )
                row = cursor.fetchone()
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


def _from_row(row: tuple[object, ...]) -> IngressQualityEvidence:
    provenance = row[5]
    if isinstance(provenance, str):
        provenance = json.loads(provenance)
    return IngressQualityEvidence(
        action_id=cast(str, row[0]),
        disposition=QualityOutcome(cast(str, row[1])),
        reason=cast(str | None, row[2]),
        evaluated_at=cast(str, row[3]),
        evaluator=cast(str | None, row[4]),
        query_provenance=cast(dict[str, object], provenance),
    )

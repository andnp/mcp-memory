"""Postgres persistence for direct mutation evidence."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator, cast

from mcp_memory.core.direct_mutation_evidence import DirectMutationEntityDelta, DirectMutationEvidence
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class PostgresDirectMutationEvidenceStore:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    def append(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence:
        existing = self.get_by_idempotency_key(evidence.idempotency_key)
        return existing if existing is not None else self.save(evidence)

    def save(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO direct_mutation_evidence (
                        evidence_id, task_id, execution_epoch, session_id, call_id, sequence,
                        tool_name, operation, idempotency_key, payload_json, ledger_json,
                        outcome, error_code, started_at, completed_at, finalized_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(evidence_id) DO UPDATE SET payload_json = EXCLUDED.payload_json,
                    ledger_json = EXCLUDED.ledger_json, outcome = EXCLUDED.outcome,
                    error_code = EXCLUDED.error_code, completed_at = EXCLUDED.completed_at,
                    finalized_at = EXCLUDED.finalized_at
                    """,
                    _values(evidence),
                )
                cursor.execute("DELETE FROM direct_mutation_entity_deltas WHERE evidence_id = %s", (evidence.evidence_id,))
                for ordinal, delta in enumerate(evidence.deltas):
                    cursor.execute(
                        """
                        INSERT INTO direct_mutation_entity_deltas (
                            evidence_id, ordinal, entity_kind, entity_id, before_revision,
                            after_revision, before_exists, after_exists, transition, snapshot_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (evidence.evidence_id, ordinal, delta.kind, delta.entity_id,
                         delta.before_revision, delta.after_revision, delta.before_exists,
                         delta.after_exists, delta.transition, json.dumps(delta.snapshot, sort_keys=True)),
                    )
            connection.commit()
        return evidence

    def get(self, evidence_id: str) -> DirectMutationEvidence | None:
        return self._fetch("SELECT * FROM direct_mutation_evidence WHERE evidence_id = %s", (evidence_id,))

    def get_by_idempotency_key(self, idempotency_key: str) -> DirectMutationEvidence | None:
        return self._fetch("SELECT * FROM direct_mutation_evidence WHERE idempotency_key = %s", (idempotency_key,))

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[DirectMutationEvidence]:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM direct_mutation_evidence WHERE task_id = %s AND execution_epoch = %s ORDER BY sequence, evidence_id",
                    (task_id, execution_epoch),
                )
                rows = cursor.fetchall()
                return [self._from_row(connection, row) for row in rows]

    def _fetch(self, query: str, params: tuple[object, ...]) -> DirectMutationEvidence | None:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return None if row is None else self._from_row(connection, row)

    def _from_row(self, connection: DbConnectionLike, row: Any) -> DirectMutationEvidence:
        with connection.cursor() as cursor:
            cursor.execute("SELECT * FROM direct_mutation_entity_deltas WHERE evidence_id = %s ORDER BY ordinal", (row["evidence_id"],))
            deltas = cast(list[Any], cursor.fetchall())
        return DirectMutationEvidence(
            evidence_id=row["evidence_id"], task_id=row["task_id"], execution_epoch=row["execution_epoch"],
            session_id=row["session_id"], call_id=row["call_id"], sequence=row["sequence"],
            tool_name=row["tool_name"], operation=row["operation"], idempotency_key=row["idempotency_key"],
            payload=_json_value(row["payload_json"]), ledger_entry=_json_value(row["ledger_json"]),
            deltas=tuple(DirectMutationEntityDelta(
                kind=item["entity_kind"], entity_id=item["entity_id"], before_revision=item["before_revision"],
                after_revision=item["after_revision"], before_exists=bool(item["before_exists"]),
                after_exists=bool(item["after_exists"]), transition=item["transition"], snapshot=_json_value(item["snapshot_json"]),
            ) for item in deltas), outcome=row["outcome"], error_code=row["error_code"],
            started_at=row["started_at"], completed_at=row["completed_at"], finalized_at=row["finalized_at"],
        )

    @contextmanager
    def _open_connection(self) -> Iterator[DbConnectionLike]:
        if self._sessions is None:
            raise RuntimeError("direct_mutation_evidence_unavailable")
        with self._sessions.open_connection() as connection:
            yield connection


def _values(evidence: DirectMutationEvidence) -> tuple[object, ...]:
    return (evidence.evidence_id, evidence.task_id, evidence.execution_epoch, evidence.session_id,
            evidence.call_id, evidence.sequence, evidence.tool_name, evidence.operation,
            evidence.idempotency_key, json.dumps(evidence.payload, sort_keys=True),
            json.dumps(evidence.ledger_entry, sort_keys=True), evidence.outcome, evidence.error_code,
            evidence.started_at, evidence.completed_at, evidence.finalized_at)


def _json_value(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    return {}

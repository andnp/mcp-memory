"""Durable storage for compact sampled curation quality evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import json
from typing import Any
from uuid import UUID

from mcp_memory.core.curation_quality import CurationQualityEvidence, CurationQualityRepository
from mcp_memory.storage.session import DbConnectionLike, SessionManager
from mcp_memory.utils.db import DatabaseManager


class SQLiteCurationQualityStore(CurationQualityRepository):
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def put_quality_evidence(self, evidence: CurationQualityEvidence) -> CurationQualityEvidence:
        connection = self._db.get_connection()
        with connection:
            connection.execute(
                """
                INSERT INTO curation_quality_evidence (
                    run_id, action_id, operation, affected_memory_ids_json,
                    policy_version, query_id, status, before_ranked_ids_json,
                    after_ranked_ids_json, retrieval_regression_count,
                    zero_result_change, payload_size_change, useful_work, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, action_id) DO UPDATE SET
                    operation = excluded.operation,
                    affected_memory_ids_json = excluded.affected_memory_ids_json,
                    policy_version = excluded.policy_version,
                    query_id = excluded.query_id,
                    status = excluded.status,
                    before_ranked_ids_json = excluded.before_ranked_ids_json,
                    after_ranked_ids_json = excluded.after_ranked_ids_json,
                    retrieval_regression_count = excluded.retrieval_regression_count,
                    zero_result_change = excluded.zero_result_change,
                    payload_size_change = excluded.payload_size_change,
                    useful_work = excluded.useful_work,
                    created_at = excluded.created_at
                """,
                _values(evidence),
            )
        return evidence

    def list_quality_evidence(
        self,
        *,
        run_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[CurationQualityEvidence]:
        bounded = max(1, min(limit, 1000))
        connection = self._db.get_connection()
        if run_id is None:
            rows = connection.execute(
                "SELECT * FROM curation_quality_evidence ORDER BY created_at DESC, action_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        else:
            rows = connection.execute(
                "SELECT * FROM curation_quality_evidence WHERE run_id = ? ORDER BY created_at DESC, action_id DESC LIMIT ?",
                (str(run_id), bounded),
            ).fetchall()
        return [_from_row(row) for row in rows]


class PostgresCurationQualityStore(CurationQualityRepository):
    def __init__(self, session_manager: SessionManager[DbConnectionLike]) -> None:
        self._sessions = session_manager

    def put_quality_evidence(self, evidence: CurationQualityEvidence) -> CurationQualityEvidence:
        with self._sessions.open_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO curation_quality_evidence (
                    run_id, action_id, operation, affected_memory_ids_json,
                    policy_version, query_id, status, before_ranked_ids_json,
                    after_ranked_ids_json, retrieval_regression_count,
                    zero_result_change, payload_size_change, useful_work, created_at
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, action_id) DO UPDATE SET
                    operation = EXCLUDED.operation,
                    affected_memory_ids_json = EXCLUDED.affected_memory_ids_json,
                    policy_version = EXCLUDED.policy_version,
                    query_id = EXCLUDED.query_id,
                    status = EXCLUDED.status,
                    before_ranked_ids_json = EXCLUDED.before_ranked_ids_json,
                    after_ranked_ids_json = EXCLUDED.after_ranked_ids_json,
                    retrieval_regression_count = EXCLUDED.retrieval_regression_count,
                    zero_result_change = EXCLUDED.zero_result_change,
                    payload_size_change = EXCLUDED.payload_size_change,
                    useful_work = EXCLUDED.useful_work,
                    created_at = EXCLUDED.created_at
                """,
                _values(evidence),
            )
        return evidence

    def list_quality_evidence(
        self,
        *,
        run_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[CurationQualityEvidence]:
        bounded = max(1, min(limit, 1000))
        query = "SELECT * FROM curation_quality_evidence"
        params: tuple[object, ...]
        if run_id is not None:
            query += " WHERE run_id = %s"
            params = (str(run_id),)
        else:
            params = ()
        query += " ORDER BY created_at DESC, action_id DESC LIMIT %s"
        params += (bounded,)
        with self._sessions.open_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query, params)
            return [_from_postgres_row(row) for row in cursor.fetchall()]


def _values(evidence: CurationQualityEvidence) -> tuple[object, ...]:
    return (
        str(evidence.run_id),
        str(evidence.action_id),
        evidence.operation,
        _json([str(value) for value in evidence.affected_memory_ids]),
        evidence.policy_version,
        evidence.query_id,
        evidence.status,
        _json([str(value) for value in evidence.before_ranked_memory_ids]),
        _json([str(value) for value in evidence.after_ranked_memory_ids]),
        evidence.retrieval_regression_count,
        evidence.zero_result_change,
        evidence.payload_size_change,
        None if evidence.useful_work is None else int(evidence.useful_work),
        evidence.created_at.isoformat(),
    )


def _from_row(row: Any) -> CurationQualityEvidence:
    return CurationQualityEvidence(
        run_id=UUID(str(row["run_id"])),
        action_id=UUID(str(row["action_id"])),
        operation=str(row["operation"]),
        affected_memory_ids=[UUID(value) for value in _json_value(row["affected_memory_ids_json"], [])],
        policy_version=str(row["policy_version"]),
        query_id=row["query_id"],
        status=str(row["status"]),
        before_ranked_memory_ids=[UUID(value) for value in _json_value(row["before_ranked_ids_json"], [])],
        after_ranked_memory_ids=[UUID(value) for value in _json_value(row["after_ranked_ids_json"], [])],
        retrieval_regression_count=row["retrieval_regression_count"],
        zero_result_change=row["zero_result_change"],
        payload_size_change=row["payload_size_change"],
        useful_work=None if row["useful_work"] is None else bool(row["useful_work"]),
        created_at=_datetime(row["created_at"]),
    )


def _from_postgres_row(row: tuple[object, ...]) -> CurationQualityEvidence:
    return CurationQualityEvidence(
        run_id=UUID(str(row[0])),
        action_id=UUID(str(row[1])),
        operation=str(row[2]),
        affected_memory_ids=[UUID(value) for value in _json_value(row[3], [])],
        policy_version=str(row[4]),
        query_id=None if row[5] is None else str(row[5]),
        status=str(row[6]),
        before_ranked_memory_ids=[UUID(value) for value in _json_value(row[7], [])],
        after_ranked_memory_ids=[UUID(value) for value in _json_value(row[8], [])],
        retrieval_regression_count=_optional_int(row[9]),
        zero_result_change=_optional_int(row[10]),
        payload_size_change=_optional_int(row[11]),
        useful_work=None if row[12] is None else bool(row[12]),
        created_at=_datetime(row[13]),
    )


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _json_value(value: object, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return [str(item) for item in parsed] if isinstance(parsed, list) else default


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))

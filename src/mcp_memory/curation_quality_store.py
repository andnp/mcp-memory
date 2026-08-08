"""Durable storage for compact sampled curation quality evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from mcp_memory.core.curation_quality import CurationQualityEvidence, CurationQualityRepository
from mcp_memory.storage.session import DbConnectionLike, SessionManager
from mcp_memory.utils.db import DatabaseManager


_POSTGRES_QUALITY_EVIDENCE_SELECT = """
    SELECT run_id, action_id, operation, affected_memory_ids_json,
           policy_version, query_id, status, before_ranked_ids_json,
           after_ranked_ids_json, retrieval_regression_count,
           zero_result_change, payload_size_change, useful_work, created_at,
           retrieval_utility_delta, acceptance_met, neutral_reason,
           content_quality_score_before, content_quality_score_after,
           content_quality_delta, content_quality_improved,
           engagement_utility_delta, engagement_evidence_json, evidence_id
    FROM curation_quality_evidence
"""


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
                    zero_result_change, payload_size_change, useful_work, created_at,
                    retrieval_utility_delta, acceptance_met, neutral_reason
                    , content_quality_score_before, content_quality_score_after,
                    content_quality_delta, content_quality_improved,
                    engagement_utility_delta, engagement_evidence_json, evidence_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    created_at = excluded.created_at,
                    retrieval_utility_delta = excluded.retrieval_utility_delta,
                    acceptance_met = excluded.acceptance_met,
                    neutral_reason = excluded.neutral_reason,
                    content_quality_score_before = excluded.content_quality_score_before,
                    content_quality_score_after = excluded.content_quality_score_after,
                    content_quality_delta = excluded.content_quality_delta,
                    content_quality_improved = excluded.content_quality_improved,
                    engagement_utility_delta = excluded.engagement_utility_delta,
                    engagement_evidence_json = excluded.engagement_evidence_json,
                    evidence_id = excluded.evidence_id
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
                    zero_result_change, payload_size_change, useful_work, created_at,
                    retrieval_utility_delta, acceptance_met, neutral_reason
                    , content_quality_score_before, content_quality_score_after,
                    content_quality_delta, content_quality_improved,
                    engagement_utility_delta, engagement_evidence_json, evidence_id
                ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
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
                    created_at = EXCLUDED.created_at,
                    retrieval_utility_delta = EXCLUDED.retrieval_utility_delta,
                    acceptance_met = EXCLUDED.acceptance_met,
                    neutral_reason = EXCLUDED.neutral_reason,
                    content_quality_score_before = EXCLUDED.content_quality_score_before,
                    content_quality_score_after = EXCLUDED.content_quality_score_after,
                    content_quality_delta = EXCLUDED.content_quality_delta,
                    content_quality_improved = EXCLUDED.content_quality_improved,
                    engagement_utility_delta = EXCLUDED.engagement_utility_delta,
                    engagement_evidence_json = EXCLUDED.engagement_evidence_json,
                    evidence_id = EXCLUDED.evidence_id
                """,
                _values(evidence),
            )
            connection.commit()
        return evidence

    def list_quality_evidence(
        self,
        *,
        run_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[CurationQualityEvidence]:
        bounded = max(1, min(limit, 1000))
        query = _POSTGRES_QUALITY_EVIDENCE_SELECT
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
        evidence.retrieval_utility_delta,
        None if evidence.acceptance_met is None else int(evidence.acceptance_met),
        evidence.neutral_reason,
        evidence.content_quality_score_before,
        evidence.content_quality_score_after,
        evidence.content_quality_delta,
        None if evidence.content_quality_improved is None else int(evidence.content_quality_improved),
        evidence.engagement_utility_delta,
        _json(evidence.engagement_evidence),
        evidence.evidence_id,
    )


def _from_row(row: Any) -> CurationQualityEvidence:
    return CurationQualityEvidence(
        run_id=UUID(str(row["run_id"])),
        action_id=UUID(str(row["action_id"])),
        evidence_id=_optional_text(_row_value(row, "evidence_id")),
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
        retrieval_utility_delta=_optional_float(_row_value(row, "retrieval_utility_delta")),
        acceptance_met=_optional_bool(_row_value(row, "acceptance_met")),
        neutral_reason=_optional_text(_row_value(row, "neutral_reason")),
        content_quality_score_before=_optional_float(_row_value(row, "content_quality_score_before")),
        content_quality_score_after=_optional_float(_row_value(row, "content_quality_score_after")),
        content_quality_delta=_optional_float(_row_value(row, "content_quality_delta")),
        content_quality_improved=_optional_bool(_row_value(row, "content_quality_improved")),
        engagement_utility_delta=_optional_float(_row_value(row, "engagement_utility_delta")),
        engagement_evidence=_json_mapping(_row_value(row, "engagement_evidence_json")),
    )


def _from_postgres_row(row: tuple[object, ...]) -> CurationQualityEvidence:
    return CurationQualityEvidence(
        run_id=UUID(str(_row_at(row, 0))),
        action_id=UUID(str(_row_at(row, 1))),
        evidence_id=_optional_text(_row_at(row, 23)),
        operation=str(_row_at(row, 2)),
        affected_memory_ids=[UUID(value) for value in _json_value(_row_at(row, 3), [])],
        policy_version=str(_row_at(row, 4)),
        query_id=_optional_text(_row_at(row, 5)),
        status=str(_row_at(row, 6)),
        before_ranked_memory_ids=[UUID(value) for value in _json_value(_row_at(row, 7), [])],
        after_ranked_memory_ids=[UUID(value) for value in _json_value(_row_at(row, 8), [])],
        retrieval_regression_count=_optional_int(_row_at(row, 9)),
        zero_result_change=_optional_int(_row_at(row, 10)),
        payload_size_change=_optional_int(_row_at(row, 11)),
        useful_work=_optional_bool(_row_at(row, 12)),
        created_at=_datetime(_row_at(row, 13)),
        retrieval_utility_delta=_optional_float(_row_at(row, 14)),
        acceptance_met=_optional_bool(_row_at(row, 15)),
        neutral_reason=_optional_text(_row_at(row, 16)),
        content_quality_score_before=_optional_float(_row_at(row, 17)),
        content_quality_score_after=_optional_float(_row_at(row, 18)),
        content_quality_delta=_optional_float(_row_at(row, 19)),
        content_quality_improved=_optional_bool(_row_at(row, 20)),
        engagement_utility_delta=_optional_float(_row_at(row, 21)),
        engagement_evidence=_json_mapping(_row_at(row, 22)),
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


def _json_mapping(value: object) -> dict[str, object]:
    if value is None:
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    return dict(parsed) if isinstance(parsed, dict) else {}


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _optional_float(value: object) -> float | None:
    return None if value is None else float(str(value))


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _row_value(row: Any, column: str) -> object:
    keys = row.keys() if hasattr(row, "keys") else ()
    if column not in keys:
        return None
    return row[column]


def _row_at(row: tuple[object, ...], index: int) -> object:
    return row[index] if len(row) > index else None


def _datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))

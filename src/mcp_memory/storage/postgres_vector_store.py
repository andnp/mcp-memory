from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from searchkernel.utils.similarity import cosine_similarity_lists

from mcp_memory.embedding_integrity_event_store import (
    EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
    EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
)
from mcp_memory.embeddings import EmbeddingRecord, is_fallback_embedding_model
from mcp_memory.storage.session import DbConnectionLike, SessionManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostgresVectorSearchCapabilities:
    pgvector_extension_installed: bool
    embedding_vector_column_present: bool

    @property
    def server_side_vector_search_available(self) -> bool:
        return self.pgvector_extension_installed and self.embedding_vector_column_present


@dataclass(frozen=True)
class PostgresEmbeddingWritePolicyState:
    fallback_persistence_policy: str = "blocked"
    blocked_fallback_write_count: int = 0
    last_blocked_fallback_model_name: str | None = None


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


def _format_vector_literal(embedding: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in embedding) + "]"


def _decode_embedding_payload(embedding_raw: object) -> list[float]:
    if isinstance(embedding_raw, str):
        embedding = list(json.loads(embedding_raw))
    elif isinstance(embedding_raw, list | tuple):
        embedding = list(embedding_raw)
    else:
        raise TypeError(f"Unexpected Postgres embedding payload type: {type(embedding_raw)!r}")
    return [float(value) for value in embedding]


class PostgresVectorStore:
    supports_candidate_filtering = True

    def __init__(
        self,
        session_manager: SessionManager[DbConnectionLike] | None,
        *,
        event_repository=None,
        fallback_row_cap: int = 1000,
    ) -> None:
        """Create a store with a bounded client-side fallback scan."""
        if fallback_row_cap < 1:
            raise ValueError("fallback_row_cap must be positive")
        self._sessions = session_manager
        self._event_repository = event_repository
        self._fallback_row_cap = fallback_row_cap
        self._search_capabilities: PostgresVectorSearchCapabilities | None = None
        self._last_search_diagnostics: dict[str, object] = {}
        self._blocked_fallback_write_count = 0
        self._last_blocked_fallback_model_name: str | None = None

    @property
    def last_search_diagnostics(self) -> dict[str, object]:
        """Return diagnostics from the most recent vector search.

        This mirrors searchkernel's read-only diagnostics surface while
        keeping the existing optional per-call diagnostics mapping intact.
        """
        return dict(self._last_search_diagnostics)

    def get_write_policy_state(self) -> PostgresEmbeddingWritePolicyState:
        return PostgresEmbeddingWritePolicyState(
            blocked_fallback_write_count=self._blocked_fallback_write_count,
            last_blocked_fallback_model_name=self._last_blocked_fallback_model_name,
        )

    def _raise_for_blocked_fallback_embedding_write(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str,
    ) -> None:
        self._blocked_fallback_write_count += 1
        self._last_blocked_fallback_model_name = model_name
        logger.warning(
            "Blocked fallback/hash embedding write in Postgres/shared mode for %s:%s model=%s",
            source_kind,
            source_id,
            model_name,
        )
        self._record_integrity_event(
            event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
            model_name=model_name,
            source_kind=source_kind,
            source_id=source_id,
            details={
                "reason": "fallback_embedding_persistence_blocked",
            },
        )
        raise ValueError(
            "Fallback/hash embeddings cannot be persisted in Postgres/shared mode; "
            f"blocked model_name {model_name!r}"
        )

    def _get_search_capabilities(self) -> PostgresVectorSearchCapabilities:
        cached = self._search_capabilities
        if cached is not None:
            return cached
        if self._sessions is None:
            capabilities = PostgresVectorSearchCapabilities(
                pgvector_extension_installed=False,
                embedding_vector_column_present=False,
            )
            self._search_capabilities = capabilities
            return capabilities
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_extension
                        WHERE extname = 'vector'
                    )
                    """
                )
                extension_row = cursor.fetchone()
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = current_schema()
                          AND table_name = 'embeddings'
                          AND column_name = 'embedding_vector'
                    )
                    """
                )
                column_row = cursor.fetchone()
        capabilities = PostgresVectorSearchCapabilities(
            pgvector_extension_installed=bool(extension_row and extension_row[0]),
            embedding_vector_column_present=bool(column_row and column_row[0]),
        )
        self._search_capabilities = capabilities
        return capabilities

    def _read_model_dimensions(
        self,
        cursor,
        *,
        source_kind: str,
        model_name: str,
    ) -> set[int]:
        cursor.execute(
            """
            SELECT DISTINCT jsonb_array_length(embedding_json)
            FROM embeddings
            WHERE source_kind = %s
              AND model_name = %s
              AND jsonb_typeof(embedding_json) = 'array'
            """,
            (source_kind, model_name),
        )
        return {
            int(row[0])
            for row in cursor.fetchall()
            if row and row[0] is not None
        }

    def scan_integrity(
        self,
        *,
        active_model_name: str | None = None,
        expected_dimension: int | None = None,
    ) -> dict[str, Any]:
        if self._sessions is None:
            return {
                "scanned_row_count": 0,
                "invalid_row_count": 0,
                "mixed_dimension_group_count": 0,
                "mixed_dimension_groups": [],
                "invalid_rows": [],
                "active_model_name": active_model_name,
                "active_model_row_count": 0,
                "active_model_rows": [],
            }

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_kind, source_id, workspace_id, model_name,
                           CASE
                               WHEN jsonb_typeof(embedding_json) = 'array'
                               THEN jsonb_array_length(embedding_json)
                               ELSE NULL
                           END AS embedding_dimension,
                           jsonb_typeof(embedding_json) AS payload_type,
                           updated_at
                    FROM embeddings
                    ORDER BY model_name ASC, source_kind ASC, source_id ASC
                    """
                )
                rows = cursor.fetchall()

        active_rows: list[dict[str, Any]] = []
        invalid_rows: list[dict[str, Any]] = []
        rows_by_model: dict[str, list[dict[str, Any]]] = {}

        for row in rows:
            source_kind = str(row[0])
            source_id = str(row[1])
            workspace_id = None if row[2] is None else str(row[2])
            model_name = str(row[3])
            dimension = int(row[4]) if isinstance(row[4], int | float | str) else None
            payload_type = None if row[5] is None else str(row[5])
            updated_at = _coerce_float(row[6])

            issues: list[str] = []
            if payload_type != "array":
                issues.append("invalid_payload_type")
            elif dimension is None or dimension < 1:
                issues.append("invalid_dimension")
            elif model_name == active_model_name and expected_dimension is not None and dimension != expected_dimension:
                issues.append("dimension_mismatch")

            row_summary: dict[str, Any] = {
                "source_kind": source_kind,
                "source_id": source_id,
                "workspace_id": workspace_id,
                "model_name": model_name,
                "dimension": dimension,
                "payload_type": payload_type,
                "updated_at": updated_at,
                "issues": issues,
            }
            rows_by_model.setdefault(model_name, []).append(row_summary)
            if model_name == active_model_name:
                active_rows.append(row_summary)
            if issues:
                invalid_rows.append(row_summary)

        mixed_dimension_groups: list[dict[str, Any]] = []
        for model_name, model_rows in rows_by_model.items():
            dimensions = sorted(
                {
                    int(row["dimension"])
                    for row in model_rows
                    if isinstance(row.get("dimension"), int)
                }
            )
            if len(dimensions) <= 1:
                continue
            source_kind_counts: dict[str, int] = {}
            for row in model_rows:
                source_kind = str(row["source_kind"])
                source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1
            mixed_dimension_groups.append(
                {
                    "model_name": model_name,
                    "dimensions": dimensions,
                    "row_count": len(model_rows),
                    "source_kind_counts": source_kind_counts,
                }
            )

        summary = {
            "scanned_row_count": len(rows),
            "invalid_row_count": len(invalid_rows),
            "mixed_dimension_group_count": len(mixed_dimension_groups),
            "mixed_dimension_groups": mixed_dimension_groups,
            "invalid_rows": invalid_rows,
            "active_model_name": active_model_name,
            "active_model_row_count": len(active_rows),
            "active_model_rows": active_rows,
        }
        self._record_integrity_event(
            event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
            model_name=active_model_name,
            scanned_row_count=int(summary["scanned_row_count"]),
            invalid_row_count=int(summary["invalid_row_count"]),
            mixed_dimension_group_count=int(summary["mixed_dimension_group_count"]),
            details={
                "active_model_name": active_model_name,
                "expected_dimension": expected_dimension,
                "active_model_row_count": int(summary["active_model_row_count"]),
            },
        )
        return summary

    def _record_integrity_event(
        self,
        *,
        event_kind: str,
        model_name: str | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
        scanned_row_count: int | None = None,
        invalid_row_count: int | None = None,
        mixed_dimension_group_count: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self._event_repository is None:
            return
        self._event_repository.record_event(
            event_kind=event_kind,
            model_name=model_name,
            source_kind=source_kind,
            source_id=source_id,
            scanned_row_count=scanned_row_count,
            invalid_row_count=invalid_row_count,
            mixed_dimension_group_count=mixed_dimension_group_count,
            details=details,
        )

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
        source_updated_at: str | None = None,
    ) -> bool:
        if is_fallback_embedding_model(model_name):
            self._raise_for_blocked_fallback_embedding_write(
                source_kind=source_kind,
                source_id=source_id,
                model_name=model_name,
            )
        if self._sessions is None:
            return False
        normalized_embedding = [float(value) for value in embedding]
        if not normalized_embedding:
            raise ValueError("embedding dimension must be positive")
        payload_json = json.dumps(normalized_embedding)
        updated_at = time.time()
        capabilities = self._get_search_capabilities()
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                if source_updated_at is not None:
                    cursor.execute(
                        "SELECT updated_at FROM memories WHERE id = %s FOR SHARE",
                        (source_id,),
                    )
                    current = cursor.fetchone()
                    if current is None or str(current[0]) != source_updated_at:
                        connection.rollback()
                        return False
                stored_dimensions = self._read_model_dimensions(
                    cursor,
                    source_kind=source_kind,
                    model_name=model_name,
                )
                current_dimension = len(normalized_embedding)
                if len(stored_dimensions) > 1:
                    raise ValueError(
                        "Refusing to upsert embedding for model_name "
                        f"{model_name!r}: existing rows already have mixed dimensions {sorted(stored_dimensions)}"
                    )
                if stored_dimensions and current_dimension not in stored_dimensions:
                    raise ValueError(
                        "Refusing to upsert embedding for model_name "
                        f"{model_name!r}: existing dimension {next(iter(stored_dimensions))} "
                        f"does not match new dimension {current_dimension}"
                    )
                if capabilities.server_side_vector_search_available:
                    if source_updated_at is not None:
                        cursor.execute(
                            """
                            INSERT INTO embeddings (
                                source_kind, source_id, workspace_id, model_name,
                                embedding_json, memory_updated_at, embedding_vector, updated_at
                            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, CAST(%s AS vector), %s)
                            ON CONFLICT (source_kind, source_id, model_name)
                            DO UPDATE SET
                                workspace_id = EXCLUDED.workspace_id,
                                embedding_json = EXCLUDED.embedding_json,
                                memory_updated_at = EXCLUDED.memory_updated_at,
                                embedding_vector = EXCLUDED.embedding_vector,
                                updated_at = EXCLUDED.updated_at
                            WHERE embeddings.memory_updated_at IS NULL
                               OR embeddings.memory_updated_at <= EXCLUDED.memory_updated_at
                            """,
                            (
                                source_kind,
                                source_id,
                                workspace_id,
                                model_name,
                                payload_json,
                                source_updated_at,
                                _format_vector_literal(normalized_embedding),
                                updated_at,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO embeddings (
                                source_kind, source_id, workspace_id, model_name, embedding_json, embedding_vector, updated_at
                            ) VALUES (%s, %s, %s, %s, %s::jsonb, CAST(%s AS vector), %s)
                            ON CONFLICT (source_kind, source_id, model_name)
                            DO UPDATE SET
                                workspace_id = EXCLUDED.workspace_id,
                                embedding_json = EXCLUDED.embedding_json,
                                embedding_vector = EXCLUDED.embedding_vector,
                                updated_at = EXCLUDED.updated_at
                            """,
                            (
                                source_kind,
                                source_id,
                                workspace_id,
                                model_name,
                                payload_json,
                                _format_vector_literal(normalized_embedding),
                                updated_at,
                            ),
                        )
                else:
                    if source_updated_at is not None:
                        cursor.execute(
                            """
                            INSERT INTO embeddings (
                                source_kind, source_id, workspace_id, model_name,
                                embedding_json, memory_updated_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                            ON CONFLICT (source_kind, source_id, model_name)
                            DO UPDATE SET
                                workspace_id = EXCLUDED.workspace_id,
                                embedding_json = EXCLUDED.embedding_json,
                                memory_updated_at = EXCLUDED.memory_updated_at,
                                updated_at = EXCLUDED.updated_at
                            WHERE embeddings.memory_updated_at IS NULL
                               OR embeddings.memory_updated_at <= EXCLUDED.memory_updated_at
                            """,
                            (
                                source_kind,
                                source_id,
                                workspace_id,
                                model_name,
                                payload_json,
                                source_updated_at,
                                updated_at,
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO embeddings (
                                source_kind, source_id, workspace_id, model_name, embedding_json, updated_at
                            ) VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                            ON CONFLICT (source_kind, source_id, model_name)
                            DO UPDATE SET
                                workspace_id = EXCLUDED.workspace_id,
                                embedding_json = EXCLUDED.embedding_json,
                                updated_at = EXCLUDED.updated_at
                            """,
                            (
                                source_kind,
                                source_id,
                                workspace_id,
                                model_name,
                                payload_json,
                                updated_at,
                            ),
                        )
                updated = int(getattr(cursor, "rowcount", 0) or 0) == 1
            connection.commit()
        return updated

    def get(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str,
    ) -> EmbeddingRecord | None:
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at, memory_updated_at
                    FROM embeddings
                    WHERE source_kind = %s AND source_id = %s AND model_name = %s
                    """,
                    (source_kind, source_id, model_name),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return EmbeddingRecord(
            source_kind=str(row[0]),
            source_id=str(row[1]),
            workspace_id=None if row[2] is None else str(row[2]),
            model_name=str(row[3]),
            embedding=_decode_embedding_payload(row[4]),
            updated_at=_coerce_float(row[5]),
            memory_updated_at=None if row[6] is None else str(row[6]),
        )

    def get_updated_at_map(
        self,
        *,
        source_kind: str,
        model_name: str,
        source_ids: list[str],
    ) -> dict[str, float]:
        if self._sessions is None:
            return {}
        normalized_source_ids = [source_id for source_id in source_ids if isinstance(source_id, str) and source_id]
        if not normalized_source_ids:
            return {}
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_id, updated_at
                    FROM embeddings
                    WHERE source_kind = %s AND model_name = %s AND source_id = ANY(%s::text[])
                    """,
                    (source_kind, model_name, normalized_source_ids),
                )
                rows = cursor.fetchall()
        return {
            str(row[0]): _coerce_float(row[1])
            for row in rows
        }

    def get_memory_updated_at_map(
        self,
        *,
        source_kind: str,
        model_name: str,
        source_ids: list[str],
    ) -> dict[str, str | None]:
        if self._sessions is None:
            return {}
        normalized_source_ids = [source_id for source_id in source_ids if isinstance(source_id, str) and source_id]
        if not normalized_source_ids:
            return {}
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT source_id, memory_updated_at
                    FROM embeddings
                    WHERE source_kind = %s AND model_name = %s AND source_id = ANY(%s::text[])
                    """,
                    (source_kind, model_name, normalized_source_ids),
                )
                rows = cursor.fetchall()
        return {str(row[0]): None if row[1] is None else str(row[1]) for row in rows}

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        candidate_ids: list[str] | None = None,
        diagnostics: dict[str, object] | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        if self._sessions is None or limit < 1:
            return []
        capabilities = self._get_search_capabilities()
        normalized_candidate_ids = [candidate_id for candidate_id in (candidate_ids or []) if candidate_id]
        if candidate_ids is not None and not normalized_candidate_ids:
            return []
        query_dimension = len(query_embedding)
        if capabilities.server_side_vector_search_available:
            query_vector = _format_vector_literal(query_embedding)
            query = (
                "SELECT source_id, 1 - (embedding_vector <=> CAST(%s AS vector)) AS score "
                "FROM embeddings WHERE source_kind = %s AND model_name = %s "
                "AND embedding_vector IS NOT NULL "
                "AND jsonb_typeof(embedding_json) = 'array' "
                "AND jsonb_array_length(embedding_json) = %s"
            )
            params: list[object] = [query_vector, source_kind, model_name, query_dimension]
            if workspace_id is not None:
                query += " AND workspace_id = %s"
                params.append(workspace_id)
            if normalized_candidate_ids:
                query += " AND source_id = ANY(%s::text[])"
                params.append(normalized_candidate_ids)
            query += " ORDER BY embedding_vector <=> CAST(%s AS vector), source_id ASC LIMIT %s"
            params.extend((query_vector, limit))
            fetch_started = time.perf_counter()
            with self._sessions.open_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, tuple(params))
                    rows = cursor.fetchall()
            fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
            scored = [(str(row[0]), _coerce_float(row[1])) for row in rows]
            search_diagnostics = {
                "backend": "postgres",
                "requested_k": limit,
                "row_count": len(rows),
                "returned": len(scored),
                "candidate_filter_count": len(normalized_candidate_ids),
                "query_dimension": query_dimension,
                "dimension_filter_applied": True,
                "search_mode": "server_side_pgvector",
                "pgvector_extension_installed": capabilities.pgvector_extension_installed,
                "embedding_vector_column_present": capabilities.embedding_vector_column_present,
                "server_side_vector_search_available": capabilities.server_side_vector_search_available,
                "raw_type_counts": {},
                "fetch_ms": round(fetch_ms, 3),
                "decode_ms": 0.0,
                "score_ms": 0.0,
                "sort_ms": 0.0,
            }
            self._last_search_diagnostics = search_diagnostics
            if diagnostics is not None:
                diagnostics.update(search_diagnostics)
            return scored
        query = (
            "SELECT source_id, embedding_json FROM embeddings "
            "WHERE source_kind = %s AND model_name = %s"
        )
        params: list[object] = [source_kind, model_name]
        if workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        if normalized_candidate_ids:
            query += " AND source_id = ANY(%s::text[])"
            params.append(normalized_candidate_ids)
        query += " ORDER BY source_id ASC LIMIT %s"
        params.append(self._fallback_row_cap)
        fetch_started = time.perf_counter()
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
        raw_type_counts: dict[str, int] = {}
        decode_started = time.perf_counter()
        decoded_rows: list[tuple[str, list[float]]] = []
        skipped_dimension_mismatch_count = 0
        skipped_invalid_embedding_count = 0
        for row in rows:
            embedding_raw = row[1]
            raw_type_name = type(embedding_raw).__name__
            raw_type_counts[raw_type_name] = raw_type_counts.get(raw_type_name, 0) + 1
            try:
                embedding = _decode_embedding_payload(embedding_raw)
            except (TypeError, ValueError):
                skipped_invalid_embedding_count += 1
                continue
            if len(embedding) != query_dimension:
                skipped_dimension_mismatch_count += 1
                continue
            decoded_rows.append((str(row[0]), embedding))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        scored: list[tuple[str, float]] = []
        score_started = time.perf_counter()
        for source_id, embedding in decoded_rows:
            score = cosine_similarity_lists(query_embedding, embedding)
            scored.append((source_id, score))
        score_ms = (time.perf_counter() - score_started) * 1000.0
        sort_started = time.perf_counter()
        scored.sort(key=lambda item: (-item[1], item[0]))
        sort_ms = (time.perf_counter() - sort_started) * 1000.0
        search_diagnostics = {
            "backend": "postgres",
            "requested_k": limit,
            "row_count": len(rows),
            "compatible_row_count": len(decoded_rows),
            "returned": min(len(scored), limit),
            "candidate_filter_count": len(normalized_candidate_ids),
            "query_dimension": query_dimension,
            "dimension_filter_applied": False,
            "skipped_dimension_mismatch_count": skipped_dimension_mismatch_count,
            "skipped_invalid_embedding_count": skipped_invalid_embedding_count,
            "fallback_row_cap": self._fallback_row_cap,
            "fallback_row_cap_applied": len(rows) >= self._fallback_row_cap,
            "search_mode": "client_python_fallback",
            "pgvector_extension_installed": capabilities.pgvector_extension_installed,
            "embedding_vector_column_present": capabilities.embedding_vector_column_present,
            "server_side_vector_search_available": capabilities.server_side_vector_search_available,
            "raw_type_counts": raw_type_counts,
            "fetch_ms": round(fetch_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "score_ms": round(score_ms, 3),
            "sort_ms": round(sort_ms, 3),
        }
        self._last_search_diagnostics = search_diagnostics
        if diagnostics is not None:
            diagnostics.update(search_diagnostics)
        return scored[:limit]

    def delete_by_model(self, *, source_kind: str, model_name: str) -> int:
        if self._sessions is None:
            return 0
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM embeddings WHERE source_kind = %s AND model_name = %s",
                    (source_kind, model_name),
                )
                deleted = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return deleted

    def delete(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str | None = None,
    ) -> int:
        if self._sessions is None:
            return 0
        query = "DELETE FROM embeddings WHERE source_kind = %s AND source_id = %s"
        params: list[object] = [source_kind, source_id]
        if model_name is not None:
            query += " AND model_name = %s"
            params.append(model_name)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                deleted = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return deleted

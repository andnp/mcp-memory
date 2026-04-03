from __future__ import annotations

from dataclasses import dataclass
import json
import time

from mcp_memory.embeddings import EmbeddingRecord, cosine_similarity
from mcp_memory.storage.session import DbConnectionLike, SessionManager


@dataclass(frozen=True)
class PostgresVectorSearchCapabilities:
    pgvector_extension_installed: bool
    embedding_vector_column_present: bool

    @property
    def server_side_vector_search_available(self) -> bool:
        return self.pgvector_extension_installed and self.embedding_vector_column_present


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


class PostgresVectorStore:
    supports_candidate_filtering = True

    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager
        self._search_capabilities: PostgresVectorSearchCapabilities | None = None

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

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
    ) -> None:
        if self._sessions is None:
            return
        normalized_embedding = [float(value) for value in embedding]
        payload_json = json.dumps(normalized_embedding)
        updated_at = time.time()
        capabilities = self._get_search_capabilities()
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                if capabilities.server_side_vector_search_available:
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
            connection.commit()

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
                    SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at
                    FROM embeddings
                    WHERE source_kind = %s AND source_id = %s AND model_name = %s
                    """,
                    (source_kind, source_id, model_name),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        embedding_raw = row[4]
        if isinstance(embedding_raw, str):
            embedding = list(json.loads(embedding_raw))
        elif isinstance(embedding_raw, list | tuple):
            embedding = list(embedding_raw)
        else:
            raise TypeError(f"Unexpected Postgres embedding payload type: {type(embedding_raw)!r}")
        return EmbeddingRecord(
            source_kind=str(row[0]),
            source_id=str(row[1]),
            workspace_id=None if row[2] is None else str(row[2]),
            model_name=str(row[3]),
            embedding=[float(value) for value in embedding],
            updated_at=_coerce_float(row[5]),
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
        if self._sessions is None:
            return []
        capabilities = self._get_search_capabilities()
        normalized_candidate_ids = [candidate_id for candidate_id in (candidate_ids or []) if candidate_id]
        if candidate_ids is not None and not normalized_candidate_ids:
            return []
        if capabilities.server_side_vector_search_available:
            query_vector = _format_vector_literal(query_embedding)
            query = (
                "SELECT source_id, 1 - (embedding_vector <=> CAST(%s AS vector)) AS score "
                "FROM embeddings WHERE source_kind = %s AND model_name = %s "
                "AND embedding_vector IS NOT NULL"
            )
            params: list[object] = [query_vector, source_kind, model_name]
            if workspace_id is not None:
                query += " AND workspace_id = %s"
                params.append(workspace_id)
            if normalized_candidate_ids:
                query += " AND source_id = ANY(%s::text[])"
                params.append(normalized_candidate_ids)
            query += " ORDER BY embedding_vector <=> CAST(%s AS vector) LIMIT %s"
            params.extend((query_vector, limit))
            fetch_started = time.perf_counter()
            with self._sessions.open_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query, tuple(params))
                    rows = cursor.fetchall()
            fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
            scored = [(str(row[0]), _coerce_float(row[1])) for row in rows]
            if diagnostics is not None:
                diagnostics.update(
                    {
                        "backend": "postgres",
                        "row_count": len(rows),
                        "candidate_filter_count": len(normalized_candidate_ids),
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
                )
            return scored
        query = "SELECT source_id, embedding_json FROM embeddings WHERE source_kind = %s AND model_name = %s"
        params: list[object] = [source_kind, model_name]
        if workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        if normalized_candidate_ids:
            query += " AND source_id = ANY(%s::text[])"
            params.append(normalized_candidate_ids)
        fetch_started = time.perf_counter()
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
        raw_type_counts: dict[str, int] = {}
        decode_started = time.perf_counter()
        decoded_rows: list[tuple[str, list[float]]] = []
        for row in rows:
            embedding_raw = row[1]
            raw_type_name = type(embedding_raw).__name__
            raw_type_counts[raw_type_name] = raw_type_counts.get(raw_type_name, 0) + 1
            if isinstance(embedding_raw, str):
                embedding = list(json.loads(embedding_raw))
            elif isinstance(embedding_raw, list | tuple):
                embedding = list(embedding_raw)
            else:
                raise TypeError(f"Unexpected Postgres embedding payload type: {type(embedding_raw)!r}")
            decoded_rows.append((str(row[0]), [float(value) for value in embedding]))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        scored: list[tuple[str, float]] = []
        score_started = time.perf_counter()
        for source_id, embedding in decoded_rows:
            score = cosine_similarity(query_embedding, embedding)
            scored.append((source_id, score))
        score_ms = (time.perf_counter() - score_started) * 1000.0
        sort_started = time.perf_counter()
        scored.sort(key=lambda item: item[1], reverse=True)
        sort_ms = (time.perf_counter() - sort_started) * 1000.0
        if diagnostics is not None:
            diagnostics.update(
                {
                    "backend": "postgres",
                    "row_count": len(rows),
                    "candidate_filter_count": len(normalized_candidate_ids),
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
            )
        return scored[:limit]

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

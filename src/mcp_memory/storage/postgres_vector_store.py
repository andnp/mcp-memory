from __future__ import annotations

import json
import time

from mcp_memory.embeddings import EmbeddingRecord, cosine_similarity
from mcp_memory.storage.session import DbConnectionLike, SessionManager


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class PostgresVectorStore:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
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
                        json.dumps(embedding),
                        time.time(),
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
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        if self._sessions is None:
            return []
        query = "SELECT source_id, embedding_json FROM embeddings WHERE source_kind = %s AND model_name = %s"
        params: list[object] = [source_kind, model_name]
        if workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            embedding_raw = row[1]
            if isinstance(embedding_raw, str):
                embedding = list(json.loads(embedding_raw))
            elif isinstance(embedding_raw, list | tuple):
                embedding = list(embedding_raw)
            else:
                raise TypeError(f"Unexpected Postgres embedding payload type: {type(embedding_raw)!r}")
            score = cosine_similarity(query_embedding, [float(value) for value in embedding])
            scored.append((str(row[0]), score))
        scored.sort(key=lambda item: item[1], reverse=True)
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
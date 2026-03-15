from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Protocol

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.utils.db import DatabaseManager


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(slots=True)
class EmbeddingRecord:
    source_kind: str
    source_id: str
    workspace_id: str | None
    model_name: str
    embedding: list[float]
    updated_at: float


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingsConfig) -> None:
        self.model_name = config.model
        self._batch_size = config.batch_size
        self._model = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]


class SQLiteVectorStore:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def upsert(
        self,
        *,
        source_kind: str,
        source_id: str,
        workspace_id: str | None,
        model_name: str,
        embedding: list[float],
    ) -> None:
        conn = self._db.get_connection()
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (
                source_kind,
                source_id,
                workspace_id,
                model_name,
                embedding_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
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
        conn.commit()

    def get(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str,
    ) -> EmbeddingRecord | None:
        row = self._db.get_connection().execute(
            """
            SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at
            FROM embeddings
            WHERE source_kind = ? AND source_id = ? AND model_name = ?
            """,
            (source_kind, source_id, model_name),
        ).fetchone()
        if row is None:
            return None
        return EmbeddingRecord(
            source_kind=str(row["source_kind"]),
            source_id=str(row["source_id"]),
            workspace_id=row["workspace_id"],
            model_name=str(row["model_name"]),
            embedding=list(json.loads(row["embedding_json"])),
            updated_at=float(row["updated_at"]),
        )

    def search(
        self,
        *,
        source_kind: str,
        model_name: str,
        query_embedding: list[float],
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        query = (
            "SELECT source_id, embedding_json FROM embeddings "
            "WHERE source_kind = ? AND model_name = ?"
        )
        params: list[object] = [source_kind, model_name]
        if workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        rows = self._db.get_connection().execute(query, params).fetchall()
        scored: list[tuple[str, float]] = []
        for row in rows:
            embedding = list(json.loads(row["embedding_json"]))
            score = cosine_similarity(query_embedding, embedding)
            scored.append((str(row["source_id"]), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def delete(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str | None = None,
    ) -> int:
        query = "DELETE FROM embeddings WHERE source_kind = ? AND source_id = ?"
        params: list[object] = [source_kind, source_id]
        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        conn = self._db.get_connection()
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount


def build_embedder(config: EmbeddingsConfig) -> Embedder | None:
    if not config.enabled:
        return None
    return SentenceTransformerEmbedder(config)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
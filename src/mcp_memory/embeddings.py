from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Protocol

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.utils.db import DatabaseManager


logger = logging.getLogger(__name__)


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
        self._model: Any | None = None
        self._use_fallback = False
        self._fallback = HashingEmbedder(model_name=f"hash:{config.model}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._model is None:
            model = _load_sentence_transformer(self.model_name, local_files_only=True)
            if model is None:
                self._use_fallback = True
            else:
                self._model = model
        if self._use_fallback:
            return self._fallback.embed(texts)
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def cache_model(self) -> bool:
        if self._model is not None and not self._use_fallback:
            return True
        model = _load_sentence_transformer(self.model_name, local_files_only=False)
        if model is None:
            return False
        self._model = model
        self._use_fallback = False
        return True


class HashingEmbedder:
    def __init__(self, model_name: str = "hashing-local", dimensions: int = 64) -> None:
        self.model_name = model_name
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_text_to_unit_vector(text, self._dimensions) for text in texts]


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
    return SentenceTransformerEmbedder(config)


def _load_sentence_transformer(model_name: str, *, local_files_only: bool) -> Any | None:
    if local_files_only and not _is_model_cached_locally(model_name):
        return None
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        model_path = snapshot_download(repo_id=model_name, local_files_only=local_files_only)
        return SentenceTransformer(model_path, local_files_only=True)
    except Exception as exc:
        mode = "cached" if local_files_only else "download"
        logger.warning("Failed to load %s embedding model %s: %s", mode, model_name, exc)
        return None


def _is_model_cached_locally(model_name: str) -> bool:
    hub_cache = os.getenv("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        cache_root = Path(hub_cache)
    else:
        hf_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "hub"

    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    refs_dir = repo_dir / "refs"
    return snapshots_dir.exists() and any(snapshots_dir.iterdir()) or refs_dir.exists()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _hash_text_to_unit_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = [token for token in text.lower().split() if token]
    if not tokens:
        return vector
    for token in tokens:
        digest = blake2b(token.encode("utf-8"), digest_size=16).digest()
        slot = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[slot] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
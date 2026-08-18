from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any, Callable, ClassVar, cast

from searchkernel.indexing.embedding_cache import SQLiteEmbeddingCache
from searchkernel.indexing.semantic import embedding_identity
from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.utils.similarity import cosine_similarity_lists

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.utils.db import DatabaseManager

logger = logging.getLogger(__name__)

_MCP_MEMORY_TORCH_NUM_THREADS_ENV = "MCP_MEMORY_TORCH_NUM_THREADS"
_MCP_MEMORY_TORCH_INTEROP_THREADS_ENV = "MCP_MEMORY_TORCH_INTEROP_THREADS"
_DEFAULT_TORCH_NUM_THREADS = 4
_DEFAULT_TORCH_INTEROP_THREADS = 1
_TORCH_THREAD_CAP_INITIALIZED = False
_TORCH_THREAD_CAP_LOCK = threading.Lock()
_OLLAMA_REQUEST_SEMAPHORES: dict[tuple[str, str, int], threading.BoundedSemaphore] = {}
_OLLAMA_REQUEST_SEMAPHORES_LOCK = threading.Lock()


@dataclass(slots=True)
class EmbeddingRecord:
    source_kind: str
    source_id: str
    workspace_id: str | None
    model_name: str
    embedding: list[float]
    updated_at: float
    memory_updated_at: str | None = None


@dataclass(slots=True)
class EmbedderStatus:
    model_name: str
    backend: str
    model_cached: bool
    configured_model_name: str | None = None


class SentenceTransformerEmbedder:
    def __init__(self, config: EmbeddingsConfig) -> None:
        self._configured_model_name = config.model
        self._batch_size = config.batch_size
        self._model: Any | None = None
        self._use_fallback = False
        self._fallback = HashingEmbedder(model_name=f"hash:{config.model}")
        self._load_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        if self._use_fallback:
            return self._fallback.model_name
        return self._configured_model_name

    @property
    def configured_model_name(self) -> str:
        return self._configured_model_name

    @property
    def encoder_namespace(self) -> str:
        """Return the stable namespace used by searchkernel caches."""
        provider_namespace = getattr(self._model, "encoder_namespace", None)
        if isinstance(provider_namespace, str) and provider_namespace:
            return provider_namespace
        return self.model_name

    @property
    def supports_persistent_embedding_cache(self) -> bool:
        return True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._ensure_model_loaded(allow_download=False):
            return self._fallback.embed(texts)
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(map(float, vector)) for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if not self._ensure_model_loaded(allow_download=False):
            return self._fallback.embed([text])[0]
        assert self._model is not None
        query_embedder = getattr(self._model, "embed_query", None)
        if callable(query_embedder):
            return list(map(float, cast(Sequence[float], query_embedder(text))))
        query_encoder = getattr(self._model, "encode_query", None)
        if callable(query_encoder):
            vector = query_encoder(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return list(map(float, cast(Sequence[Sequence[float]], vector)[0]))
        vectors = self._model.encode(
            [text],
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return list(map(float, vectors[0]))

    def cache_model(self) -> bool:
        return self._ensure_model_loaded(allow_download=True)

    def status(self) -> EmbedderStatus:
        return EmbedderStatus(
            model_name=self.model_name,
            backend="fallback" if self._use_fallback else "sentence-transformer",
            model_cached=_is_model_cached_locally(self._configured_model_name),
            configured_model_name=self._configured_model_name,
        )

    def _ensure_model_loaded(self, *, allow_download: bool) -> bool:
        if self._model is not None:
            return True
        if self._use_fallback and not allow_download:
            return False

        with self._load_lock:
            if self._model is not None:
                return True
            if self._use_fallback and not allow_download:
                return False

            model = self._load_model(allow_download=allow_download)
            if model is None:
                if not allow_download:
                    self._use_fallback = True
                return False

            self._model = model
            self._use_fallback = False
            return True

    def _load_model(self, *, allow_download: bool) -> Any | None:
        cached_locally = _is_model_cached_locally(self._configured_model_name)
        if cached_locally:
            model = _load_sentence_transformer(self._configured_model_name, local_files_only=True)
            if model is not None:
                return model
        if not allow_download or os.environ.get("MCP_MEMORY_TEST_MODE") == "1":
            return None
        return _load_sentence_transformer(self._configured_model_name, local_files_only=False)


class OllamaEmbedder:
    def __init__(self, config: EmbeddingsConfig) -> None:
        from searchkernel.adapters.embedding import OllamaEmbeddingProvider

        self._configured_model_name = config.model
        semaphore_key = (
            config.ollama_base_url,
            config.model,
            config.ollama_max_concurrency,
        )
        with _OLLAMA_REQUEST_SEMAPHORES_LOCK:
            self._request_semaphore = _OLLAMA_REQUEST_SEMAPHORES.setdefault(
                semaphore_key,
                threading.BoundedSemaphore(config.ollama_max_concurrency),
            )
        self._provider = OllamaEmbeddingProvider(
            config.model, base_url=config.ollama_base_url
        )

    @property
    def model_name(self) -> str:
        return self._configured_model_name

    @property
    def dim(self) -> int:
        dimension = getattr(self._provider, "dim", None)
        if not isinstance(dimension, int) or dimension < 1:
            raise AttributeError("Ollama embedding dimension is unavailable")
        return dimension

    @property
    def configured_model_name(self) -> str:
        return self._configured_model_name

    @property
    def encoder_namespace(self) -> str:
        provider_namespace = getattr(self._provider, "encoder_namespace", None)
        if isinstance(provider_namespace, str) and provider_namespace:
            return provider_namespace
        return self.model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._request_semaphore:
            return self._provider.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        with self._request_semaphore:
            query_embedder = getattr(self._provider, "embed_query", None)
            if callable(query_embedder):
                return list(map(float, cast(Sequence[float], query_embedder(text))))
            vectors = self._provider.embed([text])
            if len(vectors) != 1:
                raise ValueError("Ollama embedder must return one query vector")
            return list(map(float, vectors[0]))

    def status(self) -> EmbedderStatus:
        return EmbedderStatus(
            model_name=self.model_name,
            backend="ollama",
            model_cached=True,
            configured_model_name=self._configured_model_name,
        )


class HashingEmbedder:
    def __init__(self, model_name: str = "hashing-local", dimensions: int = 64) -> None:
        self.model_name = model_name
        self._dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_text_to_unit_vector(text, self._dimensions) for text in texts]

    @property
    def encoder_namespace(self) -> str:
        return self.model_name

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def status(self) -> EmbedderStatus:
        return EmbedderStatus(
            model_name=self.model_name,
            backend="hashing",
            model_cached=True,
        )


class SQLiteCachedEmbeddingProvider:
    """Add best-effort content-hash reuse to a local embedding provider."""

    def __init__(self, provider: EmbeddingBatchProvider, cache_path: str | Path) -> None:
        self._provider = provider
        self._cache_path = Path(cache_path)
        self._cache: Any | None = None
        self._cache_namespace: str | None = None
        self._cache_disabled = False

    @property
    def model_name(self) -> str:
        return self._provider.model_name

    @property
    def configured_model_name(self) -> str | None:
        configured_model_name = getattr(self._provider, "configured_model_name", None)
        return configured_model_name if isinstance(configured_model_name, str) else None

    @property
    def encoder_namespace(self) -> str:
        namespace = getattr(self._provider, "encoder_namespace", None)
        return namespace if isinstance(namespace, str) and namespace else self.model_name

    @property
    def dim(self) -> int:
        dimension = getattr(self._provider, "dim", None)
        if isinstance(dimension, int) and dimension > 0:
            return dimension
        dimensions = getattr(self._provider, "_dimensions", None)
        if isinstance(dimensions, int) and dimensions > 0:
            return dimensions
        raise AttributeError("embedding dimension is not available")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if is_fallback_embedding_model(self.model_name):
            return self._provider.embed(texts)
        cache = self._get_cache()
        if cache is None:
            return self._provider.embed(texts)

        namespace = cache.encoder_namespace
        content_hashes = [_embedding_content_hash(text, namespace) for text in texts]
        try:
            cached = cache.get_many(content_hashes)
        except Exception as exc:
            self._disable_cache(exc)
            return self._provider.embed(texts)

        vectors_by_hash = {content_hash: list(vector) for content_hash, vector in cached.items()}
        missing_texts: dict[str, str] = {
            content_hash: text
            for content_hash, text in zip(content_hashes, texts, strict=True)
            if content_hash not in vectors_by_hash
        }
        if missing_texts:
            computed = self._provider.embed(list(missing_texts.values()))
            if len(computed) != len(missing_texts):
                raise ValueError(
                    f"Embedding provider {self.model_name!r} returned {len(computed)} "
                    f"vectors for {len(missing_texts)} inputs"
                )
            computed_by_hash = dict(
                zip(missing_texts, computed, strict=True)
            )
            vectors_by_hash.update(
                {content_hash: list(vector) for content_hash, vector in computed_by_hash.items()}
            )
            self._put_computed_vectors(
                computed_by_hash,
                source_namespace=namespace,
                texts_by_hash=missing_texts,
            )

        return [vectors_by_hash[content_hash] for content_hash in content_hashes]

    def embed_query(self, text: str) -> list[float]:
        query_embedder = getattr(self._provider, "embed_query", None)
        if callable(query_embedder):
            return list(map(float, cast(Sequence[float], query_embedder(text))))
        vectors = self._provider.embed([text])
        if len(vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        return list(map(float, vectors[0]))

    def status(self) -> EmbedderStatus | Any:
        status = getattr(self._provider, "status", None)
        if callable(status):
            return status()
        return describe_embedder(self._provider)

    def _get_cache(self) -> Any | None:
        if self._cache_disabled:
            return None
        namespace = self.encoder_namespace
        if self._cache is not None and self._cache_namespace == namespace:
            return self._cache
        if self._cache is not None:
            close = getattr(self._cache, "close", None)
            if callable(close):
                close()
        try:
            self._cache = SQLiteEmbeddingCache(
                self._cache_path,
                namespace,
                dimension=0,
            )
        except Exception as exc:
            self._disable_cache(exc)
            return None
        self._cache_namespace = namespace
        return self._cache

    def _put_computed_vectors(
        self,
        vectors: dict[str, list[float]],
        *,
        source_namespace: str,
        texts_by_hash: dict[str, str],
    ) -> None:
        if is_fallback_embedding_model(self.model_name):
            return
        cache = self._get_cache()
        if cache is None:
            return
        vectors_to_store = vectors
        if cache.encoder_namespace != source_namespace:
            vectors_to_store = {
                _embedding_content_hash(texts_by_hash[content_hash], cache.encoder_namespace): vector
                for content_hash, vector in vectors.items()
            }
        try:
            cache.put_many(vectors_to_store)
        except Exception as exc:
            self._disable_cache(exc)

    def _disable_cache(self, exc: Exception) -> None:
        logger.warning("Embedding cache unavailable; continuing without reuse: %s", exc)
        self._cache_disabled = True
        self._cache = None
        self._cache_namespace = None


def with_local_embedding_cache(
    embedder: EmbeddingBatchProvider | None,
    cache_path: str | Path | None,
) -> EmbeddingBatchProvider | None:
    """Wrap only providers that explicitly opt into local content caching."""
    if embedder is None or cache_path is None:
        return embedder
    if not getattr(embedder, "supports_persistent_embedding_cache", False):
        return embedder
    return SQLiteCachedEmbeddingProvider(embedder, cache_path)


def _embedding_content_hash(text: str, encoder_namespace: str) -> str:
    return embedding_identity(text, encoder_namespace)


class SQLiteVectorStore:
    supports_candidate_filtering: ClassVar[bool] = True

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
        source_updated_at: str | None = None,
    ) -> bool:
        conn = self._db.get_connection()
        if source_updated_at is not None:
            owns_transaction = not conn.in_transaction
            if owns_transaction:
                conn.execute("BEGIN IMMEDIATE")
            try:
                current = conn.execute(
                    "SELECT updated_at FROM memories WHERE id = ?",
                    (source_id,),
                ).fetchone()
                if current is None or str(current["updated_at"]) != source_updated_at:
                    if owns_transaction:
                        conn.rollback()
                    return False
                cursor = conn.execute(
                    """
                    INSERT INTO embeddings (
                        source_kind, source_id, workspace_id, model_name,
                        embedding_json, memory_updated_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source_kind, source_id, model_name) DO UPDATE SET
                        workspace_id = excluded.workspace_id,
                        embedding_json = excluded.embedding_json,
                        memory_updated_at = excluded.memory_updated_at,
                        updated_at = excluded.updated_at
                    WHERE embeddings.memory_updated_at IS NULL
                       OR embeddings.memory_updated_at <= excluded.memory_updated_at
                    """,
                    (
                        source_kind,
                        source_id,
                        workspace_id,
                        model_name,
                        json.dumps(embedding),
                        source_updated_at,
                        time.time(),
                    ),
                )
                if owns_transaction:
                    conn.commit()
                return cursor.rowcount == 1
            except Exception:
                if owns_transaction:
                    conn.rollback()
                raise
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
        return True

    def get(
        self,
        *,
        source_kind: str,
        source_id: str,
        model_name: str,
    ) -> EmbeddingRecord | None:
        row = self._db.get_connection().execute(
            """
            SELECT source_kind, source_id, workspace_id, model_name, embedding_json, updated_at, memory_updated_at
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
            memory_updated_at=None if row["memory_updated_at"] is None else str(row["memory_updated_at"]),
        )

    def get_updated_at_map(
        self,
        *,
        source_kind: str,
        model_name: str,
        source_ids: list[str],
    ) -> dict[str, float]:
        normalized_source_ids = [source_id for source_id in source_ids if isinstance(source_id, str) and source_id]
        if not normalized_source_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_source_ids)
        rows = self._db.get_connection().execute(
            f"""
            SELECT source_id, updated_at
            FROM embeddings
            WHERE source_kind = ? AND model_name = ? AND source_id IN ({placeholders})
            """,
            [source_kind, model_name, *normalized_source_ids],
        ).fetchall()
        return {
            str(row["source_id"]): float(row["updated_at"])
            for row in rows
        }

    def get_memory_updated_at_map(
        self,
        *,
        source_kind: str,
        model_name: str,
        source_ids: list[str],
    ) -> dict[str, str | None]:
        normalized_source_ids = [source_id for source_id in source_ids if isinstance(source_id, str) and source_id]
        if not normalized_source_ids:
            return {}
        placeholders = ",".join("?" for _ in normalized_source_ids)
        rows = self._db.get_connection().execute(
            f"""
            SELECT source_id, memory_updated_at
            FROM embeddings
            WHERE source_kind = ? AND model_name = ? AND source_id IN ({placeholders})
            """,
            [source_kind, model_name, *normalized_source_ids],
        ).fetchall()
        return {
            str(row["source_id"]): None if row["memory_updated_at"] is None else str(row["memory_updated_at"])
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
        normalized_candidate_ids = [candidate_id for candidate_id in (candidate_ids or []) if candidate_id]
        if candidate_ids is not None and not normalized_candidate_ids:
            return []
        query = (
            "SELECT source_id, embedding_json FROM embeddings "
            "WHERE source_kind = ? AND model_name = ?"
        )
        params: list[object] = [source_kind, model_name]
        if normalized_candidate_ids:
            placeholders = ",".join("?" for _ in normalized_candidate_ids)
            query += f" AND source_id IN ({placeholders})"
            params.extend(normalized_candidate_ids)
        if workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        fetch_started = time.perf_counter()
        rows = self._db.get_connection().execute(query, params).fetchall()
        fetch_ms = (time.perf_counter() - fetch_started) * 1000.0
        raw_type_counts: dict[str, int] = {}
        decode_started = time.perf_counter()
        decoded_rows: list[tuple[str, list[float]]] = []
        for row in rows:
            embedding_raw = row["embedding_json"]
            raw_type_name = type(embedding_raw).__name__
            raw_type_counts[raw_type_name] = raw_type_counts.get(raw_type_name, 0) + 1
            embedding = list(json.loads(embedding_raw))
            decoded_rows.append((str(row["source_id"]), embedding))
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        scored: list[tuple[str, float]] = []
        score_started = time.perf_counter()
        for source_id, embedding in decoded_rows:
            score = cosine_similarity_lists(query_embedding, embedding)
            scored.append((source_id, score))
        score_ms = (time.perf_counter() - score_started) * 1000.0
        sort_started = time.perf_counter()
        scored.sort(key=lambda item: item[1], reverse=True)
        sort_ms = (time.perf_counter() - sort_started) * 1000.0
        if diagnostics is not None:
            diagnostics.update(
                {
                    "backend": "sqlite",
                    "row_count": len(rows),
                    "candidate_filter_count": len(normalized_candidate_ids),
                    "raw_type_counts": raw_type_counts,
                    "fetch_ms": round(fetch_ms, 3),
                    "decode_ms": round(decode_ms, 3),
                    "score_ms": round(score_ms, 3),
                    "sort_ms": round(sort_ms, 3),
                }
            )
        return scored[:limit]

    def delete_by_model(self, *, source_kind: str, model_name: str) -> int:
        conn = self._db.get_connection()
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE source_kind = ? AND model_name = ?",
            (source_kind, model_name),
        )
        conn.commit()
        return cursor.rowcount

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


def build_embedder(config: EmbeddingsConfig) -> EmbeddingBatchProvider | None:
    if config.provider == "ollama":
        return OllamaEmbedder(config)
    return SentenceTransformerEmbedder(config)


def is_fallback_embedding_model(model_name: str) -> bool:
    normalized_model_name = model_name.strip().lower()
    return normalized_model_name.startswith("hash:") or normalized_model_name == "hashing-local"


def describe_embedder(embedder: Any) -> EmbedderStatus | None:
    if embedder is None:
        return None
    status = getattr(embedder, "status", None)
    if callable(status):
        result = status()
        if isinstance(result, EmbedderStatus):
            return result
    model_name = getattr(embedder, "model_name", None)
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    return EmbedderStatus(
        model_name=model_name,
        backend=type(embedder).__name__,
        model_cached=False,
        configured_model_name=getattr(embedder, "configured_model_name", None),
    )


def _load_sentence_transformer(model_name: str, *, local_files_only: bool) -> Any | None:
    if local_files_only and not _is_model_cached_locally(model_name):
        return None
    _cap_torch_threads_before_sentence_transformer_load()
    try:
        from huggingface_hub import snapshot_download
        from sentence_transformers import SentenceTransformer

        model_path = snapshot_download(repo_id=model_name, local_files_only=local_files_only)
        return SentenceTransformer(model_path, local_files_only=True)
    except Exception as exc:
        mode = "cached" if local_files_only else "download"
        logger.warning("Failed to load %s embedding model %s: %s", mode, model_name, exc)
        return None


def _cap_torch_threads_before_sentence_transformer_load() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    global _TORCH_THREAD_CAP_INITIALIZED
    if _TORCH_THREAD_CAP_INITIALIZED:
        return

    with _TORCH_THREAD_CAP_LOCK:
        if _TORCH_THREAD_CAP_INITIALIZED:
            return
        _TORCH_THREAD_CAP_INITIALIZED = True

        try:
            import torch
        except ImportError:
            return

        _cap_torch_thread_count(
            current_threads=torch.get_num_threads(),
            desired_threads=_read_torch_thread_cap_from_env(
                _MCP_MEMORY_TORCH_NUM_THREADS_ENV,
                _DEFAULT_TORCH_NUM_THREADS,
            ),
            set_threads=torch.set_num_threads,
        )

        try:
            current_interop_threads = torch.get_num_interop_threads()
        except RuntimeError:
            return

        try:
            _cap_torch_thread_count(
                current_threads=current_interop_threads,
                desired_threads=_read_torch_thread_cap_from_env(
                    _MCP_MEMORY_TORCH_INTEROP_THREADS_ENV,
                    _DEFAULT_TORCH_INTEROP_THREADS,
                ),
                set_threads=torch.set_num_interop_threads,
            )
        except RuntimeError:
            return


def _cap_torch_thread_count(
    *,
    current_threads: int,
    desired_threads: int,
    set_threads: Callable[[int], None],
) -> None:
    if current_threads <= desired_threads:
        return
    set_threads(desired_threads)


def _read_torch_thread_cap_from_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        return default
    try:
        parsed_value = int(raw_value)
    except ValueError:
        return default
    if parsed_value < 1:
        return default
    return parsed_value


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

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
import sqlite3
from threading import Event, Lock
from time import time
from collections.abc import Sequence
from typing import Any


logger = logging.getLogger(__name__)
_CACHE_SCHEMA_VERSION = 1
_RECENT_CACHE_METRICS_WINDOW_MINUTES = 15
_INFLIGHT_SEARCH_FOLLOWER_WAIT_TIMEOUT_SECONDS = 1.0
_METRIC_COLUMNS = (
    "search_requests",
    "fresh_exact_search_hits",
    "stale_exact_search_fallbacks",
    "projection_fallbacks",
    "read_requests",
    "validated_read_hits",
    "read_validation_mismatches",
    "read_validation_failures",
    "warmed_projection_rows",
)


@dataclass(frozen=True)
class SharedReadCacheRecentMetricsSnapshot:
    window_minutes: int = _RECENT_CACHE_METRICS_WINDOW_MINUTES
    search_requests: int = 0
    fresh_exact_search_hits: int = 0
    stale_exact_search_fallbacks: int = 0
    projection_fallbacks: int = 0
    read_requests: int = 0
    validated_read_hits: int = 0
    read_validation_mismatches: int = 0
    read_validation_failures: int = 0
    warmed_projection_rows: int = 0
    fresh_exact_search_hit_rate: float = 0.0
    stale_exact_search_fallback_rate: float = 0.0
    projection_fallback_rate: float = 0.0
    validated_read_hit_rate: float = 0.0
    read_validation_mismatch_rate: float = 0.0
    read_validation_failure_rate: float = 0.0


@dataclass(frozen=True)
class SharedReadCacheSearchRequest:
    query: str
    workspace_id: str | None
    limit: int
    adaptive_limit: bool
    memory_type: str | None
    status: str | None
    include_superseded: bool

    def normalized_params(self) -> dict[str, Any]:
        return {
            "cache_schema_version": _CACHE_SCHEMA_VERSION,
            "query": self.query,
            "workspace_id": self.workspace_id,
            "limit": self.limit,
            "adaptive_limit": self.adaptive_limit,
            "memory_type": self.memory_type,
            "status": self.status,
            "include_superseded": self.include_superseded,
        }


@dataclass(frozen=True)
class SharedReadCacheReadEntry:
    payload: dict[str, Any]
    validation_token: str | None


@dataclass(frozen=True)
class SharedReadCacheProjectionUpsert:
    memory_id: str
    payload: dict[str, Any]
    validation_token: str | None = None


@dataclass(frozen=True)
class SharedReadCacheProjectionEntry:
    memory_id: str
    payload: dict[str, Any]
    validation_token: str | None
    cached_at: float


@dataclass(frozen=True)
class SharedReadCacheRecordThoughtOutboxEntry:
    outbox_id: int
    content: str
    workspace_id: str | None
    timestamp: float
    cached_at: float


@dataclass
class _SharedReadCacheInFlightSearchState:
    created_at: float = field(default_factory=time)
    completed: Event = field(default_factory=Event)
    payload: dict[str, Any] | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class SharedReadCacheInFlightSearch:
    cache_key: str
    is_leader: bool
    _state: _SharedReadCacheInFlightSearchState


@dataclass(frozen=True)
class SharedReadCacheMetricsSnapshot:
    search_requests: int = 0
    fresh_exact_search_hits: int = 0
    stale_exact_search_fallbacks: int = 0
    projection_fallbacks: int = 0
    read_requests: int = 0
    validated_read_hits: int = 0
    read_validation_mismatches: int = 0
    read_validation_failures: int = 0
    warmed_projection_rows: int = 0
    cached_search_result_count: int = 0
    cached_read_record_count: int = 0
    cached_projection_count: int = 0
    fresh_exact_search_hit_rate: float = 0.0
    stale_exact_search_fallback_rate: float = 0.0
    projection_fallback_rate: float = 0.0
    validated_read_hit_rate: float = 0.0
    read_validation_mismatch_rate: float = 0.0
    read_validation_failure_rate: float = 0.0
    recent: SharedReadCacheRecentMetricsSnapshot = field(default_factory=SharedReadCacheRecentMetricsSnapshot)


class SharedReadCache:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._inflight_searches: dict[str, _SharedReadCacheInFlightSearchState] = {}
        self._inflight_searches_lock = Lock()
        self._initialize()

    def close(self) -> None:
        return None

    def enqueue_record_thought_outbox_entry(
        self,
        *,
        content: str,
        workspace_id: str | None,
        timestamp: float,
        max_entries: int,
    ) -> SharedReadCacheRecordThoughtOutboxEntry | None:
        try:
            with self._connect() as connection:
                cached_at = time()
                cursor = connection.execute(
                    """
                    INSERT INTO record_thought_outbox (content, workspace_id, timestamp, cached_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (content, workspace_id, timestamp, cached_at),
                )
                outbox_id = cursor.lastrowid
                if outbox_id is None:
                    raise RuntimeError("record_thought_outbox_insert_missing_rowid")
                count_row = connection.execute(
                    "SELECT COUNT(*) FROM record_thought_outbox",
                ).fetchone()
                entry_count = 0 if count_row is None else _coerce_int(count_row[0])
                overflow = max(entry_count - max_entries, 0)
                if overflow > 0:
                    connection.execute(
                        """
                        DELETE FROM record_thought_outbox
                        WHERE id IN (
                            SELECT id
                            FROM record_thought_outbox
                            ORDER BY cached_at ASC, id ASC
                            LIMIT ?
                        )
                        """,
                        (overflow,),
                    )
                connection.commit()
        except Exception:
            logger.warning("Shared read cache writeback outbox write failed", exc_info=True)
            return None
        return SharedReadCacheRecordThoughtOutboxEntry(
            outbox_id=int(outbox_id),
            content=content,
            workspace_id=workspace_id,
            timestamp=timestamp,
            cached_at=cached_at,
        )

    def list_record_thought_outbox_entries(
        self,
        *,
        limit: int,
    ) -> list[SharedReadCacheRecordThoughtOutboxEntry]:
        rows = self._fetchall(
            """
            SELECT id, content, workspace_id, timestamp, cached_at
            FROM record_thought_outbox
            ORDER BY cached_at ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        )
        return [
            SharedReadCacheRecordThoughtOutboxEntry(
                outbox_id=_coerce_int(row[0]),
                content=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                timestamp=float(row[3]),
                cached_at=float(row[4]),
            )
            for row in rows
        ]

    def delete_record_thought_outbox_entries(self, outbox_ids: list[int]) -> None:
        normalized_ids = [outbox_id for outbox_id in outbox_ids if isinstance(outbox_id, int) and outbox_id > 0]
        if not normalized_ids:
            return
        placeholders = ",".join("?" for _ in normalized_ids)
        self._execute(
            f"DELETE FROM record_thought_outbox WHERE id IN ({placeholders})",
            tuple(normalized_ids),
        )

    def count_record_thought_outbox_entries(self) -> int:
        row = self._fetchone("SELECT COUNT(*) FROM record_thought_outbox", ())
        return 0 if row is None else _coerce_int(row[0])

    def load_search_response(self, request: SharedReadCacheSearchRequest) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT payload_json FROM cached_search_results WHERE cache_key = ?",
            (self._search_cache_key(request),),
        )
        return self._decode_payload(row[0]) if row is not None else None

    def load_fresh_search_response(
        self,
        request: SharedReadCacheSearchRequest,
        *,
        ttl_seconds: float,
    ) -> dict[str, Any] | None:
        row = self._fetchone(
            "SELECT payload_json, cached_at FROM cached_search_results WHERE cache_key = ?",
            (self._search_cache_key(request),),
        )
        if row is None:
            return None
        payload_json, cached_at = row
        if not isinstance(cached_at, int | float):
            return None
        if time() - float(cached_at) > ttl_seconds:
            return None
        return self._decode_payload(payload_json)

    def store_search_response(self, request: SharedReadCacheSearchRequest, payload: dict[str, Any]) -> None:
        params_json = self._serialize_json(request.normalized_params())
        self._execute(
            """
            INSERT INTO cached_search_results (cache_key, params_json, payload_json, cached_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                params_json = excluded.params_json,
                payload_json = excluded.payload_json,
                cached_at = excluded.cached_at
            """,
            (
                self._search_cache_key(request),
                params_json,
                self._serialize_json(payload),
                time(),
            ),
        )

    def begin_inflight_search(self, request: SharedReadCacheSearchRequest) -> SharedReadCacheInFlightSearch:
        cache_key = self._search_cache_key(request)
        with self._inflight_searches_lock:
            state = self._inflight_searches.get(cache_key)
            if state is None:
                state = _SharedReadCacheInFlightSearchState()
                self._inflight_searches[cache_key] = state
                return SharedReadCacheInFlightSearch(cache_key=cache_key, is_leader=True, _state=state)
        return SharedReadCacheInFlightSearch(cache_key=cache_key, is_leader=False, _state=state)

    def wait_for_inflight_search(self, entry: SharedReadCacheInFlightSearch) -> dict[str, Any]:
        if not entry._state.completed.wait(timeout=_INFLIGHT_SEARCH_FOLLOWER_WAIT_TIMEOUT_SECONDS):
            with self._inflight_searches_lock:
                current = self._inflight_searches.get(entry.cache_key)
                if (
                    current is not None
                    and current is entry._state
                    and current.payload is None
                    and current.error is None
                    and not current.completed.is_set()
                    and (time() - current.created_at) >= _INFLIGHT_SEARCH_FOLLOWER_WAIT_TIMEOUT_SECONDS
                ):
                    self._inflight_searches.pop(entry.cache_key, None)
            raise TimeoutError("Timed out waiting for in-flight shared read-cache search")
        if entry._state.payload is not None:
            return entry._state.payload
        if entry._state.error is not None:
            raise entry._state.error
        raise RuntimeError("In-flight search completed without payload or error")

    def finish_inflight_search(
        self,
        entry: SharedReadCacheInFlightSearch,
        *,
        payload: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        entry._state.payload = payload
        entry._state.error = error
        entry._state.completed.set()
        with self._inflight_searches_lock:
            current = self._inflight_searches.get(entry.cache_key)
            if current is entry._state:
                self._inflight_searches.pop(entry.cache_key, None)

    def load_read_entry(self, memory_id: str) -> SharedReadCacheReadEntry | None:
        row = self._fetchone(
            "SELECT payload_json, validation_token FROM cached_memory_records WHERE memory_id = ?",
            (memory_id,),
        )
        if row is None:
            return None
        payload = self._decode_payload(row[0])
        if payload is None:
            return None
        validation_token = row[1]
        return SharedReadCacheReadEntry(
            payload=payload,
            validation_token=validation_token if isinstance(validation_token, str) else None,
        )

    def load_read_response(self, memory_id: str) -> dict[str, Any] | None:
        entry = self.load_read_entry(memory_id)
        return None if entry is None else entry.payload

    def store_read_response(
        self,
        memory_id: str,
        payload: dict[str, Any],
        *,
        validation_token: str | None = None,
    ) -> None:
        self._execute(
            """
            INSERT INTO cached_memory_records (memory_id, payload_json, validation_token, cached_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                validation_token = excluded.validation_token,
                cached_at = excluded.cached_at
            """,
            (memory_id, self._serialize_json(payload), validation_token, time()),
        )

    def load_projection_entry(self, memory_id: str) -> SharedReadCacheProjectionEntry | None:
        row = self._fetchone(
            "SELECT payload_json, validation_token, cached_at FROM cached_memory_projections WHERE memory_id = ?",
            (memory_id,),
        )
        if row is None:
            return None
        return self._projection_entry_from_row(memory_id, row)

    def load_projection_entries(self, memory_ids: list[str]) -> list[SharedReadCacheProjectionEntry]:
        normalized_ids = [memory_id for memory_id in memory_ids if isinstance(memory_id, str) and memory_id]
        if not normalized_ids:
            return []
        placeholders = ",".join("?" for _ in normalized_ids)
        rows = self._fetchall(
            (
                "SELECT memory_id, payload_json, validation_token, cached_at "
                f"FROM cached_memory_projections WHERE memory_id IN ({placeholders})"
            ),
            tuple(normalized_ids),
        )
        entries_by_id = {
            entry.memory_id: entry
            for entry in (
                self._projection_entry_from_row(str(row[0]), row[1:])
                for row in rows
            )
            if entry is not None
        }
        return [entries_by_id[memory_id] for memory_id in normalized_ids if memory_id in entries_by_id]

    def search_projection_payloads(
        self,
        request: SharedReadCacheSearchRequest,
    ) -> list[dict[str, Any]]:
        query_tokens = _projection_query_tokens(request.query)
        if not query_tokens or request.limit <= 0:
            return []
        rows = self._fetchall(
            "SELECT memory_id, payload_json, validation_token, cached_at FROM cached_memory_projections",
            (),
        )
        matches: list[tuple[int, float, str, dict[str, Any]]] = []
        for row in rows:
            entry = self._projection_entry_from_row(str(row[0]), row[1:])
            if entry is None or entry.validation_token is None:
                continue
            if not _projection_matches_request(entry.payload, request):
                continue
            hit_count = _projection_hit_count(entry.payload, query_tokens)
            if hit_count <= 0:
                continue
            matches.append((hit_count, entry.cached_at, entry.memory_id, entry.payload))
        matches.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [payload for _hit_count, _cached_at, _memory_id, payload in matches[: request.limit]]

    def store_projection_entries(self, entries: list[SharedReadCacheProjectionUpsert]) -> None:
        if not entries:
            return
        serialized_entries = [
            (entry.memory_id, self._serialize_json(entry.payload), entry.validation_token, time())
            for entry in entries
            if entry.memory_id
        ]
        if not serialized_entries:
            return
        self._executemany(
            """
            INSERT INTO cached_memory_projections (memory_id, payload_json, validation_token, cached_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                payload_json = excluded.payload_json,
                validation_token = excluded.validation_token,
                cached_at = excluded.cached_at
            """,
            serialized_entries,
        )

    def delete_projection_entries(self, memory_ids: list[str]) -> None:
        normalized_ids = [memory_id for memory_id in memory_ids if isinstance(memory_id, str) and memory_id]
        if not normalized_ids:
            return
        placeholders = ",".join("?" for _ in normalized_ids)
        self._execute(
            f"DELETE FROM cached_memory_projections WHERE memory_id IN ({placeholders})",
            tuple(normalized_ids),
        )

    def increment_metric(self, metric_name: str, *, amount: int = 1) -> None:
        if metric_name not in _METRIC_COLUMNS or amount <= 0:
            return
        bucket_start_minute = _bucket_start_minute(time())
        self._execute_script(
            f"""
            INSERT INTO cache_metric_minute_buckets (bucket_start_minute, {metric_name})
            VALUES ({bucket_start_minute}, {amount})
            ON CONFLICT(bucket_start_minute) DO UPDATE SET
                {metric_name} = {metric_name} + {amount};

            UPDATE cache_metrics SET {metric_name} = {metric_name} + {amount} WHERE singleton = 1;
            """
        )

    def get_metrics_snapshot(self, *, now: float | None = None) -> SharedReadCacheMetricsSnapshot:
        row = self._fetchone(
            (
                "SELECT "
                "search_requests, fresh_exact_search_hits, stale_exact_search_fallbacks, projection_fallbacks, "
                "read_requests, validated_read_hits, read_validation_mismatches, read_validation_failures, "
                "warmed_projection_rows, "
                "(SELECT COUNT(*) FROM cached_search_results) AS cached_search_result_count, "
                "(SELECT COUNT(*) FROM cached_memory_records) AS cached_read_record_count, "
                "(SELECT COUNT(*) FROM cached_memory_projections) AS cached_projection_count "
                "FROM cache_metrics WHERE singleton = 1"
            ),
            (),
        )
        if row is None:
            return SharedReadCacheMetricsSnapshot()

        recent_snapshot = self.get_recent_metrics_snapshot(now=now)
        snapshot = SharedReadCacheMetricsSnapshot(
            search_requests=_coerce_int(row[0]),
            fresh_exact_search_hits=_coerce_int(row[1]),
            stale_exact_search_fallbacks=_coerce_int(row[2]),
            projection_fallbacks=_coerce_int(row[3]),
            read_requests=_coerce_int(row[4]),
            validated_read_hits=_coerce_int(row[5]),
            read_validation_mismatches=_coerce_int(row[6]),
            read_validation_failures=_coerce_int(row[7]),
            warmed_projection_rows=_coerce_int(row[8]),
            cached_search_result_count=_coerce_int(row[9]),
            cached_read_record_count=_coerce_int(row[10]),
            cached_projection_count=_coerce_int(row[11]),
            recent=recent_snapshot,
        )
        return replace(
            snapshot,
            fresh_exact_search_hit_rate=_safe_rate(snapshot.fresh_exact_search_hits, snapshot.search_requests),
            stale_exact_search_fallback_rate=_safe_rate(snapshot.stale_exact_search_fallbacks, snapshot.search_requests),
            projection_fallback_rate=_safe_rate(snapshot.projection_fallbacks, snapshot.search_requests),
            validated_read_hit_rate=_safe_rate(snapshot.validated_read_hits, snapshot.read_requests),
            read_validation_mismatch_rate=_safe_rate(snapshot.read_validation_mismatches, snapshot.read_requests),
            read_validation_failure_rate=_safe_rate(snapshot.read_validation_failures, snapshot.read_requests),
        )

    def get_recent_metrics_snapshot(
        self,
        *,
        now: float | None = None,
        window_minutes: int = _RECENT_CACHE_METRICS_WINDOW_MINUTES,
    ) -> SharedReadCacheRecentMetricsSnapshot:
        effective_window_minutes = max(window_minutes, 1)
        bucket_end = _bucket_start_minute(time() if now is None else now)
        bucket_start = bucket_end - ((effective_window_minutes - 1) * 60)
        row = self._fetchone(
            (
                "SELECT "
                "COALESCE(SUM(search_requests), 0), "
                "COALESCE(SUM(fresh_exact_search_hits), 0), "
                "COALESCE(SUM(stale_exact_search_fallbacks), 0), "
                "COALESCE(SUM(projection_fallbacks), 0), "
                "COALESCE(SUM(read_requests), 0), "
                "COALESCE(SUM(validated_read_hits), 0), "
                "COALESCE(SUM(read_validation_mismatches), 0), "
                "COALESCE(SUM(read_validation_failures), 0), "
                "COALESCE(SUM(warmed_projection_rows), 0) "
                "FROM cache_metric_minute_buckets "
                "WHERE bucket_start_minute >= ? AND bucket_start_minute <= ?"
            ),
            (bucket_start, bucket_end),
        )
        if row is None:
            return SharedReadCacheRecentMetricsSnapshot(window_minutes=effective_window_minutes)

        snapshot = SharedReadCacheRecentMetricsSnapshot(
            window_minutes=effective_window_minutes,
            search_requests=_coerce_int(row[0]),
            fresh_exact_search_hits=_coerce_int(row[1]),
            stale_exact_search_fallbacks=_coerce_int(row[2]),
            projection_fallbacks=_coerce_int(row[3]),
            read_requests=_coerce_int(row[4]),
            validated_read_hits=_coerce_int(row[5]),
            read_validation_mismatches=_coerce_int(row[6]),
            read_validation_failures=_coerce_int(row[7]),
            warmed_projection_rows=_coerce_int(row[8]),
        )
        return replace(
            snapshot,
            fresh_exact_search_hit_rate=_safe_rate(snapshot.fresh_exact_search_hits, snapshot.search_requests),
            stale_exact_search_fallback_rate=_safe_rate(snapshot.stale_exact_search_fallbacks, snapshot.search_requests),
            projection_fallback_rate=_safe_rate(snapshot.projection_fallbacks, snapshot.search_requests),
            validated_read_hit_rate=_safe_rate(snapshot.validated_read_hits, snapshot.read_requests),
            read_validation_mismatch_rate=_safe_rate(snapshot.read_validation_mismatches, snapshot.read_requests),
            read_validation_failure_rate=_safe_rate(snapshot.read_validation_failures, snapshot.read_requests),
        )

    def _initialize(self) -> None:
        self._execute_script(
            """
            CREATE TABLE IF NOT EXISTS cached_search_results (
                cache_key TEXT PRIMARY KEY,
                params_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cached_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cached_memory_records (
                memory_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                validation_token TEXT,
                cached_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cached_memory_projections (
                memory_id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                validation_token TEXT,
                cached_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS record_thought_outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                workspace_id TEXT,
                timestamp REAL NOT NULL,
                cached_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cache_metrics (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                search_requests INTEGER NOT NULL DEFAULT 0,
                fresh_exact_search_hits INTEGER NOT NULL DEFAULT 0,
                stale_exact_search_fallbacks INTEGER NOT NULL DEFAULT 0,
                projection_fallbacks INTEGER NOT NULL DEFAULT 0,
                read_requests INTEGER NOT NULL DEFAULT 0,
                validated_read_hits INTEGER NOT NULL DEFAULT 0,
                read_validation_mismatches INTEGER NOT NULL DEFAULT 0,
                read_validation_failures INTEGER NOT NULL DEFAULT 0,
                warmed_projection_rows INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS cache_metric_minute_buckets (
                bucket_start_minute INTEGER PRIMARY KEY,
                search_requests INTEGER NOT NULL DEFAULT 0,
                fresh_exact_search_hits INTEGER NOT NULL DEFAULT 0,
                stale_exact_search_fallbacks INTEGER NOT NULL DEFAULT 0,
                projection_fallbacks INTEGER NOT NULL DEFAULT 0,
                read_requests INTEGER NOT NULL DEFAULT 0,
                validated_read_hits INTEGER NOT NULL DEFAULT 0,
                read_validation_mismatches INTEGER NOT NULL DEFAULT 0,
                read_validation_failures INTEGER NOT NULL DEFAULT 0,
                warmed_projection_rows INTEGER NOT NULL DEFAULT 0
            );

            INSERT INTO cache_metrics (singleton)
            VALUES (1)
            ON CONFLICT(singleton) DO NOTHING;
            """
        )
        self._ensure_column("cached_memory_records", "validation_token", "TEXT")
        self._ensure_column("cached_memory_projections", "validation_token", "TEXT")

    def _search_cache_key(self, request: SharedReadCacheSearchRequest) -> str:
        normalized = self._serialize_json(request.normalized_params())
        return sha256(normalized.encode("utf-8")).hexdigest()

    def _execute_script(self, script: str) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(script)
        except Exception:
            logger.warning("Shared read cache schema initialization failed", exc_info=True)

    def _ensure_column(self, table_name: str, column_name: str, column_definition: str) -> None:
        try:
            with self._connect() as connection:
                columns = {
                    str(row["name"])
                    for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
                }
                if column_name in columns:
                    return
                connection.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
                )
                connection.commit()
        except Exception:
            logger.warning(
                "Shared read cache schema migration failed for %s.%s",
                table_name,
                column_name,
                exc_info=True,
            )

    def _execute(self, query: str, params: tuple[object, ...]) -> None:
        try:
            with self._connect() as connection:
                connection.execute(query, params)
                connection.commit()
        except Exception:
            logger.warning("Shared read cache write failed", exc_info=True)

    def _executemany(self, query: str, params: Sequence[tuple[object, ...]]) -> None:
        try:
            with self._connect() as connection:
                connection.executemany(query, params)
                connection.commit()
        except Exception:
            logger.warning("Shared read cache write failed", exc_info=True)

    def _fetchone(self, query: str, params: tuple[object, ...]) -> tuple[Any, ...] | None:
        try:
            with self._connect() as connection:
                row = connection.execute(query, params).fetchone()
        except Exception:
            logger.warning("Shared read cache read failed", exc_info=True)
            return None
        return tuple(row) if row is not None else None

    def _fetchall(self, query: str, params: tuple[object, ...]) -> list[tuple[Any, ...]]:
        try:
            with self._connect() as connection:
                rows = connection.execute(query, params).fetchall()
        except Exception:
            logger.warning("Shared read cache read failed", exc_info=True)
            return []
        return [tuple(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _decode_payload(self, payload_json: Any) -> dict[str, Any] | None:
        if not isinstance(payload_json, str):
            return None
        payload = json.loads(payload_json)
        return payload if isinstance(payload, dict) else None

    def _projection_entry_from_row(
        self,
        memory_id: str,
        row: tuple[Any, ...],
    ) -> SharedReadCacheProjectionEntry | None:
        payload = self._decode_payload(row[0])
        cached_at = row[2]
        if payload is None or not isinstance(cached_at, int | float):
            return None
        validation_token = row[1]
        return SharedReadCacheProjectionEntry(
            memory_id=memory_id,
            payload=payload,
            validation_token=validation_token if isinstance(validation_token, str) else None,
            cached_at=float(cached_at),
        )

    def _serialize_json(self, value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _projection_query_tokens(query: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", query.lower()):
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _projection_matches_request(
    payload: dict[str, Any],
    request: SharedReadCacheSearchRequest,
) -> bool:
    if request.workspace_id is not None:
        workspace_ids = payload.get("workspace_ids")
        if not isinstance(workspace_ids, list) or request.workspace_id not in workspace_ids:
            return False
    if request.memory_type is not None and payload.get("memory_type") != request.memory_type:
        return False
    status = payload.get("status")
    if request.status is not None and status != request.status:
        return False
    if not request.include_superseded and status == "superseded":
        return False
    return True


def _projection_hit_count(payload: dict[str, Any], query_tokens: list[str]) -> int:
    title = str(payload.get("title") or "").lower()
    summary = str(payload.get("summary") or "").lower()
    tags = [str(tag).lower() for tag in payload.get("tags", []) if isinstance(tag, str)]
    hit_count = 0
    for token in query_tokens:
        if token in title or token in summary or any(token in tag for tag in tags):
            hit_count += 1
    return hit_count


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return 0


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _bucket_start_minute(value: float) -> int:
    return int(value // 60) * 60

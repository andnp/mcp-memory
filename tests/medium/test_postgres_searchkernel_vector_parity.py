"""Optional Docker-backed parity benchmark for the two Postgres vector stores.

Run with::

    uv run pytest -s tests/medium/test_postgres_searchkernel_vector_parity.py

The benchmark is intentionally isolated from the normal Postgres fixture: the
current mcp-memory schema and searchkernel's per-model vector-table schema are
created in separate temporary schemas inside a pgvector container.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import socket
import time
from time import perf_counter
from typing import Any
from urllib.parse import quote
import uuid

import psycopg
from psycopg import sql
import pytest

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from searchkernel.domain import Record


pytestmark = pytest.mark.medium

_POSTGRES_USER = "postgres"
_POSTGRES_PASSWORD = "parity"
_POSTGRES_IMAGE = "pgvector/pgvector:pg17"
_MODEL_NAME = "parity-model"
_DIMENSION = 4


def _free_localhost_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_postgres(container: Any, *, timeout_seconds: int = 60) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        container.reload()
        if container.status in {"dead", "exited"}:
            raise RuntimeError("pgvector container exited before becoming ready")
        result = container.exec_run(
            ["pg_isready", "-h", "127.0.0.1", "-p", "5432", "-U", _POSTGRES_USER]
        )
        if result.exit_code == 0:
            return
        time.sleep(0.5)
    raise TimeoutError("pgvector container did not become ready within 60 seconds")


def _schema_dsn(base_dsn: str, schema: str) -> str:
    options = quote(f"-c search_path={schema},public", safe="")
    separator = "&" if "?" in base_dsn else "?"
    return f"{base_dsn}{separator}options={options}"


@pytest.fixture(scope="module")
def pgvector_base_dsn() -> Iterator[str]:
    """Start pgvector, or skip this optional benchmark when Docker is absent."""
    docker = pytest.importorskip("docker", reason="Docker SDK is required for this benchmark")
    client: Any | None = None
    container: Any | None = None
    try:
        try:
            docker_client: Any = docker.from_env()
            docker_client.ping()
        except Exception as exc:
            pytest.skip(f"Docker is unavailable for the Postgres parity benchmark: {exc}")
        client = docker_client
        assert client is not None

        try:
            image = client.images.get(_POSTGRES_IMAGE)
        except docker.errors.ImageNotFound:
            image = client.images.pull("pgvector/pgvector", "pg17")

        port = _free_localhost_port()
        container = client.containers.run(
            image,
            detach=True,
            environment={
                "POSTGRES_DB": "postgres",
                "POSTGRES_PASSWORD": _POSTGRES_PASSWORD,
                "POSTGRES_USER": _POSTGRES_USER,
            },
            name=f"mcp-memory-vector-parity-{uuid.uuid4().hex[:10]}",
            ports={"5432/tcp": port},
            tmpfs={"/var/lib/postgresql/data": "rw,size=512m"},
        )
        _wait_for_postgres(container)
        yield f"postgresql://{_POSTGRES_USER}:{_POSTGRES_PASSWORD}@127.0.0.1:{port}/postgres"
    except docker.errors.DockerException as exc:
        pytest.skip(f"Could not start pgvector container: {exc}")
    finally:
        if container is not None:
            try:
                container.remove(force=True)
            except Exception:
                pass
        if client is not None:
            client.close()


@dataclass
class _ParityBackends:
    mcp_manager: PostgresConnectionManager
    searchkernel_pool: Any


@pytest.fixture
def parity_backends(pgvector_base_dsn: str) -> Iterator[_ParityBackends]:
    try:
        from searchkernel.adapters.stores.pgvector import (
            PostgresConnection,
            _create_schema,
        )
    except ModuleNotFoundError as exc:
        pytest.skip(f"searchkernel pgvector dependencies are unavailable: {exc}")

    suffix = uuid.uuid4().hex[:10]
    mcp_schema = f"mcp_parity_{suffix}"
    searchkernel_schema = f"kernel_parity_{suffix}"
    with psycopg.connect(pgvector_base_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(mcp_schema))
            )
            cursor.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(searchkernel_schema))
            )

    mcp_config = PostgresStorageConfig(
        dsn=_schema_dsn(pgvector_base_dsn, mcp_schema),
        pool_min=1,
        pool_max=2,
    )
    try:
        searchkernel_pool = PostgresConnection(
            _schema_dsn(pgvector_base_dsn, searchkernel_schema),
            min_connections=1,
            max_connections=2,
        )
    except ImportError as exc:
        pytest.skip(f"searchkernel pgvector dependencies are unavailable: {exc}")
    _create_schema(searchkernel_pool)
    ensure_postgres_schema(mcp_config)

    try:
        with PostgresConnectionManager(mcp_config) as mcp_manager:
            yield _ParityBackends(mcp_manager, searchkernel_pool)
    finally:
        searchkernel_pool.close()


def _records() -> tuple[Record, ...]:
    timestamp = datetime(2026, 8, 2, 12, tzinfo=UTC)
    return (
        Record(
            workspace_id="workspace-a",
            source_kind="memory",
            source_id="auth-a",
            title="Authentication policy",
            body="Authentication policy and token rotation.",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
        Record(
            workspace_id="workspace-a",
            source_kind="memory",
            source_id="auth-b",
            title="Authentication migration",
            body="Authentication migration notes.",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[0.8, 0.2, 0.0, 0.0],
        ),
        Record(
            workspace_id="workspace-a",
            source_kind="note",
            source_id="auth-note",
            title="Authentication note",
            body="A note about authentication.",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[0.7, 0.3, 0.0, 0.0],
        ),
        Record(
            workspace_id="workspace-b",
            source_kind="memory",
            source_id="auth-other",
            title="Other workspace authentication",
            body="Authentication in another workspace.",
            created_at=timestamp,
            updated_at=timestamp,
            embedding=[1.0, 0.0, 0.0, 0.0],
        ),
    )


def _upsert_mcp(store: PostgresVectorStore, records: tuple[Record, ...]) -> None:
    for record in records:
        assert record.embedding is not None
        store.upsert(
            source_kind=record.source_kind,
            source_id=record.source_id,
            workspace_id=record.workspace_id,
            model_name=_MODEL_NAME,
            embedding=list(record.embedding),
        )


def test_postgres_vector_backend_capability_benchmark(
    parity_backends: _ParityBackends,
) -> None:
    """Compare filtering, bounded ANN diagnostics, identity, and search latency."""
    from searchkernel.adapters.stores.pgvector import PGVectorStore

    records = _records()
    mcp_store = PostgresVectorStore(parity_backends.mcp_manager)
    searchkernel_store = PGVectorStore(
        parity_backends.searchkernel_pool,
        hnsw_iterative_scan="off",
        hnsw_max_scan_tuples=3,
        overfetch_multiplier=2.0,
        max_scan_rounds=2,
    )
    _upsert_mcp(mcp_store, records)
    searchkernel_store.upsert(list(records), model_name=_MODEL_NAME, dim=_DIMENSION)

    query = [1.0, 0.05, 0.0, 0.0]
    candidate = records[0]
    assert candidate.embedding is not None

    # Warm capability/schema probes before timing the comparable search calls.
    mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=query,
        workspace_id="workspace-a",
        limit=1,
    )
    searchkernel_store.search(
        query,
        k=1,
        model_name=_MODEL_NAME,
        dim=_DIMENSION,
        filters={"workspace_id": "workspace-a", "source_kinds": ["memory"]},
    )

    mcp_diagnostics: dict[str, object] = {}
    mcp_started = perf_counter()
    mcp_candidate_hits = mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=query,
        candidate_ids=[candidate.source_id],
        diagnostics=mcp_diagnostics,
        workspace_id="workspace-a",
        limit=2,
    )
    mcp_latency_ms = (perf_counter() - mcp_started) * 1000.0

    searchkernel_started = perf_counter()
    searchkernel_candidate_hits = searchkernel_store.search(
        query,
        k=2,
        model_name=_MODEL_NAME,
        dim=_DIMENSION,
        filters={
            "candidate_ids": [candidate.storage_key],
            "workspace_id": "workspace-a",
            "source_kinds": ["memory"],
        },
    )
    searchkernel_latency_ms = (perf_counter() - searchkernel_started) * 1000.0
    searchkernel_diagnostics = searchkernel_store.last_search_diagnostics

    assert [source_id for source_id, _score in mcp_candidate_hits] == ["auth-a"]
    assert [hit.source_id for hit in searchkernel_candidate_hits] == ["auth-a"]
    assert all(hit.workspace_id == "workspace-a" for hit in searchkernel_candidate_hits)
    assert mcp_diagnostics["search_mode"] == "server_side_pgvector"
    assert mcp_diagnostics["candidate_filter_count"] == 1
    assert mcp_diagnostics["query_dimension"] == _DIMENSION
    assert mcp_diagnostics["dimension_filter_applied"] is True
    assert mcp_diagnostics["row_count"] == 1
    assert "scan_limit" not in mcp_diagnostics
    assert searchkernel_diagnostics["requested_k"] == 2
    assert searchkernel_diagnostics["scan_rounds"] <= 2
    assert searchkernel_diagnostics["scan_limit"] <= 3
    assert searchkernel_diagnostics["scan_bound_hit"] is True
    assert searchkernel_diagnostics["under_returned"] is True

    mcp_workspace_hits = mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=query,
        workspace_id="workspace-a",
        limit=10,
    )
    searchkernel_workspace_hits = searchkernel_store.search(
        query,
        k=10,
        model_name=_MODEL_NAME,
        dim=_DIMENSION,
        filters={"workspace_id": "workspace-a", "source_kinds": ["memory"]},
    )
    assert {source_id for source_id, _score in mcp_workspace_hits} == {
        "auth-a",
        "auth-b",
    }
    assert {hit.source_id for hit in searchkernel_workspace_hits} == {
        "auth-a",
        "auth-b",
    }
    assert all(hit.workspace_id == "workspace-a" for hit in searchkernel_workspace_hits)

    model_three = Record(
        workspace_id="workspace-a",
        source_kind="memory",
        source_id="model-three",
        title="Three-dimensional model",
        body="Model and dimension isolation fixture.",
        created_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        updated_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        embedding=[1.0, 0.0, 0.0],
    )
    assert model_three.embedding is not None
    mcp_store.upsert(
        source_kind=model_three.source_kind,
        source_id=model_three.source_id,
        workspace_id=model_three.workspace_id,
        model_name="model-three",
        embedding=list(model_three.embedding),
    )
    searchkernel_store.upsert([model_three], model_name="model-three", dim=3)
    mcp_three_hits = mcp_store.search(
        source_kind="memory",
        model_name="model-three",
        query_embedding=[1.0, 0.0, 0.0],
        workspace_id="workspace-a",
        limit=10,
    )
    searchkernel_three_hits = searchkernel_store.search(
        [1.0, 0.0, 0.0],
        k=10,
        model_name="model-three",
        dim=3,
        filters={"workspace_id": "workspace-a", "source_kinds": ["memory"]},
    )
    assert [source_id for source_id, _score in mcp_three_hits] == ["model-three"]
    assert [hit.source_id for hit in searchkernel_three_hits] == ["model-three"]
    assert mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=[1.0, 0.0],
        workspace_id="workspace-a",
        limit=10,
    ) == []
    assert searchkernel_store.search(
        [1.0, 0.0],
        k=10,
        model_name=_MODEL_NAME,
        dim=2,
        filters={"workspace_id": "workspace-a", "source_kinds": ["memory"]},
    ) == []

    collision_a = Record(
        workspace_id="workspace-a",
        source_kind="memory",
        source_id="collision",
        title="Collision A",
        body="Same source ID in workspace A.",
        created_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        updated_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        embedding=[0.95, 0.05, 0.0, 0.0],
    )
    collision_b = Record(
        workspace_id="workspace-b",
        source_kind="memory",
        source_id="collision",
        title="Collision B",
        body="Same source ID in workspace B.",
        created_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        updated_at=datetime(2026, 8, 2, 12, tzinfo=UTC),
        embedding=[0.95, 0.05, 0.0, 0.0],
    )
    _upsert_mcp(mcp_store, (collision_a, collision_b))
    searchkernel_store.upsert(
        [collision_a, collision_b], model_name=_MODEL_NAME, dim=_DIMENSION
    )
    mcp_collision_a = mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=query,
        workspace_id="workspace-a",
        candidate_ids=["collision"],
        limit=2,
    )
    mcp_collision_b = mcp_store.search(
        source_kind="memory",
        model_name=_MODEL_NAME,
        query_embedding=query,
        workspace_id="workspace-b",
        candidate_ids=["collision"],
        limit=2,
    )
    searchkernel_collision_a = searchkernel_store.search(
        query,
        k=2,
        model_name=_MODEL_NAME,
        dim=_DIMENSION,
        filters={"workspace_id": "workspace-a", "candidate_ids": [collision_a.storage_key]},
    )
    searchkernel_collision_b = searchkernel_store.search(
        query,
        k=2,
        model_name=_MODEL_NAME,
        dim=_DIMENSION,
        filters={"workspace_id": "workspace-b", "candidate_ids": [collision_b.storage_key]},
    )
    assert mcp_collision_a == []
    assert [source_id for source_id, _score in mcp_collision_b] == ["collision"]
    assert [hit.storage_key for hit in searchkernel_collision_a] == [collision_a.storage_key]
    assert [hit.storage_key for hit in searchkernel_collision_b] == [collision_b.storage_key]

    report = {
        "mcp_memory": {
            "candidate_hits": [source_id for source_id, _score in mcp_candidate_hits],
            "diagnostics": mcp_diagnostics,
            "latency_ms": round(mcp_latency_ms, 3),
            "identity_collision": {
                "workspace_a": [source_id for source_id, _score in mcp_collision_a],
                "workspace_b": [source_id for source_id, _score in mcp_collision_b],
            },
        },
        "searchkernel": {
            "candidate_hits": [hit.storage_key for hit in searchkernel_candidate_hits],
            "diagnostics": searchkernel_diagnostics,
            "latency_ms": round(searchkernel_latency_ms, 3),
            "identity_collision": {
                "workspace_a": [hit.storage_key for hit in searchkernel_collision_a],
                "workspace_b": [hit.storage_key for hit in searchkernel_collision_b],
            },
        },
        "comparison": {
            "model_dimension_filtering": "both backends isolate model and dimension",
            "candidate_bound": "searchkernel bounded scan diagnostics; mcp-memory direct LIMIT",
            "identity": "searchkernel preserves workspace in canonical identity; mcp-memory keys by source kind/id/model",
        },
    }
    print("POSTGRES_VECTOR_PARITY " + json.dumps(report, sort_keys=True))

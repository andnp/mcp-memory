from __future__ import annotations

from collections.abc import Generator
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass

import pytest

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.postgres_connection import load_postgres_driver_modules


POSTGRES_IMAGE = os.getenv("MCP_MEMORY_TEST_POSTGRES_IMAGE", "postgres")
POSTGRES_TAG = os.getenv("MCP_MEMORY_TEST_POSTGRES_TAG", "17")
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "password"


def _get_free_localhost_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _sanitize_node_id(node_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", node_id)
    return sanitized[:48].strip("_") or "test"


def _wait_for_postgres_ready(container, *, timeout_seconds: int = 60, check_interval: float = 1.0) -> None:
    started_at = time.time()
    while time.time() - started_at < timeout_seconds:
        try:
            container.reload()
            if container.status == "running":
                result = container.exec_run(f"pg_isready -h localhost -p 5432 -U {POSTGRES_USER}")
                if getattr(result, "exit_code", 1) == 0:
                    return
        except Exception:
            pass
        time.sleep(check_interval)
    container_id = container.id[:12] if getattr(container, "id", None) else "unknown"
    raise TimeoutError(f"Postgres container {container_id} did not become ready within {timeout_seconds} seconds")


def _wait_for_database_connection(dsn: str, *, retries: int = 15, base_delay: float = 0.2) -> None:
    psycopg, _ = load_postgres_driver_modules()
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with psycopg.connect(dsn, autocommit=True) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
            return
        except Exception as exc:  # pragma: no cover - integration retry path
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2**attempt))
    if last_error is not None:  # pragma: no cover - defensive
        raise last_error


@dataclass(frozen=True)
class PostgresContainerInfo:
    host: str
    port: int

    @property
    def admin_dsn(self) -> str:
        return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{self.host}:{self.port}/postgres"


@pytest.fixture(scope="session")
def postgres_container() -> Generator[PostgresContainerInfo, None, None]:
    docker = pytest.importorskip("docker")
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:  # pragma: no cover - depends on local docker availability
        pytest.skip(f"Docker daemon unavailable for Postgres integration tests: {exc}")

    port = _get_free_localhost_port()
    name = f"mcp-memory-test-postgres-{uuid.uuid4().hex[:12]}"

    try:
        image = client.images.get(f"{POSTGRES_IMAGE}:{POSTGRES_TAG}")
    except docker.errors.ImageNotFound:
        image = client.images.pull(POSTGRES_IMAGE, POSTGRES_TAG)

    container = client.containers.run(
        image,
        detach=True,
        ports={"5432/tcp": port},
        environment={
            "POSTGRES_USER": POSTGRES_USER,
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
        },
        name=name,
    )

    try:
        _wait_for_postgres_ready(container)
        yield PostgresContainerInfo(host="127.0.0.1", port=port)
    finally:
        try:
            container.stop(timeout=5)
        except Exception:
            pass
        try:
            container.remove(v=True, force=True)
        except Exception:
            pass


@pytest.fixture
def postgres_test_db_name(request: pytest.FixtureRequest) -> str:
    timestamp = int(time.time() * 1000)
    unique_id = uuid.uuid4().hex[:12]
    return f"test_{_sanitize_node_id(request.node.nodeid)}_{unique_id}_{timestamp}"


@pytest.fixture
def postgres_storage_config(
    postgres_container: PostgresContainerInfo,
    postgres_test_db_name: str,
) -> Generator[PostgresStorageConfig, None, None]:
    psycopg, _ = load_postgres_driver_modules()
    admin_dsn = postgres_container.admin_dsn
    test_dsn = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{postgres_container.host}:{postgres_container.port}/{postgres_test_db_name}"

    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE DATABASE "{postgres_test_db_name}"')

    _wait_for_database_connection(test_dsn)

    try:
        yield PostgresStorageConfig(dsn=test_dsn)
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                    (postgres_test_db_name,),
                )
                cursor.execute(f'DROP DATABASE IF EXISTS "{postgres_test_db_name}"')
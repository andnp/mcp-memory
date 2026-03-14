import asyncio
import json
from pathlib import Path

import pytest

from mcp_memory.core.journal import System1Journal
from mcp_memory.utils.db import DatabaseManager
from tests.sdk.mcp import FakeToolRuntime
from tests.sdk.providers import FakeAIProvider, FakeSubprocessInstaller


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "small: fast, isolated tests with no network or model loading",
    )
    config.addinivalue_line(
        "markers",
        "medium: in-process integration tests using real local components",
    )
    config.addinivalue_line(
        "markers",
        "large: end-to-end tests that may start full processes",
    )


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    return tmp_path / ".memories"


@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory.db"


@pytest.fixture
def db_manager(temp_db_path: Path) -> DatabaseManager:
    manager = DatabaseManager(temp_db_path)
    try:
        yield manager
    finally:
        manager.close()


@pytest.fixture
def system1_journal(db_manager: DatabaseManager) -> System1Journal:
    return System1Journal(db_manager)


@pytest.fixture
def fake_ai_provider() -> FakeAIProvider:
    return FakeAIProvider()


@pytest.fixture
def seed_data() -> dict:
    seed_file = Path(__file__).parent / "fixtures" / "seed_data.json"
    return json.loads(seed_file.read_text(encoding="utf-8"))


@pytest.fixture
def seeded_db(db_manager: DatabaseManager, seed_data: dict) -> DatabaseManager:
    conn = db_manager.get_connection()

    for document in seed_data["documents"]:
        conn.execute(
            "INSERT INTO documents (doc_id, file_path, content_hash, mtime, status, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                document["doc_id"],
                document["file_path"],
                document["content_hash"],
                document["mtime"],
                document["status"],
                document["indexed_at"],
            ),
        )

    for chunk in seed_data["chunks"]:
        conn.execute(
            "INSERT INTO chunks (chunk_id, doc_id, content, metadata, vector, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                chunk["chunk_id"],
                chunk["doc_id"],
                chunk["content"],
                json.dumps(chunk["metadata"]),
                None,
                1710000800.0,
            ),
        )

    for entry in seed_data["journal_entries"]:
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp, status) VALUES (?, ?, ?)",
            (entry["content"], entry["timestamp"], entry["status"]),
        )

    for state in seed_data["system_state"]:
        conn.execute(
            "INSERT INTO system_state (key, value) VALUES (?, ?)",
            (state["key"], state["value"]),
        )

    conn.commit()
    return db_manager


@pytest.fixture
def fake_tool_runtime() -> FakeToolRuntime:
    return FakeToolRuntime()


@pytest.fixture
def install_fake_subprocess(monkeypatch) -> FakeSubprocessInstaller:
    installer = FakeSubprocessInstaller()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", installer)
    return installer
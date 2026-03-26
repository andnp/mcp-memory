import asyncio
import json
import os
from collections.abc import Generator
from pathlib import Path

import pytest

from mcp_memory.core.journal import System1Journal
from mcp_memory.utils.db import DatabaseManager
from tests.sdk.mcp import FakeToolRuntime
from tests.sdk.providers import FakeAIProvider, FakeSubprocessInstaller


def pytest_configure(config) -> None:
    os.environ["MCP_MEMORY_TEST_MODE"] = "1"
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
def db_manager(temp_db_path: Path) -> Generator[DatabaseManager, None, None]:
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

    for entry in seed_data["journal_entries"]:
        conn.execute(
            "INSERT INTO system1_journal (content, timestamp, status) VALUES (?, ?, ?)",
            (entry["content"], entry["timestamp"], entry["status"]),
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

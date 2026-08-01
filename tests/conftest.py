import asyncio
import json
import os
from collections.abc import Generator
from pathlib import Path

import pytest
import tomlkit

from mcp_memory.config import ensure_default_config_exists, resolve_default_config_path
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


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home_dir = tmp_path / "home"
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    home_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_dir))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_dir))
    monkeypatch.setenv("XDG_STATE_HOME", str(state_dir))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    config_path = ensure_default_config_exists(resolve_default_config_path())
    document = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    daemon_table = document.setdefault("daemon", tomlkit.table())
    daemon_table["port"] = 0
    config_path.write_text(tomlkit.dumps(document), encoding="utf-8")


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

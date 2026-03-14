from pathlib import Path

import pytest

from mcp_memory.core.journal import System1Journal
from mcp_memory.utils.db import DatabaseManager


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
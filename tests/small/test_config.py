from pathlib import Path

import pytest

from mcp_memory.config import MemoryConfig, ProjectConfig


pytestmark = pytest.mark.small


def test_project_config_requires_absolute_path() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        ProjectConfig(name="demo", path="relative/path")


def test_project_config_normalizes_absolute_path(tmp_path: Path) -> None:
    project = ProjectConfig(name="demo_project", path=str(tmp_path))

    assert project.path == str(tmp_path.resolve())


def test_memory_config_rejects_invalid_storage_strategy() -> None:
    with pytest.raises(ValueError, match="storage_strategy"):
        MemoryConfig(storage_strategy="elsewhere")
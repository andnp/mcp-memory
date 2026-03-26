from __future__ import annotations

from pathlib import Path

import pytest

from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository


@pytest.mark.medium
def test_create_runtime_initializes_task_execution_attempt_repository(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    runtime = create_runtime(workspace_root_override=None, cwd=tmp_path / "workspace")
    try:
        assert runtime.task_execution_attempts is not None
        assert isinstance(runtime.task_execution_attempts, TaskExecutionAttemptRepository)

        attempt = runtime.task_execution_attempts.start_attempt(
            task_id="runtime-task",
            execution_epoch=1,
            task_name="memory-curator",
            request_id="runtime-req",
            subprocess_pid=123,
            provider_key="test-provider",
            provider_name="Test Provider",
            model_name="test-model",
            started_at=5.0,
        )

        assert attempt.workspace_id == runtime.workspace_id
        assert runtime.task_execution_attempts.get_attempt(task_id="runtime-task", execution_epoch=1).task_id == "runtime-task"
    finally:
        runtime.close()

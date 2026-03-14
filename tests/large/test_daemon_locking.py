from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from mcp_memory.daemon import DaemonLifecycleController, DaemonLockTimeoutError


pytestmark = pytest.mark.large


def _hold_runtime(
    project_path: str,
    home_path: str,
    data_home_path: str,
    ready_queue,
    release_event,
):
    os.environ["HOME"] = home_path
    os.environ["XDG_DATA_HOME"] = data_home_path

    async def runner():
        controller = DaemonLifecycleController(
            shutdown_grace_seconds=30.0,
            lock_timeout_seconds=2.0,
        )
        runtime = await controller.acquire_runtime(
            project_override=project_path,
            cwd=Path(project_path),
        )
        ready_queue.put("ready")
        await asyncio.to_thread(release_event.wait)
        await controller.release_runtime(runtime)
        await controller.shutdown_now()

    asyncio.run(runner())


def test_controller_times_out_when_another_process_holds_lock(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    home_path = tmp_path / "home"
    data_home_path = tmp_path / "data"
    home_path.mkdir()
    data_home_path.mkdir()

    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    release_event = ctx.Event()
    process = ctx.Process(
        target=_hold_runtime,
        args=(
            str(workspace),
            str(home_path),
            str(data_home_path),
            ready_queue,
            release_event,
        ),
    )

    original_home = os.environ.get("HOME")
    original_data_home = os.environ.get("XDG_DATA_HOME")

    process.start()
    try:
        assert ready_queue.get(timeout=10) == "ready"

        os.environ["HOME"] = str(home_path)
        os.environ["XDG_DATA_HOME"] = str(data_home_path)

        async def attempt_acquire():
            controller = DaemonLifecycleController(
                shutdown_grace_seconds=0.01,
                lock_timeout_seconds=0.2,
            )
            with pytest.raises(DaemonLockTimeoutError):
                await controller.acquire_runtime(
                    project_override=str(workspace),
                    cwd=workspace,
                )

        asyncio.run(attempt_acquire())
    finally:
        release_event.set()
        process.join(timeout=10)
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home

        if original_data_home is None:
            os.environ.pop("XDG_DATA_HOME", None)
        else:
            os.environ["XDG_DATA_HOME"] = original_data_home

    assert process.exitcode == 0
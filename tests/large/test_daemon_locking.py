from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from mcp_memory.daemon import DaemonLockTimeoutError, FilesystemLock

pytestmark = pytest.mark.large


def _hold_lock(lock_path: str, ready_queue, release_event) -> None:
    lock = FilesystemLock(Path(lock_path))
    lock.acquire(timeout_seconds=2.0)
    ready_queue.put("ready")
    release_event.wait()
    lock.release()


def test_filesystem_lock_times_out_when_another_process_holds_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "daemon.lock"
    ctx = mp.get_context("spawn")
    ready_queue = ctx.Queue()
    release_event = ctx.Event()
    process = ctx.Process(
        target=_hold_lock,
        args=(str(lock_path), ready_queue, release_event),
    )

    process.start()
    try:
        assert ready_queue.get(timeout=10) == "ready"
        contender = FilesystemLock(lock_path)
        with pytest.raises(DaemonLockTimeoutError):
            contender.acquire(timeout_seconds=0.2)
    finally:
        release_event.set()
        process.join(timeout=10)

    assert process.exitcode == 0

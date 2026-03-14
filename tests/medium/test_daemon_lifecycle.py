from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mcp_memory.daemon import DaemonLifecycleController
from mcp_memory.mcp.runtime import RuntimeSpec


pytestmark = pytest.mark.medium


class FakeRuntime:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class FakeWorker:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls: list[float] = []

    async def start(self) -> None:
        self.start_calls += 1

    async def stop(self, grace_period_seconds: float) -> None:
        self.stop_calls.append(grace_period_seconds)


def _make_spec(memory_path: Path):
    return RuntimeSpec(config=None, project_name="demo", memory_path=memory_path)


@pytest.mark.asyncio
async def test_controller_reuses_runtime_and_refcounts_clients(tmp_path: Path):
    created: list[FakeRuntime] = []
    spec = _make_spec(tmp_path / "memories")

    def resolve_spec(project_override: str | None, cwd: Path | None):
        return spec

    def build_runtime(runtime_spec: RuntimeSpec):
        runtime = FakeRuntime()
        created.append(runtime)
        return runtime

    controller = DaemonLifecycleController(
        shutdown_grace_seconds=0.05,
        runtime_spec_resolver=resolve_spec,
        runtime_factory=build_runtime,
        worker_factory=lambda runtime: None,
    )

    runtime_one = await controller.acquire_runtime()
    runtime_two = await controller.acquire_runtime()

    assert runtime_one is runtime_two
    assert controller.client_count == 2
    assert len(created) == 1

    await controller.release_runtime(runtime_one)
    await asyncio.sleep(0.08)

    assert created[0].close_calls == 0
    assert controller.has_runtime is True

    await controller.release_runtime(runtime_two)
    await asyncio.sleep(0.08)

    assert created[0].close_calls == 1
    assert controller.has_runtime is False


@pytest.mark.asyncio
async def test_controller_cancels_pending_shutdown_when_client_reconnects(tmp_path: Path):
    created: list[FakeRuntime] = []
    spec = _make_spec(tmp_path / "memories")

    def resolve_spec(project_override: str | None, cwd: Path | None):
        return spec

    def build_runtime(runtime_spec: RuntimeSpec):
        runtime = FakeRuntime()
        created.append(runtime)
        return runtime

    controller = DaemonLifecycleController(
        shutdown_grace_seconds=0.15,
        runtime_spec_resolver=resolve_spec,
        runtime_factory=build_runtime,
        worker_factory=lambda runtime: None,
    )

    runtime_one = await controller.acquire_runtime()
    await controller.release_runtime(runtime_one)
    await asyncio.sleep(0.05)

    runtime_two = await controller.acquire_runtime()
    await asyncio.sleep(0.15)

    assert runtime_two is runtime_one
    assert created[0].close_calls == 0
    assert controller.client_count == 1

    await controller.release_runtime(runtime_two)
    await asyncio.sleep(0.18)

    assert created[0].close_calls == 1


@pytest.mark.asyncio
async def test_controller_starts_and_stops_runtime_worker(tmp_path: Path):
    created_workers: list[FakeWorker] = []
    spec = _make_spec(tmp_path / "memories")

    def resolve_spec(project_override: str | None, cwd: Path | None):
        return spec

    def build_runtime(runtime_spec: RuntimeSpec):
        return FakeRuntime()

    def build_worker(runtime):
        worker = FakeWorker()
        created_workers.append(worker)
        return worker

    controller = DaemonLifecycleController(
        shutdown_grace_seconds=0.05,
        runtime_spec_resolver=resolve_spec,
        runtime_factory=build_runtime,
        worker_factory=build_worker,
    )

    runtime = await controller.acquire_runtime()
    await controller.release_runtime(runtime)
    await asyncio.sleep(0.08)

    assert len(created_workers) == 1
    assert created_workers[0].start_calls == 1
    assert created_workers[0].stop_calls == [0.05]
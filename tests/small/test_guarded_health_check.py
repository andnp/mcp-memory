from __future__ import annotations

import pytest

from mcp_memory.core.ports.embedding_maintenance import EmbeddingDatabaseHealthError
from mcp_memory.storage.guarded_health_check import GuardedHealthCheck

pytestmark = pytest.mark.small


class _DriverError(Exception):
    pass


def test_run_integrity_check_runs_probe_then_operation() -> None:
    calls: list[str] = []
    guard = GuardedHealthCheck(
        exception_type=_DriverError,
        reopen=lambda: calls.append("reopen"),
        probe=lambda: calls.append("probe"),
    )

    guard.run_integrity_check(lambda: calls.append("operation"))

    assert calls == ["probe", "operation"]


def test_run_integrity_check_wraps_driver_error_as_domain_error() -> None:
    guard = GuardedHealthCheck(exception_type=_DriverError, reopen=lambda: None)

    def _fail() -> None:
        raise _DriverError("boom")

    with pytest.raises(EmbeddingDatabaseHealthError, match="boom"):
        guard.run_integrity_check(_fail)


def test_run_integrity_check_propagates_probe_domain_error_unwrapped() -> None:
    def _probe() -> None:
        raise EmbeddingDatabaseHealthError("probe failed")

    guard = GuardedHealthCheck(exception_type=_DriverError, reopen=lambda: None, probe=_probe)

    with pytest.raises(EmbeddingDatabaseHealthError, match="probe failed"):
        guard.run_integrity_check(lambda: None)


def test_retry_after_reopen_recovers_on_success() -> None:
    calls: list[str] = []
    guard = GuardedHealthCheck(
        exception_type=_DriverError,
        reopen=lambda: calls.append("reopen"),
    )

    recovered = guard.retry_after_reopen(lambda: calls.append("operation"))

    assert recovered is True
    assert calls == ["reopen", "operation"]


def test_retry_after_reopen_returns_false_when_operation_still_fails() -> None:
    def _fail() -> None:
        raise _DriverError("still broken")

    guard = GuardedHealthCheck(exception_type=_DriverError, reopen=lambda: None)

    recovered = guard.retry_after_reopen(_fail)

    assert recovered is False


def test_retry_after_reopen_returns_false_when_reopen_raises_os_error() -> None:
    def _reopen() -> None:
        raise OSError("cannot reopen")

    guard = GuardedHealthCheck(exception_type=_DriverError, reopen=_reopen)

    recovered = guard.retry_after_reopen(lambda: None)

    assert recovered is False

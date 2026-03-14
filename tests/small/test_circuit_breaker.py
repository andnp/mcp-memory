import pytest

from mcp_memory.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)


pytestmark = pytest.mark.small


def test_circuit_breaker_opens_after_failure_threshold() -> None:
    breaker = CircuitBreaker(
        failure_threshold=2,
        recovery_timeout=60.0,
        success_threshold=1,
        window_duration=60.0,
    )

    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom-1")))
    with pytest.raises(ValueError):
        breaker.call(lambda: (_ for _ in ()).throw(ValueError("boom-2")))

    assert breaker.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        breaker.call(lambda: "nope")


def test_circuit_breaker_recovers_after_timeout_and_successes(monkeypatch) -> None:
    breaker = CircuitBreaker(
        failure_threshold=1,
        recovery_timeout=5.0,
        success_threshold=2,
        window_duration=60.0,
    )

    current_time = {"value": 100.0}
    monkeypatch.setattr("mcp_memory.utils.circuit_breaker.time.time", lambda: current_time["value"])

    with pytest.raises(RuntimeError):
        breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    assert breaker.state == CircuitState.OPEN

    current_time["value"] = 106.0
    assert breaker.call(lambda: "first-success") == "first-success"
    assert breaker.state == CircuitState.HALF_OPEN

    assert breaker.call(lambda: "second-success") == "second-success"
    assert breaker.state == CircuitState.CLOSED
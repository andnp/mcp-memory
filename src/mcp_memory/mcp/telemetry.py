from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
from logging import getLogger
from threading import Lock
from time import time
from uuid import uuid4

from mcp_memory.application.ports import MemoryReadContext
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository

_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0
_DEFAULT_SEARCH_DIAGNOSTICS_SAMPLE_RATE = 0.1
_MAX_DIAGNOSTIC_LABELS = 32
_MAX_DIAGNOSTIC_LABEL_LENGTH = 48
_MAX_COUNTER_VALUE = 2_147_483_647
logger = getLogger(__name__)


class SearchDiagnosticsSampler:
    """Keep a deterministic, bounded aggregate of structured search diagnostics."""

    def __init__(
        self,
        *,
        sample_rate: float | None = None,
        max_labels: int = _MAX_DIAGNOSTIC_LABELS,
    ) -> None:
        configured_rate = _configured_search_diagnostics_sample_rate()
        self.sample_rate = (
            configured_rate if sample_rate is None else float(sample_rate)
        )
        if not 0.0 <= self.sample_rate <= 1.0:
            raise ValueError("search diagnostics sample_rate must be in [0.0, 1.0]")
        if max_labels < 1:
            raise ValueError("search diagnostics max_labels must be positive")
        self._max_labels = max_labels
        self._lock = Lock()
        self._observed_searches = 0
        self._sampled_searches = 0
        self._serialized_diagnostics = 0
        self._degraded_count = 0
        self._planner_decisions: Counter[str] = Counter()
        self._cache_status: Counter[str] = Counter()
        self._failure_stages: Counter[str] = Counter()
        self._planner_available = False
        self._cache_available = False
        self._failure_stages_available = False
        self._diagnostic_serialization_failures = 0

    def observe(self, invocation_id: str, diagnostics: object) -> None:
        with self._lock:
            self._observed_searches = _bounded_increment(self._observed_searches)
        if not self.should_sample(invocation_id):
            return
        with self._lock:
            self._sampled_searches = _bounded_increment(self._sampled_searches)
        try:
            payload = _diagnostic_payload(diagnostics)
        except Exception:
            with self._lock:
                self._diagnostic_serialization_failures = _bounded_increment(
                    self._diagnostic_serialization_failures
                )
            return

        with self._lock:
            self._serialized_diagnostics = _bounded_increment(self._serialized_diagnostics)
            degraded = payload.get("degraded")
            if isinstance(degraded, bool):
                self._degraded_count += int(degraded)

            planner_decisions = payload.get("lane_decisions")
            if isinstance(planner_decisions, Mapping):
                self._planner_available = True
                self._count_labels(
                    self._planner_decisions,
                    "enabled",
                    planner_decisions.get("enabled"),
                )
                self._count_labels(
                    self._planner_decisions,
                    "skipped",
                    planner_decisions.get("skipped"),
                )

            cache_diagnostics = payload.get("cache_diagnostics")
            if isinstance(cache_diagnostics, Sequence) and not isinstance(
                cache_diagnostics, (str, bytes)
            ):
                self._cache_available = True
                for diagnostic in cache_diagnostics:
                    if isinstance(diagnostic, str):
                        self._increment_label(
                            self._cache_status, _cache_status_label(diagnostic)
                        )

            failures = payload.get("failures")
            if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
                self._failure_stages_available = True
                for failure in failures:
                    if isinstance(failure, Mapping) and isinstance(failure.get("stage"), str):
                        self._increment_label(
                            self._failure_stages,
                            _safe_diagnostic_label(failure["stage"]),
                        )

    def should_sample(self, invocation_id: str) -> bool:
        if self.sample_rate <= 0.0:
            return False
        if self.sample_rate >= 1.0:
            return True
        digest = int.from_bytes(sha256(invocation_id.encode("utf-8")).digest()[:8], "big")
        return digest / 2**64 < self.sample_rate

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            observed = self._observed_searches
            sampled = self._sampled_searches
            serialized = self._serialized_diagnostics
            degraded = self._degraded_count
            return {
                "available": True,
                "sample_rate": self.sample_rate,
                "observed_searches": observed,
                "sampled_searches": sampled,
                "serialized_diagnostics": serialized,
                "sampled_rate": _safe_rate(sampled, observed),
                "degraded_count": degraded if serialized else None,
                "degraded_rate": _safe_rate(degraded, serialized),
                "planner_decisions": (
                    dict(self._planner_decisions) if self._planner_available else None
                ),
                "cache_status": dict(self._cache_status) if self._cache_available else None,
                "failure_stages": (
                    dict(self._failure_stages)
                    if self._failure_stages_available
                    else None
                ),
                "diagnostic_serialization_failures": self._diagnostic_serialization_failures,
            }

    def _count_labels(
        self,
        counter: Counter[str],
        prefix: str,
        values: object,
    ) -> None:
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return
        for value in values:
            if isinstance(value, str):
                self._increment_label(counter, f"{prefix}:{_safe_diagnostic_label(value)}")

    def _increment_label(self, counter: Counter[str], label: str) -> None:
        if label not in counter and len(counter) >= self._max_labels:
            label = "other"
        counter[label] = min(counter[label] + 1, _MAX_COUNTER_VALUE)


def _configured_search_diagnostics_sample_rate() -> float:
    raw_rate = os.environ.get("MCP_MEMORY_SEARCH_DIAGNOSTICS_SAMPLE_RATE")
    if raw_rate is None:
        return _DEFAULT_SEARCH_DIAGNOSTICS_SAMPLE_RATE
    try:
        return float(raw_rate)
    except ValueError:
        logger.warning("Invalid search diagnostics sample rate; using default")
        return _DEFAULT_SEARCH_DIAGNOSTICS_SAMPLE_RATE


def _diagnostic_payload(diagnostics: object) -> Mapping[str, object]:
    if isinstance(diagnostics, Mapping):
        return diagnostics
    to_payload = getattr(diagnostics, "to_payload", None)
    if callable(to_payload):
        payload = to_payload()
    else:
        model_dump = getattr(diagnostics, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("search diagnostics are not serializable")
        payload = model_dump()
    if not isinstance(payload, Mapping):
        raise TypeError("search diagnostics payload is not a mapping")
    return payload


def _safe_diagnostic_label(value: str) -> str:
    normalized = "_".join(value.strip().lower().split())
    return normalized[:_MAX_DIAGNOSTIC_LABEL_LENGTH] or "unknown"


def _cache_status_label(value: str) -> str:
    normalized = value.lower()
    for label in ("fresh", "stale", "projection", "hit", "miss", "fallback"):
        if label in normalized:
            return label
    return "other"


def _bounded_increment(value: int) -> int:
    return min(value + 1, _MAX_COUNTER_VALUE)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def get_search_diagnostics_sampler(
    telemetry: object,
    *,
    sample_rate: float | None = None,
) -> SearchDiagnosticsSampler:
    sampler = getattr(telemetry, "_search_diagnostics_sampler", None)
    if isinstance(sampler, SearchDiagnosticsSampler):
        return sampler
    sampler = SearchDiagnosticsSampler(sample_rate=sample_rate)
    try:
        setattr(telemetry, "_search_diagnostics_sampler", sampler)
    except (AttributeError, TypeError):
        # Runtime test doubles with slots cannot retain auxiliary state.
        return sampler
    return sampler


def record_search_diagnostics(
    telemetry: object,
    *,
    invocation_id: str,
    diagnostics: object,
) -> None:
    get_search_diagnostics_sampler(telemetry).observe(invocation_id, diagnostics)


def search_diagnostics_snapshot(telemetry: object | None) -> dict[str, object]:
    if telemetry is None:
        return {"available": False}
    return get_search_diagnostics_sampler(telemetry).snapshot()

def _retrieval_telemetry_repository(
    ctx: MemoryReadContext,
) -> RetrievalTelemetryRepository:
    repository = ctx.retrieval_telemetry
    if repository is None:
        repository = RetrievalTelemetryRepository(
            ctx.db_manager,
            workspace_id=ctx.workspace_id,
            storage_backend=ctx.storage_backend,
        )
        ctx.retrieval_telemetry = repository
    return repository


def _log_slow_memory_tool_operation(
    ctx: MemoryReadContext,
    *,
    tool_name: str,
    duration_ms: float,
    data: dict[str, object],
) -> None:
    if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
        return
    runtime_logs = getattr(ctx, "runtime_logs", None)
    if runtime_logs is not None:
        try:
            runtime_logs.write_log(
                source="memory-tool",
                logger_name=__name__,
                level="WARNING",
                message=f"Slow {tool_name} operation",
                created_at=time(),
                data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)}
                | data,
            )
            return
        except (
            Exception
        ):  # pragma: no cover - defensive fallback for locked telemetry/log stores
            logger.warning(
                "Failed to persist slow %s log; falling back to process logger",
                tool_name,
                exc_info=True,
            )
    logger.warning("Slow %s operation: %.3fms %s", tool_name, duration_ms, data)


def _record_search_invocation(
    ctx: MemoryReadContext,
    *,
    caller_kind: str,
    query: str,
    surfaced_memory_ids: list[str],
    graph_provenance: Mapping[str, object] | None = None,
    duration_ms: float,
    diagnostics: object | None = None,
) -> None:
    repository = _retrieval_telemetry_repository(ctx)
    invocation_id = str(uuid4())
    repository.record_search(
        invocation_id=invocation_id,
        caller_kind=caller_kind,
        query=query,
        surfaced_memory_ids=surfaced_memory_ids,
        graph_provenance=graph_provenance,
        duration_ms=duration_ms,
    )
    if diagnostics is not None:
        record_search_diagnostics(
            repository,
            invocation_id=invocation_id,
            diagnostics=diagnostics,
        )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="search_memory_records",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "query": query,
            "result_count": len(surfaced_memory_ids),
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )


def _record_read_invocation(
    ctx: MemoryReadContext,
    *,
    caller_kind: str,
    memory_id: str,
    duration_ms: float,
) -> None:
    repository = _retrieval_telemetry_repository(ctx)
    repository.record_read(
        invocation_id=str(uuid4()),
        caller_kind=caller_kind,
        memory_id=memory_id,
        duration_ms=duration_ms,
    )
    try:
        repository.flush()
    except Exception:
        logger.debug(
            "Read telemetry flush failed; continuing without blocking the read response",
            exc_info=True,
        )
    _log_slow_memory_tool_operation(
        ctx,
        tool_name="read_memory_record",
        duration_ms=duration_ms,
        data={
            "caller_kind": caller_kind,
            "memory_id": memory_id,
            "storage_backend": ctx.storage_backend or "sqlite",
        },
    )


class McpRetrievalTelemetryAdapter:
    def record_search(
        self,
        ctx: MemoryReadContext,
        *,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        graph_provenance: Mapping[str, object] | None = None,
        duration_ms: float,
        diagnostics: object | None = None,
    ) -> None:
        _record_search_invocation(
            ctx,
            caller_kind=caller_kind,
            query=query,
            surfaced_memory_ids=surfaced_memory_ids,
            graph_provenance=graph_provenance,
            duration_ms=duration_ms,
            diagnostics=diagnostics,
        )

    def record_read(
        self,
        ctx: MemoryReadContext,
        *,
        caller_kind: str,
        memory_id: str,
        duration_ms: float,
    ) -> None:
        _record_read_invocation(
            ctx,
            caller_kind=caller_kind,
            memory_id=memory_id,
            duration_ms=duration_ms,
        )

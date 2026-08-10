"""Run the labeled search-quality corpus in a local SQLite process."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast

from mcp_memory.daemon import ensure_daemon_started
from mcp_memory.daemon_transport import request_daemon_json
from mcp_memory.config import Config
from mcp_memory.embeddings import SQLiteVectorStore
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.utils.db import DatabaseManager

from .corpus import SearchQualityCase, SearchQualityCorpus, load_corpus
from .metrics import SearchObservation, SearchQualityMetrics, evaluate_corpus


class TopicEmbedder:
    """Fixed topic encoder that requires no model download or service."""

    model_name = "benchmark-topics"
    dim = 6
    _topics = {
        "authentication": 0,
        "auth": 0,
        "token": 0,
        "credential": 0,
        "security": 0,
        "database": 1,
        "sqlite": 1,
        "postgres": 1,
        "storage": 1,
        "persistence": 1,
        "search": 2,
        "retrieval": 2,
        "ranking": 2,
        "query": 2,
        "fts": 2,
        "daemon": 3,
        "worker": 3,
        "runtime": 3,
        "background": 3,
        "lifecycle": 3,
        "embedding": 4,
        "semantic": 4,
        "vector": 4,
        "model": 4,
        "graph": 5,
        "links": 5,
        "relationships": 5,
        "authority": 5,
    }

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode text into normalized topic vectors."""
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            for token in text.lower().replace("-", " ").split():
                topic = self._topics.get(token.strip(".,?!:;"))
                if topic is not None:
                    vector[topic] += 1.0
            norm = math.sqrt(sum(value * value for value in vector))
            vectors.append([value / norm for value in vector] if norm else vector)
        return vectors


SearchCaseRunner = Callable[[SearchQualityCase], SearchObservation]
LiveRequest = Callable[[SearchQualityCase, float], object]
ResultLabeler = Callable[[dict[str, object]], str | None]
LiveStatus = Literal[
    "ok",
    "degraded",
    "unavailable",
    "timeout",
    "malformed_response",
    "daemon_error",
    "error",
]


@dataclass(frozen=True, slots=True)
class LiveCaseResult:
    """Privacy-safe outcome metadata for one live corpus request."""

    evaluation_label: str
    status: LiveStatus
    scored: bool
    latency_ms: float | None
    request_id: int | str | None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class LiveSearchQualityReport:
    """Quality metrics plus non-payload live daemon outcome metadata."""

    corpus_version: str
    metrics: SearchQualityMetrics
    cases: tuple[LiveCaseResult, ...]

    def to_mapping(self) -> dict[str, object]:
        """Serialize metrics without query text or result payloads."""
        return {
            "corpus_version": self.corpus_version,
            "metrics": self.metrics.to_mapping(),
            "cases": [
                {
                    "evaluation_label": case.evaluation_label,
                    "status": case.status,
                    "scored": case.scored,
                    "latency_ms": case.latency_ms,
                    "request_id": case.request_id,
                    "error_code": case.error_code,
                }
                for case in self.cases
            ],
        }


def run_search_quality(
    corpus: SearchQualityCorpus,
    search_case: SearchCaseRunner,
) -> SearchQualityMetrics:
    """Evaluate a corpus through a caller-provided in-process search function."""
    observations = {
        case.evaluation_label: search_case(case) for case in corpus.entries
    }
    return evaluate_corpus(corpus, observations)


def run_live_daemon(
    corpus: SearchQualityCorpus | None = None,
    *,
    workspace_root: str | None = None,
    timeout_seconds: float = 60.0,
    request_fn: LiveRequest | None = None,
    result_labeler: ResultLabeler | None = None,
) -> LiveSearchQualityReport:
    """Evaluate a corpus through the daemon transport without failing fast."""
    selected_corpus = corpus or load_corpus()
    labeler = result_labeler or _default_result_label
    if request_fn is None:
        try:
            metadata = ensure_daemon_started()
        except Exception as exc:
            return _unavailable_report(selected_corpus, _error_code(exc))

        def daemon_request(case: SearchQualityCase, request_timeout: float) -> object:
            return _request_daemon_case(
                metadata,
                case,
                workspace_root=workspace_root,
                timeout_seconds=request_timeout,
            )

        live_request: LiveRequest = daemon_request
    else:
        live_request = request_fn

    observations: dict[str, SearchObservation] = {}
    case_results: list[LiveCaseResult] = []
    for case in selected_corpus.entries:
        started = time.perf_counter()
        try:
            payload = _decode_daemon_response(live_request(case, timeout_seconds))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            diagnostics = _diagnostics(payload)
            result_labels = _result_labels(payload, labeler)
            observations[case.evaluation_label] = SearchObservation(
                result_labels=result_labels,
                latency_ms=elapsed_ms,
                semantic_abstained=_semantic_abstention(diagnostics),
            )
            degraded = diagnostics.get("degraded") is True
            case_results.append(
                LiveCaseResult(
                    evaluation_label=case.evaluation_label,
                    status="degraded" if degraded else "ok",
                    scored=bool(result_labels),
                    latency_ms=elapsed_ms,
                    request_id=_request_id(diagnostics),
                )
            )
        except TimeoutError:
            _record_live_failure(
                case,
                observations,
                case_results,
                status="timeout",
                error_code="request_timeout",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except (OSError, ConnectionError, RuntimeError) as exc:
            _record_live_failure(
                case,
                observations,
                case_results,
                status="unavailable",
                error_code=_error_code(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except _DaemonResponseError:
            _record_live_failure(
                case,
                observations,
                case_results,
                status="daemon_error",
                error_code="daemon_error_response",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            _record_live_failure(
                case,
                observations,
                case_results,
                status="malformed_response",
                error_code="malformed_response",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            _record_live_failure(
                case,
                observations,
                case_results,
                status="error",
                error_code=_error_code(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    return LiveSearchQualityReport(
        corpus_version=selected_corpus.version,
        metrics=evaluate_corpus(selected_corpus, observations),
        cases=tuple(case_results),
    )


class _DaemonResponseError(ValueError):
    """Signal a valid response envelope containing a daemon error."""


def _request_daemon_case(
    metadata: object,
    case: SearchQualityCase,
    *,
    workspace_root: str | None,
    timeout_seconds: float,
) -> object:
    payload: dict[str, object] = {
        "query": case.query,
        "limit": case.acceptable_top_k,
        "debug": True,
    }
    if workspace_root is not None:
        payload["__workspace_root"] = workspace_root
    return request_daemon_json(
        metadata,
        "/internal/tools/search_memory_records",
        payload,
        timeout_seconds=timeout_seconds,
    )


def _decode_daemon_response(response: object) -> dict[str, object]:
    if not isinstance(response, dict):
        raise ValueError("daemon response must be an object")
    payload: object = response
    contents = response.get("contents")
    if contents is not None:
        if not isinstance(contents, list):
            raise ValueError("daemon contents must be a list")
        text = next(
            (
                item.get("text")
                for item in contents
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ),
            None,
        )
        if text is None:
            raise ValueError("daemon response text is missing")
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("daemon search payload must be an object")
    decoded = cast(dict[str, object], payload)
    status = decoded.get("status")
    if status == "error":
        raise _DaemonResponseError("daemon returned an error")
    if status is not None and status != "ok":
        raise _DaemonResponseError("daemon returned an unknown status")
    if not isinstance(decoded.get("results"), list):
        raise ValueError("daemon search results must be a list")
    return decoded


def _diagnostics(payload: dict[str, object]) -> dict[str, object]:
    value = payload.get("search_diagnostics")
    return value if isinstance(value, dict) else {}


def _request_id(diagnostics: dict[str, object]) -> int | str | None:
    transport = diagnostics.get("transport")
    if not isinstance(transport, dict):
        return None
    request_id = transport.get("request_id")
    return request_id if isinstance(request_id, (int, str)) else None


def _semantic_abstention(diagnostics: dict[str, object]) -> bool | None:
    value = diagnostics.get("semantic_abstained")
    return value if isinstance(value, bool) else None


def _default_result_label(result: dict[str, object]) -> str | None:
    value = result.get("evaluation_label")
    return value if isinstance(value, str) and value else None


def _result_labels(
    payload: dict[str, object],
    labeler: ResultLabeler,
) -> tuple[str, ...]:
    results = cast(list[object], payload["results"])
    labels: list[str] = []
    for result in results:
        if isinstance(result, dict):
            label = labeler(cast(dict[str, object], result))
            if label is not None:
                labels.append(label)
    return tuple(labels)


def _record_live_failure(
    case: SearchQualityCase,
    observations: dict[str, SearchObservation],
    case_results: list[LiveCaseResult],
    *,
    status: LiveStatus,
    error_code: str,
    latency_ms: float,
) -> None:
    observations[case.evaluation_label] = SearchObservation(
        result_labels=(),
        latency_ms=latency_ms,
    )
    case_results.append(
        LiveCaseResult(
            evaluation_label=case.evaluation_label,
            status=status,
            scored=False,
            latency_ms=latency_ms,
            request_id=None,
            error_code=error_code,
        )
    )


def _unavailable_report(
    corpus: SearchQualityCorpus,
    error_code: str,
) -> LiveSearchQualityReport:
    observations = {
        case.evaluation_label: SearchObservation(()) for case in corpus.entries
    }
    return LiveSearchQualityReport(
        corpus_version=corpus.version,
        metrics=evaluate_corpus(corpus, observations),
        cases=tuple(
            LiveCaseResult(
                evaluation_label=case.evaluation_label,
                status="unavailable",
                scored=False,
                latency_ms=None,
                request_id=None,
                error_code=error_code,
            )
            for case in corpus.entries
        ),
    )


def _error_code(exc: Exception) -> str:
    if isinstance(exc, RuntimeError) and str(exc) == "daemon_not_started":
        return "daemon_not_started"
    return type(exc).__name__.lower()


def run_in_process(
    corpus: SearchQualityCorpus | None = None,
) -> SearchQualityMetrics:
    """Run the default corpus against temporary relational and vector stores."""
    selected_corpus = corpus or load_corpus()
    with TemporaryDirectory() as directory:
        manager = DatabaseManager(Path(directory) / "search-quality.db")
        try:
            repository = RelationalMemoryRepository(manager)
            label_to_id = _seed_records(repository)
            service = RelationalMemorySearchService(
                repository,
                Config(),
                embedder=TopicEmbedder(),
                vector_store=SQLiteVectorStore(manager),
                db_manager=manager,
            )
            id_to_label = {memory_id: label for label, memory_id in label_to_id.items()}

            def search_case(case: SearchQualityCase) -> SearchObservation:
                started = time.perf_counter()
                results = service.search_memories(
                    case.query,
                    workspace_id=case.workspace,
                    limit=case.acceptable_top_k,
                    side_effect_free=True,
                )
                return SearchObservation(
                    result_labels=tuple(
                        id_to_label.get(result.memory_id, "unknown")
                        for result in results
                    ),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                )

            return run_search_quality(selected_corpus, search_case)
        finally:
            manager.close()


def _seed_records(repository: RelationalMemoryRepository) -> dict[str, str]:
    records = (
        (
            "authentication-token-rotation",
            "Authentication token rotation",
            "Rotate credentials after the security review.",
            "auth security",
        ),
        (
            "sqlite-wal-persistence",
            "SQLite WAL persistence",
            "Use SQLite storage with WAL for durable local persistence.",
            "database storage",
        ),
        (
            "postgres-vector-ranking",
            "Postgres vector ranking",
            "Use Postgres server-side vector ranking for semantic retrieval.",
            "database search semantic vector",
        ),
        (
            "search-ranking-ladder",
            "Search ranking ladder",
            "Keyword and semantic search ranking should preserve exact query intent.",
            "search ranking retrieval",
        ),
        (
            "daemon-worker-lifecycle",
            "Daemon worker lifecycle",
            "Background worker lifecycle and daemon shutdown behavior.",
            "daemon runtime worker",
        ),
        (
            "graph-authority-links",
            "Graph authority links",
            "Relationship links provide authority support for graph-aware ranking.",
            "graph relationships authority",
        ),
        (
            "embedding-repair-queue",
            "Embedding repair queue",
            "Repair missing semantic vector embeddings in the background.",
            "embedding semantic vector",
        ),
        (
            "generic-architecture-note",
            "Generic architecture note",
            "Architecture decisions and implementation details.",
            "architecture",
        ),
    )
    label_to_id: dict[str, str] = {}
    for label, title, content, tags in records:
        created = repository.create_memory(
            title,
            content,
            ["workspace"],
            summary=content,
            tags=tags.split(),
            memory_type="fact",
        )
        if created is None:
            raise RuntimeError(f"failed to create benchmark record: {label}")
        label_to_id[label] = created.id
    return label_to_id

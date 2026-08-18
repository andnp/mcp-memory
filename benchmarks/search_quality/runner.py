"""Run the labeled search-quality corpus in a local SQLite process."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Protocol, TypedDict, cast

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.daemon import ensure_daemon_started
from mcp_memory.daemon_transport import request_daemon_json
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


class SearchQualityRepository(Protocol):
    def create_memory(
        self,
        title: str,
        content: str,
        workspace_ids: list[str],
        *,
        summary: str,
        tags: list[str],
        memory_type: str,
    ) -> MemoryRecord | None: ...


SearchCaseRunner = Callable[[SearchQualityCase], SearchObservation]
PolicyName = Literal["baseline", "calibrated_fusion", "query_expansion", "reranking"]
PolicyStatus = Literal["completed", "skipped"]
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
        scored_case_count = sum(case.scored for case in self.cases)
        return {
            "corpus_version": self.corpus_version,
            "metrics": self.metrics.to_mapping(),
            "scored_case_count": scored_case_count,
            "quality_scored": scored_case_count > 0,
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


@dataclass(frozen=True, slots=True)
class SearchPolicyObservation:
    """One policy result with its externally observed degradation state."""

    observation: SearchObservation
    degraded: bool = False


PolicyCaseRunner = Callable[[SearchQualityCase], SearchPolicyObservation]
PolicyBuilder = Callable[[], PolicyCaseRunner]
RerankingPolicyBuilder = Callable[[object], PolicyCaseRunner]
LOCAL_POLICY_VERSION = "mcp-memory-search-quality-v1"


@dataclass(frozen=True, slots=True)
class SearchPolicyIdentity:
    """Stable metadata for one benchmark policy configuration."""

    policy: PolicyName
    identity: str
    fingerprint: str

    def to_mapping(self) -> dict[str, str]:
        """Serialize policy metadata without runtime objects."""
        return {
            "policy": self.policy,
            "identity": self.identity,
            "fingerprint": self.fingerprint,
        }


class PolicyComparisonKwargs(TypedDict):
    """Typed arguments shared by local policy builders and comparisons."""

    baseline_builder: PolicyBuilder
    calibrated_fusion_builder: PolicyBuilder | None
    query_expansion_builder: PolicyBuilder | None
    reranking_builder: RerankingPolicyBuilder | None
    policy_identities: Mapping[PolicyName, SearchPolicyIdentity]


@dataclass(frozen=True, slots=True)
class LocalPolicyBuilders:
    """Lazy local builders and metadata for deterministic comparisons."""

    baseline_builder: PolicyBuilder
    calibrated_fusion_builder: PolicyBuilder | None
    query_expansion_builder: PolicyBuilder | None
    reranking_builder: RerankingPolicyBuilder | None
    identities: Mapping[PolicyName, SearchPolicyIdentity]

    def comparison_kwargs(self) -> PolicyComparisonKwargs:
        """Return builders in the shape accepted by ``run_policy_comparison``."""
        return {
            "baseline_builder": self.baseline_builder,
            "calibrated_fusion_builder": self.calibrated_fusion_builder,
            "query_expansion_builder": self.query_expansion_builder,
            "reranking_builder": self.reranking_builder,
            "policy_identities": self.identities,
        }

    def metadata(self) -> dict[str, dict[str, str]]:
        """Return stable metadata keyed by policy name."""
        return {
            policy: identity.to_mapping()
            for policy, identity in self.identities.items()
        }


@dataclass(frozen=True, slots=True)
class SearchPolicyReport:
    """Metrics and degradation outcomes for one policy run."""

    policy: PolicyName
    status: PolicyStatus
    metrics: SearchQualityMetrics | None
    degraded_case_count: int
    degradation_rate: float | None
    skip_reason: str | None = None
    diagnostics_complete: bool = False
    duplicate_case_count: int = 0
    observations: Mapping[str, SearchObservation] = field(default_factory=dict)
    policy_fingerprint: str | None = None
    corpus_fingerprint: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize one policy without search payloads."""
        return {
            "policy": self.policy,
            "status": self.status,
            "metrics": None if self.metrics is None else self.metrics.to_mapping(),
            "degraded_case_count": self.degraded_case_count,
            "degradation_rate": self.degradation_rate,
            "skip_reason": self.skip_reason,
            "diagnostics_complete": self.diagnostics_complete,
            "duplicate_case_count": self.duplicate_case_count,
            "policy_fingerprint": self.policy_fingerprint,
            "corpus_fingerprint": self.corpus_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class SearchPolicyAcceptanceThresholds:
    """Named thresholds for accepting an experimental search policy."""

    min_broad_hit_at_5_delta: float = 0.0
    min_historical_hit_at_5_delta: float = 0.0
    max_degradation_rate_delta: float = 0.0
    max_p95_latency_ms: float | None = None
    require_diagnostics: bool = False
    max_duplicate_case_count: int | None = None
    max_semantic_abstention_rate_delta: float | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize thresholds for a future canary consumer."""
        mapping: dict[str, object] = {
            "min_broad_hit_at_5_delta": self.min_broad_hit_at_5_delta,
            "min_historical_hit_at_5_delta": self.min_historical_hit_at_5_delta,
            "max_degradation_rate_delta": self.max_degradation_rate_delta,
            "max_p95_latency_ms": self.max_p95_latency_ms,
        }
        if self.require_diagnostics:
            mapping["require_diagnostics"] = True
        if self.max_duplicate_case_count is not None:
            mapping["max_duplicate_case_count"] = self.max_duplicate_case_count
        if self.max_semantic_abstention_rate_delta is not None:
            mapping["max_semantic_abstention_rate_delta"] = (
                self.max_semantic_abstention_rate_delta
            )
        return mapping


@dataclass(frozen=True, slots=True)
class SearchPolicyAcceptanceDecision:
    """Stable, provider-free result of comparing one policy with baseline."""

    accepted: bool
    candidate_policy: PolicyName
    baseline_policy: PolicyName
    reasons: tuple[str, ...]
    thresholds: SearchPolicyAcceptanceThresholds
    diagnostics_complete: bool | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize the decision with deterministic reason ordering."""
        mapping: dict[str, object] = {
            "accepted": self.accepted,
            "candidate_policy": self.candidate_policy,
            "baseline_policy": self.baseline_policy,
            "reasons": list(self.reasons),
            "thresholds": self.thresholds.to_mapping(),
        }
        if self.diagnostics_complete is not None:
            mapping["diagnostics_complete"] = self.diagnostics_complete
        return mapping


def evaluate_policy_acceptance(
    candidate: SearchPolicyReport,
    baseline: SearchPolicyReport,
    *,
    thresholds: SearchPolicyAcceptanceThresholds | None = None,
) -> SearchPolicyAcceptanceDecision:
    """Evaluate a candidate report against deterministic acceptance gates."""
    resolved_thresholds = thresholds or SearchPolicyAcceptanceThresholds()
    reasons: list[str] = []
    if baseline.status != "completed":
        reasons.append("baseline_not_completed")
    if candidate.status != "completed":
        reasons.append("candidate_not_completed")
    if baseline.metrics is None:
        reasons.append("baseline_metrics_missing")
    if candidate.metrics is None:
        reasons.append("candidate_metrics_missing")
    diagnostics_complete: bool | None = None
    if (
        resolved_thresholds.require_diagnostics
        and baseline.status == "completed"
        and candidate.status == "completed"
    ):
        diagnostics_complete = (
            baseline.diagnostics_complete and candidate.diagnostics_complete
        )
        if not baseline.diagnostics_complete:
            reasons.append("baseline_diagnostics_missing")
        if not candidate.diagnostics_complete:
            reasons.append("candidate_diagnostics_missing")
    if (
        resolved_thresholds.max_duplicate_case_count is not None
        and baseline.status == "completed"
        and candidate.status == "completed"
    ):
        if (
            baseline.duplicate_case_count
            > resolved_thresholds.max_duplicate_case_count
        ):
            reasons.append("baseline_duplicate_case_count_above_threshold")
        if (
            candidate.duplicate_case_count
            > resolved_thresholds.max_duplicate_case_count
        ):
            reasons.append("candidate_duplicate_case_count_above_threshold")
    if baseline.metrics is not None and candidate.metrics is not None:
        if baseline.metrics.corpus_version != candidate.metrics.corpus_version:
            reasons.append("corpus_version_mismatch")
        if baseline.metrics.query_count != candidate.metrics.query_count:
            reasons.append("query_count_mismatch")
        summaries = (
            ("exact", "exact_hit_at_1_not_preserved"),
            ("broad", "broad_hit_at_5_below_threshold"),
            ("historical", "historical_hit_at_5_below_threshold"),
        )
        for query_class, failure in summaries:
            baseline_summary = baseline.metrics.by_query_class.get(query_class)
            candidate_summary = candidate.metrics.by_query_class.get(query_class)
            if baseline_summary is None or candidate_summary is None:
                reasons.append(f"{query_class}_metrics_missing")
                continue
            if baseline_summary.query_count < 1 or candidate_summary.query_count < 1:
                reasons.append(f"{query_class}_metrics_insufficient")
                continue
            if query_class == "exact":
                if candidate_summary.hit_at_1 < baseline_summary.hit_at_1:
                    reasons.append(failure)
            elif query_class == "broad":
                if (
                    candidate_summary.hit_at_5 - baseline_summary.hit_at_5
                    < resolved_thresholds.min_broad_hit_at_5_delta
                ):
                    reasons.append(failure)
            elif (
                candidate_summary.hit_at_5 - baseline_summary.hit_at_5
                < resolved_thresholds.min_historical_hit_at_5_delta
            ):
                reasons.append(failure)
        if baseline.degradation_rate is None or candidate.degradation_rate is None:
            reasons.append("degradation_rate_missing")
        elif (
            candidate.degradation_rate - baseline.degradation_rate
            > resolved_thresholds.max_degradation_rate_delta
        ):
            reasons.append("degradation_rate_above_threshold")
        if resolved_thresholds.max_p95_latency_ms is not None:
            if candidate.metrics.latency_p95_ms is None:
                reasons.append("candidate_p95_latency_missing")
            elif candidate.metrics.latency_p95_ms > resolved_thresholds.max_p95_latency_ms:
                reasons.append("candidate_p95_latency_above_threshold")
        if resolved_thresholds.max_semantic_abstention_rate_delta is not None:
            baseline_rate = baseline.metrics.semantic_abstention_rate
            candidate_rate = candidate.metrics.semantic_abstention_rate
            if baseline_rate is None:
                reasons.append("baseline_semantic_abstention_rate_missing")
            if candidate_rate is None:
                reasons.append("candidate_semantic_abstention_rate_missing")
            if baseline_rate is not None and candidate_rate is not None:
                if (
                    candidate_rate - baseline_rate
                    > resolved_thresholds.max_semantic_abstention_rate_delta
                ):
                    reasons.append(
                        "candidate_semantic_abstention_rate_above_threshold"
                    )
    return SearchPolicyAcceptanceDecision(
        accepted=not reasons,
        candidate_policy=candidate.policy,
        baseline_policy=baseline.policy,
        reasons=tuple(reasons),
        thresholds=resolved_thresholds,
        diagnostics_complete=diagnostics_complete,
    )


@dataclass(frozen=True, slots=True)
class SearchPolicyComparison:
    """Deterministic baseline and experiment results for one corpus."""

    corpus_version: str
    policies: tuple[SearchPolicyReport, ...]
    corpus_fingerprint: str = ""

    @property
    def by_policy(self) -> dict[str, SearchPolicyReport]:
        """Return policy reports keyed by their stable names."""
        return {report.policy: report for report in self.policies}

    def to_mapping(self) -> dict[str, object]:
        """Serialize the comparison in policy execution order."""
        return {
            "corpus_version": self.corpus_version,
            "corpus_fingerprint": self.corpus_fingerprint,
            "policies": [report.to_mapping() for report in self.policies],
        }


def run_policy_comparison(
    corpus: SearchQualityCorpus | None = None,
    *,
    baseline_builder: PolicyBuilder,
    calibrated_fusion_builder: PolicyBuilder | None = None,
    query_expansion_builder: PolicyBuilder | None = None,
    reranking_builder: RerankingPolicyBuilder | None = None,
    reranker: object | None = None,
    policy_identities: Mapping[PolicyName, SearchPolicyIdentity] | None = None,
) -> SearchPolicyComparison:
    """Run the same corpus against isolated baseline and policy builders.

    Experimental builders are supplied by the caller so this deterministic
    benchmark never changes production configuration or downloads providers.
    Reranking is skipped unless both its builder and reranker are supplied.
    """
    selected_corpus = corpus or load_corpus()
    corpus_fingerprint = _corpus_fingerprint(selected_corpus)
    identities = policy_identities or {}
    policies: tuple[
        tuple[PolicyName, PolicyBuilder | RerankingPolicyBuilder | None, object | None],
        ...,
    ] = (
        ("baseline", baseline_builder, None),
        ("calibrated_fusion", calibrated_fusion_builder, None),
        ("query_expansion", query_expansion_builder, None),
        ("reranking", reranking_builder, reranker),
    )
    reports: list[SearchPolicyReport] = []
    for policy, builder, supplied_reranker in policies:
        if builder is None:
            reports.append(
                _skipped_policy(
                    policy,
                    "builder_not_supplied",
                    policy_fingerprint=_policy_fingerprint(policy, identities),
                    corpus_fingerprint=corpus_fingerprint,
                )
            )
            continue
        if policy == "reranking" and supplied_reranker is None:
            reports.append(
                _skipped_policy(
                    policy,
                    "reranker_not_supplied",
                    policy_fingerprint=_policy_fingerprint(policy, identities),
                    corpus_fingerprint=corpus_fingerprint,
                )
            )
            continue
        if policy == "reranking":
            runner = cast(RerankingPolicyBuilder, builder)(
                cast(object, supplied_reranker)
            )
        else:
            runner = cast(PolicyBuilder, builder)()
        reports.append(
            _run_policy(
                selected_corpus,
                policy,
                runner,
                policy_fingerprint=_policy_fingerprint(policy, identities),
                corpus_fingerprint=corpus_fingerprint,
            )
        )
    return SearchPolicyComparison(
        selected_corpus.version,
        tuple(reports),
        corpus_fingerprint,
    )


def build_local_policy_builders(
    repository: RelationalMemoryRepository,
    label_to_id: Mapping[str, str],
    *,
    db_manager: DatabaseManager,
    vector_store: SQLiteVectorStore,
    embedder: TopicEmbedder | None = None,
    config: Config | None = None,
    requested_policies: Collection[PolicyName] = (),
) -> LocalPolicyBuilders:
    """Build lazy policy factories over deterministic local dependencies.

    Baseline is always available. Experimental services are constructed only
    when their policy is requested and each factory call creates a fresh
    service. Reranking remains absent because this local adapter has no real
    reranker dependency to supply.
    """
    valid_policies: set[PolicyName] = {
        "baseline",
        "calibrated_fusion",
        "query_expansion",
        "reranking",
    }
    requested = set(requested_policies)
    unknown = requested - valid_policies
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unknown search policy: {names}")
    requested.add("baseline")

    resolved_embedder = embedder or TopicEmbedder()
    resolved_config = config or Config()
    id_to_label = {memory_id: label for label, memory_id in label_to_id.items()}
    identities: dict[PolicyName, SearchPolicyIdentity] = {}
    builders: dict[PolicyName, PolicyBuilder] = {}
    for policy in ("baseline", "calibrated_fusion", "query_expansion"):
        if policy not in requested:
            continue
        policy_config = _local_policy_config(resolved_config, policy)
        identities[policy] = SearchPolicyIdentity(
            policy=policy,
            identity=f"{LOCAL_POLICY_VERSION}:{policy}",
            fingerprint=(
                policy_config.searchkernel.active_feature_fingerprint()
                or "baseline"
            ),
        )
        builders[policy] = _local_policy_builder(
            repository,
            id_to_label,
            db_manager=db_manager,
            vector_store=vector_store,
            embedder=resolved_embedder,
            config=policy_config,
        )

    return LocalPolicyBuilders(
        baseline_builder=builders["baseline"],
        calibrated_fusion_builder=builders.get("calibrated_fusion"),
        query_expansion_builder=builders.get("query_expansion"),
        reranking_builder=None,
        identities=identities,
    )


def _local_policy_config(config: Config, policy: PolicyName) -> Config:
    if policy == "reranking":
        raise ValueError("reranking has no local deterministic builder")
    searchkernel = replace(
        config.searchkernel,
        calibrated_fusion_enabled=policy == "calibrated_fusion",
        query_expansion_enabled=policy == "query_expansion",
        rerank_policy="disabled",
        rerank_budget=0,
    )
    return replace(config, searchkernel=searchkernel)


def _local_policy_builder(
    repository: RelationalMemoryRepository,
    id_to_label: Mapping[str, str],
    *,
    db_manager: DatabaseManager,
    vector_store: SQLiteVectorStore,
    embedder: TopicEmbedder,
    config: Config,
) -> PolicyBuilder:
    def build() -> PolicyCaseRunner:
        service = RelationalMemorySearchService(
            repository,
            config,
            embedder=embedder,
            vector_store=vector_store,
            db_manager=db_manager,
        )

        def search_case(case: SearchQualityCase) -> SearchPolicyObservation:
            started = time.perf_counter()
            results, diagnostics = service.search_memories_with_diagnostics(
                case.query,
                workspace_id=case.workspace,
                limit=case.acceptable_top_k,
                debug=True,
                side_effect_free=True,
            )
            observation = SearchObservation(
                result_labels=tuple(
                    id_to_label.get(result.memory_id, "unknown")
                    for result in results
                ),
                latency_ms=(time.perf_counter() - started) * 1000.0,
                semantic_abstained=diagnostics.semantic_abstained,
                diagnostics=diagnostics.to_payload(),
            )
            return SearchPolicyObservation(observation, diagnostics.degraded)

        return search_case

    return build


_REQUIRED_DIAGNOSTIC_KEYS = frozenset(
    {
        "timing_ms",
        "candidate_counts",
        "degraded",
        "lane_decisions",
        "overlap",
        "duplicate_candidate_ids",
        "multi_lane_candidate_ids",
        "final_duplicate_ids",
        "semantic",
    }
)


def _corpus_fingerprint(corpus: SearchQualityCorpus) -> str:
    encoded = json.dumps(
        corpus.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _policy_fingerprint(
    policy: PolicyName,
    identities: Mapping[PolicyName, SearchPolicyIdentity],
) -> str:
    identity = identities.get(policy)
    if identity is not None:
        return identity.fingerprint
    encoded = json.dumps({"policy": policy}, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _diagnostics_complete(diagnostics: Mapping[str, object] | None) -> bool:
    return diagnostics is not None and diagnostics.keys() >= _REQUIRED_DIAGNOSTIC_KEYS


def _diagnostic_mapping(
    diagnostics: Mapping[str, object],
    key: str,
) -> Mapping[str, object] | None:
    value = diagnostics.get(key)
    return value if isinstance(value, Mapping) else None


def _diagnostic_list(diagnostics: Mapping[str, object], key: str) -> list[object]:
    value = diagnostics.get(key)
    return list(value) if isinstance(value, list) else []


def _enrich_observation(
    observation: SearchObservation,
    *,
    policy_fingerprint: str,
    corpus_fingerprint: str,
    degraded: bool,
) -> SearchObservation:
    diagnostics = observation.diagnostics
    if diagnostics is None:
        return replace(
            observation,
            policy_fingerprint=policy_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            diagnostics_complete=False,
            degraded=degraded,
        )

    semantic = _diagnostic_mapping(diagnostics, "semantic")
    semantic_rate_value = diagnostics.get("semantic_abstention_rate")
    if semantic is not None:
        semantic_rate_value = semantic.get("abstention_rate", semantic_rate_value)
    semantic_rate = (
        float(semantic_rate_value)
        if isinstance(semantic_rate_value, (int, float))
        else None
    )
    timing = _diagnostic_mapping(diagnostics, "timing_ms")
    stage_timings = (
        {key: float(value) for key, value in timing.items() if isinstance(value, (int, float))}
        if timing is not None
        else None
    )
    overlap = _diagnostic_mapping(diagnostics, "overlap")
    duplicate_semantics: dict[str, object] = {
        "duplicate_candidate_ids": _diagnostic_list(
            diagnostics, "duplicate_candidate_ids"
        ),
        "multi_lane_candidate_ids": _diagnostic_list(
            diagnostics, "multi_lane_candidate_ids"
        ),
        "final_duplicate_ids": _diagnostic_list(diagnostics, "final_duplicate_ids"),
    }
    if overlap is not None:
        duplicate_semantics["overlap"] = dict(overlap)
    return replace(
        observation,
        policy_fingerprint=policy_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        lane_decisions=_diagnostic_mapping(diagnostics, "lane_decisions"),
        stage_timings_ms=stage_timings,
        diagnostics_complete=_diagnostics_complete(diagnostics),
        degraded=degraded,
        duplicate_semantics=duplicate_semantics,
        semantic_abstention_rate=semantic_rate,
        semantic_abstention_available=semantic_rate is not None,
    )


def _run_policy(
    corpus: SearchQualityCorpus,
    policy: PolicyName,
    runner: PolicyCaseRunner,
    *,
    policy_fingerprint: str,
    corpus_fingerprint: str,
) -> SearchPolicyReport:
    observations: dict[str, SearchObservation] = {}
    degraded_case_count = 0
    diagnostics_complete = True
    duplicate_case_count = 0
    for case in corpus.entries:
        outcome = runner(case)
        observation = _enrich_observation(
            outcome.observation,
            policy_fingerprint=policy_fingerprint,
            corpus_fingerprint=corpus_fingerprint,
            degraded=outcome.degraded,
        )
        observations[case.evaluation_label] = observation
        degraded_case_count += int(outcome.degraded)
        diagnostics_complete &= observation.diagnostics_complete
        diagnostics = observation.diagnostics
        duplicate_ids = None if diagnostics is None else diagnostics.get(
            "final_duplicate_ids"
        )
        duplicate_case_count += int(bool(duplicate_ids))
    return SearchPolicyReport(
        policy=policy,
        status="completed",
        metrics=evaluate_corpus(corpus, observations),
        degraded_case_count=degraded_case_count,
        degradation_rate=degraded_case_count / len(corpus.entries)
        if corpus.entries
        else 0.0,
        diagnostics_complete=diagnostics_complete,
        duplicate_case_count=duplicate_case_count,
        observations=observations,
        policy_fingerprint=policy_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
    )


def _skipped_policy(
    policy: PolicyName,
    reason: str,
    *,
    policy_fingerprint: str,
    corpus_fingerprint: str,
) -> SearchPolicyReport:
    return SearchPolicyReport(
        policy=policy,
        status="skipped",
        metrics=None,
        degraded_case_count=0,
        degradation_rate=None,
        skip_reason=reason,
        policy_fingerprint=policy_fingerprint,
        corpus_fingerprint=corpus_fingerprint,
    )


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
                diagnostics=dict(diagnostics),
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
            label_to_id = seed_search_quality_records(repository)
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
                workspace_id = None if case.workspace == "global" else case.workspace
                results = service.search_memories(
                    case.query,
                    workspace_id=workspace_id,
                    ranking_workspace_id=(
                        "workspace" if case.workspace == "global" else None
                    ),
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


def seed_search_quality_records(
    repository: SearchQualityRepository,
    *,
    workspace: str = "workspace",
) -> dict[str, str]:
    records: tuple[tuple[str, str, str, str], ...] = (
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
            "daemon-runtime-history",
            "Previous daemon runtime evidence",
            "Historical daemon runtime evidence: the previous worker shutdown followed a stalled background task.",
            "daemon runtime history worker",
        ),
        (
            "daemon-current-health",
            "Current daemon health",
            "Current daemon health: the background worker is running normally.",
            "daemon runtime health worker",
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
        (
            "search-quality-follow-up",
            "Search quality follow-up",
            "Search-quality follow-up evidence: searchkernel version 0.21 showed broad-query recall loss during latency spikes with transport and runtime warnings.",
            "search quality searchkernel query runtime",
        ),
        (
            "search-quality-follow-up-decoy-runtime",
            "Search quality runtime report",
            "Another project reported searchkernel version transport and runtime warnings during a latency review; broad query recall remained under observation.",
            "search quality searchkernel query runtime",
        ),
        (
            "search-quality-follow-up-decoy-transport",
            "Search quality transport report",
            "A separate benchmark project tracked searchkernel version, broad query recall, latency, transport warnings, and runtime behavior.",
            "search quality searchkernel query runtime",
        ),
        (
            "global-project-ranking",
            "Shared deployment guidance",
            "Shared search retrieval deployment guidance for the active project workspace.",
            "deployment guidance project search retrieval",
        ),
        (
            "global-other-project",
            "Shared deployment guidance elsewhere",
            "Shared deployment guidance for another project workspace.",
            "deployment guidance project",
        ),
    )
    workspace_ids_by_label = {
        "global-project-ranking": [workspace],
        "global-other-project": ["other-workspace"],
        "search-quality-follow-up": [workspace],
        "search-quality-follow-up-decoy-runtime": ["other-workspace"],
        "search-quality-follow-up-decoy-transport": ["benchmark-project"],
    }
    label_to_id: dict[str, str] = {}
    for label, title, content, tags in records:
        created = repository.create_memory(
            title,
            content,
            workspace_ids_by_label.get(label, [workspace]),
            summary=content,
            tags=str(tags).split(),
            memory_type="fact",
        )
        if created is None:
            raise RuntimeError(f"failed to create benchmark record: {label}")
        label_to_id[label] = created.id
    return label_to_id

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.core.ports import SearchHealthPort
from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.management.models import (
    NerdMetricsPayload,
    QualityCleanupCandidatePayload,
    QualityCleanupCandidatesPayload,
    QualityCleanupCriterionPayload,
    QualityCleanupRecommendationPayload,
    SelectorStatsPayload,
)
from mcp_memory.management.query_runner import PostgresManagementQueryAdapter, SQLiteManagementQueryAdapter
from mcp_memory.management.scope_policy import ScopePolicyKind, resolve_workspace_id_for_policy
from mcp_memory.management.selector_stats_reporting import build_selector_stats_payload

_LOW_CONVERSION_DEFAULT_MIN_SEARCH_COUNT = 3
_LOW_CONVERSION_DEFAULT_MAX_RATE = 0.25
_QUALITY_CLEANUP_CRITERIA: dict[str, tuple[str, str, int]] = {
    "trace_like_memory_count": (
        "Trace-like memory",
        "Title matches an operational trace/journal pattern that should usually be curated or archived.",
        60,
    ),
    "generic_summary_count": (
        "Generic summary",
        "Summary starts with a generic 'Covers ...' pattern that hides the concrete takeaway.",
        35,
    ),
    "untagged_observation_count": (
        "Untagged observation",
        "Observation memory has no tags, which makes targeted retrieval and cleanup harder.",
        30,
    ),
    "oversized_memory_count": (
        "Oversized memory",
        "Memory content is at least 4 KB, making it a good cleanup or split candidate.",
        20,
    ),
}
_QUALITY_CLEANUP_RECOMMENDATIONS: dict[str, tuple[str, str, int]] = {
    "trace_like_memory_count": (
        "Archive or curate",
        "This looks trace-like enough that it may belong in archival/cleanup flow rather than as a durable canonical memory.",
        60,
    ),
    "generic_summary_count": (
        "Resummarize",
        "Rewrite the summary to state the concrete takeaway so retrieval value is obvious before opening the record.",
        35,
    ),
    "untagged_observation_count": (
        "Retag",
        "Add concrete subsystem or topic tags so the memory is easier to retrieve and maintain.",
        30,
    ),
    "oversized_memory_count": (
        "Split or trim",
        "Break the memory into narrower focused records or trim excess detail if the size is obscuring the durable point.",
        20,
    ),
    "low_conversion": (
        "Review title and summary",
        "The memory is surfaced often but rarely opened, so its title/summary or overall relevance signal likely needs improvement.",
        50,
    ),
}


@dataclass(frozen=True)
class AnalyticsServiceDependencies:
    db_manager: Any
    workspace_id: str | None
    storage_backend: str | None
    task_queue: Any
    provider_usage_repo: Any
    config: Any
    ai_json_provider: Any
    ai_agent_provider: Any
    ai_provider_registry: Any
    relational_search: MemorySearchPort | None
    search_health: SearchHealthPort | None = None


class AnalyticsService:
    def __init__(self, dependencies: AnalyticsServiceDependencies) -> None:
        self._dependencies = dependencies

    def get_selector_stats(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        limit: int = 200,
        now: float | None = None,
    ) -> SelectorStatsPayload:
        dependencies = self._dependencies
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        runs = build_recent_agent_runs(
            dependencies.db_manager,
            effective_workspace_id,
            limit=limit,
            detail_level="compact",
        )
        return build_selector_stats_payload(
            runs,
            window_hours=window_hours,
            run_limit=limit,
            now=now,
        )

    def get_nerd_metrics(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
    ) -> NerdMetricsPayload:
        dependencies = self._dependencies
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        query_adapter = (
            PostgresManagementQueryAdapter()
            if dependencies.storage_backend == "postgres"
            else SQLiteManagementQueryAdapter()
        )
        return build_nerd_metrics(
            db_manager=dependencies.db_manager,
            query_adapter=query_adapter,
            workspace_id=effective_workspace_id,
            task_queue=dependencies.task_queue,
            provider_usage_repo=dependencies.provider_usage_repo,
            config=dependencies.config,
            ai_json_provider=dependencies.ai_json_provider,
            ai_agent_provider=dependencies.ai_agent_provider,
            ai_provider_registry=dependencies.ai_provider_registry,
            relational_search=dependencies.relational_search,
            search_health=dependencies.search_health,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )

    def list_quality_cleanup_candidates(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
        limit: int = 25,
        low_conversion_min_search_count: int = _LOW_CONVERSION_DEFAULT_MIN_SEARCH_COUNT,
        low_conversion_max_conversion_rate: float = _LOW_CONVERSION_DEFAULT_MAX_RATE,
    ) -> QualityCleanupCandidatesPayload:
        nerd_metrics = self.get_nerd_metrics(
            scope=scope,
            workspace_id=workspace_id,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )
        generated_at = nerd_metrics.generated_at
        candidates_by_id: dict[str, QualityCleanupCandidatePayload] = {}

        for signal in nerd_metrics.quality_drilldown.signals:
            criteria_config = _QUALITY_CLEANUP_CRITERIA.get(signal.key)
            if criteria_config is None or signal.count <= 0:
                continue
            label, rationale, weight = criteria_config
            for record in signal.records:
                candidate = candidates_by_id.setdefault(
                    record.memory_id,
                    QualityCleanupCandidatePayload(
                        memory_id=record.memory_id,
                        title=record.title,
                        summary=record.summary,
                        memory_type=record.memory_type,
                        status=record.status,
                        updated_at=record.updated_at,
                        tags=list(record.tags),
                    ),
                )
                if candidate.updated_at is None:
                    candidate.updated_at = record.updated_at
                if candidate.summary is None:
                    candidate.summary = record.summary
                if not candidate.tags:
                    candidate.tags = list(record.tags)
                _append_quality_cleanup_criterion(
                    candidate,
                    QualityCleanupCriterionPayload(
                        key=signal.key,
                        label=label,
                        rationale=rationale,
                        weight=weight,
                    ),
                )
                _append_quality_cleanup_recommendation(candidate, signal.key)

        for record in nerd_metrics.retrieval.low_conversion_memories:
            if record.search_count < low_conversion_min_search_count:
                continue
            if record.conversion_rate > low_conversion_max_conversion_rate:
                continue
            candidate = candidates_by_id.setdefault(
                record.memory_id,
                QualityCleanupCandidatePayload(
                    memory_id=record.memory_id,
                    title=record.title,
                    memory_type=record.memory_type,
                    status=record.status,
                    tags=list(record.tags),
                ),
            )
            if not candidate.tags:
                candidate.tags = list(record.tags)
            candidate.search_count = record.search_count
            candidate.read_count = record.read_count
            candidate.converted_search_count = record.converted_search_count
            candidate.conversion_rate = record.conversion_rate
            candidate.last_search_at = record.last_search_at
            candidate.last_read_at = record.last_read_at
            _append_quality_cleanup_criterion(
                candidate,
                QualityCleanupCriterionPayload(
                    key="low_conversion",
                    label="Low conversion",
                    rationale=(
                        f"Surfaced at least {low_conversion_min_search_count} times but converts at or below "
                        f"{low_conversion_max_conversion_rate:.2f}."
                    ),
                    weight=50,
                    search_count=record.search_count,
                    read_count=record.read_count,
                    converted_search_count=record.converted_search_count,
                    conversion_rate=record.conversion_rate,
                ),
            )
            _append_quality_cleanup_recommendation(candidate, "low_conversion")

        all_candidates = sorted(
            candidates_by_id.values(),
            key=lambda candidate: (
                -candidate.priority_score,
                -(candidate.search_count or 0),
                candidate.title.lower(),
                candidate.memory_id,
            ),
        )
        candidates = all_candidates[: max(limit, 0)]
        return QualityCleanupCandidatesPayload(
            generated_at=generated_at,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            low_conversion_min_search_count=low_conversion_min_search_count,
            low_conversion_max_conversion_rate=low_conversion_max_conversion_rate,
            total_candidates=len(all_candidates),
            candidates=candidates,
        )

    def _resolve_scoped_workspace_id(
        self,
        *,
        scope: str | None,
        workspace_id: str | None,
    ) -> str | None:
        return resolve_workspace_id_for_policy(
            ScopePolicyKind.SERVICE_SCOPED_DEFAULT,
            scope=scope,
            workspace_id=workspace_id,
            current_workspace_id=self._dependencies.workspace_id,
        )


def _append_quality_cleanup_criterion(
    candidate: QualityCleanupCandidatePayload,
    criterion: QualityCleanupCriterionPayload,
) -> None:
    if any(existing.key == criterion.key for existing in candidate.criteria):
        return
    candidate.criteria.append(criterion)
    candidate.criteria.sort(key=lambda item: (-item.weight, item.key))
    candidate.priority_score = sum(item.weight for item in candidate.criteria)


def _append_quality_cleanup_recommendation(
    candidate: QualityCleanupCandidatePayload,
    criterion_key: str,
) -> None:
    recommendation = _QUALITY_CLEANUP_RECOMMENDATIONS.get(criterion_key)
    if recommendation is None:
        return
    label, rationale, weight = recommendation
    if any(existing.key == criterion_key for existing in candidate.recommendations):
        return
    candidate.recommendations.append(
        QualityCleanupRecommendationPayload(
            key=criterion_key,
            label=label,
            rationale=rationale,
            weight=weight,
        )
    )
    candidate.recommendations.sort(key=lambda item: (-item.weight, item.key))

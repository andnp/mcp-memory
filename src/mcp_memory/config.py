from __future__ import annotations

from dataclasses import dataclass, field, fields
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal, cast
import logging
import os
import re
import tempfile
import tomllib
from urllib.parse import urlsplit

import tomlkit


logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "mcp-memory"
GLOBAL_DAEMON_IDENTITY = "global"
StorageBackendKind = Literal["sqlite", "postgres"]
StorageCacheMode = Literal["readonly", "writeback"]
MemoryGCMode = Literal["report-only", "delete"]
IngressEvidenceMode = Literal["off", "shadow", "enforce"]
IngressReplayPolicy = Literal["legacy", "detect", "enforce"]
IngressQualityAdmission = Literal["disabled", "manual", "canary"]
SearchKernelExpansionPolicy = Literal["vector", "synonym"]
SearchKernelRerankPolicy = Literal["disabled", "default"]


@dataclass
class MemoryRecencyConfig:
    boost_window_days: int
    max_boost_amount: float
    boost_decay_rate: float

    def __post_init__(self) -> None:
        if self.boost_window_days < 0:
            raise ValueError("boost_window_days must be non-negative")
        if not (0.0 <= self.max_boost_amount <= 0.5):
            raise ValueError("max_boost_amount must be in [0.0, 0.5]")
        if not (0.0 < self.boost_decay_rate < 1.0):
            raise ValueError("boost_decay_rate must be in (0.0, 1.0)")


@dataclass
class MemoryConfig:
    enabled: bool = True
    checkpoint_interval_ops: int = 10
    checkpoint_interval_secs: int = 300
    recency_journal: MemoryRecencyConfig = field(
        default_factory=lambda: MemoryRecencyConfig(14, 0.2, 0.95)
    )
    recency_plan: MemoryRecencyConfig = field(
        default_factory=lambda: MemoryRecencyConfig(30, 0.15, 0.97)
    )
    recency_fact: MemoryRecencyConfig = field(
        default_factory=lambda: MemoryRecencyConfig(180, 0.05, 0.99)
    )
    recency_observation: MemoryRecencyConfig = field(
        default_factory=lambda: MemoryRecencyConfig(14, 0.2, 0.95)
    )
    recency_reflection: MemoryRecencyConfig = field(
        default_factory=lambda: MemoryRecencyConfig(180, 0.05, 0.99)
    )

    def __post_init__(self) -> None:
        if self.checkpoint_interval_ops < 1:
            raise ValueError("checkpoint_interval_ops must be >= 1")
        if self.checkpoint_interval_secs < 0:
            raise ValueError("checkpoint_interval_secs must be >= 0")

    def get_recency_config(self, memory_type: str) -> MemoryRecencyConfig:
        return {
            "journal": self.recency_journal,
            "plan": self.recency_plan,
            "fact": self.recency_fact,
            "observation": self.recency_observation,
            "reflection": self.recency_reflection,
        }.get(memory_type, self.recency_journal)


@dataclass
class AIConfig:
    provider: str = "none"
    model: str = "gpt-5.6-luna"
    timeout_seconds: float = 900.0
    max_retries: int = 0
    provider_trust_class: str | None = None
    provider_allowlisted: bool | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"none", "copilot-sdk"}:
            raise ValueError("ai.provider must be one of 'none' or 'copilot-sdk'")
        if self.timeout_seconds <= 0:
            raise ValueError("ai.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("ai.max_retries must be >= 0")
        if self.provider_trust_class is not None and self.provider_trust_class not in {
            "local",
            "trusted_external",
            "external",
        }:
            raise ValueError(
                "ai.provider_trust_class must be one of 'local', 'trusted_external', or 'external'"
            )
        if self.provider_allowlisted is not None and not isinstance(self.provider_allowlisted, bool):
            raise ValueError("ai.provider_allowlisted must be a boolean")


@dataclass
class ProviderRoutingConfig:
    profiles: dict[str, AIConfig] = field(default_factory=dict)
    task_routes: dict[str, list[str]] = field(default_factory=dict)
    task_classes: dict[str, str] = field(default_factory=dict)
    profile_daily_call_limits: dict[str, int] = field(default_factory=dict)
    model_burst_call_limit: int = 2
    model_burst_window_seconds: float = 300.0
    default_json_route: list[str] = field(default_factory=list)
    default_agentic_route: list[str] = field(default_factory=list)
    fallback_to_json_only: bool = False
    low_priority_task_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        normalized_profiles: dict[str, AIConfig] = {}
        for key, value in self.profiles.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("provider_routing.profiles keys must be non-empty strings")
            normalized_profiles[key.strip()] = value
        self.profiles = normalized_profiles
        self.task_routes = {
            str(task_name).strip(): [str(item).strip() for item in route if isinstance(item, str) and str(item).strip()]
            for task_name, route in self.task_routes.items()
            if isinstance(task_name, str) and task_name.strip() and isinstance(route, list)
        }
        self.task_classes = {
            str(task_name).strip(): str(task_class).strip()
            for task_name, task_class in self.task_classes.items()
            if isinstance(task_name, str) and task_name.strip() and isinstance(task_class, str) and str(task_class).strip()
        }
        self.profile_daily_call_limits = {
            str(profile_key).strip(): int(limit)
            for profile_key, limit in self.profile_daily_call_limits.items()
            if isinstance(profile_key, str) and profile_key.strip()
        }
        for profile_key, limit in self.profile_daily_call_limits.items():
            if limit < 1:
                raise ValueError(f"provider_routing.profile_daily_call_limits[{profile_key!r}] must be >= 1")
        if self.model_burst_call_limit < 1:
            raise ValueError("provider_routing.model_burst_call_limit must be >= 1")
        if self.model_burst_window_seconds <= 0:
            raise ValueError("provider_routing.model_burst_window_seconds must be > 0")
        self.default_json_route = [item.strip() for item in self.default_json_route if isinstance(item, str) and item.strip()]
        self.default_agentic_route = [item.strip() for item in self.default_agentic_route if isinstance(item, str) and item.strip()]
        self.low_priority_task_names = [item.strip() for item in self.low_priority_task_names if isinstance(item, str) and item.strip()]


@dataclass
class IngestSuppressionWindow:
    start_hour: int
    end_hour: int
    days_of_week: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0 <= self.start_hour <= 23):
            raise ValueError("ingest_suppression.windows.start_hour must be in [0, 23]")
        if not (0 <= self.end_hour <= 23):
            raise ValueError("ingest_suppression.windows.end_hour must be in [0, 23]")
        normalized_days: list[int] = []
        for day in self.days_of_week:
            if not isinstance(day, int) or not (0 <= day <= 6):
                raise ValueError("ingest_suppression.windows.days_of_week must contain integers in [0, 6]")
            if day not in normalized_days:
                normalized_days.append(day)
        self.days_of_week = normalized_days


@dataclass
class IngestSuppressionConfig:
    enabled: bool = False
    windows: list[IngestSuppressionWindow] = field(default_factory=list)


@dataclass
class IngestEscalationConfig:
    enabled: bool = True
    deterministic_first: bool = True
    agentic_pending_count_threshold: int = 8
    novelty_threshold: float = 0.6
    preview_entry_limit: int = 8

    def __post_init__(self) -> None:
        if self.agentic_pending_count_threshold < 1:
            raise ValueError("ingest_escalation.agentic_pending_count_threshold must be >= 1")
        if not (0.0 <= self.novelty_threshold <= 1.0):
            raise ValueError("ingest_escalation.novelty_threshold must be in [0.0, 1.0]")
        if self.preview_entry_limit < 1:
            raise ValueError("ingest_escalation.preview_entry_limit must be >= 1")


@dataclass
class CurationConfig:
    """Curator runtime configuration."""


@dataclass
class MaintenanceConfig:
    archived_memory_retention_days: int = 90
    memory_gc_batch_size: int = 100
    dangling_link_gc_batch_size: int = 100
    memory_gc_mode: MemoryGCMode = "report-only"

    def __post_init__(self) -> None:
        if self.archived_memory_retention_days < 1:
            raise ValueError("maintenance.archived_memory_retention_days must be >= 1")
        if self.memory_gc_batch_size < 1:
            raise ValueError("maintenance.memory_gc_batch_size must be >= 1")
        if self.dangling_link_gc_batch_size < 1:
            raise ValueError("maintenance.dangling_link_gc_batch_size must be >= 1")
        if self.memory_gc_mode not in {"report-only", "delete"}:
            raise ValueError("maintenance.memory_gc_mode must be 'report-only' or 'delete'")



@dataclass
class DaemonConfig:
    host: str = "127.0.0.1"
    port: int = 4242
    auto_start_timeout_seconds: float = 10.0
    shutdown_grace_seconds: float = 5.0
    healthcheck_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.port < 0 or self.port > 65535:
            raise ValueError("daemon.port must be in [0, 65535]")
        if self.auto_start_timeout_seconds <= 0:
            raise ValueError("daemon.auto_start_timeout_seconds must be > 0")
        if self.shutdown_grace_seconds < 0:
            raise ValueError("daemon.shutdown_grace_seconds must be >= 0")
        if self.healthcheck_interval_seconds <= 0:
            raise ValueError("daemon.healthcheck_interval_seconds must be > 0")


@dataclass
class SQLiteStorageConfig:
    path: str = ""


@dataclass
class PostgresStorageConfig:
    dsn: str = ""
    pool_min: int = 1
    pool_max: int = 10
    statement_timeout_ms: int = 30000
    lock_timeout_ms: int = 5000
    application_name: str = DEFAULT_APP_NAME

    def __post_init__(self) -> None:
        if self.pool_min < 1:
            raise ValueError("storage.postgres.pool_min must be >= 1")
        if self.pool_max < self.pool_min:
            raise ValueError("storage.postgres.pool_max must be >= storage.postgres.pool_min")
        if self.statement_timeout_ms < 1:
            raise ValueError("storage.postgres.statement_timeout_ms must be >= 1")
        if self.lock_timeout_ms < 1:
            raise ValueError("storage.postgres.lock_timeout_ms must be >= 1")
        if not self.application_name.strip():
            raise ValueError("storage.postgres.application_name must be non-empty")


@dataclass
class StorageCacheConfig:
    enabled: bool = False
    mode: StorageCacheMode = "readonly"
    max_cached_search_docs: int = 50000
    max_outbox_entries: int = 10000

    def __post_init__(self) -> None:
        if self.mode not in {"readonly", "writeback"}:
            raise ValueError("storage.cache.mode must be 'readonly' or 'writeback'")
        if self.max_cached_search_docs < 1:
            raise ValueError("storage.cache.max_cached_search_docs must be >= 1")
        if self.max_outbox_entries < 1:
            raise ValueError("storage.cache.max_outbox_entries must be >= 1")


@dataclass
class StorageConfig:
    backend: StorageBackendKind = "sqlite"
    sqlite: SQLiteStorageConfig = field(default_factory=SQLiteStorageConfig)
    postgres: PostgresStorageConfig = field(default_factory=PostgresStorageConfig)
    cache: StorageCacheConfig = field(default_factory=StorageCacheConfig)

    def __post_init__(self) -> None:
        if self.backend not in {"sqlite", "postgres"}:
            raise ValueError("storage.backend must be 'sqlite' or 'postgres'")
        if self.backend == "postgres":
            normalized_dsn = self.postgres.dsn.strip()
            if not normalized_dsn:
                raise ValueError("storage.postgres.dsn must be set when storage.backend is 'postgres'")
            scheme = urlsplit(normalized_dsn).scheme.lower()
            if scheme not in {"postgres", "postgresql"}:
                raise ValueError("storage.postgres.dsn must use the postgres:// or postgresql:// scheme")


@dataclass
class BackupsConfig:
    enabled: bool = True
    interval_seconds: float = 3600.0
    max_snapshots: int = 24
    create_startup_snapshot: bool = True
    warn_on_shared_storage: bool = True

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("backups.interval_seconds must be > 0")
        if self.max_snapshots < 1:
            raise ValueError("backups.max_snapshots must be >= 1")


@dataclass
class EmbeddingsConfig:
    provider: str = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    ollama_base_url: str = "http://localhost:11434"
    ollama_max_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("embeddings.batch_size must be >= 1")
        if self.ollama_max_concurrency < 1:
            raise ValueError("embeddings.ollama_max_concurrency must be >= 1")
        if self.provider not in ("sentence-transformers", "ollama"):
            raise ValueError(
                "embeddings.provider must be 'sentence-transformers' or 'ollama'"
            )


@dataclass
class LoggingConfig:
    max_runtime_logs: int = 5000
    max_log_age_days: int = 14
    retention_check_interval_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.max_runtime_logs < 1:
            raise ValueError("logging.max_runtime_logs must be >= 1")
        if self.max_log_age_days < 1:
            raise ValueError("logging.max_log_age_days must be >= 1")
        if self.retention_check_interval_seconds <= 0:
            raise ValueError("logging.retention_check_interval_seconds must be > 0")


@dataclass
class SearchRankingConfig:
    rrf_k: float = 60.0
    calibration_threshold: float = 0.035
    calibration_steepness: float = 150.0
    workspace_multiplier: float = 1.2
    semantic_only_abstain_threshold: float = 0.8
    semantic_only_keyword_penalty: float = 0.65
    keyword_coverage_floor: float = 0.6
    keyword_low_coverage_penalty: float = 0.4
    graph_expansion_only_penalty: float = 0.35
    degradation_multiplier: float = 0.3
    access_half_life_days: float = 7.0
    access_bonus_scale: float = 0.1
    authority_link_step: float = 0.02
    authority_link_cap: int = 10
    adaptive_result_max: int = 15
    adaptive_result_score_ratio_floor: float = 0.7
    adaptive_result_min_score: float = 0.35
    adaptive_result_max_score_gap: float = 0.08

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("search_ranking.rrf_k must be > 0")
        if not (0.0 <= self.calibration_threshold <= 1.0):
            raise ValueError("search_ranking.calibration_threshold must be in [0.0, 1.0]")
        if self.calibration_steepness <= 0:
            raise ValueError("search_ranking.calibration_steepness must be > 0")
        if self.workspace_multiplier < 1.0:
            raise ValueError("search_ranking.workspace_multiplier must be >= 1.0")
        if not (0.0 <= self.semantic_only_abstain_threshold <= 1.0):
            raise ValueError("search_ranking.semantic_only_abstain_threshold must be in [0.0, 1.0]")
        if not (0.0 <= self.semantic_only_keyword_penalty <= 1.0):
            raise ValueError("search_ranking.semantic_only_keyword_penalty must be in [0.0, 1.0]")
        if not (0.0 < self.keyword_coverage_floor <= 1.0):
            raise ValueError("search_ranking.keyword_coverage_floor must be in (0.0, 1.0]")
        if not (0.0 <= self.keyword_low_coverage_penalty <= 1.0):
            raise ValueError("search_ranking.keyword_low_coverage_penalty must be in [0.0, 1.0]")
        if not (0.0 <= self.graph_expansion_only_penalty <= 1.0):
            raise ValueError("search_ranking.graph_expansion_only_penalty must be in [0.0, 1.0]")
        if not (0.0 <= self.degradation_multiplier <= 1.0):
            raise ValueError("search_ranking.degradation_multiplier must be in [0.0, 1.0]")
        if self.access_half_life_days <= 0:
            raise ValueError("search_ranking.access_half_life_days must be > 0")
        if self.access_bonus_scale < 0:
            raise ValueError("search_ranking.access_bonus_scale must be >= 0")
        if self.authority_link_step < 0:
            raise ValueError("search_ranking.authority_link_step must be >= 0")
        if self.authority_link_cap < 0:
            raise ValueError("search_ranking.authority_link_cap must be >= 0")
        if self.adaptive_result_max < 1:
            raise ValueError("search_ranking.adaptive_result_max must be >= 1")
        if not (0.0 <= self.adaptive_result_score_ratio_floor <= 1.0):
            raise ValueError("search_ranking.adaptive_result_score_ratio_floor must be in [0.0, 1.0]")
        if not (0.0 <= self.adaptive_result_min_score <= 1.0):
            raise ValueError("search_ranking.adaptive_result_min_score must be in [0.0, 1.0]")
        if not (0.0 <= self.adaptive_result_max_score_gap <= 1.0):
            raise ValueError("search_ranking.adaptive_result_max_score_gap must be in [0.0, 1.0]")


@dataclass
class SearchKernelConfig:
    failure_mode: str = "lenient"
    calibrated_fusion_enabled: bool = False
    query_expansion_enabled: bool = False
    query_expansion_policy: SearchKernelExpansionPolicy = "vector"
    rerank_policy: SearchKernelRerankPolicy = "disabled"
    rerank_budget: int = 0

    def __post_init__(self) -> None:
        if self.failure_mode not in {"strict", "lenient"}:
            raise ValueError("searchkernel.failure_mode must be strict or lenient")
        if not isinstance(self.calibrated_fusion_enabled, bool):
            raise ValueError("searchkernel.calibrated_fusion_enabled must be a boolean")
        if not isinstance(self.query_expansion_enabled, bool):
            raise ValueError("searchkernel.query_expansion_enabled must be a boolean")
        if self.query_expansion_policy not in {"vector", "synonym"}:
            raise ValueError(
                "searchkernel.query_expansion_policy must be vector or synonym"
            )
        if self.rerank_policy not in {"disabled", "default"}:
            raise ValueError(
                "searchkernel.rerank_policy must be disabled or default"
            )
        if self.rerank_budget < 0:
            raise ValueError("searchkernel.rerank_budget must be >= 0")
        if self.rerank_policy == "disabled" and self.rerank_budget != 0:
            raise ValueError(
                "searchkernel.rerank_budget must be 0 when rerank_policy is disabled"
            )

    def active_feature_fingerprint(self) -> str | None:
        """Return only enabled policy inputs for derivative-cache isolation."""
        features: list[str] = []
        if self.calibrated_fusion_enabled:
            features.append("calibrated-fusion")
        if self.query_expansion_enabled:
            features.append(f"query-expansion:{self.query_expansion_policy}")
        if self.rerank_policy != "disabled":
            features.append(f"rerank:{self.rerank_policy}:{self.rerank_budget}")
        return None if not features else ";".join(features)


@dataclass
class Config:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    backups: BackupsConfig = field(default_factory=BackupsConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    search_ranking: SearchRankingConfig = field(default_factory=SearchRankingConfig)
    searchkernel: SearchKernelConfig = field(default_factory=SearchKernelConfig)
    provider_routing: ProviderRoutingConfig = field(default_factory=ProviderRoutingConfig)
    ingest_suppression: IngestSuppressionConfig = field(default_factory=IngestSuppressionConfig)
    ingest_escalation: IngestEscalationConfig = field(default_factory=IngestEscalationConfig)
    ingress_evidence_mode: IngressEvidenceMode = "off"
    ingress_replay_policy: IngressReplayPolicy = "legacy"
    ingress_quality_admission: IngressQualityAdmission = "disabled"
    curation: CurationConfig = field(default_factory=CurationConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)

    def __post_init__(self) -> None:
        if self.ingress_evidence_mode not in {"off", "shadow", "enforce"}:
            raise ValueError("ingress_evidence_mode must be 'off', 'shadow', or 'enforce'")
        if self.ingress_replay_policy not in {"legacy", "detect", "enforce"}:
            raise ValueError("ingress_replay_policy must be 'legacy', 'detect', or 'enforce'")
        if self.ingress_quality_admission not in {"disabled", "manual", "canary"}:
            raise ValueError("ingress_quality_admission must be 'disabled', 'manual', or 'canary'")


def _load_dataclass_from_dict(cls: type[Any], data: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in data:
            continue
        value = data[field_info.name]
        if isinstance(value, dict) and hasattr(field_info.type, "__dataclass_fields__"):
            value = _load_dataclass_from_dict(cast(type[Any], field_info.type), value)
        kwargs[field_info.name] = value
    return cls(**kwargs)


def _load_memory_config(data: dict[str, Any]) -> MemoryConfig:
    kwargs: dict[str, Any] = {}
    for key in {"enabled", "checkpoint_interval_ops", "checkpoint_interval_secs"}:
        if key in data:
            kwargs[key] = data[key]

    for key in {
        "recency_journal",
        "recency_plan",
        "recency_fact",
        "recency_observation",
        "recency_reflection",
    }:
        if key in data and isinstance(data[key], dict):
            kwargs[key] = _load_dataclass_from_dict(MemoryRecencyConfig, data[key])

    return MemoryConfig(**kwargs)


def _load_provider_routing_config(data: dict[str, Any]) -> ProviderRoutingConfig:
    profiles: dict[str, AIConfig] = {}
    raw_profiles = data.get("profiles", {})
    if isinstance(raw_profiles, dict):
        for key, value in raw_profiles.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            profiles[key] = _load_dataclass_from_dict(AIConfig, value)

    def _normalize_route_map(value: object) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for key, items in value.items():
            if not isinstance(key, str):
                continue
            if isinstance(items, list):
                normalized[key] = [str(item).strip() for item in items if isinstance(item, str) and str(item).strip()]
            elif isinstance(items, str) and items.strip():
                normalized[key] = [items.strip()]
        return normalized

    def _normalize_route_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    raw_limits = data.get("profile_daily_call_limits", {})
    profile_daily_call_limits = {
        str(key): int(value)
        for key, value in raw_limits.items()
        if isinstance(raw_limits, dict) and isinstance(key, str) and isinstance(value, int)
    }

    return ProviderRoutingConfig(
        profiles=profiles,
        task_routes=_normalize_route_map(data.get("task_routes", {})),
        task_classes={
            str(key).strip(): str(value).strip()
            for key, value in data.get("task_classes", {}).items()
            if isinstance(data.get("task_classes", {}), dict) and isinstance(key, str) and isinstance(value, str)
        },
        profile_daily_call_limits=profile_daily_call_limits,
        model_burst_call_limit=int(data.get("model_burst_call_limit", 2)),
        model_burst_window_seconds=float(data.get("model_burst_window_seconds", 300.0)),
        default_json_route=_normalize_route_list(data.get("default_json_route", [])),
        default_agentic_route=_normalize_route_list(data.get("default_agentic_route", [])),
        fallback_to_json_only=bool(data.get("fallback_to_json_only", False)),
        low_priority_task_names=_normalize_route_list(data.get("low_priority_task_names", [])),
    )


def _load_storage_config(data: dict[str, Any]) -> StorageConfig:
    sqlite_data = data.get("sqlite", {})
    postgres_data = data.get("postgres", {})
    cache_data = data.get("cache", {})
    return StorageConfig(
        backend=cast(StorageBackendKind, str(data.get("backend", "sqlite"))),
        sqlite=_load_dataclass_from_dict(SQLiteStorageConfig, sqlite_data if isinstance(sqlite_data, dict) else {}),
        postgres=_load_dataclass_from_dict(PostgresStorageConfig, postgres_data if isinstance(postgres_data, dict) else {}),
        cache=_load_dataclass_from_dict(StorageCacheConfig, cache_data if isinstance(cache_data, dict) else {}),
    )


def _default_provider_routing_data() -> dict[str, Any]:
    return {
        "profiles": {
            "copilot-strong": {
                "provider": "copilot-sdk",
                "model": "gpt-5.6-luna",
                "timeout_seconds": 900,
                "max_retries": 0,
            },
            "copilot-mini": {
                "provider": "copilot-sdk",
                "model": "gpt-5.6-luna",
                "timeout_seconds": 900,
                "max_retries": 0,
            },
        },
        "task_routes": {
            "ingest-system1": ["copilot-mini"],
            "deduplicator": ["copilot-mini"],
            "memory-curator": ["copilot-strong", "copilot-mini"],
        },
        "task_classes": {
            "ingest-system1": "cheap_agentic",
            "deduplicator": "cheap_agentic",
            "memory-curator": "premium_agentic",
            "graph-linker": "cheap_json",
            "conflict-detector": "cheap_json",
            "defragmenter": "cheap_json",
            "taxonomist": "cheap_json",
            "fact-checker": "deterministic",
            "project-manager": "deterministic",
            "summarize-memory": "deterministic",
            "sweeper": "deterministic",
        },
        "profile_daily_call_limits": {
            "copilot-strong": 20,
            "copilot-mini": 50,
        },
        "model_burst_call_limit": 2,
        "model_burst_window_seconds": 300.0,
        "default_json_route": ["copilot-mini"],
        "default_agentic_route": ["copilot-mini"],
        "fallback_to_json_only": False,
        "low_priority_task_names": ["graph-linker", "conflict-detector", "defragmenter", "taxonomist"],
    }


def _merge_provider_routing_defaults(raw_provider_routing: object) -> dict[str, Any]:
    defaults = _default_provider_routing_data()
    if not isinstance(raw_provider_routing, dict):
        return defaults
    raw_provider_routing = cast(dict[str, Any], raw_provider_routing)

    merged = dict(defaults)
    for key in (
        "model_burst_call_limit",
        "model_burst_window_seconds",
        "default_json_route",
        "default_agentic_route",
        "fallback_to_json_only",
        "low_priority_task_names",
    ):
        if key in raw_provider_routing:
            merged[key] = raw_provider_routing[key]

    for key in ("profiles", "task_routes", "task_classes", "profile_daily_call_limits"):
        default_mapping = defaults.get(key, {})
        raw_mapping = raw_provider_routing.get(key)
        if isinstance(default_mapping, dict) and isinstance(raw_mapping, dict):
            merged[key] = {**default_mapping, **raw_mapping}
        elif key in raw_provider_routing:
            merged[key] = raw_mapping

    return merged


def _provider_routing_data_for_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Resolve provider routing while honoring the legacy top-level AI switch.

    Older configs used ``[ai].provider = "none"`` without a routing section.
    In that shape, injecting the modern routing defaults would unexpectedly
    construct Copilot providers.  An explicitly present ``provider_routing``
    section remains authoritative and keeps its normal default backfilling.
    """
    raw_provider_routing = raw.get("provider_routing")
    raw_ai = raw.get("ai")
    if (
        "provider_routing" not in raw
        and isinstance(raw_ai, dict)
        and raw_ai.get("provider") == "none"
    ):
        return {}
    return _merge_provider_routing_defaults(raw_provider_routing)


def _load_ingest_suppression_config(data: dict[str, Any]) -> IngestSuppressionConfig:
    windows: list[IngestSuppressionWindow] = []
    raw_windows = data.get("windows", [])
    if isinstance(raw_windows, list):
        for item in raw_windows:
            if not isinstance(item, dict):
                continue
            days = item.get("days_of_week", [])
            windows.append(
                IngestSuppressionWindow(
                    start_hour=int(item.get("start_hour", 0)),
                    end_hour=int(item.get("end_hour", 0)),
                    days_of_week=[int(day) for day in days] if isinstance(days, list) else [],
                )
            )
    return IngestSuppressionConfig(
        enabled=bool(data.get("enabled", False)),
        windows=windows,
    )


def _load_ingest_escalation_config(data: dict[str, Any]) -> IngestEscalationConfig:
    return IngestEscalationConfig(
        enabled=bool(data.get("enabled", True)),
        deterministic_first=bool(data.get("deterministic_first", True)),
        agentic_pending_count_threshold=int(data.get("agentic_pending_count_threshold", 8)),
        novelty_threshold=float(data.get("novelty_threshold", 0.6)),
        preview_entry_limit=int(data.get("preview_entry_limit", 8)),
    )


def resolve_default_config_path() -> Path:
    return Path.home() / ".config" / DEFAULT_APP_NAME / "config.toml"


def resolve_global_data_dir() -> Path:
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share"))


def resolve_state_dir() -> Path:
    return Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / DEFAULT_APP_NAME


def resolve_backup_dir() -> Path:
    return resolve_global_data_dir() / DEFAULT_APP_NAME / "backups"


def ensure_default_config_exists(config_path: Path | None = None) -> Path:
    resolved_path = config_path or resolve_default_config_path()
    if resolved_path.exists():
        return resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    document = tomlkit.document()
    document["storage"] = {
        "backend": "sqlite",
        "sqlite": {
            "path": "",
        },
        "postgres": {
            "dsn": "",
            "pool_min": 1,
            "pool_max": 10,
            "statement_timeout_ms": 30000,
            "lock_timeout_ms": 5000,
            "application_name": DEFAULT_APP_NAME,
        },
        "cache": {
            "enabled": False,
            "mode": "readonly",
            "max_cached_search_docs": 50000,
            "max_outbox_entries": 10000,
        },
    }
    document["daemon"] = {
        "host": "127.0.0.1",
        "port": 4242,
        "auto_start_timeout_seconds": 10.0,
        "shutdown_grace_seconds": 5.0,
        "healthcheck_interval_seconds": 0.05,
    }
    document["backups"] = {
        "enabled": True,
        "interval_seconds": 3600.0,
        "max_snapshots": 24,
        "create_startup_snapshot": True,
        "warn_on_shared_storage": True,
    }
    document["maintenance"] = {
        "archived_memory_retention_days": 90,
        "memory_gc_batch_size": 100,
        "dangling_link_gc_batch_size": 100,
        "memory_gc_mode": "report-only",
    }
    document["embeddings"] = {
        "provider": "sentence-transformers",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
        "ollama_base_url": "http://localhost:11434",
        "ollama_max_concurrency": 1,
    }
    document["logging"] = {
        "max_runtime_logs": 5000,
        "max_log_age_days": 14,
        "retention_check_interval_seconds": 60.0,
    }
    document["search_ranking"] = {
        "rrf_k": 60.0,
        "calibration_threshold": 0.035,
        "calibration_steepness": 150.0,
        "workspace_multiplier": 1.2,
        "degradation_multiplier": 0.3,
        "access_half_life_days": 7.0,
        "access_bonus_scale": 0.1,
        "authority_link_step": 0.02,
        "authority_link_cap": 10,
        "adaptive_result_max": 15,
        "adaptive_result_score_ratio_floor": 0.7,
        "adaptive_result_min_score": 0.35,
        "adaptive_result_max_score_gap": 0.08,
    }
    document["searchkernel"] = {
        "failure_mode": "lenient",
        "calibrated_fusion_enabled": False,
        "query_expansion_enabled": False,
        "query_expansion_policy": "vector",
        "rerank_policy": "disabled",
        "rerank_budget": 0,
    }
    document["provider_routing"] = _default_provider_routing_data()
    document["ingest_suppression"] = {
        "enabled": False,
        "windows": [],
    }
    document["ingest_escalation"] = {
        "enabled": True,
        "deterministic_first": True,
        "agentic_pending_count_threshold": 8,
        "novelty_threshold": 0.6,
        "preview_entry_limit": 8,
    }
    document["ingress_evidence_mode"] = "off"
    document["ingress_replay_policy"] = "legacy"
    document["ingress_quality_admission"] = "disabled"
    memory_table = tomlkit.table()
    memory_table.update({
        "enabled": True,
        "checkpoint_interval_ops": 10,
        "checkpoint_interval_secs": 300,
    })
    document["memory"] = memory_table
    for key, value in {
        "recency_journal": (14, 0.2, 0.95),
        "recency_plan": (30, 0.15, 0.97),
        "recency_fact": (180, 0.05, 0.99),
        "recency_observation": (14, 0.2, 0.95),
        "recency_reflection": (180, 0.05, 0.99),
    }.items():
        memory_table[key] = {
            "boost_window_days": value[0],
            "max_boost_amount": value[1],
            "boost_decay_rate": value[2],
        }

    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=resolved_path.parent,
        delete=False,
        suffix=".tmp",
        encoding="utf-8",
    ) as handle:
        tmp_path = Path(handle.name)
        tomlkit.dump(document, handle)

    tmp_path.replace(resolved_path)
    logger.info("Created default config at %s", resolved_path)
    return resolved_path


def load_config(config_path: Path | None = None) -> Config:
    resolved_path = ensure_default_config_exists(config_path)
    with resolved_path.open("rb") as handle:
        raw = tomllib.load(handle)

    return Config(
        memory=_load_memory_config(raw.get("memory", {})),
        storage=_load_storage_config(raw.get("storage", {})),
        daemon=_load_dataclass_from_dict(DaemonConfig, raw.get("daemon", {})),
        backups=_load_dataclass_from_dict(BackupsConfig, raw.get("backups", {})),
        embeddings=_load_dataclass_from_dict(EmbeddingsConfig, raw.get("embeddings", {})),
        logging=_load_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
        search_ranking=_load_dataclass_from_dict(SearchRankingConfig, raw.get("search_ranking", {})),
        searchkernel=_load_dataclass_from_dict(SearchKernelConfig, raw.get("searchkernel", {})),
        provider_routing=_load_provider_routing_config(_provider_routing_data_for_config(raw)),
        ingest_suppression=_load_ingest_suppression_config(raw.get("ingest_suppression", {})),
        ingest_escalation=_load_ingest_escalation_config(raw.get("ingest_escalation", {})),
        ingress_evidence_mode=cast(IngressEvidenceMode, raw.get("ingress_evidence_mode", "off")),
        ingress_replay_policy=cast(IngressReplayPolicy, raw.get("ingress_replay_policy", "legacy")),
        ingress_quality_admission=cast(IngressQualityAdmission, raw.get("ingress_quality_admission", "disabled")),
        curation=_load_dataclass_from_dict(CurationConfig, raw.get("curation", {})),
        maintenance=_load_dataclass_from_dict(MaintenanceConfig, raw.get("maintenance", {})),
    )


def resolve_workspace_root(
    cwd: Path | None = None,
    workspace_root: str | None = None,
) -> Path:
    base_path = Path(workspace_root).expanduser().resolve() if workspace_root else (cwd or Path.cwd()).expanduser().resolve()
    git_root = _find_git_root(base_path)
    return git_root or base_path


def resolve_workspace_id(
    cwd: Path | None = None,
    workspace_root: str | None = None,
) -> str:
    resolved_root = resolve_workspace_root(cwd, workspace_root)
    digest = sha1(str(resolved_root).encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", resolved_root.name or "workspace")
    slug = re.sub(r"-+", "-", slug).strip("-") or "workspace"
    return f"{slug}-{digest}"


def resolve_memory_path(config: Config) -> Path:
    configured_path = config.storage.sqlite.path.strip()
    if configured_path:
        return Path(configured_path).expanduser()
    return resolve_global_data_dir() / DEFAULT_APP_NAME / "memories"


def resolve_daemon_metadata_path(workspace_id: str | None = None) -> Path:
    del workspace_id
    metadata_dir = resolve_state_dir() / "daemons"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / "daemon.json"


def resolve_daemon_startup_log_path(workspace_id: str | None = None) -> Path:
    del workspace_id
    daemon_dir = resolve_state_dir() / "daemons"
    daemon_dir.mkdir(parents=True, exist_ok=True)
    return daemon_dir / "daemon.log"


def resolve_daemon_lock_path(workspace_id: str | None = None) -> Path:
    del workspace_id
    lock_dir = resolve_state_dir() / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / "daemon.lock"


def resolve_daemon_socket_path() -> Path:
    socket_dir = resolve_state_dir() / "sockets"
    socket_dir.mkdir(parents=True, exist_ok=True)
    preferred = socket_dir / "daemon.sock"
    if len(str(preferred)) <= 80:
        return preferred
    digest = sha1(str(resolve_state_dir()).encode("utf-8")).hexdigest()[:12]
    return Path("/tmp") / f"mcp-memory-{digest}.sock"


def _find_git_root(start_path: Path) -> Path | None:
    current = start_path if start_path.is_dir() else start_path.parent
    while True:
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent

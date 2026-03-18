from __future__ import annotations

from dataclasses import dataclass, field, fields
from hashlib import sha1
from pathlib import Path
from typing import Any, cast
import logging
import os
import re
import tempfile
import tomllib

import tomlkit


logger = logging.getLogger(__name__)

DEFAULT_APP_NAME = "mcp-memory"
GLOBAL_DAEMON_IDENTITY = "global"


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
    model: str = "gemini-3-flash-preview"
    timeout_seconds: float = 900.0
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.provider not in {"none", "gemini-cli", "copilot-cli", "opencode", "ollama"}:
            raise ValueError("ai.provider must be one of 'none', 'gemini-cli', 'copilot-cli', 'opencode', or 'ollama'")
        if self.timeout_seconds <= 0:
            raise ValueError("ai.timeout_seconds must be > 0")
        if self.max_retries < 0:
            raise ValueError("ai.max_retries must be >= 0")


@dataclass
class ProviderRoutingConfig:
    profiles: dict[str, AIConfig] = field(default_factory=dict)
    task_routes: dict[str, list[str]] = field(default_factory=dict)
    profile_daily_call_limits: dict[str, int] = field(default_factory=dict)
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
        self.profile_daily_call_limits = {
            str(profile_key).strip(): int(limit)
            for profile_key, limit in self.profile_daily_call_limits.items()
            if isinstance(profile_key, str) and profile_key.strip()
        }
        for profile_key, limit in self.profile_daily_call_limits.items():
            if limit < 1:
                raise ValueError(f"provider_routing.profile_daily_call_limits[{profile_key!r}] must be >= 1")
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
class GeminiCLIConfig:
    command: str = "gemini"


@dataclass
class CopilotCLIConfig:
    command: str = "copilot"


@dataclass
class OpenCodeCLIConfig:
    command: str = "opencode"


@dataclass
class OllamaCLIConfig:
    command: str = "ollama"


@dataclass
class DaemonConfig:
    host: str = "127.0.0.1"
    auto_start_timeout_seconds: float = 10.0
    shutdown_grace_seconds: float = 5.0
    healthcheck_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.auto_start_timeout_seconds <= 0:
            raise ValueError("daemon.auto_start_timeout_seconds must be > 0")
        if self.shutdown_grace_seconds < 0:
            raise ValueError("daemon.shutdown_grace_seconds must be >= 0")
        if self.healthcheck_interval_seconds <= 0:
            raise ValueError("daemon.healthcheck_interval_seconds must be > 0")


@dataclass
class EmbeddingsConfig:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("embeddings.batch_size must be >= 1")


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
    degradation_multiplier: float = 0.3
    access_half_life_days: float = 7.0
    access_bonus_scale: float = 0.1
    authority_link_step: float = 0.02
    authority_link_cap: int = 10

    def __post_init__(self) -> None:
        if self.rrf_k <= 0:
            raise ValueError("search_ranking.rrf_k must be > 0")
        if not (0.0 <= self.calibration_threshold <= 1.0):
            raise ValueError("search_ranking.calibration_threshold must be in [0.0, 1.0]")
        if self.calibration_steepness <= 0:
            raise ValueError("search_ranking.calibration_steepness must be > 0")
        if self.workspace_multiplier < 1.0:
            raise ValueError("search_ranking.workspace_multiplier must be >= 1.0")
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


@dataclass
class Config:
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    gemini_cli: GeminiCLIConfig = field(default_factory=GeminiCLIConfig)
    copilot_cli: CopilotCLIConfig = field(default_factory=CopilotCLIConfig)
    opencode: OpenCodeCLIConfig = field(default_factory=OpenCodeCLIConfig)
    ollama: OllamaCLIConfig = field(default_factory=OllamaCLIConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    search_ranking: SearchRankingConfig = field(default_factory=SearchRankingConfig)
    provider_routing: ProviderRoutingConfig = field(default_factory=ProviderRoutingConfig)
    ingest_suppression: IngestSuppressionConfig = field(default_factory=IngestSuppressionConfig)
    ingest_escalation: IngestEscalationConfig = field(default_factory=IngestEscalationConfig)


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
        profile_daily_call_limits=profile_daily_call_limits,
        default_json_route=_normalize_route_list(data.get("default_json_route", [])),
        default_agentic_route=_normalize_route_list(data.get("default_agentic_route", [])),
        fallback_to_json_only=bool(data.get("fallback_to_json_only", False)),
        low_priority_task_names=_normalize_route_list(data.get("low_priority_task_names", [])),
    )


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


def ensure_default_config_exists(config_path: Path | None = None) -> Path:
    resolved_path = config_path or resolve_default_config_path()
    if resolved_path.exists():
        return resolved_path

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    document = tomlkit.document()
    document["ai"] = {
        "provider": "none",
        "model": "gemini-3-flash-preview",
        "timeout_seconds": 900,
        "max_retries": 0,
    }
    document["gemini_cli"] = {"command": "gemini"}
    document["copilot_cli"] = {"command": "copilot"}
    document["opencode"] = {"command": "opencode"}
    document["ollama"] = {"command": "ollama"}
    document["daemon"] = {
        "host": "127.0.0.1",
        "auto_start_timeout_seconds": 10.0,
        "shutdown_grace_seconds": 5.0,
        "healthcheck_interval_seconds": 0.05,
    }
    document["embeddings"] = {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 32,
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
    }
    document["provider_routing"] = {
        "profiles": {
            "copilot-strong": {
                "provider": "copilot-cli",
                "model": "gpt-5.4",
                "timeout_seconds": 900,
                "max_retries": 0,
            },
            "copilot-mini": {
                "provider": "copilot-cli",
                "model": "gpt-5-mini",
                "timeout_seconds": 900,
                "max_retries": 0,
            },
            "gemini-cheap": {
                "provider": "gemini-cli",
                "model": "gemini-3-flash-preview",
                "timeout_seconds": 900,
                "max_retries": 0,
            }
        },
        "task_routes": {
            "ingest-system1": ["copilot-mini", "gemini-cheap"],
            "deduplicator": ["copilot-mini", "gemini-cheap"],
            "memory-curator": ["copilot-strong", "gemini-cheap"],
        },
        "profile_daily_call_limits": {
            "copilot-strong": 20,
            "copilot-mini": 50,
            "gemini-cheap": 100,
        },
        "default_json_route": [],
        "default_agentic_route": [],
        "fallback_to_json_only": False,
        "low_priority_task_names": [],
    }
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
        ai=_load_dataclass_from_dict(AIConfig, raw.get("ai", {})),
        gemini_cli=_load_dataclass_from_dict(GeminiCLIConfig, raw.get("gemini_cli", {})),
        copilot_cli=_load_dataclass_from_dict(CopilotCLIConfig, raw.get("copilot_cli", {})),
        opencode=_load_dataclass_from_dict(OpenCodeCLIConfig, raw.get("opencode", {})),
        ollama=_load_dataclass_from_dict(OllamaCLIConfig, raw.get("ollama", {})),
        daemon=_load_dataclass_from_dict(DaemonConfig, raw.get("daemon", {})),
        embeddings=_load_dataclass_from_dict(EmbeddingsConfig, raw.get("embeddings", {})),
        logging=_load_dataclass_from_dict(LoggingConfig, raw.get("logging", {})),
        search_ranking=_load_dataclass_from_dict(SearchRankingConfig, raw.get("search_ranking", {})),
        provider_routing=_load_provider_routing_config(raw.get("provider_routing", {})),
        ingest_suppression=_load_ingest_suppression_config(raw.get("ingest_suppression", {})),
        ingest_escalation=_load_ingest_escalation_config(raw.get("ingest_escalation", {})),
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
    del config
    return resolve_global_data_dir() / DEFAULT_APP_NAME / "memories"


def resolve_daemon_metadata_path(workspace_id: str | None = None) -> Path:
    del workspace_id
    metadata_dir = resolve_state_dir() / "daemons"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / "daemon.json"


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

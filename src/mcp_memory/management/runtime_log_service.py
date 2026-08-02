from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from mcp_memory.management.memory_service import _USE_SERVICE_WORKSPACE
from mcp_memory.management.models import (
    RuntimeLogListPayload,
    RuntimeLogPayload,
    RuntimeLogPrunePayload,
    RuntimeLogSummaryPayload,
)
from mcp_memory.runtime_log_store import _AllWorkspacesSentinel


@dataclass(frozen=True)
class RuntimeLogServiceDependencies:
    db_manager: Any
    runtime_logs: Any
    workspace_id: str | None
    dashboard_static_root: Path
    dashboard_static_path: Path
    dashboard_dist_path: Path
    dashboard_asset_root: Path


def _resolve_log_workspace_id(
    service_workspace_id: str | None,
    workspace_id: str | None | object,
) -> str | None | _AllWorkspacesSentinel:
    if workspace_id is _USE_SERVICE_WORKSPACE:
        return service_workspace_id
    return cast(str | None | _AllWorkspacesSentinel, workspace_id)


def _ensure_dashboard_base_href(html: str) -> str:
    if "<base " in html:
        return html
    if "</head>" not in html:
        return html
    return html.replace("</head>", '    <base href="/dashboard/">\n  </head>', 1)


class RuntimeLogService:
    def __init__(self, dependencies: RuntimeLogServiceDependencies) -> None:
        self._dependencies = dependencies

    @property
    def dashboard_static_root(self) -> Path:
        return self._dependencies.dashboard_static_root

    def list_logs(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
    ) -> RuntimeLogListPayload:
        dependencies = self._dependencies
        if dependencies.db_manager is None:
            return RuntimeLogListPayload()
        effective_workspace_id = _resolve_log_workspace_id(dependencies.workspace_id, workspace_id)
        return RuntimeLogListPayload(
            logs=[
                RuntimeLogPayload(
                    id=record.id,
                    created_at=record.created_at,
                    level=record.level,
                    logger_name=record.logger_name,
                    source=record.source,
                    message=record.message,
                    data=record.data,
                )
                for record in dependencies.runtime_logs.list_logs(
                    workspace_id=effective_workspace_id,
                    level=level,
                    logger_name=logger_name,
                    source=source,
                    query=query,
                    after=after,
                    before=before,
                    limit=limit,
                )
            ]
        )

    def summarize_logs(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> RuntimeLogSummaryPayload:
        dependencies = self._dependencies
        if dependencies.db_manager is None:
            return RuntimeLogSummaryPayload()
        effective_workspace_id = _resolve_log_workspace_id(dependencies.workspace_id, workspace_id)
        summary = dependencies.runtime_logs.summarize_logs(
            workspace_id=effective_workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        return RuntimeLogSummaryPayload(
            total=summary.total,
            by_level=summary.by_level,
            by_source=summary.by_source,
        )

    def prune_logs(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        max_runtime_logs: int | None = None,
        max_log_age_days: int | None = None,
    ) -> RuntimeLogPrunePayload:
        dependencies = self._dependencies
        config = dependencies.runtime_logs.retention_policy
        effective_workspace_id = _resolve_log_workspace_id(dependencies.workspace_id, workspace_id)
        deleted = dependencies.runtime_logs.prune_logs(
            workspace_id=effective_workspace_id,
            max_runtime_logs=max_runtime_logs,
            max_age_days=max_log_age_days,
        )
        resolved_max_runtime_logs = config.max_runtime_logs if max_runtime_logs is None else max_runtime_logs
        resolved_max_log_age_days = config.max_log_age_days if max_log_age_days is None else max_log_age_days
        return RuntimeLogPrunePayload(
            deleted=deleted,
            max_runtime_logs=resolved_max_runtime_logs,
            max_log_age_days=resolved_max_log_age_days,
        )

    def load_dashboard_html(self) -> str:
        dependencies = self._dependencies
        html_path = (
            dependencies.dashboard_dist_path
            if dependencies.dashboard_dist_path.exists()
            else dependencies.dashboard_static_path
        )
        return _ensure_dashboard_base_href(html_path.read_text(encoding="utf-8"))

    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None:
        asset_root = self._dependencies.dashboard_asset_root
        if not asset_path.strip() or not asset_root.exists():
            return None
        candidate = (asset_root / asset_path).resolve()
        resolved_asset_root = asset_root.resolve()
        if resolved_asset_root not in candidate.parents and candidate != resolved_asset_root:
            return None
        if not candidate.is_file():
            return None
        return candidate

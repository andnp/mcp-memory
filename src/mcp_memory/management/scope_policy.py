from __future__ import annotations

from enum import Enum


class ScopePolicyKind(str, Enum):
    GLOBAL_ONLY = "global_only"
    GLOBAL_DEFAULT_FILTERABLE = "global_default_filterable"
    GLOBAL_DEFAULT_RANKING_CONTEXT = "global_default_ranking_context"
    SERVICE_SCOPED_DEFAULT = "service_scoped_default"


GLOBAL_ONLY_ENDPOINTS = frozenset({
    "/api/health",
    "/api/overview",
})

GLOBAL_DEFAULT_FILTERABLE_ENDPOINTS = frozenset({
    "/api/metrics/nerd",
    "/api/quality-cleanup",
    "/api/selector-stats",
    "/api/tasks",
    "/api/memories",
    "/api/logs",
    "/api/logs/summary",
    "/api/ai-conversations",
    "/api/admin/logs/prune",
})

GLOBAL_DEFAULT_RANKING_CONTEXT_ENDPOINTS = frozenset({
    "/api/memories/search",
})


def scope_policy_for_endpoint(path: str) -> ScopePolicyKind:
    if path in GLOBAL_ONLY_ENDPOINTS:
        return ScopePolicyKind.GLOBAL_ONLY
    if path in GLOBAL_DEFAULT_RANKING_CONTEXT_ENDPOINTS:
        return ScopePolicyKind.GLOBAL_DEFAULT_RANKING_CONTEXT
    if path in GLOBAL_DEFAULT_FILTERABLE_ENDPOINTS:
        return ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE
    return ScopePolicyKind.SERVICE_SCOPED_DEFAULT


def resolve_workspace_id_for_policy(
    policy: ScopePolicyKind,
    *,
    scope: str | None,
    workspace_id: str | None,
    current_workspace_id: str | None,
) -> str | None:
    if policy is ScopePolicyKind.GLOBAL_ONLY:
        return None

    if workspace_id is not None:
        return workspace_id

    if policy in {
        ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE,
        ScopePolicyKind.GLOBAL_DEFAULT_RANKING_CONTEXT,
    }:
        if scope is None or scope == "global":
            return None
        if scope == "workspace":
            return require_current_workspace_id(current_workspace_id)
        raise ValueError("scope_must_be_global_or_workspace")

    if policy is ScopePolicyKind.SERVICE_SCOPED_DEFAULT:
        if scope is None:
            return current_workspace_id
        if scope == "global":
            return None
        if scope == "workspace":
            return require_current_workspace_id(current_workspace_id)
        raise ValueError("scope_must_be_global_or_workspace")

    raise ValueError(f"unsupported_scope_policy:{policy}")


def require_current_workspace_id(current_workspace_id: str | None) -> str:
    if current_workspace_id is None:
        raise ValueError("workspace_id_required_for_workspace_scope")
    return current_workspace_id
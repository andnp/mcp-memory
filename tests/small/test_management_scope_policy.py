from types import SimpleNamespace
from typing import cast

import pytest

import mcp_memory.cli as cli
from mcp_memory.management.scope_policy import (
    ScopePolicyKind,
    resolve_workspace_id_for_policy,
    scope_policy_for_endpoint,
)
from mcp_memory.management.service import ManagementService

pytestmark = pytest.mark.small


@pytest.mark.parametrize(
    ("scope", "workspace_id"),
    [
        (None, None),
        ("global", None),
        ("workspace", None),
        ("global", "workspace-b"),
        ("workspace", "workspace-b"),
    ],
)
def test_global_only_policy_always_resolves_to_none(scope: str | None, workspace_id: str | None) -> None:
    assert resolve_workspace_id_for_policy(
        ScopePolicyKind.GLOBAL_ONLY,
        scope=scope,
        workspace_id=workspace_id,
        current_workspace_id="workspace-a",
    ) is None


@pytest.mark.parametrize(
    ("scope", "workspace_id", "current_workspace_id", "expected"),
    [
        (None, None, "workspace-a", None),
        ("global", None, "workspace-a", None),
        ("workspace", None, "workspace-a", "workspace-a"),
        ("workspace", "workspace-b", "workspace-a", "workspace-b"),
        ("global", "workspace-b", "workspace-a", "workspace-b"),
    ],
)
def test_global_default_filterable_policy_preserves_precedence(
    scope: str | None,
    workspace_id: str | None,
    current_workspace_id: str | None,
    expected: str | None,
) -> None:
    assert resolve_workspace_id_for_policy(
        ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE,
        scope=scope,
        workspace_id=workspace_id,
        current_workspace_id=current_workspace_id,
    ) == expected


@pytest.mark.parametrize(
    ("scope", "workspace_id", "current_workspace_id", "expected"),
    [
        (None, None, "workspace-a", None),
        ("global", None, "workspace-a", None),
        ("workspace", None, "workspace-a", "workspace-a"),
        ("workspace", "workspace-b", "workspace-a", "workspace-b"),
        ("global", "workspace-b", "workspace-a", "workspace-b"),
    ],
)
def test_global_default_ranking_context_policy_preserves_precedence(
    scope: str | None,
    workspace_id: str | None,
    current_workspace_id: str | None,
    expected: str | None,
) -> None:
    assert resolve_workspace_id_for_policy(
        ScopePolicyKind.GLOBAL_DEFAULT_RANKING_CONTEXT,
        scope=scope,
        workspace_id=workspace_id,
        current_workspace_id=current_workspace_id,
    ) == expected


def test_global_default_filterable_policy_rejects_invalid_scope() -> None:
    with pytest.raises(ValueError, match="scope_must_be_global_or_workspace"):
        resolve_workspace_id_for_policy(
            ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE,
            scope="bogus",
            workspace_id=None,
            current_workspace_id="workspace-a",
        )


def test_global_default_filterable_policy_requires_current_workspace_for_workspace_scope() -> None:
    with pytest.raises(ValueError, match="workspace_id_required_for_workspace_scope"):
        resolve_workspace_id_for_policy(
            ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE,
            scope="workspace",
            workspace_id=None,
            current_workspace_id=None,
        )


def test_global_default_ranking_context_policy_requires_current_workspace_for_workspace_scope() -> None:
    with pytest.raises(ValueError, match="workspace_id_required_for_workspace_scope"):
        resolve_workspace_id_for_policy(
            ScopePolicyKind.GLOBAL_DEFAULT_RANKING_CONTEXT,
            scope="workspace",
            workspace_id=None,
            current_workspace_id=None,
        )


@pytest.mark.parametrize(
    ("scope", "workspace_id", "current_workspace_id", "expected"),
    [
        (None, None, "workspace-a", "workspace-a"),
        ("global", None, "workspace-a", None),
        ("workspace", None, "workspace-a", "workspace-a"),
        (None, "workspace-b", "workspace-a", "workspace-b"),
        ("workspace", "workspace-b", None, "workspace-b"),
    ],
)
def test_service_scoped_default_policy_preserves_precedence(
    scope: str | None,
    workspace_id: str | None,
    current_workspace_id: str | None,
    expected: str | None,
) -> None:
    assert resolve_workspace_id_for_policy(
        ScopePolicyKind.SERVICE_SCOPED_DEFAULT,
        scope=scope,
        workspace_id=workspace_id,
        current_workspace_id=current_workspace_id,
    ) == expected


def test_service_scoped_default_policy_requires_current_workspace_for_workspace_scope() -> None:
    with pytest.raises(ValueError, match="workspace_id_required_for_workspace_scope"):
        resolve_workspace_id_for_policy(
            ScopePolicyKind.SERVICE_SCOPED_DEFAULT,
            scope="workspace",
            workspace_id=None,
            current_workspace_id=None,
        )


def test_scope_policy_for_endpoint_matches_current_management_contract() -> None:
    assert scope_policy_for_endpoint("/api/health") is ScopePolicyKind.GLOBAL_ONLY
    assert scope_policy_for_endpoint("/api/overview") is ScopePolicyKind.GLOBAL_ONLY
    assert scope_policy_for_endpoint("/api/memories/search") is ScopePolicyKind.GLOBAL_DEFAULT_RANKING_CONTEXT
    assert scope_policy_for_endpoint("/api/logs") is ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE
    assert scope_policy_for_endpoint("/api/admin/search/repair") is ScopePolicyKind.SERVICE_SCOPED_DEFAULT


def test_cli_operator_workspace_filter_uses_global_default_policy_and_trims_explicit_workspace_id() -> None:
    service = cast(ManagementService, SimpleNamespace(workspace_id="workspace-a"))

    assert cli._resolve_operator_workspace_filter(service, scope="global", workspace_id=None) is None
    assert cli._resolve_operator_workspace_filter(service, scope="workspace", workspace_id=None) == "workspace-a"
    assert cli._resolve_operator_workspace_filter(service, scope="workspace", workspace_id="  workspace-b  ") == "workspace-b"


def test_cli_operator_workspace_filter_requires_current_workspace_for_workspace_scope() -> None:
    with pytest.raises(ValueError, match="workspace_id_required_for_workspace_scope"):
        cli._resolve_operator_workspace_filter(
            cast(ManagementService, SimpleNamespace(workspace_id=None)),
            scope="workspace",
            workspace_id=None,
        )
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcp_memory.embedding_integrity_event_store import EmbeddingIntegrityEventRepository
from mcp_memory.management.models import TransportDiagnosticsPayload
from mcp_memory.management.service import (
    _USE_SERVICE_WORKSPACE,
    _build_default_embedding_integrity_events,
    _build_default_provider_usage,
    _coerce_transport_diagnostics_payload,
    _embedding_integrity_snapshot_payload,
    _resolve_log_workspace_id,
    _resolve_service_workspace_id,
)
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.storage.noop import NoopProviderUsageRepository
from mcp_memory.storage.postgres_embedding_integrity_event_store import PostgresEmbeddingIntegrityEventRepository

pytestmark = pytest.mark.small


class _SQLiteDbManagerStub:
    def get_connection(self) -> object:
        raise AssertionError("helper construction should not touch the database")


class _PostgresSessionManagerStub:
    def open_connection(self) -> object:
        raise AssertionError("helper construction should not open a session")


def _make_context_stub(**overrides: object) -> Any:
    values: dict[str, object] = {
        "workspace_id": "workspace-a",
        "storage_backend": None,
        "db_manager": _SQLiteDbManagerStub(),
        "config": None,
    }
    values.update(overrides)
    return cast(Any, SimpleNamespace(**values))


@pytest.mark.parametrize(
    ("storage_backend", "db_manager", "expected_type"),
    [
        (None, _SQLiteDbManagerStub(), ProviderUsageRepository),
        ("sqlite", object(), NoopProviderUsageRepository),
        ("postgres", _SQLiteDbManagerStub(), NoopProviderUsageRepository),
    ],
    ids=["sqlite-with-connection", "sqlite-without-connection", "postgres-uses-noop"],
)
def test_build_default_provider_usage_selects_repository_by_backend_and_connection(
    storage_backend: str | None,
    db_manager: object,
    expected_type: type[object],
) -> None:
    repository = _build_default_provider_usage(
        _make_context_stub(storage_backend=storage_backend, db_manager=db_manager)
    )

    assert isinstance(repository, expected_type)


@pytest.mark.parametrize(
    ("storage_backend", "db_manager", "expected_type"),
    [
        (None, None, None),
        (None, _SQLiteDbManagerStub(), EmbeddingIntegrityEventRepository),
        ("postgres", _PostgresSessionManagerStub(), PostgresEmbeddingIntegrityEventRepository),
        ("postgres", object(), None),
    ],
    ids=["no-db-manager", "sqlite-repository", "postgres-repository", "postgres-without-session-manager"],
)
def test_build_default_embedding_integrity_events_selects_repository_by_connection_shape(
    storage_backend: str | None,
    db_manager: object | None,
    expected_type: type[object] | None,
) -> None:
    repository = _build_default_embedding_integrity_events(
        _make_context_stub(storage_backend=storage_backend, db_manager=db_manager)
    )

    if expected_type is None:
        assert repository is None
    else:
        assert isinstance(repository, expected_type)


def test_workspace_id_resolvers_distinguish_service_default_from_explicit_override() -> None:
    assert _resolve_log_workspace_id("workspace-a", _USE_SERVICE_WORKSPACE) == "workspace-a"
    assert _resolve_log_workspace_id("workspace-a", None) is None
    assert _resolve_service_workspace_id("workspace-a", _USE_SERVICE_WORKSPACE) == "workspace-a"
    assert _resolve_service_workspace_id("workspace-a", "workspace-b") == "workspace-b"


def test_coerce_transport_diagnostics_payload_handles_none_payload_dict_and_junk() -> None:
    existing = TransportDiagnosticsPayload(current_in_flight_count=2)

    empty_payload = _coerce_transport_diagnostics_payload(None)
    existing_payload = _coerce_transport_diagnostics_payload(existing)
    dict_payload = _coerce_transport_diagnostics_payload(
        {
            "current_in_flight_count": 3,
            "queued_waiter_count": 1,
            "active_requests": [
                {
                    "request_id": 7,
                    "path": "/health",
                    "phase": "running",
                    "age_ms": 12.5,
                }
            ],
        }
    )
    junk_payload = _coerce_transport_diagnostics_payload("not-a-payload")

    assert empty_payload == TransportDiagnosticsPayload()
    assert existing_payload is existing
    assert dict_payload.current_in_flight_count == 3
    assert dict_payload.queued_waiter_count == 1
    assert len(dict_payload.active_requests) == 1
    assert dict_payload.active_requests[0].path == "/health"
    assert dict_payload.active_requests[0].phase == "running"
    assert junk_payload == TransportDiagnosticsPayload()


def test_embedding_integrity_snapshot_payload_handles_missing_and_present_records() -> None:
    assert _embedding_integrity_snapshot_payload(None) is None

    payload = _embedding_integrity_snapshot_payload(
        SimpleNamespace(
            created_at=101.5,
            model_name="mini-embed",
            source_kind="memory",
            source_id="memory-1",
            scanned_row_count=5,
            invalid_row_count=2,
            mixed_dimension_group_count=1,
        )
    )

    assert payload is not None
    assert payload.created_at == 101.5
    assert payload.model_name == "mini-embed"
    assert payload.source_kind == "memory"
    assert payload.source_id == "memory-1"
    assert payload.scanned_row_count == 5
    assert payload.invalid_row_count == 2
    assert payload.mixed_dimension_group_count == 1
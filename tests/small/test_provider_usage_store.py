from __future__ import annotations

import pytest

from mcp_memory.core.providers.interfaces import ProviderTokenUsage
from mcp_memory.provider_usage_store import ProviderUsageRepository
from tests.small.provider_usage_conversation_contract import (
    assert_normalizes_invalid_persisted_conversation_payloads,
    assert_preserves_first_terminal_conversation_finalization,
)

pytestmark = pytest.mark.small


def test_provider_usage_repository_preserves_first_terminal_conversation_finalization(db_manager) -> None:
    assert_preserves_first_terminal_conversation_finalization(
        lambda: ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    )


def test_provider_usage_repository_normalizes_invalid_persisted_conversation_payloads(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    connection = db_manager.get_connection()
    next_request_number = 0

    def load_conversation(raw_payload: object):
        nonlocal next_request_number
        next_request_number += 1
        request_id = f"req-invalid-payload-{next_request_number}"
        connection.execute(
            "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                1,
                "workspace-a",
                "memory-curator",
                "task-invalid-payload",
                "gemini-cli",
                "Gemini CLI",
                "gemini-2.5-pro",
                111,
                "prompt",
                "response",
                raw_payload,
                "success",
                None,
                100.0,
                100.0,
                0.0,
            ),
        )
        connection.commit()
        return repository.get_conversation(request_id)[0]

    assert_normalizes_invalid_persisted_conversation_payloads(load_conversation)


def test_provider_usage_repository_counts_conversation_statuses_since(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    for request_id, status, completed_at, workspace_id in (
        ("req-1", "success", 100.0, "workspace-a"),
        ("req-2", "error", 110.0, "workspace-a"),
        ("req-3", "success", 120.0, "workspace-a"),
        ("req-4", "running", 120.0, "workspace-b"),
    ):
        ProviderUsageRepository(db_manager, workspace_id=workspace_id).record_conversation(
            request_id=request_id,
            attempt=1,
            task_name="task",
            task_id=None,
            provider_key="provider",
            provider_name="Provider",
            model_name="model",
            subprocess_pid=None,
            prompt_text="prompt",
            response_text="response",
            parsed=None,
            status=status,
            error_text=None,
            started_at=completed_at,
            completed_at=completed_at,
        )

    assert repository.count_conversation_statuses_since(after=105.0, workspace_id="workspace-a") == {
        "error": 1,
        "success": 1,
    }


def test_provider_usage_repository_summarizes_token_usage(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="memory-curator",
        task_id="token-task",
        request_id="token-request",
        subprocess_pid=None,
        provider_key="copilot-sdk",
        provider_name="Copilot SDK",
        model_name="gpt-5.4-mini",
        status="success",
        duration_seconds=1.5,
        created_at=900.0,
        error_text=None,
        token_usage=ProviderTokenUsage(
            input_tokens=100,
            output_tokens=30,
            cached_input_tokens=12,
            cache_write_tokens=3,
            reasoning_tokens=5,
            total_tokens=130,
            source="test",
        ),
    )

    summary = repository.summarize_usage(now=1000.0)[0]

    assert summary.input_tokens_last_day == 100
    assert summary.output_tokens_last_day == 30
    assert summary.cached_input_tokens_last_day == 12
    assert summary.cache_write_tokens_last_day == 3
    assert summary.reasoning_tokens_last_day == 5
    assert summary.total_tokens_last_day == 130
    assert summary.token_usage_source == "test"


def test_provider_usage_repository_round_trips_curation_packet_id(db_manager) -> None:
    """Provider calls and conversations retain their curator packet identity.

    The packet ID is persisted independently on both provider telemetry rows.
    """
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")

    repository.record_call(
        task_name="memory-curator",
        task_id="curator-task",
        request_id="packet-request",
        curation_packet_id="packet-123",
        subprocess_pid=None,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        status="success",
        duration_seconds=1.0,
        created_at=100.0,
        error_text=None,
    )
    repository.record_conversation(
        request_id="packet-conversation",
        attempt=1,
        task_name="memory-curator",
        task_id="curator-task",
        curation_packet_id="packet-123",
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        subprocess_pid=None,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="success",
        error_text=None,
        started_at=100.0,
        completed_at=101.0,
    )

    provider_row = db_manager.get_connection().execute(
        "SELECT curation_packet_id FROM provider_usage WHERE request_id = ?",
        ("packet-request",),
    ).fetchone()
    conversation = repository.get_conversation("packet-conversation")[0]

    assert tuple(provider_row) == ("packet-123",)
    assert conversation.curation_packet_id == "packet-123"


def test_provider_usage_repository_defaults_omitted_curation_packet_id_to_none(db_manager) -> None:
    """Existing callers that omit packet attribution persist a null value.

    Legacy non-curator provider usage remains explicitly unattributed.
    """
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")

    repository.record_call(
        task_name="worker",
        task_id=None,
        request_id="unattributed-request",
        subprocess_pid=None,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        status="success",
        duration_seconds=1.0,
        created_at=100.0,
        error_text=None,
    )
    repository.record_conversation(
        request_id="unattributed-conversation",
        attempt=1,
        task_name="worker",
        task_id=None,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        subprocess_pid=None,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="success",
        error_text=None,
        started_at=100.0,
        completed_at=101.0,
    )

    provider_row = db_manager.get_connection().execute(
        "SELECT curation_packet_id FROM provider_usage WHERE request_id = ?",
        ("unattributed-request",),
    ).fetchone()
    conversation = repository.get_conversation("unattributed-conversation")[0]

    assert tuple(provider_row) == (None,)
    assert conversation.curation_packet_id is None


def test_provider_usage_repository_derives_attempt_identity_from_lifecycle(db_manager) -> None:
    db_manager.get_connection().execute(
        "INSERT INTO task_execution_attempts (task_id, execution_epoch, workspace_id, task_name, request_id, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("curator-task", 7, "workspace-a", "memory-curator", "request-7", "running", 1.0),
    )
    db_manager.get_connection().commit()
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")

    repository.record_call(
        task_name="memory-curator",
        task_id="curator-task",
        request_id="request-7",
        attempt=2,
        subprocess_pid=None,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        status="success",
        duration_seconds=1.0,
        created_at=2.0,
        error_text=None,
    )

    row = db_manager.get_connection().execute(
        "SELECT execution_epoch, attempt_identity FROM provider_usage WHERE request_id = ?",
        ("request-7",),
    ).fetchone()
    assert tuple(row) == (7, "curator-task:7:request-7:2")


def test_provider_usage_repository_keeps_missing_identity_unattributed(db_manager) -> None:
    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="memory-curator",
        task_id=None,
        request_id=None,
        subprocess_pid=None,
        provider_key="provider",
        provider_name="Provider",
        model_name="model",
        status="skipped",
        duration_seconds=0.0,
        created_at=2.0,
        error_text=None,
    )

    row = db_manager.get_connection().execute(
        "SELECT task_id, execution_epoch, request_id, attempt_identity FROM provider_usage ORDER BY id DESC LIMIT 1",
    ).fetchone()
    assert tuple(row) == (None, None, None, None)

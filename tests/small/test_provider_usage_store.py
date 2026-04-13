from __future__ import annotations

import pytest

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
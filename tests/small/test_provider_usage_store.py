from __future__ import annotations

import pytest

from mcp_memory.provider_usage_store import ProviderUsageRepository
from tests.small.provider_usage_conversation_contract import (
    assert_preserves_first_terminal_conversation_finalization,
)


pytestmark = pytest.mark.small


def test_provider_usage_repository_preserves_first_terminal_conversation_finalization(db_manager) -> None:
    assert_preserves_first_terminal_conversation_finalization(
        lambda: ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    )
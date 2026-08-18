from collections import deque

import pytest
from copilot.session_events import AssistantMessageData

from mcp_memory.core.providers import copilot_sdk as copilot_sdk_module
from mcp_memory.core.providers.copilot_sdk import CopilotSDKProvider
from tests.sdk.providers import (
    FakeAIProvider,
    FakeCopilotClient,
    FakeCopilotClientFactory,
    FakeCopilotSession,
    FakeCopilotSessionEvent,
)

pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_fake_ai_provider_can_fail_then_recover() -> None:
    provider = FakeAIProvider(
        responses=[{"actions": []}],
        error_sequence=[TimeoutError("slow provider"), None],
    )

    with pytest.raises(TimeoutError, match="slow provider"):
        await provider.ask("first request")

    result = await provider.ask("second request")

    assert result == {"actions": []}
    assert provider.call_count == 2
    assert provider.prompts == ["first request", "second request"]


@pytest.mark.asyncio
async def test_copilot_sdk_provider_recovers_after_timeout(monkeypatch) -> None:
    call_count = 0
    timeout_session = FakeCopilotSession(error=TimeoutError("slow provider"))
    success_session = FakeCopilotSession(
        events=deque([FakeCopilotSessionEvent(data=AssistantMessageData(content='{"actions": []}', message_id="m1"))])
    )
    sessions = deque([timeout_session, success_session])

    class _RotatingClient(FakeCopilotClient):
        async def create_session(self, **kwargs):
            nonlocal call_count
            call_count += 1
            self.session = sessions.popleft()
            return self.session

    factory = FakeCopilotClientFactory(client=_RotatingClient(session=timeout_session))
    monkeypatch.setattr(copilot_sdk_module, "CopilotClient", factory)

    provider = CopilotSDKProvider(model="gpt-5.4-mini", max_retries=1)
    result = await provider.ask("retry after timeout")

    assert result == {"actions": []}
    assert call_count == 2


@pytest.mark.asyncio
async def test_copilot_sdk_provider_raises_after_exhausted_retries_on_timeout(monkeypatch) -> None:
    session = FakeCopilotSession(error=TimeoutError("still broken"))
    factory = FakeCopilotClientFactory(client=FakeCopilotClient(session=session))
    monkeypatch.setattr(copilot_sdk_module, "CopilotClient", factory)

    provider = CopilotSDKProvider(model="gpt-5.4-mini", max_retries=1)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        await provider.ask("this still fails")

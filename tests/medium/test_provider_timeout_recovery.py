import asyncio

import pytest

from mcp_memory.core.providers import GeminiCLIProvider
from tests.sdk.providers import FakeAIProvider, FakeAsyncProcess


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
async def test_gemini_cli_provider_recovers_after_timeout(
    install_fake_subprocess,
) -> None:
    timeout_process = FakeAsyncProcess(raise_timeout=True)
    success_process = FakeAsyncProcess(stdout_text='{"actions": []}')
    install_fake_subprocess.add(timeout_process, success_process)

    provider = GeminiCLIProvider(command="gemini", max_retries=1)

    result = await provider.ask("retry after timeout")

    assert result == {"actions": []}
    assert timeout_process.killed is True
    assert timeout_process.waited is True
    assert len(install_fake_subprocess.calls) == 2


@pytest.mark.asyncio
async def test_gemini_cli_provider_raises_after_exhausted_retries(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(raise_timeout=True),
        FakeAsyncProcess(stderr_text="still broken", returncode=1),
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=1)

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        await provider.ask("this still fails")
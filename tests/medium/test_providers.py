import pytest

from mcp_memory.core.providers import GeminiCLIProvider
from tests.sdk.providers import FakeAsyncProcess


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_gemini_cli_provider_returns_parsed_json(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(FakeAsyncProcess(stdout_text='{"actions": []}'))

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    result = await provider.ask("summarize these memories")

    assert result == {"actions": []}
    assert len(install_fake_subprocess.calls) == 1
    args, kwargs = install_fake_subprocess.calls[0]
    assert args[:3] == ("gemini", "ask", "--json")
    assert args[3] == "summarize these memories"
    assert kwargs == {"stdout": -1, "stderr": -1}


@pytest.mark.asyncio
async def test_gemini_cli_provider_parses_embedded_json(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stdout_text='debug output\n{"actions": [{"type": "ignore", "entry_indices": [0]}]}\n')
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    result = await provider.ask("decide what to ignore")

    assert result["actions"][0]["type"] == "ignore"
    assert result["actions"][0]["entry_indices"] == [0]


@pytest.mark.asyncio
async def test_gemini_cli_provider_retries_after_failed_process(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stderr_text="temporary failure", returncode=1),
        FakeAsyncProcess(stdout_text='{"actions": []}'),
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=1)

    result = await provider.ask("retry this request")

    assert result == {"actions": []}
    assert len(install_fake_subprocess.calls) == 2
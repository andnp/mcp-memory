import pytest

from mcp_memory.config import AIConfig, Config, CopilotCLIConfig, GeminiCLIConfig, OllamaCLIConfig, OpenCodeCLIConfig
from mcp_memory.core.providers import (
    CopilotCLIProvider,
    GeminiCLIProvider,
    OllamaCLIProvider,
    OpenCodeCLIProvider,
    build_ai_provider_from_config,
)
from tests.sdk.providers import FakeAsyncProcess


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_gemini_cli_provider_returns_parsed_json(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(FakeAsyncProcess(stdout_text='{"actions": []}'))

    provider = GeminiCLIProvider(
        command="gemini",
        model="gemini-3-flash-preview",
        max_retries=0,
    )

    result = await provider.ask("summarize these memories")

    assert result == {"actions": []}
    assert len(install_fake_subprocess.calls) == 1
    args, kwargs = install_fake_subprocess.calls[0]
    assert args[:5] == ("gemini", "--model", "gemini-3-flash-preview", "ask", "--json")
    assert args[5] == "summarize these memories"
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


@pytest.mark.asyncio
async def test_other_cli_providers_use_expected_commands(install_fake_subprocess) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stdout_text='{"ok": true}'),
        FakeAsyncProcess(stdout_text='{"ok": true}'),
        FakeAsyncProcess(stdout_text='{"ok": true}'),
    )

    copilot = CopilotCLIProvider(command="copilot", model="gpt-5.4", max_retries=0)
    opencode = OpenCodeCLIProvider(command="opencode", model="gpt-5.4-mini", max_retries=0)
    ollama = OllamaCLIProvider(command="ollama", model="llama3.1:8b", max_retries=0)

    assert await copilot.ask("link these memories") == {"ok": True}
    assert await opencode.ask("summarize this") == {"ok": True}
    assert await ollama.ask("detect conflicts") == {"ok": True}

    assert install_fake_subprocess.calls[0][0] == (
        "copilot",
        "ask",
        "--model",
        "gpt-5.4",
        "--json",
        "link these memories",
    )
    assert install_fake_subprocess.calls[1][0] == (
        "opencode",
        "ask",
        "--model",
        "gpt-5.4-mini",
        "--json",
        "summarize this",
    )
    assert install_fake_subprocess.calls[2][0] == (
        "ollama",
        "run",
        "llama3.1:8b",
        "detect conflicts",
    )


def test_build_ai_provider_from_config_supports_expanded_providers() -> None:
    gemini = build_ai_provider_from_config(
        Config(ai=AIConfig(provider="gemini-cli"), gemini_cli=GeminiCLIConfig(command="gemini"))
    )
    copilot = build_ai_provider_from_config(
        Config(ai=AIConfig(provider="copilot-cli"), copilot_cli=CopilotCLIConfig(command="copilot"))
    )
    opencode = build_ai_provider_from_config(
        Config(ai=AIConfig(provider="opencode"), opencode=OpenCodeCLIConfig(command="opencode"))
    )
    ollama = build_ai_provider_from_config(
        Config(ai=AIConfig(provider="ollama"), ollama=OllamaCLIConfig(command="ollama"))
    )

    assert isinstance(gemini, GeminiCLIProvider)
    assert isinstance(copilot, CopilotCLIProvider)
    assert isinstance(opencode, OpenCodeCLIProvider)
    assert isinstance(ollama, OllamaCLIProvider)
import asyncio

import pytest

from mcp_memory.config import AIConfig, Config, CopilotCLIConfig, GeminiCLIConfig, OllamaCLIConfig, OpenCodeCLIConfig
from mcp_memory.core.providers import (
    CopilotCLIProvider,
    GeminiCLIProvider,
    OllamaCLIProvider,
    OpenCodeCLIProvider,
    build_ai_provider_from_config,
)
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.provider_usage_store import ProviderUsageRepository
from tests.sdk.providers import FakeAsyncProcess


pytestmark = pytest.mark.medium


@pytest.mark.asyncio
async def test_gemini_cli_provider_returns_parsed_json(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stdout_text='{"session_id":"abc","response":"{\\"actions\\": []}"}')
    )

    provider = GeminiCLIProvider(
        command="gemini",
        model="gemini-3-flash-preview",
        max_retries=0,
    )

    result = await provider.ask("summarize these memories")

    assert result == {"actions": []}
    assert len(install_fake_subprocess.calls) == 1
    args, kwargs = install_fake_subprocess.calls[0]
    assert args[:6] == (
        "gemini",
        "--model",
        "gemini-3-flash-preview",
        "--output-format",
        "json",
        "--prompt",
    )
    assert args[6] == "summarize these memories"
    assert kwargs == {"stdout": -1, "stderr": -1}


@pytest.mark.asyncio
async def test_gemini_cli_provider_preserves_plain_json_objects(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(FakeAsyncProcess(stdout_text='{"actions": []}'))

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    result = await provider.ask("plain json still works")

    assert result == {"actions": []}


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


@pytest.mark.asyncio
async def test_instrumented_provider_records_task_name_with_usage_context(db_manager) -> None:
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"ok": True, "prompt": prompt}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="memory-curator")

    result = await provider.ask("clean things up")
    rows = db_manager.get_connection().execute(
        "SELECT task_name, provider_key, status FROM provider_usage ORDER BY id DESC LIMIT 1"
    ).fetchall()

    assert result["ok"] is True
    assert len(rows) == 1
    assert rows[0]["task_name"] == "memory-curator"
    assert rows[0]["provider_key"] == "gemini-cli"
    assert rows[0]["status"] == "success"


@pytest.mark.asyncio
async def test_instrumented_provider_records_subprocess_and_conversation(
    db_manager,
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(FakeAsyncProcess(pid=7777, stdout_text='{"actions": []}'))

    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("graph-linker", workspace_id="workspace-a", available_at=0.0, task_id="graph-linker-1")
    assert queue.claim_next(now=1.0) is not None

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        GeminiCLIProvider(command="gemini", model="gemini-3-flash-preview", max_retries=0),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_queue=queue,
    ).with_usage_context(task_name="graph-linker", task_id=task.id, workspace_id="workspace-a")

    result = await provider.ask("link related memories")

    usage_row = db_manager.get_connection().execute(
        "SELECT task_id, request_id, subprocess_pid, status FROM provider_usage ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conversation_row = db_manager.get_connection().execute(
        "SELECT request_id, task_id, task_name, subprocess_pid, prompt_text, response_text, status FROM ai_conversations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    refreshed = queue.get_task(task.id)

    assert result == {"actions": []}
    assert usage_row is not None
    assert usage_row["task_id"] == task.id
    assert usage_row["subprocess_pid"] == 7777
    assert usage_row["status"] == "success"
    assert conversation_row is not None
    assert conversation_row["request_id"] == usage_row["request_id"]
    assert conversation_row["task_id"] == task.id
    assert conversation_row["task_name"] == "graph-linker"
    assert conversation_row["subprocess_pid"] == 7777
    assert conversation_row["prompt_text"] == "link related memories"
    assert conversation_row["response_text"] == '{"actions": []}'
    assert refreshed.subprocess_pid is None
    assert refreshed.active_request_id is None


@pytest.mark.asyncio
async def test_instrumented_provider_persists_running_conversation_before_finish(db_manager) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask(self, prompt: str) -> dict[str, object]:
            assert self._observer is not None
            self._observer(
                {
                    "event": "started",
                    "attempt": 1,
                    "prompt": prompt,
                    "subprocess_pid": 31337,
                    "started_at": 10.0,
                }
            )
            started.set()
            await release.wait()
            self._observer(
                {
                    "event": "finished",
                    "attempt": 1,
                    "status": "success",
                    "prompt": prompt,
                    "subprocess_pid": 31337,
                    "raw_text": '{"ok": true}',
                    "parsed": {"ok": True},
                    "error": None,
                    "started_at": 10.0,
                    "completed_at": 12.0,
                    "duration_seconds": 2.0,
                }
            )
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="memory-curator", task_id="task-123", workspace_id="workspace-a")

    task = asyncio.create_task(provider.ask("long running prompt"))
    await started.wait()
    running_rows = repository.list_conversations(status="running")
    assert len(running_rows) == 1
    assert running_rows[0].task_id == "task-123"
    assert running_rows[0].subprocess_pid == 31337
    assert running_rows[0].prompt_text == "long running prompt"

    release.set()
    result = await task
    finished_rows = repository.list_conversations(status="success")

    assert result == {"ok": True}
    assert len(finished_rows) == 1
    assert finished_rows[0].response_text == '{"ok": true}'


@pytest.mark.asyncio
async def test_provider_usage_queries_are_global_by_default(db_manager) -> None:
    repository_a = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository_b = ProviderUsageRepository(db_manager, workspace_id="workspace-b")

    repository_a.record_call(
        task_name="memory-curator",
        task_id="task-a",
        request_id="req-a",
        subprocess_pid=None,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        status="success",
        duration_seconds=0.1,
        created_at=10.0,
        error_text=None,
    )
    repository_b.record_call(
        task_name="graph-linker",
        task_id="task-b",
        request_id="req-b",
        subprocess_pid=None,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        status="success",
        duration_seconds=0.2,
        created_at=11.0,
        error_text=None,
    )

    summary = repository_a.summarize_usage(now=12.0)

    assert {(item.task_name, item.calls_last_day) for item in summary} == {
        ("graph-linker", 1),
        ("memory-curator", 1),
    }
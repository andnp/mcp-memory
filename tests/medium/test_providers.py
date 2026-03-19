import asyncio
import json
import signal
import time

import pytest

from mcp_memory.config import AIConfig, Config, CopilotCLIConfig, GeminiCLIConfig, OllamaCLIConfig, OpenCodeCLIConfig
from mcp_memory.core.providers import (
    AgenticRunResult,
    CopilotCLIAgenticProvider,
    CopilotCLIProvider,
    GeminiCLIAgenticProvider,
    GeminiCLIProvider,
    OllamaCLIProvider,
    OpenCodeCLIProvider,
    build_agentic_ai_provider_from_config,
    build_ai_provider_from_config,
    build_json_ai_provider_from_config,
)
from mcp_memory.core.providers.instrumented import InstrumentedAIProvider
from mcp_memory.core.providers._json_cli import ProviderBackoffError
from mcp_memory.core.providers.interfaces import ProviderBudgetExceeded
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
async def test_gemini_cli_provider_reports_signal_name_for_negative_exit(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stderr_text="terminated externally", returncode=-signal.SIGTERM)
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    with pytest.raises(RuntimeError, match="failed after 1 attempts") as exc_info:
        await provider.ask("retry this request")

    assert str(exc_info.value) == (
        "Gemini CLI failed after 1 attempts: "
        "Terminated by signal SIGTERM (15): terminated externally"
    )


@pytest.mark.asyncio
async def test_gemini_cli_provider_reports_unknown_negative_exit_signal_number(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stderr_text="terminated externally", returncode=-999)
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    with pytest.raises(RuntimeError, match="failed after 1 attempts") as exc_info:
        await provider.ask("retry this request")

    assert str(exc_info.value) == (
        "Gemini CLI failed after 1 attempts: "
        "Terminated by signal 999: terminated externally"
    )


@pytest.mark.asyncio
async def test_gemini_cli_provider_raises_backoff_error_for_quota_exhaustion(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stderr_text=(
                "TerminalQuotaError: You have exhausted your capacity on this model. "
                "Your quota will reset after 3h50m47s.\nretryDelayMs: 13847179.189666"
            ),
            returncode=1,
        )
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    with pytest.raises(ProviderBackoffError) as exc_info:
        await provider.ask("retry this request")

    assert exc_info.value.retry_delay_seconds == pytest.approx(13847.179189666)


@pytest.mark.asyncio
async def test_gemini_agentic_provider_uses_yolo_mode_and_allowed_mcp_server(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stdout_text='{"summary": "Performed maintenance."}')
    )

    provider = GeminiCLIAgenticProvider(
        command="gemini",
        model="gemini-3-flash-preview",
        max_retries=0,
        cwd="/tmp/workspace",
    )

    result = await provider.run_agent("Clean up the memory store.")

    assert result.status == "success"
    assert result.summary == "Performed maintenance."
    assert len(install_fake_subprocess.calls) == 1
    args, kwargs = install_fake_subprocess.calls[0]
    assert args == (
        "gemini",
        "--model",
        "gemini-3-flash-preview",
        "--prompt",
        "Clean up the memory store.",
        "--output-format",
        "json",
        "--approval-mode",
        "yolo",
        "--allowed-mcp-server-names",
        "mcp-memory-internal",
    )
    assert kwargs == {"stdout": -1, "stderr": -1, "cwd": "/tmp/workspace"}


@pytest.mark.asyncio
async def test_gemini_agentic_provider_extracts_nested_summary_from_mixed_response(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stdout_text=json.dumps(
                {
                    "response": (
                        "I performed maintenance.\n\n"
                        '{"summary": "Split oversized records into focused entries."}'
                    )
                }
            )
        )
    )

    provider = GeminiCLIAgenticProvider(
        command="gemini",
        model="gemini-3-flash-preview",
        max_retries=0,
    )

    result = await provider.run_agent("Clean up the memory store.")

    assert result.status == "success"
    assert result.summary == "Split oversized records into focused entries."


@pytest.mark.asyncio
async def test_gemini_agentic_provider_raises_backoff_error_for_quota_exhaustion(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stderr_text=(
                "TerminalQuotaError: You have exhausted your capacity on this model. "
                "Your quota will reset after 59m59s.\nretryDelayMs: 3599000"
            ),
            returncode=1,
        )
    )

    provider = GeminiCLIAgenticProvider(command="gemini", max_retries=0)

    with pytest.raises(ProviderBackoffError) as exc_info:
        await provider.run_agent("Clean up the memory store.")

    assert exc_info.value.retry_delay_seconds == pytest.approx(3599.0)


@pytest.mark.asyncio
async def test_other_cli_providers_use_expected_commands(install_fake_subprocess) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(stdout_text='{"type":"assistant.message","data":{"content":"{\\"ok\\": true}"}}'),
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
        "--model",
        "gpt-5.4",
        "--output-format",
        "json",
        "--stream",
        "off",
        "--silent",
        "--no-ask-user",
        "--no-custom-instructions",
        "--prompt",
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


def test_split_provider_builders_keep_json_and_agentic_paths_distinct() -> None:
    config = Config(ai=AIConfig(provider="gemini-cli"), gemini_cli=GeminiCLIConfig(command="gemini"))

    json_provider = build_json_ai_provider_from_config(config)
    agentic_provider = build_agentic_ai_provider_from_config(config)

    assert isinstance(json_provider, GeminiCLIProvider)
    assert isinstance(agentic_provider, GeminiCLIAgenticProvider)


def test_split_provider_builders_support_copilot_agentic() -> None:
    config = Config(ai=AIConfig(provider="copilot-cli"), copilot_cli=CopilotCLIConfig(command="copilot"))

    json_provider = build_json_ai_provider_from_config(config)
    agentic_provider = build_agentic_ai_provider_from_config(config, workspace_root=None)

    assert isinstance(json_provider, CopilotCLIProvider)
    assert isinstance(agentic_provider, CopilotCLIAgenticProvider)


@pytest.mark.asyncio
async def test_copilot_agentic_provider_uses_autopilot_and_inline_mcp_config(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stdout_text='{"type":"assistant.message","data":{"content":"{\\"summary\\": \\\"Performed maintenance.\\\"}"}}'
        )
    )

    provider = CopilotCLIAgenticProvider(
        command="copilot",
        model="gpt-5-mini",
        max_retries=0,
        cwd="/tmp/workspace",
    )

    result = await provider.run_agent("Clean up the memory store.")

    assert result.status == "success"
    assert result.summary == "Performed maintenance."
    args, kwargs = install_fake_subprocess.calls[0]
    assert args[:14] == (
        "copilot",
        "--model",
        "gpt-5-mini",
        "--output-format",
        "json",
        "--stream",
        "off",
        "--silent",
        "--no-ask-user",
        "--no-custom-instructions",
        "--autopilot",
        "--allow-all-tools",
        "--max-autopilot-continues",
        "12",
    )
    assert args[14] == "--available-tools"
    available_tools = args[15]
    assert "mcp-memory-internal-internal_search_memory_records" in available_tools
    assert "mcp-memory-internal-internal_read_memory_record" in available_tools
    assert args[16] == "--additional-mcp-config"
    mcp_config = json.loads(args[17])
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["type"] == "stdio"
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["command"] == "uv"
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["args"] == [
        "run",
        "mcp-memory",
        "internal-run",
        "--workspace-root",
        "/tmp/workspace",
    ]
    assert args[18:20] == ("--prompt", "Clean up the memory store.")
    assert kwargs == {"stdout": -1, "stderr": -1, "cwd": "/tmp/workspace"}


@pytest.mark.asyncio
async def test_copilot_agentic_provider_prefers_json_result_over_trailing_non_json_messages(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stdout_text="\n".join(
                [
                    '{"type":"assistant.message","data":{"content":"planning to search"}}',
                    '{"type":"assistant.message","data":{"content":"{\\"summary\\": \\\"Structured answer.\\\"}"}}',
                    '{"type":"assistant.message","data":{"content":"Task completed but no task_complete tool exists."}}',
                ]
            )
        )
    )

    provider = CopilotCLIAgenticProvider(command="copilot", max_retries=0)

    result = await provider.run_agent("Return a structured answer.")

    assert result.summary == "Structured answer."
    assert result.parsed == {"summary": "Structured answer."}


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
async def test_instrumented_provider_prefers_ask_json_when_available(db_manager) -> None:
    class _Provider:
        def __init__(self) -> None:
            self.ask_calls = 0
            self.ask_json_calls = 0

        async def ask(self, prompt: str) -> dict[str, object]:
            self.ask_calls += 1
            return {"path": "ask", "prompt": prompt}

        async def ask_json(self, prompt: str) -> dict[str, object]:
            self.ask_json_calls += 1
            return {"path": "ask_json", "prompt": prompt}

    provider_impl = _Provider()
    provider = InstrumentedAIProvider(
        provider_impl,
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    )

    result = await provider.ask_json("clean json mode")

    assert result["path"] == "ask_json"
    assert provider_impl.ask_json_calls == 1
    assert provider_impl.ask_calls == 0


@pytest.mark.asyncio
async def test_instrumented_provider_can_forward_agentic_runs(db_manager) -> None:
    class _Provider:
        async def run_agent(self, prompt: str) -> AgenticRunResult:
            return AgenticRunResult(status="success", summary=prompt)

    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    )

    result = await provider.run_agent("agent mode")

    assert result.status == "success"
    assert result.summary == "agent mode"


@pytest.mark.asyncio
async def test_instrumented_provider_reports_agentic_capability_from_wrapped_provider(db_manager) -> None:
    class _JsonOnlyProvider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"ok": True, "prompt": prompt}

    class _AgenticProvider:
        async def run_agent(self, prompt: str) -> AgenticRunResult:
            return AgenticRunResult(status="success", summary=prompt)

    json_provider = InstrumentedAIProvider(
        _JsonOnlyProvider(),
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="copilot-mini",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
    )
    agentic_provider = InstrumentedAIProvider(
        _AgenticProvider(),
        usage_repository=ProviderUsageRepository(db_manager, workspace_id="workspace-a"),
        provider_key="gemini-cli:agentic",
        provider_name="Gemini CLI Agentic",
        model_name="gemini-3-flash-preview",
    )

    assert json_provider.supports_agentic() is False
    assert agentic_provider.supports_agentic() is True


@pytest.mark.asyncio
async def test_instrumented_provider_enforces_daily_budget_before_call(db_manager) -> None:
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"ok": True, "prompt": prompt}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="summarize-memory",
        task_id="task-a",
        request_id="req-a",
        subprocess_pid=None,
        provider_key="copilot-mini",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
        status="success",
        duration_seconds=0.1,
        created_at=time.time(),
        error_text=None,
    )
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="copilot-mini",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
        daily_call_limit=1,
    )

    with pytest.raises(ProviderBudgetExceeded) as exc_info:
        await provider.ask("cheap but capped")

    assert exc_info.value.provider_key == "copilot-mini"
    assert exc_info.value.daily_call_limit == 1


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

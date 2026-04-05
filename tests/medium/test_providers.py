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
from mcp_memory.core.providers.interfaces import ProviderAuthenticationRequired
from mcp_memory.core.providers.interfaces import ProviderBudgetExceeded
from mcp_memory.core.providers.interfaces import ProviderAdmissionDeferred
from mcp_memory.core.providers.interfaces import ProviderRateLimitExceeded
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.task_execution_store import TaskExecutionAttemptRepository
from tests.sdk.providers import FakeAsyncProcess


pytestmark = pytest.mark.medium


class _DelayedFakeProcess(FakeAsyncProcess):
    def __init__(self, *, delay_seconds: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.delay_seconds = delay_seconds

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(self.delay_seconds)
        return await super().communicate()


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
async def test_gemini_cli_provider_raises_authentication_required_for_browser_auth_prompt(
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stdout_text="Opening authentication page in your browser. Do you want to continue? [Y/n]:"
        )
    )

    provider = GeminiCLIProvider(command="gemini", max_retries=0)

    with pytest.raises(ProviderAuthenticationRequired) as exc_info:
        await provider.ask("retry this request")

    assert exc_info.value.provider_name == "Gemini CLI"
    assert exc_info.value.error_text == "Interactive authentication required"


@pytest.mark.asyncio
async def test_gemini_cli_provider_emits_heartbeat_while_waiting(
    install_fake_subprocess,
    monkeypatch,
) -> None:
    install_fake_subprocess.add(
        _DelayedFakeProcess(delay_seconds=0.03, pid=8888, stdout_text='{"actions": []}')
    )
    monkeypatch.setattr(
        "mcp_memory.core.providers._json_cli.PROVIDER_SUBPROCESS_HEARTBEAT_SECONDS",
        0.01,
    )
    events: list[dict[str, object]] = []

    provider = GeminiCLIProvider(command="gemini", max_retries=0).with_observer(events.append)

    result = await provider.ask("wait for heartbeat")

    heartbeat_events = [event for event in events if event.get("event") == "heartbeat"]
    assert result == {"actions": []}
    assert heartbeat_events
    assert heartbeat_events[0]["subprocess_pid"] == 8888
    elapsed_seconds = heartbeat_events[0]["elapsed_seconds"]
    assert isinstance(elapsed_seconds, int | float)
    assert float(elapsed_seconds) >= 0.0


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
async def test_copilot_agentic_provider_uses_inline_mcp_config_without_autopilot(
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
    assert args[:11] == (
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
        "--allow-all-tools",
    )
    assert args[11] == "--available-tools"
    available_tools = args[12]
    assert "mcp-memory-internal-task_complete" in available_tools
    assert "mcp-memory-internal-internal_task_complete" in available_tools
    assert "mcp-memory-internal-internal_search_memory_records" in available_tools
    assert "mcp-memory-internal-internal_read_memory_record" in available_tools
    assert "mcp-memory-internal-internal_get_compatible_work_batch" in available_tools
    assert "mcp-memory-internal-internal_complete_work_item" in available_tools
    assert args[13] == "--additional-mcp-config"
    mcp_config = json.loads(args[14])
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["type"] == "stdio"
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["command"] == "uv"
    assert mcp_config["mcpServers"]["mcp-memory-internal"]["args"] == [
        "run",
        "mcp-memory",
        "internal-run",
        "--workspace-root",
        "/tmp/workspace",
    ]
    assert args[15:17] == ("--prompt", "Clean up the memory store.")
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
async def test_instrumented_provider_enforces_model_burst_limit_before_call(db_manager) -> None:
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"ok": True, "prompt": prompt}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="graph-linker",
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
        provider_key="copilot-strong",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
        daily_call_limit=20,
        model_burst_call_limit=1,
        model_burst_window_seconds=600.0,
    )

    with pytest.raises(ProviderRateLimitExceeded) as exc_info:
        await provider.ask("cheap but burst-limited")

    assert exc_info.value.model_name == "gpt-5-mini"
    assert exc_info.value.calls_in_window == 1
    assert exc_info.value.burst_call_limit == 1


@pytest.mark.asyncio
async def test_instrumented_provider_exposes_structured_admission_decision(db_manager) -> None:
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            return {"ok": True, "prompt": prompt}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    repository.record_call(
        task_name="graph-linker",
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
        provider_key="copilot-strong",
        provider_name="Copilot CLI",
        model_name="gpt-5-mini",
        daily_call_limit=20,
        model_burst_call_limit=1,
        model_burst_window_seconds=600.0,
    )

    decision = provider.admission_decision()

    assert decision.allowed is False
    assert decision.reason == "model_burst_limit_exceeded"


@pytest.mark.asyncio
async def test_instrumented_provider_persists_upstream_backoff_and_later_reports_admission_denial(db_manager) -> None:
    class _Provider:
        async def ask(self, prompt: str) -> dict[str, object]:
            raise ProviderBackoffError("quota will reset after 1m", retry_delay_seconds=60.0)

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="memory-curator", task_id="task-backoff", workspace_id="workspace-a")

    with pytest.raises(ProviderBackoffError):
        await provider.ask("trigger backoff")

    state = repository.get_active_admission_state(
        provider_key="gemini-cli",
        model_name="gemini-3-flash-preview",
    )
    assert state is not None
    assert state.reason_code == "provider_quota_exhausted"

    with pytest.raises(ProviderAdmissionDeferred, match="provider_admission_deferred"):
        await provider.ask("blocked while backoff active")

    rows = db_manager.get_connection().execute(
        "SELECT status, reason_category, reason_code FROM provider_usage ORDER BY id DESC LIMIT 2"
    ).fetchall()
    assert [row["status"] for row in rows] == ["skipped", "error"]
    assert rows[0]["reason_category"] == "upstream"
    assert rows[0]["reason_code"] == "provider_quota_exhausted"


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
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        GeminiCLIProvider(command="gemini", model="gemini-3-flash-preview", max_retries=0),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_queue=queue,
        task_execution_attempts=attempt_repository,
    ).with_usage_context(
        task_name="graph-linker",
        task_id=task.id,
        execution_epoch=1,
        workspace_id="workspace-a",
    )

    result = await provider.ask("link related memories")

    usage_row = db_manager.get_connection().execute(
        "SELECT task_id, request_id, subprocess_pid, status FROM provider_usage ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conversation_row = db_manager.get_connection().execute(
        "SELECT request_id, task_id, task_name, subprocess_pid, prompt_text, response_text, status FROM ai_conversations ORDER BY id DESC LIMIT 1"
    ).fetchone()
    attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)
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
    assert attempt.request_id == usage_row["request_id"]
    assert attempt.subprocess_pid == 7777
    assert attempt.status == "success"
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
async def test_instrumented_provider_heartbeat_refreshes_task_and_running_conversation(db_manager) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    started_at = 3_000_000_000.0
    heartbeat_at = started_at + 2.5
    completed_at = started_at + 4.0

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
                    "started_at": started_at,
                }
            )
            started.set()
            await asyncio.sleep(0)
            self._observer(
                {
                    "event": "heartbeat",
                    "attempt": 1,
                    "prompt": prompt,
                    "subprocess_pid": 31337,
                    "started_at": started_at,
                    "heartbeat_at": heartbeat_at,
                    "elapsed_seconds": 2.5,
                }
            )
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
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": 4.0,
                }
            )
            return {"ok": True}

    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("memory-curator", available_at=0.0, task_id="heartbeat-task")
    assert queue.claim_next(now=1.0) is not None

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_queue=queue,
        task_execution_attempts=attempt_repository,
    ).with_usage_context(
        task_name="memory-curator",
        task_id=task.id,
        execution_epoch=1,
        workspace_id="workspace-a",
    )

    provider_task = asyncio.create_task(provider.ask("keep going"))
    await started.wait()

    for _ in range(20):
        if queue.get_task(task.id).updated_at == heartbeat_at:
            break
        await asyncio.sleep(0)

    refreshed_task = queue.get_task(task.id)
    running_conversation = repository.list_conversations(status="running", limit=1)[0]
    running_attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)
    assert refreshed_task.updated_at == pytest.approx(heartbeat_at)
    assert running_conversation.completed_at == pytest.approx(heartbeat_at)
    assert running_conversation.duration_seconds == pytest.approx(2.5)
    assert running_attempt.last_heartbeat_at == pytest.approx(heartbeat_at)
    assert running_attempt.status == "running"

    release.set()
    result = await provider_task
    finished_attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)

    assert result == {"ok": True}
    assert finished_attempt.status == "success"
    assert finished_attempt.completed_at == pytest.approx(completed_at)


@pytest.mark.asyncio
async def test_instrumented_provider_records_attempt_telemetry_after_recovery_terminalization(
    db_manager,
) -> None:
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
                    "event": "heartbeat",
                    "attempt": 1,
                    "prompt": prompt,
                    "subprocess_pid": 31337,
                    "started_at": 10.0,
                    "heartbeat_at": 11.0,
                    "elapsed_seconds": 1.0,
                }
            )
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

    queue = SQLiteTaskQueue(db_manager)
    task = queue.enqueue("memory-curator", available_at=0.0, task_id="observer-race-task")
    assert queue.claim_next(now=1.0) is not None

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_queue=queue,
        task_execution_attempts=attempt_repository,
    ).with_usage_context(
        task_name="memory-curator",
        task_id=task.id,
        execution_epoch=1,
        workspace_id="workspace-a",
    )

    provider_task = asyncio.create_task(provider.ask("race the reconciler"))
    await started.wait()
    queue.fail_permanently(task.id, "Task was abandoned without an active provider subprocess", failed_at=10.5)

    release.set()
    result = await provider_task
    attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)

    assert result == {"ok": True}
    assert queue.get_task(task.id).status == "failed"
    assert attempt.status == "success"
    assert attempt.completed_at == pytest.approx(12.0)
    assert attempt.termination_reason is None


@pytest.mark.asyncio
async def test_instrumented_provider_reopens_attempt_for_later_provider_call_in_same_execution_epoch(
    db_manager,
) -> None:
    call_count = 0

    class _Provider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _Provider(observer)

        async def ask(self, prompt: str) -> dict[str, object]:
            nonlocal call_count
            assert self._observer is not None
            call_count += 1
            call_number = call_count
            started_at = 10.0 if call_number == 1 else 20.0
            completed_at = 12.0 if call_number == 1 else 24.0
            subprocess_pid = 1111 if call_number == 1 else 2222
            self._observer(
                {
                    "event": "started",
                    "attempt": 1,
                    "prompt": prompt,
                    "subprocess_pid": subprocess_pid,
                    "started_at": started_at,
                }
            )
            self._observer(
                {
                    "event": "finished",
                    "attempt": 1,
                    "status": "success",
                    "prompt": prompt,
                    "subprocess_pid": subprocess_pid,
                    "raw_text": json.dumps({"call": call_number}),
                    "parsed": {"call": call_number},
                    "error": None,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": completed_at - started_at,
                }
            )
            return {"call": call_number}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    attempt_repository = TaskExecutionAttemptRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        task_execution_attempts=attempt_repository,
    ).with_usage_context(
        task_name="ingest-system1",
        task_id="task-multi-call",
        execution_epoch=3,
        workspace_id="workspace-a",
    )

    first_result = await provider.ask("first provider call")
    second_result = await provider.ask("second provider call")

    latest_usage = db_manager.get_connection().execute(
        "SELECT request_id FROM provider_usage WHERE task_id = ? ORDER BY id DESC LIMIT 1",
        ("task-multi-call",),
    ).fetchone()
    attempt = attempt_repository.get_attempt(task_id="task-multi-call", execution_epoch=3)

    assert first_result == {"call": 1}
    assert second_result == {"call": 2}
    assert latest_usage is not None
    assert attempt.request_id == latest_usage["request_id"]
    assert attempt.subprocess_pid == 2222
    assert attempt.status == "success"
    assert attempt.started_at == pytest.approx(20.0)
    assert attempt.last_heartbeat_at == pytest.approx(24.0)
    assert attempt.completed_at == pytest.approx(24.0)
    assert attempt.error_text is None
    assert attempt.termination_reason is None


@pytest.mark.asyncio
async def test_instrumented_provider_finalizes_running_conversation_on_success_without_finished_event(db_manager) -> None:
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
                    "subprocess_pid": 2121,
                    "started_at": 10.0,
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
    ).with_usage_context(task_name="memory-curator", task_id="task-success-no-finish", workspace_id="workspace-a")

    result = await provider.ask("finish without observer terminal event")
    conversation = repository.list_conversations(task_name="memory-curator", limit=1)[0]

    assert result == {"ok": True}
    assert conversation.task_id == "task-success-no-finish"
    assert conversation.status == "success"
    assert conversation.parsed == {"ok": True}
    assert conversation.response_text == '{"ok": true}'


@pytest.mark.asyncio
async def test_instrumented_agentic_provider_finalizes_running_conversation_on_success_without_finished_event(db_manager) -> None:
    class _AgenticProvider:
        def __init__(self, observer=None) -> None:
            self._observer = observer

        def with_observer(self, observer):
            return _AgenticProvider(observer)

        async def run_agent(self, prompt: str) -> AgenticRunResult:
            assert self._observer is not None
            self._observer(
                {
                    "event": "started",
                    "attempt": 1,
                    "prompt": prompt,
                    "subprocess_pid": 5252,
                    "started_at": 20.0,
                }
            )
            return AgenticRunResult(
                status="success",
                summary="done",
                raw_text='{"summary":"done"}',
                parsed={"summary": "done"},
            )

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _AgenticProvider(),
        usage_repository=repository,
        provider_key="copilot-mini:agentic",
        provider_name="Copilot CLI Agentic",
        model_name="gpt-5-mini",
    ).with_usage_context(task_name="deduplicator", task_id="task-agentic-success-no-finish", workspace_id="workspace-a")

    result = await provider.run_agent("agentic finish without observer terminal event")
    conversation = repository.list_conversations(task_name="deduplicator", limit=1)[0]

    assert result.status == "success"
    assert conversation.task_id == "task-agentic-success-no-finish"
    assert conversation.status == "success"
    assert conversation.parsed == {"summary": "done"}
    assert conversation.response_text == '{"summary":"done"}'


@pytest.mark.asyncio
async def test_instrumented_provider_finalizes_running_conversation_on_error(db_manager) -> None:
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
                    "subprocess_pid": 4242,
                    "started_at": 10.0,
                }
            )
            raise RuntimeError("Provider subprocess 4242 exited unexpectedly")

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="ingest-system1", task_id="task-error", workspace_id="workspace-a")

    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        await provider.ask("recover from this")

    conversations = repository.get_conversation(
        db_manager.get_connection().execute(
            "SELECT request_id FROM ai_conversations WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            ("task-error",),
        ).fetchone()["request_id"]
    )

    assert len(conversations) == 1
    assert conversations[0].status == "error"
    assert conversations[0].task_id == "task-error"
    assert conversations[0].error_text == "Provider subprocess 4242 exited unexpectedly"


@pytest.mark.asyncio
async def test_instrumented_provider_records_auth_required_reason_for_browser_auth_prompt(
    db_manager,
    install_fake_subprocess,
) -> None:
    install_fake_subprocess.add(
        FakeAsyncProcess(
            stdout_text="Opening authentication page in your browser. Do you want to continue? [Y/n]:"
        )
    )

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        GeminiCLIProvider(command="gemini", model="gemini-3-flash-preview", max_retries=0),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="taxonomist", task_id="task-auth", workspace_id="workspace-a")

    with pytest.raises(ProviderAuthenticationRequired):
        await provider.ask("taxonomy pass")

    usage_row = db_manager.get_connection().execute(
        "SELECT status, reason_category, reason_code, error_text FROM provider_usage ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conversation = repository.list_conversations(task_name="taxonomist", limit=1)[0]

    assert usage_row is not None
    assert usage_row["status"] == "auth_required"
    assert usage_row["reason_category"] == "auth"
    assert usage_row["reason_code"] == "interactive_auth_required"
    assert usage_row["error_text"] == "Interactive authentication required"
    assert conversation.status == "auth_required"
    assert conversation.reason_category == "auth"
    assert conversation.reason_code == "interactive_auth_required"
    assert conversation.response_text == "Opening authentication page in your browser. Do you want to continue? [Y/n]:"


@pytest.mark.asyncio
async def test_instrumented_provider_finalizes_running_conversation_on_cancellation(db_manager) -> None:
    started = asyncio.Event()

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
                    "subprocess_pid": 5151,
                    "started_at": 20.0,
                }
            )
            started.set()
            await asyncio.sleep(60.0)
            return {"ok": True}

    repository = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    provider = InstrumentedAIProvider(
        _Provider(),
        usage_repository=repository,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
    ).with_usage_context(task_name="memory-curator", task_id="task-cancel", workspace_id="workspace-a")

    provider_task = asyncio.create_task(provider.ask("cancel me"))
    await started.wait()
    provider_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await provider_task

    conversation = repository.list_conversations(task_name="memory-curator", limit=1)[0]
    assert conversation.task_id == "task-cancel"
    assert conversation.status == "cancelled"
    assert conversation.error_text == "Command cancelled"


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

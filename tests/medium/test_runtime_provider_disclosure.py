from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from mcp_memory.config import AIConfig, Config, ProviderRoutingConfig
from mcp_memory.mcp import runtime

pytestmark = pytest.mark.medium


class _FakeProvider:
    provider_name = "fake provider"


def _registry_for(config: Config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MCP_MEMORY_TEST_MODE", raising=False)
    monkeypatch.setattr(runtime, "build_json_ai_provider", lambda *_args: _FakeProvider())
    monkeypatch.setattr(runtime, "build_agentic_ai_provider", lambda *_args: None)
    storage = SimpleNamespace(
        provider_usage=object(),
        task_execution_attempts=None,
        task_queue=None,
    )
    return runtime._build_provider_registry(
        config=config,
        workspace_root=Path("."),
        workspace_id="workspace",
        storage=cast(Any, storage),
    )


def test_runtime_external_provider_defaults_to_not_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Config(
        provider_routing=ProviderRoutingConfig(
            profiles={"external": AIConfig(provider="copilot-sdk")},
        )
    )

    provider = cast(Any, _registry_for(config, monkeypatch)["external"]["json"])

    assert provider._provider_trust_class == "external"
    assert provider._provider_allowlisted is False


def test_runtime_trust_metadata_preserves_local_and_explicit_external_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Config(
        provider_routing=ProviderRoutingConfig(
            profiles={
                "local": AIConfig(provider="copilot-sdk", provider_trust_class="local"),
                "trusted": AIConfig(
                    provider="copilot-sdk",
                    provider_trust_class="trusted_external",
                    provider_allowlisted=True,
                ),
            },
        )
    )

    registry = _registry_for(config, monkeypatch)

    local = cast(Any, registry["local"]["json"])
    trusted = cast(Any, registry["trusted"]["json"])
    assert local._provider_trust_class == "local"
    assert local._provider_allowlisted is True
    assert trusted._provider_trust_class == "trusted_external"
    assert trusted._provider_allowlisted is True

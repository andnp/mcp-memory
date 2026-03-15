from __future__ import annotations

import time

from mcp_memory.provider_usage_store import ProviderUsageRepository


class InstrumentedAIProvider:
    def __init__(
        self,
        provider,
        *,
        usage_repository: ProviderUsageRepository,
        provider_key: str,
        provider_name: str,
        model_name: str,
    ) -> None:
        self._provider = provider
        self._usage_repository = usage_repository
        self._provider_key = provider_key
        self._provider_name = provider_name
        self._model_name = model_name

    async def ask(self, prompt: str) -> dict:
        started_at = time.time()
        try:
            response = await self._provider.ask(prompt)
        except Exception as exc:
            self._usage_repository.record_call(
                provider_key=self._provider_key,
                provider_name=self._provider_name,
                model_name=self._model_name,
                status="error",
                duration_seconds=max(time.time() - started_at, 0.0),
                created_at=time.time(),
                error_text=str(exc),
            )
            raise
        self._usage_repository.record_call(
            provider_key=self._provider_key,
            provider_name=self._provider_name,
            model_name=self._model_name,
            status="success",
            duration_seconds=max(time.time() - started_at, 0.0),
            created_at=time.time(),
            error_text=None,
        )
        return response
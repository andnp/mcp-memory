from __future__ import annotations


class NoopProviderUsageRepository:
    def __init__(self, *, workspace_id: str | None) -> None:
        self._workspace_id = workspace_id

    def summarize_usage(self, *, workspace_id: str | None | object = None):
        del workspace_id
        return []

    def list_conversations(
        self,
        *,
        workspace_id: str | None | object = None,
        request_id: str | None = None,
        task_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ):
        del workspace_id, request_id, task_id, task_name, status, limit
        return []

    def list_active_admission_states(
        self,
        *,
        now: float | None = None,
        provider_key: str | None = None,
        model_name: str | None = None,
    ):
        del now, provider_key, model_name
        return []

    def count_recent_calls(
        self,
        *,
        provider_keys: list[str],
        now: float | None = None,
        window_seconds: float = 86400.0,
    ) -> int:
        del provider_keys, now, window_seconds
        return 0

    def count_recent_model_calls(
        self,
        *,
        model_names: list[str],
        now: float | None = None,
        window_seconds: float = 600.0,
    ) -> int:
        del model_names, now, window_seconds
        return 0

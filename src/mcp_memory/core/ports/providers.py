"""Provider-neutral persistence contracts used by application services."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class ProviderConversationLike(Protocol):
    @property
    def request_id(self) -> str: ...

    @property
    def task_id(self) -> str | None: ...

    @property
    def completed_at(self) -> float: ...


class ProviderAdmissionStateLike(Protocol):
    @property
    def reason_code(self) -> str: ...

    @property
    def reason_category(self) -> str: ...

    @property
    def error_text(self) -> str | None: ...

    @property
    def retry_delay_seconds(self) -> float | None: ...

    @property
    def active_until(self) -> float: ...


class ProviderUsagePort(Protocol):
    def get_active_admission_state(
        self,
        *,
        provider_key: str,
        model_name: str,
        now: float | None = None,
    ) -> ProviderAdmissionStateLike | None: ...

    def count_recent_calls(
        self,
        *,
        provider_keys: list[str],
        now: float | None = None,
        window_seconds: float = 86400.0,
    ) -> int: ...

    def count_recent_model_calls(
        self,
        *,
        model_names: list[str],
        now: float | None = None,
        window_seconds: float = 600.0,
    ) -> int: ...

    def record_call(
        self,
        *,
        task_name: str | None,
        task_id: str | None,
        request_id: str | None,
        subprocess_pid: int | None,
        provider_key: str,
        provider_name: str,
        model_name: str,
        status: str,
        duration_seconds: float,
        created_at: float,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
    ) -> None: ...

    def record_conversation(
        self,
        *,
        request_id: str,
        attempt: int,
        task_name: str | None,
        task_id: str | None,
        provider_key: str,
        provider_name: str,
        model_name: str,
        subprocess_pid: int | None,
        prompt_text: str,
        response_text: str,
        parsed: Mapping[str, object] | None,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        started_at: float,
        completed_at: float,
    ) -> None: ...

    def touch_running_conversation(
        self,
        *,
        request_id: str,
        completed_at: float | None = None,
        subprocess_pid: int | None | object = ...,
    ) -> int: ...

    def finalize_running_conversation(
        self,
        *,
        request_id: str,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        completed_at: float | None = None,
        response_text: str | object = ...,
        parsed: Mapping[str, object] | None | object = ...,
    ) -> int: ...

    def upsert_admission_state(
        self,
        *,
        provider_key: str,
        model_name: str,
        reason_category: str,
        reason_code: str,
        error_text: str | None,
        retry_delay_seconds: float | None,
        active_until: float,
        updated_at: float,
    ) -> None: ...

    def clear_admission_state(self, *, provider_key: str, model_name: str) -> None: ...

    def list_conversations(
        self,
        *,
        workspace_id: str | None | object = ...,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> Sequence[ProviderConversationLike]: ...

    def reconcile_running_task_conversations(
        self,
        *,
        task_id: str,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        completed_at: float | None = None,
    ) -> int: ...


class ProviderPolicyEventPort(Protocol):
    def record_event(
        self,
        *,
        task_name: str,
        task_id: str | None,
        event_kind: str,
        warning_kind: str | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        route_key: str | None = None,
        candidate_routes: list[str] | None = None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        warning_suppressed: bool = False,
        created_at: float | None = None,
    ) -> None: ...


class TaskExecutionAttemptRecordLike(Protocol):
    @property
    def started_at(self) -> float: ...

    @property
    def last_heartbeat_at(self) -> float | None: ...

    @property
    def subprocess_pid(self) -> int | None: ...


class TaskExecutionAttemptPort(Protocol):
    def start_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        task_name: str | None,
        request_id: str | None,
        subprocess_pid: int | None,
        provider_key: str | None,
        provider_name: str | None,
        model_name: str | None,
        started_at: float | None = None,
        status: str = "running",
    ) -> TaskExecutionAttemptRecordLike: ...

    def heartbeat_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        heartbeat_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
    ) -> TaskExecutionAttemptRecordLike: ...

    def finish_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        status: str,
        completed_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
        error_text: str | None = None,
        termination_reason: str | None = None,
    ) -> TaskExecutionAttemptRecordLike: ...

    def get_attempt(self, *, task_id: str, execution_epoch: int) -> TaskExecutionAttemptRecordLike: ...


class NullProviderUsagePort:
    """Provider-neutral no-op used when runtime persistence is not configured."""

    def get_active_admission_state(
        self,
        *,
        provider_key: str,
        model_name: str,
        now: float | None = None,
    ) -> None:
        del provider_key, model_name, now
        return None

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

    def record_call(self, **kwargs: object) -> None:
        del kwargs

    def record_conversation(self, **kwargs: object) -> None:
        del kwargs

    def touch_running_conversation(self, **kwargs: object) -> int:
        del kwargs
        return 0

    def finalize_running_conversation(self, **kwargs: object) -> int:
        del kwargs
        return 0

    def upsert_admission_state(self, **kwargs: object) -> None:
        del kwargs

    def clear_admission_state(self, **kwargs: object) -> None:
        del kwargs

    def list_conversations(
        self,
        *,
        workspace_id: str | None | object = ...,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> Sequence[ProviderConversationLike]:
        del workspace_id, request_id, task_name, status, limit
        return []

    def reconcile_running_task_conversations(self, **kwargs: object) -> int:
        del kwargs
        return 0


__all__ = [
    "ProviderAdmissionStateLike",
    "ProviderConversationLike",
    "ProviderPolicyEventPort",
    "ProviderUsagePort",
    "NullProviderUsagePort",
    "TaskExecutionAttemptPort",
    "TaskExecutionAttemptRecordLike",
]

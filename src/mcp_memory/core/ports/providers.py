"""Provider-neutral persistence contracts used by application services."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol


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
        status: str | None = None,
        limit: int = 50,
    ) -> list[Any]: ...

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


__all__ = [
    "ProviderAdmissionStateLike",
    "ProviderUsagePort",
    "TaskExecutionAttemptPort",
    "TaskExecutionAttemptRecordLike",
]

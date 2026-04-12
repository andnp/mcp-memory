from __future__ import annotations

from enum import StrEnum


class RecoveryAction(StrEnum):
    FINALIZE_CANCELLATION = "finalize_cancellation"
    RETRY_DEAD_SUBPROCESS = "retry_dead_subprocess"
    FAIL_ABANDONED = "fail_abandoned"
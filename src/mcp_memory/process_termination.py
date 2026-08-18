from __future__ import annotations

import os
import signal
import time
from dataclasses import dataclass
from typing import Callable, Literal

TerminationScope = Literal["pid", "process_group"]


@dataclass(frozen=True)
class ProcessTerminationResult:
    signal_sequence: tuple[str, ...] = ()
    process_group_id: int | None = None
    escalated_to_sigkill: bool = False

    @property
    def signal_sent(self) -> bool:
        return bool(self.signal_sequence)


def wait_for_process_exit(
    pid: int,
    *,
    deadline: float,
    poll_interval_seconds: float,
    is_process_running: Callable[[int], bool],
) -> None:
    while time.monotonic() < deadline:
        if not is_process_running(pid):
            return
        time.sleep(poll_interval_seconds)
    raise RuntimeError(f"Timed out waiting for process exit: pid={pid}")


def send_process_signal(
    pid: int,
    sig: signal.Signals,
    *,
    scope: TerminationScope,
    suppress_permission_errors: bool = False,
) -> tuple[int | None, bool]:
    process_group_id = None
    if scope == "process_group":
        try:
            process_group_id = os.getpgid(pid)
        except (AttributeError, ProcessLookupError, OSError):
            process_group_id = None

        if process_group_id is not None and process_group_id > 0:
            try:
                os.killpg(process_group_id, sig)
                return process_group_id, True
            except PermissionError:
                if suppress_permission_errors:
                    return process_group_id, False
                raise
            except (AttributeError, ProcessLookupError, OSError):
                pass

    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return process_group_id, False
    except PermissionError:
        if suppress_permission_errors:
            return process_group_id, False
        raise
    return process_group_id, True


def terminate_process(
    pid: int,
    *,
    deadline: float,
    poll_interval_seconds: float,
    is_process_running: Callable[[int], bool],
    send_signal: Callable[[int, signal.Signals], int | None | tuple[int | None, bool]],
    wait_for_exit: Callable[..., None],
) -> ProcessTerminationResult:
    wait_timeout_seconds = max(deadline - time.monotonic(), poll_interval_seconds)
    process_group_id, signal_sent = _normalize_signal_dispatch(send_signal(pid, signal.SIGTERM))
    signal_sequence: list[str] = []
    if signal_sent:
        signal_sequence.append(signal.SIGTERM.name)
    else:
        return ProcessTerminationResult(process_group_id=process_group_id)

    try:
        wait_for_exit(
            pid,
            deadline=deadline,
            poll_interval_seconds=poll_interval_seconds,
        )
        return ProcessTerminationResult(
            signal_sequence=tuple(signal_sequence),
            process_group_id=process_group_id,
            escalated_to_sigkill=False,
        )
    except RuntimeError:
        if not is_process_running(pid):
            return ProcessTerminationResult(
                signal_sequence=tuple(signal_sequence),
                process_group_id=process_group_id,
                escalated_to_sigkill=False,
            )

        sigkill_group_id, sigkill_sent = _normalize_signal_dispatch(send_signal(pid, signal.SIGKILL))
        process_group_id = sigkill_group_id or process_group_id
        if not sigkill_sent:
            return ProcessTerminationResult(
                signal_sequence=tuple(signal_sequence),
                process_group_id=process_group_id,
                escalated_to_sigkill=False,
            )

        signal_sequence.append(signal.SIGKILL.name)
        wait_for_exit(
            pid,
            deadline=time.monotonic() + wait_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        return ProcessTerminationResult(
            signal_sequence=tuple(signal_sequence),
            process_group_id=process_group_id,
            escalated_to_sigkill=True,
        )


def _normalize_signal_dispatch(dispatch: int | None | tuple[int | None, bool]) -> tuple[int | None, bool]:
    if isinstance(dispatch, tuple):
        return dispatch
    return dispatch, True

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeAIProvider:
    responses: list[dict[str, Any]] = field(default_factory=lambda: [{"actions": []}])
    error: Exception | None = None
    error_sequence: list[Exception | None] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    call_count: int = 0

    async def ask(self, prompt: str) -> dict[str, Any]:
        self.prompts.append(prompt)
        self.call_count += 1

        if self.error_sequence:
            sequence_index = min(self.call_count - 1, len(self.error_sequence) - 1)
            sequence_error = self.error_sequence[sequence_index]
            if sequence_error is not None:
                raise sequence_error

        if self.error is not None:
            raise self.error

        if not self.responses:
            return {"actions": []}

        if len(self.responses) == 1:
            return self.responses[0]

        index = min(self.call_count - 1, len(self.responses) - 1)
        return self.responses[index]


class ConsolidationResponseFactory:
    @staticmethod
    def create(entry_indices: list[int], title: str, content: str) -> dict[str, Any]:
        return {
            "type": "create",
            "entry_indices": entry_indices,
            "title": title,
            "content": content,
        }

    @staticmethod
    def ignore(entry_indices: list[int]) -> dict[str, Any]:
        return {
            "type": "ignore",
            "entry_indices": entry_indices,
        }

    @classmethod
    def actions(cls, *actions: dict[str, Any]) -> dict[str, Any]:
        return {"actions": list(actions)}

@dataclass
class FakeAsyncProcess:
    pid: int = 4242
    stdout_text: str = ""
    stderr_text: str = ""
    returncode: int = 0
    raise_timeout: bool = False
    killed: bool = False
    waited: bool = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self.raise_timeout:
            raise asyncio.TimeoutError
        return (
            self.stdout_text.encode("utf-8"),
            self.stderr_text.encode("utf-8"),
        )

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> None:
        self.waited = True


@dataclass
class FakeSubprocessInstaller:
    processes: deque[FakeAsyncProcess] = field(default_factory=deque)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def add(self, *processes: FakeAsyncProcess) -> None:
        self.processes.extend(processes)

    async def __call__(self, *args: Any, **kwargs: Any) -> FakeAsyncProcess:
        self.calls.append((args, kwargs))
        if not self.processes:
            raise AssertionError("No fake subprocesses configured")
        return self.processes.popleft()

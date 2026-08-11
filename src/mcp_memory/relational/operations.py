from __future__ import annotations

class ReadMemoryRecordOperation:
    def __init__(self, search_service) -> None:
        self._search_service = search_service

    def execute(self, memory_id: str):
        return self._search_service.read_memory(memory_id)

from pathlib import Path

from mcp_memory.core.storage import get_memory_file_path


def normalize_memory_name(memory_id: str) -> str:
    if memory_id.startswith("memory:"):
        return memory_id.split("memory:", 1)[1]
    return memory_id


def resolve_memory_path(memory_root: Path, memory_id: str) -> Path | None:
    candidate = get_memory_file_path(memory_root, normalize_memory_name(memory_id))
    if candidate.exists():
        return candidate
    return None
from pathlib import Path


def ensure_memory_dirs(memory_path: Path) -> None:
    memory_path.mkdir(parents=True, exist_ok=True)
    (memory_path / "indices").mkdir(exist_ok=True)
    (memory_path / ".trash").mkdir(exist_ok=True)

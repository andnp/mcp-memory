from pathlib import Path


def normalize_path(path: str, docs_root: Path) -> str:
    candidate = Path(path)
    root = docs_root.resolve()

    try:
        return str(candidate.resolve().relative_to(root))
    except ValueError:
        return candidate.name if candidate.is_absolute() else str(candidate)


def matches_any_excluded(
    path: str,
    excluded_files: set[str],
    docs_root: Path,
) -> bool:
    normalized = normalize_path(path, docs_root)
    return (
        path in excluded_files
        or normalized in excluded_files
        or Path(normalized).name in excluded_files
    )
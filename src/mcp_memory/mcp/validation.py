from __future__ import annotations

from typing import Any


def require_string(arguments: dict[str, Any], field_name: str) -> str:
    value = arguments.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def optional_string(arguments: dict[str, Any], field_name: str) -> str | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    return normalized or None


def optional_positive_int(arguments: dict[str, Any], field_name: str, default: int) -> int:
    value = arguments.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return value


def string_list(
    arguments: dict[str, Any],
    field_name: str,
    required: bool = False,
) -> list[str]:
    value = arguments.get(field_name)
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings")
    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must be a list of strings")
        normalized = item.strip()
        if normalized:
            normalized_values.append(normalized)
    if required and not normalized_values:
        raise ValueError(f"{field_name} is required")
    return normalized_values


def optional_object(arguments: dict[str, Any], field_name: str) -> dict[str, object] | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def optional_bool(arguments: dict[str, Any], field_name: str, default: bool = False) -> bool:
    value = arguments.get(field_name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def validate_ingest_mutation_payload(
    *,
    entry_ids: list[int],
    content: str,
    title: str | None = None,
) -> None:
    if not entry_ids:
        raise ValueError("ingest mutations require at least one claimed entry id")
    if not content.strip():
        raise ValueError("ingest mutations require non-empty content")
    if title is not None and not title.strip():
        raise ValueError("ingest mutations require non-empty title")
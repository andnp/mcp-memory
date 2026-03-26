from __future__ import annotations

import json
from typing import Any


def coerce_text_summary(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def normalize_tag_values(tags: list[str]) -> list[str]:
    aliases = {
        "unit-test": "testing",
        "unit_tests": "testing",
        "tests": "testing",
        "test": "testing",
        "authn": "auth",
        "authz": "auth",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized_tag = tag.strip().lower().replace("_", "-")
        normalized_tag = aliases.get(normalized_tag, normalized_tag)
        if not normalized_tag or normalized_tag in seen:
            continue
        seen.add(normalized_tag)
        normalized.append(normalized_tag)
    return sorted(normalized)

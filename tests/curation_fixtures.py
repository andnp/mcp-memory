from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "curation"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "fixture_id",
    "description",
    "memories",
    "links",
    "allowed_actions",
    "forbidden_actions",
    "required_invariants",
    "acceptable_retention_reasons",
}
_MEMORY_KEYS = {"id", "project", "title", "content", "summary", "type", "status", "tags", "protected"}
_LINK_KEYS = {"source_id", "target_id", "type", "context"}
_ACTION_KEYS = {"operation", "target_ids"}


@dataclass(frozen=True)
class CurationFixture:
    fixture_id: str
    description: str
    memories: tuple[dict[str, Any], ...]
    links: tuple[dict[str, Any], ...]
    allowed_actions: tuple[dict[str, Any], ...]
    forbidden_actions: tuple[dict[str, Any], ...]
    required_invariants: tuple[str, ...]
    acceptable_retention_reasons: tuple[str, ...]


def load_curation_fixture(path: Path) -> CurationFixture:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _validate_fixture(payload, path)
    return CurationFixture(
        fixture_id=payload["fixture_id"],
        description=payload["description"],
        memories=tuple(payload["memories"]),
        links=tuple(payload["links"]),
        allowed_actions=tuple(payload["allowed_actions"]),
        forbidden_actions=tuple(payload["forbidden_actions"]),
        required_invariants=tuple(payload["required_invariants"]),
        acceptable_retention_reasons=tuple(payload["acceptable_retention_reasons"]),
    )


def load_all_curation_fixtures(root: Path = FIXTURE_ROOT) -> tuple[CurationFixture, ...]:
    paths = sorted(root.glob("*.json"))
    return tuple(load_curation_fixture(path) for path in paths)


def _validate_fixture(payload: Any, path: Path) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: fixture must be an object")
    if set(payload) != _TOP_LEVEL_KEYS:
        raise ValueError(f"{path}: unexpected or missing top-level keys")
    if payload["schema_version"] != 1:
        raise ValueError(f"{path}: unsupported schema version")
    for key in ("fixture_id", "description"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{path}: {key} must be a non-empty string")
    for key in ("memories", "links", "allowed_actions", "forbidden_actions", "required_invariants", "acceptable_retention_reasons"):
        if not isinstance(payload[key], list):
            raise ValueError(f"{path}: {key} must be a list")

    memory_ids: set[str] = set()
    for memory in payload["memories"]:
        if not isinstance(memory, dict) or set(memory) != _MEMORY_KEYS:
            raise ValueError(f"{path}: invalid memory shape")
        memory_id = memory["id"]
        if not isinstance(memory_id, str) or not memory_id or memory_id in memory_ids:
            raise ValueError(f"{path}: memory IDs must be unique non-empty strings")
        memory_ids.add(memory_id)
        if not all(isinstance(memory[key], str) for key in ("project", "title", "content", "summary", "type", "status")):
            raise ValueError(f"{path}: memory text fields must be strings")
        if not isinstance(memory["tags"], list) or not all(isinstance(tag, str) for tag in memory["tags"]):
            raise ValueError(f"{path}: memory tags must be strings")
        if not isinstance(memory["protected"], bool):
            raise ValueError(f"{path}: memory protected must be boolean")

    for link in payload["links"]:
        if not isinstance(link, dict) or set(link) != _LINK_KEYS:
            raise ValueError(f"{path}: invalid link shape")
        if not all(isinstance(link[key], str) and link[key] for key in _LINK_KEYS):
            raise ValueError(f"{path}: links must contain non-empty strings")
        if link["source_id"] not in memory_ids or link["target_id"] not in memory_ids:
            raise ValueError(f"{path}: links must reference fixture memories")

    for key in ("allowed_actions", "forbidden_actions"):
        for action in payload[key]:
            if not isinstance(action, dict) or set(action) != _ACTION_KEYS:
                raise ValueError(f"{path}: invalid {key} shape")
            if not isinstance(action["operation"], str) or not action["operation"]:
                raise ValueError(f"{path}: action operation must be non-empty")
            if not isinstance(action["target_ids"], list) or not action["target_ids"]:
                raise ValueError(f"{path}: action target_ids must be non-empty")
            if not all(target in memory_ids for target in action["target_ids"]):
                raise ValueError(f"{path}: action targets must reference fixture memories")

    for key in ("required_invariants", "acceptable_retention_reasons"):
        if not all(isinstance(value, str) and value for value in payload[key]):
            raise ValueError(f"{path}: {key} must contain non-empty strings")

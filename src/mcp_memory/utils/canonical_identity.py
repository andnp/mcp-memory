"""Domain-agnostic canonical-JSON normalization and deterministic identity derivation.

Backend-neutral mechanics only: no knowledge of any particular record shape,
domain vocabulary, or storage backend belongs here.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID, uuid5


def normalize_text(value: str, *, collapse: bool = False) -> str:
    """Apply NFC unicode normalization and canonical line endings."""
    value = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    return value.strip() if collapse else value


def normalize(value: Any) -> Any:
    """Recursively normalize a value into canonical-JSON-safe form."""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support non-finite numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON object keys must be strings")
            normalized_key = normalize_text(key)
            if normalized_key in result:
                raise ValueError(f"duplicate canonical JSON key: {normalized_key!r}")
            result[normalized_key] = normalize(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [normalize(item) for item in value]
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonicalize(value: Any) -> str:
    """Return the canonical JSON string used for deterministic hashing and identity."""
    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any, *, version: str = "v1") -> str:
    """Return a versioned, deterministic content hash of value."""
    digest = hashlib.sha256(canonicalize(value).encode("utf-8")).hexdigest()
    return f"{version}:{digest}"


def stable_uuid5(namespace: UUID, *parts: Any) -> UUID:
    """Derive a deterministic UUID5 from one or more canonicalized parts."""
    name = parts[0] if len(parts) == 1 else list(parts)
    return uuid5(namespace, canonicalize(name))

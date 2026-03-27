from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageBootstrapState:
    backend: str
    schema_metadata_present: bool
    schema_version: int | None
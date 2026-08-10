"""Versioned, privacy-safe search-quality corpus definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class QueryClass(StrEnum):
    """Stable categories used to compare search quality by intent."""

    BROAD = "broad"
    EXACT = "exact"
    HISTORICAL = "historical"
    DEGRADED = "degraded"
    RELATIONAL = "relational"


@dataclass(frozen=True, slots=True)
class SearchQualityCase:
    """One labeled query with workspace and expected-result metadata."""

    case_id: str
    evaluation_label: str
    query: str
    query_class: QueryClass
    intent: str
    workspace: str
    expected_labels: tuple[str, ...]
    acceptable_top_k: int = 5

    def __post_init__(self) -> None:
        """Reject entries that cannot produce a meaningful evaluation."""
        for field_name in (
            "case_id",
            "evaluation_label",
            "query",
            "intent",
            "workspace",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.expected_labels:
            raise ValueError("expected_labels must not be empty")
        if self.acceptable_top_k < 1:
            raise ValueError("acceptable_top_k must be positive")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SearchQualityCase:
        """Build a case from the versioned JSON representation."""
        return cls(
            case_id=data["id"],
            evaluation_label=data["evaluation_label"],
            query=data["query"],
            query_class=QueryClass(data["query_class"]),
            intent=data["intent"],
            workspace=data["workspace"],
            expected_labels=tuple(data["expected_labels"]),
            acceptable_top_k=data.get("acceptable_top_k", 5),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the case without memory content or generated IDs."""
        return {
            "id": self.case_id,
            "evaluation_label": self.evaluation_label,
            "query": self.query,
            "query_class": self.query_class.value,
            "intent": self.intent,
            "workspace": self.workspace,
            "expected_labels": list(self.expected_labels),
            "acceptable_top_k": self.acceptable_top_k,
        }


@dataclass(frozen=True, slots=True)
class SearchQualityCorpus:
    """Immutable corpus whose version is part of every evaluation result."""

    version: str
    entries: tuple[SearchQualityCase, ...]

    def __post_init__(self) -> None:
        """Require unique case and evaluation labels within one corpus."""
        if not self.version.strip():
            raise ValueError("version must not be empty")
        case_ids = [entry.case_id for entry in self.entries]
        labels = [entry.evaluation_label for entry in self.entries]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique")
        if len(labels) != len(set(labels)):
            raise ValueError("evaluation_label values must be unique")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> SearchQualityCorpus:
        """Build a corpus from the versioned JSON representation."""
        return cls(
            version=data["version"],
            entries=tuple(
                SearchQualityCase.from_mapping(entry)
                for entry in data.get("entries", [])
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize the corpus without private record payloads."""
        return {
            "version": self.version,
            "entries": [entry.to_mapping() for entry in self.entries],
        }


DEFAULT_CORPUS_PATH = Path(__file__).with_name("corpus.v1.json")


def load_corpus(path: Path = DEFAULT_CORPUS_PATH) -> SearchQualityCorpus:
    """Load a versioned search-quality corpus from JSON."""
    with path.open(encoding="utf-8") as stream:
        return SearchQualityCorpus.from_mapping(json.load(stream))

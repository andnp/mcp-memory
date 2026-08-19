import math
from uuid import UUID

import pytest
from mcp_memory.utils.canonical_identity import canonicalize, normalize, normalize_text, stable_hash, stable_uuid5

NAMESPACE = UUID("2f5b4bb8-4f1b-5e2f-8f9d-6a7d8e9c0b1a")


def test_canonicalize_normalizes_unicode_newlines_and_sorts_keys() -> None:
    assert canonicalize({"b": "é\r", "a": "x"}) == '{"a":"x","b":"é\\n"}'


def test_canonicalize_rejects_non_finite_floats() -> None:
    with pytest.raises(ValueError):
        canonicalize({"value": math.inf})


def test_canonicalize_rejects_duplicate_keys_after_normalization() -> None:
    with pytest.raises(ValueError):
        normalize({"á": 1, "á": 2})


def test_normalize_text_collapses_only_when_requested() -> None:
    assert normalize_text("  x  ") == "  x  "
    assert normalize_text("  x  ", collapse=True) == "x"


def test_stable_hash_is_deterministic_and_versioned() -> None:
    first = stable_hash({"a": 1, "b": 2})
    second = stable_hash({"b": 2, "a": 1})
    assert first == second
    assert first.startswith("v1:")
    assert stable_hash({"a": 1, "b": 3}) != first


def test_stable_uuid5_is_deterministic_and_namespace_sensitive() -> None:
    first = stable_uuid5(NAMESPACE, {"a": 1})
    second = stable_uuid5(NAMESPACE, {"a": 1})
    other_namespace = stable_uuid5(UUID(int=0), {"a": 1})
    assert first == second
    assert first != other_namespace
    assert stable_uuid5(NAMESPACE, {"a": 1}, {"b": 2}) != stable_uuid5(NAMESPACE, {"b": 2}, {"a": 1})

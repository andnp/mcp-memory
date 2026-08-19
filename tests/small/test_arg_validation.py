from __future__ import annotations

import pytest

from mcp_memory.toolkit.arg_validation import (
    optional_bool,
    optional_object,
    optional_positive_int,
    optional_string,
    require_string,
    string_list,
)

pytestmark = pytest.mark.small


def test_require_string_normalizes_and_strips() -> None:
    assert require_string({"field": "  value  "}, "field") == "value"


def test_require_string_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        require_string({"field": 1}, "field")


def test_require_string_rejects_blank() -> None:
    with pytest.raises(ValueError):
        require_string({"field": "   "}, "field")


def test_optional_string_returns_none_when_absent() -> None:
    assert optional_string({}, "field") is None


def test_optional_string_returns_none_for_blank() -> None:
    assert optional_string({"field": "   "}, "field") is None


def test_optional_string_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        optional_string({"field": 1}, "field")


def test_optional_positive_int_uses_default() -> None:
    assert optional_positive_int({}, "field", 5) == 5


def test_optional_positive_int_rejects_bool() -> None:
    with pytest.raises(TypeError):
        optional_positive_int({"field": True}, "field", 5)


def test_optional_positive_int_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        optional_positive_int({"field": 0}, "field", 5)


def test_string_list_filters_blank_entries() -> None:
    assert string_list({"field": ["a", "  ", " b "]}, "field") == ["a", "b"]


def test_string_list_defaults_to_empty_when_absent() -> None:
    assert string_list({}, "field") == []


def test_string_list_required_rejects_missing() -> None:
    with pytest.raises(ValueError):
        string_list({}, "field", required=True)


def test_string_list_required_rejects_all_blank() -> None:
    with pytest.raises(ValueError):
        string_list({"field": ["  "]}, "field", required=True)


def test_string_list_rejects_non_string_items() -> None:
    with pytest.raises(TypeError):
        string_list({"field": ["a", 1]}, "field")


def test_optional_object_returns_copy() -> None:
    original = {"a": 1}
    result = optional_object({"field": original}, "field")
    assert result == original
    assert result is not original


def test_optional_object_returns_none_when_absent() -> None:
    assert optional_object({}, "field") is None


def test_optional_object_rejects_non_dict() -> None:
    with pytest.raises(TypeError):
        optional_object({"field": "not-a-dict"}, "field")


def test_optional_bool_uses_default() -> None:
    assert optional_bool({}, "field", True) is True


def test_optional_bool_rejects_non_bool() -> None:
    with pytest.raises(TypeError):
        optional_bool({"field": "true"}, "field")

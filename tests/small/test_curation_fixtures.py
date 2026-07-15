from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import tests.curation_fixtures as curation_fixtures
from tests.curation_fixtures import FIXTURE_ROOT, load_all_curation_fixtures, load_curation_fixture


pytestmark = pytest.mark.small


EXPECTED_FIXTURES = {
    "bad-summary",
    "contradiction",
    "cross-project-false-friend",
    "duplicate",
    "focused-oversized",
    "link",
    "no-op",
    "protected",
    "split",
}


def test_golden_fixtures_load_and_validate() -> None:
    fixtures = load_all_curation_fixtures()

    assert {fixture.fixture_id for fixture in fixtures} == EXPECTED_FIXTURES
    assert len(fixtures) == len(EXPECTED_FIXTURES)
    assert all(fixture.memories for fixture in fixtures)
    assert all(fixture.required_invariants for fixture in fixtures)
    assert all(fixture.acceptable_retention_reasons for fixture in fixtures)


def test_golden_fixture_loading_is_sorted_and_repeatable() -> None:
    first = load_all_curation_fixtures()
    second = load_all_curation_fixtures()

    assert first == second
    assert [fixture.fixture_id for fixture in first] == sorted(EXPECTED_FIXTURES)


def test_loader_rejects_schema_drift(tmp_path: Path) -> None:
    source = FIXTURE_ROOT / "no-op.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    candidate = tmp_path / source.name
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="top-level keys"):
        load_curation_fixture(candidate)


def test_fixture_loader_has_no_production_or_provider_dependency() -> None:
    loader_source = inspect.getsource(curation_fixtures)

    assert "mcp_memory" not in loader_source
    assert "subprocess" not in loader_source
    assert "http" not in loader_source.lower()

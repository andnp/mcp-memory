import pytest

from mcp_memory.core.ports.memory import MemoryLink
from mcp_memory.core.search_graph import GraphCandidateExpander


def test_expand_filters_links_and_applies_seed_and_neighbor_limits():
    links = {
        "seed_high": [
            MemoryLink("seed_high", "ignored", "UNSUPPORTED", ""),
            *[
                MemoryLink("seed_high", f"high_{index}", "DEPENDS_ON", "")
                for index in range(11)
            ],
        ],
        "seed_middle": [MemoryLink("seed_middle", "middle", "AMENDS", "")],
        "seed_low": [MemoryLink("seed_low", "low", "CONTRADICTS", "")],
        "seed_fourth": [MemoryLink("seed_fourth", "fourth", "DEPENDS_ON", "")],
    }

    expanded = GraphCandidateExpander(lambda seed_id: links[seed_id]).expand(
        {
            "seed_low": 0.1,
            "seed_high": 0.9,
            "seed_middle": 0.5,
            "seed_fourth": 0.05,
        }
    )

    assert set(expanded) == {"middle", "low", *(f"high_{index}" for index in range(10))}


def test_expand_keeps_highest_duplicate_contribution_with_provenance():
    links = {
        "first": [MemoryLink("first", "target", "AMENDS", "")],
        "winner": [MemoryLink("winner", "target", "DEPENDS_ON", "")],
    }

    expanded = GraphCandidateExpander(lambda seed_id: links[seed_id]).expand(
        {"first": 0.8, "winner": 0.9}
    )

    assert expanded["target"].rrf_score == pytest.approx(0.63)
    assert expanded["target"].seed_id == "winner"
    assert expanded["target"].link_type == "DEPENDS_ON"


def test_expand_keeps_first_provenance_on_equal_contributions():
    links = {
        "first": [MemoryLink("first", "target", "DEPENDS_ON", "")],
        "second": [MemoryLink("second", "target", "DEPENDS_ON", "")],
    }

    expanded = GraphCandidateExpander(lambda seed_id: links[seed_id]).expand(
        {"first": 0.8, "second": 0.8}
    )

    assert expanded["target"].rrf_score == pytest.approx(0.56)
    assert expanded["target"].seed_id == "first"
    assert expanded["target"].link_type == "DEPENDS_ON"

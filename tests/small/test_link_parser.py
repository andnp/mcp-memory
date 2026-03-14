import pytest

from mcp_memory.core.link_parser import extract_links, normalize_tag


pytestmark = pytest.mark.small


def test_extract_links_infers_memory_and_external_edge_types() -> None:
    content = (
        "This depends on [[memory:roadmap]] for the next step. "
        "Also fix the bug tracked in [[src/server.py]]."
    )

    links = extract_links(content)

    assert len(links) == 2
    assert links[0].target == "memory:roadmap"
    assert links[0].is_memory_link is True
    assert links[0].edge_type == "DEPENDS_ON"
    assert links[1].target == "src/server.py"
    assert links[1].is_memory_link is False
    assert links[1].edge_type == "debugs"


def test_normalize_tag_trims_and_lowercases() -> None:
    assert normalize_tag("  Testing-Strategy  ") == "testing-strategy"
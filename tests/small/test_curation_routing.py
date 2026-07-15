import pytest

from mcp_memory.core.curation_routing import (
    CurationFinding,
    CurationOperation,
    MaintenanceFamily,
    NeedsDifferentSpecialist,
    primary_family_for_finding,
    primary_family_for_operation,
    route_finding,
)


@pytest.mark.parametrize(
    ("finding", "family"),
    [
        ("summary", MaintenanceFamily.SUMMARIZER),
        ("taxonomy", MaintenanceFamily.TAXONOMIST),
        ("graph", MaintenanceFamily.GRAPH_LINKER),
        ("conflict", MaintenanceFamily.CONFLICT_REVIEW),
        ("duplicate", MaintenanceFamily.DEDUPLICATOR),
        ("defragmentation", MaintenanceFamily.DEFRAGMENTER),
        ("plan_aging", MaintenanceFamily.PROJECT_MANAGER),
        ("fact_degradation", MaintenanceFamily.FACT_CHECKER),
        ("curator_owned_split", MaintenanceFamily.CURATOR),
    ],
)
def test_findings_route_to_documented_primary_family(
    finding: str, family: MaintenanceFamily
) -> None:
    assert primary_family_for_finding(finding) is family

    result = route_finding(finding)
    assert isinstance(result, NeedsDifferentSpecialist)
    assert result.result == "needs_different_specialist"
    assert result.finding is CurationFinding(finding)
    assert result.primary_family is family


@pytest.mark.parametrize(
    ("operation", "family"),
    [
        (CurationOperation.NORMALIZE_MEMORY, MaintenanceFamily.CURATOR),
        (CurationOperation.REWRITE_MEMORY, MaintenanceFamily.CURATOR),
        (CurationOperation.CREATE_LINK, MaintenanceFamily.GRAPH_LINKER),
        (CurationOperation.REMOVE_LINK, MaintenanceFamily.GRAPH_LINKER),
        (CurationOperation.MERGE_MEMORIES, MaintenanceFamily.DEDUPLICATOR),
        (CurationOperation.SPLIT_MEMORY, MaintenanceFamily.CURATOR),
        (CurationOperation.ARCHIVE_MEMORY, MaintenanceFamily.CURATOR),
    ],
)
def test_operations_route_to_documented_primary_family(
    operation: CurationOperation, family: MaintenanceFamily
) -> None:
    assert primary_family_for_operation(operation) is family


def test_unknown_routing_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        primary_family_for_finding("unknown")
    with pytest.raises(ValueError):
        primary_family_for_operation("unknown")

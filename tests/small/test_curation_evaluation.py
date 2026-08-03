from __future__ import annotations

import inspect

from mcp_memory.core import curation_evaluation
from mcp_memory.core.curation_evaluation import (
    ReplayCase,
    ReplayResult,
    ReplaySnapshot,
    evaluate_query_replay,
)


def test_query_replay_reports_rank_irrelevance_zero_results_and_payload_delta() -> None:
    report = evaluate_query_replay(
        [
            ReplayCase(
                query_id="query-1",
                intended_memory_ids=("wanted",),
                before=ReplaySnapshot(
                    (
                        ReplayResult("noise", payload_size=4),
                        ReplayResult("wanted", payload_size=8),
                    )
                ),
                after=ReplaySnapshot((ReplayResult("wanted", payload_size=5),)),
                top_k=2,
            ),
            ReplayCase(
                query_id="query-2",
                intended_memory_ids=("missing",),
                before=ReplaySnapshot(),
                after=ReplaySnapshot((ReplayResult("noise", payload_size=3),)),
            ),
        ]
    )

    first, second = report.cases
    assert (first.intended_rank_before, first.intended_rank_after) == (2, 1)
    assert (first.irrelevant_top_results_before, first.irrelevant_top_results_after) == (1, 0)
    assert first.payload_size_change == -7
    assert second.zero_result_change == -1
    assert (report.zero_results_before, report.zero_results_after) == (1, 0)
    assert report.payload_size_change == -4


def test_query_replay_is_deterministic_and_uses_canonical_payload_size() -> None:
    case = ReplayCase(
        query_id="stable",
        intended_memory_ids=("wanted",),
        before=ReplaySnapshot((ReplayResult("wanted", {"b": 2, "a": "é"}),)),
        after=ReplaySnapshot((ReplayResult("wanted", {"a": "é", "b": 2}),)),
    )

    assert evaluate_query_replay([case]) == evaluate_query_replay([case])
    assert case.before.payload_size() == case.after.payload_size()


def test_evaluator_has_no_provider_or_public_request_path_dependency() -> None:
    source = inspect.getsource(curation_evaluation)

    assert "mcp_memory.providers" not in source
    assert "mcp_memory.server" not in source
    assert "server" not in source.lower()
    assert "http" not in source.lower()


def test_query_replay_marks_neutral_evidence_without_utility_or_regression() -> None:
    report = evaluate_query_replay(
        [
            ReplayCase(
                query_id="blank",
                intended_memory_ids=("wanted",),
                before=ReplaySnapshot((ReplayResult("wanted"),)),
                after=ReplaySnapshot(),
                query_text=" ",
            ),
            ReplayCase(
                query_id="incomplete",
                intended_memory_ids=("wanted",),
                before=ReplaySnapshot((ReplayResult("wanted"),)),
                after=ReplaySnapshot(),
                replay_complete=False,
            ),
        ]
    )

    assert [case.neutral_reason for case in report.cases] == [
        "blank_query",
        "incomplete_replay",
    ]
    assert report.retrieval_regression_count == 0
    assert report.useful_work_count == 0
    assert report.retrieval_utility_delta == 0.0

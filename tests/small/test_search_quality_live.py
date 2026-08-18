from __future__ import annotations

import json

import pytest

from benchmarks.search_quality.corpus import (
    QueryClass,
    SearchQualityCase,
    SearchQualityCorpus,
)
from benchmarks.search_quality.runner import run_live_daemon

pytestmark = pytest.mark.small


def _corpus() -> SearchQualityCorpus:
    return SearchQualityCorpus(
        version="live-test",
        entries=(
            SearchQualityCase(
                case_id="exact-case",
                evaluation_label="exact-target",
                query="exact query",
                query_class=QueryClass.EXACT,
                intent="retrieve exact target",
                workspace="workspace",
                expected_labels=("exact-target",),
            ),
            SearchQualityCase(
                case_id="historical-case",
                evaluation_label="historical-target",
                query="historical query",
                query_class=QueryClass.HISTORICAL,
                intent="retrieve historical target",
                workspace="workspace",
                expected_labels=("historical-target",),
            ),
        ),
    )


def _response(
    label: str | None,
    *,
    request_id: int = 17,
    degraded: bool = False,
    memory_id: str | None = None,
) -> dict[str, object]:
    payload = {
        "status": "ok",
        "results": [
            {
                **({"evaluation_label": label} if label is not None else {}),
                "memory_id": memory_id,
                "private": "do-not-store",
            }
        ],
        "search_diagnostics": {
            "degraded": degraded,
            "transport": {"request_id": request_id},
        },
    }
    return {"contents": [{"type": "text", "text": json.dumps(payload)}]}


def test_search_quality_live_runner_retains_timing_and_request_id() -> None:
    """Record client timing and transport identity without raw payloads."""
    calls: list[tuple[str, float]] = []

    def request(case: SearchQualityCase, timeout_seconds: float) -> object:
        calls.append((case.evaluation_label, timeout_seconds))
        return _response(case.evaluation_label, request_id=41)

    report = run_live_daemon(_corpus(), request_fn=request, timeout_seconds=3.5)
    serialized = json.dumps(report.to_mapping(), sort_keys=True)

    assert [case.status for case in report.cases] == ["ok", "ok"]
    assert [case.request_id for case in report.cases] == [41, 41]
    assert all(case.latency_ms is not None for case in report.cases)
    assert report.metrics.hit_at_1 == 1.0
    assert calls == [("exact-target", 3.5), ("historical-target", 3.5)]
    assert "do-not-store" not in serialized
    assert "exact query" not in serialized


def test_search_quality_live_runner_marks_unscored_real_results() -> None:
    """Expose that real memory IDs are unscored without an explicit labeler."""
    report = run_live_daemon(
        _corpus(),
        request_fn=lambda case, _timeout: _response(
            None,
            memory_id=f"memory-{case.case_id}",
        ),
    )

    assert report.metrics.hit_at_1 == 0.0
    assert all(not case.scored for case in report.cases)
    assert report.to_mapping()["scored_case_count"] == 0
    assert report.to_mapping()["quality_scored"] is False


def test_search_quality_live_runner_scores_memory_id_map() -> None:
    """Translate daemon memory IDs into corpus labels before evaluating quality."""
    label_map = {
        "memory-exact-case": "exact-target",
        "memory-historical-case": "historical-target",
    }

    def label_result(result: dict[str, object]) -> str | None:
        memory_id = result.get("memory_id")
        return label_map.get(memory_id) if isinstance(memory_id, str) else None

    report = run_live_daemon(
        _corpus(),
        request_fn=lambda case, _timeout: _response(
            case.evaluation_label,
            memory_id=f"memory-{case.case_id}",
        ),
        result_labeler=label_result,
    )

    assert report.metrics.hit_at_1 == 1.0
    assert all(case.scored for case in report.cases)
    assert report.to_mapping()["scored_case_count"] == 2
    assert report.to_mapping()["quality_scored"] is True


@pytest.mark.parametrize(
    ("failure", "status", "error_code"),
    (
        (OSError("socket missing"), "unavailable", "oserror"),
        (TimeoutError("request timed out"), "timeout", "request_timeout"),
    ),
)
def test_search_quality_live_runner_continues_after_transport_failure(
    failure: Exception,
    status: str,
    error_code: str,
) -> None:
    """Turn transport failures into case outcomes and continue the corpus."""
    responses: list[object] = [failure, _response("historical-target")]

    def request(_case: SearchQualityCase, _timeout_seconds: float) -> object:
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    report = run_live_daemon(_corpus(), request_fn=request)

    assert [case.status for case in report.cases] == [status, "ok"]
    assert report.cases[0].error_code == error_code
    assert report.cases[1].request_id == 17
    assert report.metrics.hit_at_1 == 0.5


def test_search_quality_live_runner_reports_malformed_response() -> None:
    """Classify malformed daemon text without aborting later searches."""
    responses: list[object] = [
        {"contents": [{"type": "text", "text": "not-json"}]},
        _response("historical-target"),
    ]

    def request(_case: SearchQualityCase, _timeout_seconds: float) -> object:
        return responses.pop(0)

    report = run_live_daemon(_corpus(), request_fn=request)

    assert [case.status for case in report.cases] == ["malformed_response", "ok"]
    assert report.cases[0].error_code == "malformed_response"
    assert report.metrics.hit_at_1 == 0.5


def test_search_quality_live_runner_reports_degraded_response() -> None:
    """Preserve useful results while marking degraded search diagnostics."""
    report = run_live_daemon(
        _corpus(),
        request_fn=lambda case, _timeout: _response(
            case.evaluation_label,
            degraded=True,
        ),
    )

    assert [case.status for case in report.cases] == ["degraded", "degraded"]
    assert all(case.scored for case in report.cases)
    assert report.metrics.hit_at_1 == 1.0


def test_search_quality_live_runner_reports_unavailable_daemon(monkeypatch) -> None:
    """Report every case when daemon startup is unavailable."""
    from benchmarks.search_quality import runner

    def unavailable() -> object:
        raise OSError("daemon unavailable")

    monkeypatch.setattr(runner, "ensure_daemon_started", unavailable)

    report = run_live_daemon(_corpus())

    assert [case.status for case in report.cases] == ["unavailable", "unavailable"]
    assert all(case.error_code == "oserror" for case in report.cases)
    assert report.metrics.hit_at_1 == 0.0

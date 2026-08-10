from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from benchmarks import search_quality_live


pytestmark = pytest.mark.small


@dataclass(frozen=True)
class _FakeCase:
    status: str


@dataclass(frozen=True)
class _FakeReport:
    cases: tuple[_FakeCase, ...]
    mapping: dict[str, object]

    def to_mapping(self) -> dict[str, object]:
        return self.mapping


def test_main_prints_privacy_safe_metrics_for_degraded_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify a degraded live run remains successful and exposes its metrics.

    The CLI must emit the runner's privacy-safe mapping without adding query
    text or result payloads to the JSON output.
    """
    calls: list[tuple[str | None, float]] = []
    report = _FakeReport(
        cases=(_FakeCase("degraded"),),
        mapping={
            "corpus_version": "test",
            "metrics": {"hit_at_1": 0.5},
            "cases": [{"status": "degraded"}],
        },
    )

    def fake_runner(*, workspace_root: str | None, timeout_seconds: float) -> _FakeReport:
        calls.append((workspace_root, timeout_seconds))
        return report

    monkeypatch.setattr(search_quality_live, "run_live_daemon", fake_runner)

    exit_code = search_quality_live.main(
        ["--timeout", "2.5", "--workspace-root", "/tmp/workspace"]
    )

    assert exit_code == 0
    assert calls == [("/tmp/workspace", 2.5)]
    output = capsys.readouterr().out
    assert json.loads(output)["metrics"] == {"hit_at_1": 0.5}
    assert "raw query text" not in output
    assert "result payload" not in output


def test_main_returns_failure_for_unavailable_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Verify daemon transport failure returns nonzero while retaining metrics.

    CI callers need a failure status, while operators still need the report
    details that explain the degraded benchmark outcome.
    """
    report = _FakeReport(
        cases=(_FakeCase("unavailable"),),
        mapping={
            "corpus_version": "test",
            "metrics": {"hit_at_1": 0.0},
            "cases": [{"status": "unavailable", "error_code": "oserror"}],
        },
    )
    monkeypatch.setattr(search_quality_live, "run_live_daemon", lambda **_: report)

    exit_code = search_quality_live.main([])

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["metrics"]["hit_at_1"] == 0.0
    assert payload["cases"][0]["error_code"] == "oserror"

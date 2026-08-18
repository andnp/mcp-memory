from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from benchmarks import search_quality_live

pytestmark = pytest.mark.small


@dataclass(frozen=True)
class _FakeCase:
    status: str
    scored: bool = False


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
        cases=(_FakeCase("degraded", scored=True),),
        mapping={
            "corpus_version": "test",
            "metrics": {"hit_at_1": 0.5},
            "cases": [{"status": "degraded"}],
        },
    )

    def fake_runner(
        *,
        workspace_root: str | None,
        timeout_seconds: float,
        result_labeler=None,
    ) -> _FakeReport:
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


def test_main_returns_unscored_exit_code_without_label_map(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail closed when transport succeeds but no quality case was scored."""
    report = _FakeReport(
        cases=(_FakeCase("ok"),),
        mapping={"quality_scored": False, "scored_case_count": 0},
    )
    monkeypatch.setattr(search_quality_live, "run_live_daemon", lambda **_: report)

    assert search_quality_live.main([]) == 2
    assert json.loads(capsys.readouterr().out)["quality_scored"] is False


def test_main_forwards_memory_id_label_map_without_serializing_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Use a local label map for scoring while keeping its contents private."""
    label_map_path = tmp_path / "labels.json"
    label_map_path.write_text('{"memory-1": "exact-target"}', encoding="utf-8")
    report = _FakeReport(
        cases=(_FakeCase("ok", scored=True),),
        mapping={"quality_scored": True, "scored_case_count": 1},
    )
    received: list[object] = []

    def fake_runner(*, result_labeler, **_kwargs) -> _FakeReport:
        received.append(result_labeler({"memory_id": "memory-1"}))
        return report

    monkeypatch.setattr(search_quality_live, "run_live_daemon", fake_runner)

    assert search_quality_live.main(["--label-map", str(label_map_path)]) == 0
    assert received == ["exact-target"]
    assert "memory-1" not in capsys.readouterr().out


def test_main_rejects_malformed_label_map(
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject malformed label-map JSON before contacting the daemon."""
    label_map_path = tmp_path / "labels.json"
    label_map_path.write_text("not-json", encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        search_quality_live.main(["--label-map", str(label_map_path)])

    assert raised.value.code == 2
    assert "invalid label map" in capsys.readouterr().err


@pytest.mark.parametrize("contents", ['["memory-1"]', '{"memory-1": []}'])
def test_main_rejects_invalid_label_map_values(
    tmp_path,
    contents: str,
) -> None:
    """Reject label maps whose JSON shape cannot identify corpus labels."""
    label_map_path = tmp_path / "labels.json"
    label_map_path.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit) as raised:
        search_quality_live.main(["--label-map", str(label_map_path)])

    assert raised.value.code == 2

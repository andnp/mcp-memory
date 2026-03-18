from __future__ import annotations

from mcp_memory.core.summaries import build_deterministic_summary


def test_build_deterministic_summary_prefers_heading_and_signal_sentence() -> None:
    summary = build_deterministic_summary(
        title="Observability & Provider Telemetry",
        content=(
            "Scratchpad intro.\n\n"
            "## Monitoring Surfaces\n"
            "Provider telemetry carries task attribution into the dashboard, API, and stats command.\n"
            "Background tasks can now be costed per task name."
        ),
        memory_type="fact",
    )

    assert "telemetry" in summary.lower()
    assert "task attribution" in summary.lower() or "dashboard" in summary.lower()


def test_build_deterministic_summary_consolidates_reflection_bullets() -> None:
    summary = build_deterministic_summary(
        title="Reflection: Product Feedback and E2E Progress",
        content=(
            "Consolidated observations:\n"
            "- Product Feedback: Search summaries were too vague.\n"
            "- Daemon-backed E2E Progress: Smoke tests now hit thin-client routing.\n"
            "- QA Lessons: Verification should inspect provider conversation logs."
        ),
        memory_type="reflection",
    )

    assert summary.startswith("Consolidates ")
    assert "Product Feedback" in summary
    assert "Daemon-backed E2E Progress" in summary


def test_build_deterministic_summary_strips_timestamp_noise() -> None:
    summary = build_deterministic_summary(
        title="Ingest repair note",
        content=(
            "- [2026-03-17 12:00] Implemented provider-aware cooldowns for Gemini CLI retries.\n"
            "- [2026-03-17 12:05] Verified the worker honors exception-specific retry delays."
        ),
        memory_type="observation",
    )

    assert "[2026-03-17" not in summary
    assert "cooldowns" in summary.lower() or "retry delays" in summary.lower()

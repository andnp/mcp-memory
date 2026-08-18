from __future__ import annotations

import pytest

from mcp_memory.core.task_handlers.curator_support import curator_summary_claims_mutating_actions

pytestmark = pytest.mark.small


def test_curator_summary_detects_actions_before_declined_follow_up() -> None:
    summary = (
        "Rewrote and retagged two records. Reviewed adjacent clusters; "
        "declined merges because the records were distinct."
    )

    assert curator_summary_claims_mutating_actions(summary)


def test_curator_summary_ignores_declined_only_cleanup() -> None:
    summary = "No mutations were made; merges and rewrites were considered but declined."

    assert not curator_summary_claims_mutating_actions(summary)


def test_curator_summary_detects_link_cleanup_actions() -> None:
    assert curator_summary_claims_mutating_actions("Removed two misleading links and added one related link.")

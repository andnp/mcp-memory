from pathlib import Path

import pytest

from mcp_memory.core.consolidation import FastConsolidation, SlowConsolidation
from tests.sdk.providers import ConsolidationResponseFactory, FakeAIProvider


pytestmark = pytest.mark.medium


def _read_markdown_files(memory_path: Path) -> list[tuple[str, str]]:
    return sorted(
        (
            path.name,
            path.read_text(encoding="utf-8"),
        )
        for path in memory_path.glob("*.md")
    )


def test_fast_consolidation_groups_related_entries(
    system1_journal,
    memory_path: Path,
) -> None:
    system1_journal.record("import bug blocks the memory tests")
    system1_journal.record("fix import bug in the server wiring")
    system1_journal.record("design dashboard palette later")

    consolidator = FastConsolidation(
        system1_journal,
        memory_path,
        similarity_threshold=0.25,
    )

    created_files = consolidator.consolidate()
    created_contents = _read_markdown_files(memory_path)

    assert len(created_files) == 2
    assert [name for name, _ in created_contents] == sorted(created_files)
    assert system1_journal.count_by_status() == {"processed": 3}
    assert any("import bug blocks the memory tests" in content for _, content in created_contents)
    assert any("design dashboard palette later" in content for _, content in created_contents)


def test_fast_consolidation_respects_batch_size(
    system1_journal,
    memory_path: Path,
) -> None:
    for index in range(5):
        system1_journal.record(f"note {index} about imports and tests")

    consolidator = FastConsolidation(system1_journal, memory_path, batch_size=2)

    created_files = consolidator.consolidate()

    assert len(created_files) == 1
    assert system1_journal.count_by_status() == {"pending": 3, "processed": 2}


@pytest.mark.asyncio
async def test_slow_consolidation_uses_fake_provider_actions(
    system1_journal,
    memory_path: Path,
) -> None:
    first = system1_journal.record("record import cleanup notes")
    second = system1_journal.record("track medium test coverage gaps")
    third = system1_journal.record("ignore this fleeting scratch note")

    provider = FakeAIProvider(
        responses=[
            ConsolidationResponseFactory.actions(
                ConsolidationResponseFactory.create(
                    [0, 1],
                    "Import Stability Plan",
                    "Bundle import cleanup with medium test coverage.",
                ),
                ConsolidationResponseFactory.ignore([2]),
            )
        ]
    )

    consolidator = SlowConsolidation(system1_journal, memory_path, provider)

    created_files = await consolidator.consolidate()
    created_contents = _read_markdown_files(memory_path)

    assert len(provider.prompts) == 1
    assert len(created_files) == 1
    assert created_files[0].startswith("import-stability-plan-")
    assert provider.prompts[0].count("[") >= 3
    assert created_contents[0][1].endswith("Bundle import cleanup with medium test coverage.")
    assert system1_journal.count_by_status() == {"processed": 3}
    recent_ids = [entry.id for entry in system1_journal.get_recent(limit=3)]
    assert recent_ids == [third.id, second.id, first.id]


@pytest.mark.asyncio
async def test_slow_consolidation_falls_back_to_fast_mode_on_provider_error(
    system1_journal,
    memory_path: Path,
) -> None:
    system1_journal.record("memory search bug needs a regression test")
    system1_journal.record("regression test for the memory search bug")

    provider = FakeAIProvider(error=RuntimeError("provider offline"))
    consolidator = SlowConsolidation(system1_journal, memory_path, provider)

    created_files = await consolidator.consolidate()
    created_contents = _read_markdown_files(memory_path)

    assert len(provider.prompts) == 1
    assert len(created_files) == 1
    assert len(created_contents) == 1
    assert "auto-consolidated" in created_contents[0][1]
    assert "memory search bug needs a regression test" in created_contents[0][1]
    assert system1_journal.count_by_status() == {"processed": 2}
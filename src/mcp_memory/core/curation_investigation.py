"""Bounded, read-only investigation before typed curator planning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any, Mapping, cast


READ_ONLY_CURATOR_INVESTIGATION_TOOLS = (
    "internal_peek_record",
    "internal_maintenance_search",
    "internal_list_relationships",
    "internal_bounded_adjacency",
)


@dataclass(frozen=True, slots=True)
class CurationInvestigationLimits:
    max_rounds: int = 6
    max_tool_calls: int = 24
    max_context_characters: int = 20_000
    max_result_characters: int = 12_000
    max_records: int = 32

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_tool_calls",
            "max_context_characters",
            "max_result_characters",
            "max_records",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CurationInvestigationResult:
    status: str
    record_ids: tuple[str, ...] = ()
    rounds: int = 0
    tool_calls: int = 0
    reason: str | None = None


async def run_curator_investigation(
    provider: Any,
    *,
    seed_records: list[Any],
    task_id: str,
    limits: CurationInvestigationLimits | None = None,
    session: Any | None = None,
) -> CurationInvestigationResult:
    """Ask an agent to discover evidence, without accepting its record payloads."""

    bounds = limits or CurationInvestigationLimits()
    run_agent = getattr(provider, "run_agent", None)
    if session is not None:
        run_agent = getattr(session, "run_agent", None)
    if not callable(run_agent):
        return CurationInvestigationResult("skipped", reason="agentic_provider_unavailable")

    prompt = _investigation_prompt(seed_records, task_id=task_id, limits=bounds)
    if session is None:
        scoped_provider = getattr(provider, "with_allowed_tool_names", None)
        if not callable(scoped_provider):
            return CurationInvestigationResult("skipped", reason="read_only_scope_unavailable")
        provider = scoped_provider(READ_ONLY_CURATOR_INVESTIGATION_TOOLS)
        run_agent = getattr(provider, "run_agent", None)
        if not callable(run_agent):
            return CurationInvestigationResult("skipped", reason="read_only_provider_unavailable")

    try:
        result = await cast(Callable[[str], Awaitable[Any]], run_agent)(prompt)
    except Exception as exc:  # investigation is advisory; typed planning still runs
        return CurationInvestigationResult("failed", reason=type(exc).__name__)

    raw_text = getattr(result, "raw_text", None)
    parsed = getattr(result, "parsed", None)
    if isinstance(raw_text, str) and len(raw_text) > bounds.max_result_characters:
        return CurationInvestigationResult("rejected", reason="result_too_large")
    if not isinstance(parsed, Mapping):
        if not isinstance(raw_text, str):
            return CurationInvestigationResult("rejected", reason="result_not_json")
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            return CurationInvestigationResult("rejected", reason="result_not_json")
    if len(json.dumps(parsed, default=str, separators=(",", ":"))) > bounds.max_result_characters:
        return CurationInvestigationResult("rejected", reason="result_too_large")

    rounds = _reported_count(parsed, "rounds", "rounds_used")
    tool_calls = _reported_count(parsed, "tool_calls", "tool_calls_used")
    if rounds > bounds.max_rounds or tool_calls > bounds.max_tool_calls:
        return CurationInvestigationResult(
            "rejected",
            rounds=rounds,
            tool_calls=tool_calls,
            reason="investigation_budget_exceeded",
        )
    record_ids = _record_ids(parsed, bounds.max_records)
    return CurationInvestigationResult(
        "completed",
        record_ids=record_ids,
        rounds=rounds,
        tool_calls=tool_calls,
    )


def _investigation_prompt(
    seed_records: list[Any], *, task_id: str, limits: CurationInvestigationLimits
) -> str:
    seed_context: list[dict[str, str]] = []
    payload = {
        "task_id": task_id,
        "limits": {
            "max_rounds": limits.max_rounds,
            "max_tool_calls": limits.max_tool_calls,
            "max_context_characters": limits.max_context_characters,
            "max_result_characters": limits.max_result_characters,
            "max_records": limits.max_records,
        },
        "seed_records": seed_context,
        "available_tools": list(READ_ONLY_CURATOR_INVESTIGATION_TOOLS),
    }
    for record in seed_records:
        candidate = {
            "memory_id": str(getattr(record, "id", "")),
            "title": str(getattr(record, "title", ""))[:80],
            "summary": str(getattr(record, "summary", ""))[:220],
        }
        candidate_context = [*seed_context, candidate]
        payload["seed_records"] = candidate_context
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        if len(encoded) > limits.max_context_characters:
            break
        seed_context.append(candidate)
    payload["seed_records"] = seed_context
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return (
        "You are the curator's exploration phase. Explore the memory base actively using "
        "multiple rounds of the listed read-only tools. Start from the seed records, then "
        "look for duplicates, better canonicals, related claims, stale traces, missing "
        "evidence, and nearby records that could form a higher-quality repair together. "
        "Read every record that may be relevant to a useful improvement, not only records "
        "needed to justify retention. Do not mutate records, links, work items, or task "
        "state. Stop within the stated round and tool-call limits. Return one JSON object "
        "with rounds, tool_calls, and record_ids for every memory you actually read or "
        "want the planner to inspect. Do not return record content; the application will "
        "re-read every ID authoritatively before planning.\n"
        + encoded
    )


def _reported_count(payload: Mapping[str, Any], *names: str) -> int:
    for name in names:
        value = payload.get(name)
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return max(value, 0)
        if isinstance(value, list):
            return len(value)
    return 0


def _record_ids(payload: Mapping[str, Any], limit: int) -> tuple[str, ...]:
    if limit == 0:
        return ()
    values: list[Any] = []
    for key in ("record_ids", "memory_ids", "investigated_memory_ids"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            values.extend(candidate)
    records = payload.get("records")
    if isinstance(records, list):
        values.extend(
            item.get("memory_id", item.get("id"))
            for item in records
            if isinstance(item, Mapping)
        )
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or value in result:
            continue
        result.append(value)
        if len(result) >= limit:
            break
    return tuple(result)

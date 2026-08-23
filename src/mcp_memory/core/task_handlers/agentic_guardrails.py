from __future__ import annotations

import re

_INGEST_TRANSIENT_MARKERS = (
    "work log",
    "worklog",
    "progress update",
    "task complete",
    "completed the task",
    "finished the task",
    "worked on",
    "did a refactor",
    "ran tests",
    "test suite",
    "changed files",
    "when i started",
    "still investigating",
    "next step",
    "in progress",
    "temporary workaround",
    "already red",
    "didn't fix",
    "did not fix",
)
_INGEST_DURABLE_MARKERS = (
    "decision",
    "policy",
    "root cause",
    "lesson",
    "reusable",
    "constraint",
    "deadline",
    "release",
    "incident",
    "postmortem",
    "failure mode",
    "required because",
    "must remain",
    "will break",
)
_INGEST_DATE_PATTERN = re.compile(
    r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b"
)


def ingest_memory_quality_error(
    *,
    title: str,
    content: str,
    summary: str | None,
) -> str | None:
    """Reject obvious execution residue unless it carries a durable claim."""
    normalized_title = " ".join(title.lower().split())
    normalized_summary = " ".join((summary or "").lower().split())
    normalized_content = " ".join(content.lower().split())
    normalized = " ".join((normalized_title, normalized_summary, normalized_content))
    content_has_completion_shape = (
        "task id:" in normalized_content
        and ("task name:" in normalized_content or normalized_content.startswith("task:"))
        and (
            "completion marker" in normalized_content
            or "completionrecorded" in normalized_content
            or "merged:" in normalized_content
            or "archived:" in normalized_content
            or "absorbed_observations:" in normalized_content
        )
    )
    if (
        normalized_title.startswith(("task_complete", "task complete"))
        or normalized_summary.startswith("task id:")
        or "completion marker" in normalized_summary
        or content_has_completion_shape
    ):
        return "routine completion/status traces must use task_complete instead of creating memories"
    if any(marker in normalized for marker in _INGEST_DURABLE_MARKERS):
        return None

    marker_count = sum(marker in normalized for marker in _INGEST_TRANSIENT_MARKERS)
    title_or_summary_is_status = normalized.startswith((
        "task complete",
        "task_complete",
        "work log",
        "progress update",
        "status update",
    ))
    dated_execution = bool(_INGEST_DATE_PATTERN.search(normalized)) and marker_count >= 2
    if title_or_summary_is_status or dated_execution:
        return "routine progress and execution residue must be omitted unless it contains a durable claim"
    return None


def build_agentic_memory_guardrails(*, include_temporal_guidance: bool = False) -> str:
    lines = [
        "Guardrails:",
        "- Prefer focused, durable memories. Small-to-medium records beat large mixed-topic blobs.",
        "- Do not merge, append, or rewrite across different projects, products, or repositories unless the memory is explicitly about their relationship.",
        "- Generic overlap in words like architecture, daemon, testing, roadmap, or migration is weak evidence; when unsure, split, link, read more, or no-op.",
        "- Mutate only when the change clearly improves retrieval quality or coherence. No-op is acceptable when no change adds clear value.",
    ]
    if include_temporal_guidance:
        lines.insert(
            3,
            "- Treat progress notes, debugging chatter, and one-off execution status as temporal context unless they contain reusable long-term knowledge.",
        )
    return "\n".join(lines)


def build_ingest_guardrails() -> str:
    return "\n".join(
        [
            build_agentic_memory_guardrails(include_temporal_guidance=True),
            "- Do not ingest dated work logs, task-complete notes, progress updates, test status, or execution residue unless they contain a durable decision, incident, constraint, root cause, or reusable lesson.",
            "- Append only when the project scope truly matches; otherwise ignore or create a narrow observation.",
        ]
    )


def build_deduplicator_guardrails() -> str:
    return "\n".join(
        [
            build_agentic_memory_guardrails(),
            "- Merge only when the records describe the same durable concept; otherwise keep them separate and preserve lineage.",
        ]
    )


def build_curator_guardrails() -> str:
    return "\n".join(
        [
            build_agentic_memory_guardrails(include_temporal_guidance=True),
            "- Prefer split-and-link over expanding a memory that already spans multiple topics, projects, or time horizons.",
        ]
    )


def build_reflection_synthesis_guardrails() -> str:
    return "\n".join(
        [
            build_agentic_memory_guardrails(include_temporal_guidance=True),
            "- Only synthesize one reflection when the inputs share the same project and durable theme.",
        ]
    )

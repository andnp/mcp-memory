from __future__ import annotations


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

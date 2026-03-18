from __future__ import annotations

import re


MAX_SUMMARY_CHARS = 220
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
_TIMESTAMP_PREFIX_RE = re.compile(r"^\s*[-*]?\s*\[[^\]]+\]\s*")
_BULLET_PREFIX_RE = re.compile(r"^\s*[-*+]\s+")
_HEADING_RE = re.compile(r"^\s{0,3}#+\s+(?P<heading>.+?)\s*$")
_SIGNAL_VERBS = {
    "add",
    "added",
    "archive",
    "archived",
    "capture",
    "captured",
    "clean",
    "completed",
    "consolidates",
    "create",
    "created",
    "document",
    "documents",
    "fix",
    "fixed",
    "harden",
    "hardened",
    "implement",
    "implemented",
    "improve",
    "improved",
    "link",
    "linked",
    "merge",
    "merged",
    "normalize",
    "normalized",
    "record",
    "records",
    "refresh",
    "refreshed",
    "repair",
    "repaired",
    "split",
    "summarize",
    "summarized",
    "surface",
    "surfaces",
    "track",
    "tracked",
    "update",
    "updated",
    "verify",
    "verified",
}


def build_deterministic_summary(
    *,
    title: str,
    content: str,
    memory_type: str | None = None,
    max_chars: int = MAX_SUMMARY_CHARS,
) -> str:
    normalized_title = _normalize_inline_text(title)
    stripped = content.strip()
    if not stripped:
        return ""

    bullet_summary = _build_bullet_summary(stripped, memory_type=memory_type, max_chars=max_chars)
    if bullet_summary is not None:
        return bullet_summary

    simple_summary = _build_simple_prose_summary(stripped, max_chars=max_chars)
    if simple_summary is not None:
        return simple_summary

    candidate_sentences = _candidate_sentences(normalized_title, stripped)
    if candidate_sentences:
        title_tokens = _token_set(normalized_title)
        ranked = sorted(
            candidate_sentences,
            key=lambda item: _sentence_score(item, title_tokens=title_tokens),
            reverse=True,
        )
        best = ranked[0]
        summary = _trim_summary(best, max_chars=max_chars)
        if summary:
            return summary

    lines = [_normalize_inline_text(_strip_line_prefix(line)) for line in stripped.splitlines()]
    lines = [line for line in lines if line]
    if lines:
        if len(lines[0]) <= max_chars:
            return lines[0]
        return _trim_summary(lines[0], max_chars=max_chars)

    normalized_content = _normalize_inline_text(stripped)
    return _trim_summary(normalized_content, max_chars=max_chars)


def _build_simple_prose_summary(content: str, *, max_chars: int) -> str | None:
    non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(non_empty_lines) != 1:
        return None
    line = non_empty_lines[0]
    if _BULLET_PREFIX_RE.match(line) or _HEADING_RE.match(line):
        return None
    sentences = _split_sentences(line)
    if len(sentences) >= 2:
        joined = f"{sentences[0]} {sentences[1]}".strip()
        if len(joined) <= max_chars:
            return joined
    if len(sentences) == 1 and len(sentences[0]) <= max_chars:
        return sentences[0]
    return None


def _build_bullet_summary(content: str, *, memory_type: str | None, max_chars: int) -> str | None:
    bullet_lines = []
    for raw_line in content.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        if _HEADING_RE.match(stripped_line):
            continue
        if _BULLET_PREFIX_RE.match(stripped_line) or _TIMESTAMP_PREFIX_RE.match(stripped_line):
            bullet_lines.append(_normalize_inline_text(_strip_line_prefix(stripped_line)))

    if len(bullet_lines) < 2:
        return None

    phrases = [_bullet_phrase(line) for line in bullet_lines]
    phrases = [phrase for phrase in phrases if phrase]
    if not phrases:
        return None

    lead = "Consolidates" if memory_type == "reflection" or content.startswith("Consolidated observations:") else "Covers"
    summary = f"{lead} {_join_phrases(phrases[:3])}."
    return _trim_summary(summary, max_chars=max_chars)


def _candidate_sentences(title: str, content: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in content.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        heading_match = _HEADING_RE.match(stripped_line)
        if heading_match is not None:
            heading = _normalize_inline_text(heading_match.group("heading"))
            if heading and heading.lower() != title.lower():
                candidates.append(heading)
            continue
        cleaned_line = _normalize_inline_text(_strip_line_prefix(stripped_line))
        if not cleaned_line:
            continue
        for sentence in _split_sentences(cleaned_line):
            if sentence:
                candidates.append(sentence)
    return candidates


def _sentence_score(sentence: str, *, title_tokens: set[str]) -> tuple[float, int, int]:
    sentence_tokens = _token_set(sentence)
    overlap = len(sentence_tokens & title_tokens)
    signal_verbs = len(sentence_tokens & _SIGNAL_VERBS)
    length = len(sentence)
    length_bonus = 2 if 40 <= length <= 180 else 1 if 20 <= length <= 220 else 0
    punctuation_bonus = 1 if ":" in sentence or ";" in sentence else 0
    return (float((overlap * 4) + (signal_verbs * 3) + length_bonus + punctuation_bonus), overlap, -abs(length - 96))


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_inline_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def _strip_line_prefix(value: str) -> str:
    stripped = _BULLET_PREFIX_RE.sub("", value, count=1)
    stripped = _TIMESTAMP_PREFIX_RE.sub("", stripped, count=1)
    return stripped.strip()


def _bullet_phrase(value: str) -> str:
    sentence = _split_sentences(value)
    lead = sentence[0] if sentence else value
    if ":" in lead:
        left, _, right = lead.partition(":")
        if 0 < len(left.strip()) <= 80:
            return _normalize_inline_text(left)
        return _normalize_inline_text(right)
    return _normalize_inline_text(lead)


def _join_phrases(phrases: list[str]) -> str:
    unique_phrases: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        normalized = phrase.strip().rstrip(". ")
        lowered = normalized.lower()
        if not normalized or lowered in seen:
            continue
        seen.add(lowered)
        unique_phrases.append(normalized)
    if not unique_phrases:
        return "key points"
    if len(unique_phrases) == 1:
        return unique_phrases[0]
    if len(unique_phrases) == 2:
        return f"{unique_phrases[0]} and {unique_phrases[1]}"
    return f"{', '.join(unique_phrases[:-1])}, and {unique_phrases[-1]}"


def _token_set(value: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(value)}


def _normalize_inline_text(value: str) -> str:
    return " ".join(value.strip().split())


def _trim_summary(value: str, *, max_chars: int) -> str:
    normalized = _normalize_inline_text(value)
    if len(normalized) <= max_chars:
        return normalized
    trimmed = normalized[: max(max_chars - 1, 0)].rstrip(" ,;:-")
    return f"{trimmed}…"

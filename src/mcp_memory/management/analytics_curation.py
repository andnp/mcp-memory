from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from mcp_memory.management.models import (
    CurationCandidateMetricsPayload,
    CurationHistoryMetricsPayload,
    CurationMetricsPayload,
    CurationProviderDisclosurePayload,
    CurationQualityMetricsPayload,
    CurationSpecialistRouteMetricsPayload,
)
from mcp_memory.management.query_runner import (
    ManagementQueryAdapter,
    ManagementQueryRunner,
)

_RESTORABLE_OPERATIONS = frozenset({"normalize_memory", "create_link"})
_VALID_PLAN_OUTCOMES = frozenset(
    {
        "applied",
        "partially_applied",
        "no_op",
        "stale_plan",
        "verification_failed",
        "deferred",
    }
)
_MUTATION_CATEGORY_BY_OPERATION = {
    "create_link": "structural_link",
    "remove_link": "structural_link",
    "merge_memories": "structural_link",
    "split_memory": "structural_link",
    "normalize_memory": "content_tag",
    "rewrite_memory": "content_tag",
    "archive_memory": "retention",
}


def build_curation_metrics(
    db_manager,
    *,
    config=None,
    window_hours: int = 24,
    now: float | None = None,
    query_adapter: ManagementQueryAdapter | None = None,
) -> CurationMetricsPayload:
    """Build curation metrics exclusively from durable curation evidence."""
    generated_at = time.time() if now is None else now
    cutoff = generated_at - (max(window_hours, 0) * 3600)
    runner = ManagementQueryRunner(db_manager, adapter=query_adapter)
    if not runner.available:
        return CurationMetricsPayload(window_hours=window_hours)

    run_rows = runner.fetchall(
        """
        SELECT run_id, state, outcome, rejection_codes_json, retry_reason,
               budget_usage_json, created_at
        FROM curation_runs
        """
    )
    runs = [row for row in run_rows if _in_window(row.get("created_at"), cutoff)]
    run_ids = {str(row["run_id"]) for row in runs}

    receipt_rows = runner.fetchall(
        """
        SELECT run_id, operation, status, error_code, applied_at, verified_at
        FROM curation_action_receipts
        """
    )
    quality_rows = (
        runner.fetchall(
            """
            SELECT run_id, operation, status, retrieval_regression_count, zero_result_change,
                   payload_size_change, useful_work, content_quality_improved,
                   content_quality_delta, created_at
            FROM curation_quality_evidence
            """
        )
        if run_ids
        else []
    )
    quality = [
        row
        for row in quality_rows
        if str(row.get("run_id")) in run_ids
        and _in_window(row.get("created_at"), cutoff)
    ]
    receipts = [
        row
        for row in receipt_rows
        if str(row.get("run_id")) in run_ids
        and (
            _in_window(row.get("applied_at"), cutoff)
            or row.get("applied_at") is None
        )
    ]

    candidate_rows = runner.fetchall(
        """
        SELECT disposition, consecutive_no_op_count, cooldown_until
        FROM curation_candidate_state
        """
    )
    candidate_dispositions = Counter(_text(row.get("disposition"), "unknown") for row in candidate_rows)
    candidate_no_op_count = sum(
        1 for row in candidate_rows if _integer(row.get("consecutive_no_op_count")) > 0
    )
    cooldown_count = sum(
        1
        for row in candidate_rows
        if _is_after(row.get("cooldown_until"), generated_at)
    )

    event_rows = runner.fetchall(
        """
        SELECT id, operation, family, provider_id, reason_code, status,
               curation_run_id, restores_event_id, created_at
        FROM memory_mutation_events
        """
    )
    events = [
        row
        for row in event_rows
        if str(row.get("curation_run_id")) in run_ids
        and _in_window(row.get("created_at"), cutoff)
    ]
    revision_rows = runner.fetchall(
        """
        SELECT event_id FROM memory_record_revisions
        UNION ALL
        SELECT event_id FROM memory_link_revisions
        """
    )
    reversible_event_ids = {str(row["event_id"]) for row in revision_rows}

    restore_rows = runner.fetchall(
        """
        SELECT status, target_event_id, event_id, created_at
        FROM memory_restore_requests
        """
    )
    restores = [row for row in restore_rows if _in_window(row.get("created_at"), cutoff)]

    route_rows = runner.fetchall(
        """
        SELECT family_key, status, payload_json, created_at
        FROM work_items
        """
    )
    routes = [
        row
        for row in route_rows
        if _in_window(row.get("created_at"), cutoff) and _is_specialist_route(row.get("payload_json"))
    ]

    run_states = Counter(_text(row.get("state"), "unknown") for row in runs)
    run_outcomes = Counter(_text(row.get("outcome"), "unknown") for row in runs if row.get("outcome") is not None)
    rejection_reasons = Counter(
        reason
        for row in runs
        for reason in _string_list(row.get("rejection_codes_json"))
    )
    operation_counts = Counter(_text(row.get("operation"), "unknown") for row in receipts)
    receipt_statuses = Counter(_text(row.get("status"), "unknown") for row in receipts)
    receipt_rejection_reasons = Counter(
        _text(row.get("error_code"), "unknown")
        for row in receipts
        if row.get("error_code") is not None
    )
    rejection_reasons.update(receipt_rejection_reasons)

    verified_receipts = sum(1 for row in receipts if row.get("status") == "verified")
    verification_failure_count = sum(
        1 for row in receipts if row.get("status") == "verification_failed"
    )
    terminal_receipts = sum(
        1
        for row in receipts
        if row.get("status") in {"verified", "verification_failed", "rejected", "stale", "failed"}
    )
    verified_yield = _ratio(verified_receipts, len(receipts))
    valid_plan_count = sum(
        1 for row in runs if _text(row.get("outcome"), "") in _VALID_PLAN_OUTCOMES
    )
    accepted_mutation_count = sum(
        _json_non_negative_int(_json_mapping(row.get("budget_usage_json")).get("accepted_mutations"))
        for row in runs
    )
    provider_failure_count = sum(
        1 for row in runs if row.get("outcome") == "provider_failed"
    )
    retry_count = sum(_retry_count(row) for row in runs)
    mutation_categories = Counter(
        _mutation_category(_text(row.get("operation"), "unknown"))
        for row in receipts
    )
    structural_only_quality = [
        row
        for row in quality
        if row.get("status") == "structural_only"
        or _text(row.get("operation"), "") in {"create_link", "remove_link"}
    ]
    evaluated_quality = [
        row
        for row in quality
        if row.get("status") == "evaluated"
        and _text(row.get("operation"), "") not in {"create_link", "remove_link"}
    ]
    quality_observed = [
        row
        for row in quality
        if (
            row in evaluated_quality
            or row.get("content_quality_delta") is not None
        )
    ]
    useful_work_observed = [row for row in quality_observed if row.get("useful_work") is not None]
    useful_work_count = sum(1 for row in useful_work_observed if row.get("useful_work") in (1, True))
    content_quality = [
        row for row in quality if row.get("content_quality_delta") is not None
    ]

    history_event_count = len(events)
    restorable_event_count = sum(
        1
        for row in events
        if row.get("status") == "applied"
        and _text(row.get("operation"), "") in _RESTORABLE_OPERATIONS
        and str(row.get("id")) in reversible_event_ids
    )
    provider_counts = Counter(
        _text(row.get("provider_id"), "unknown")
        for row in events
        if row.get("provider_id") is not None
    )

    return CurationMetricsPayload(
        window_hours=window_hours,
        run_states=dict(sorted(run_states.items())),
        run_outcomes=dict(sorted(run_outcomes.items())),
        operation_counts=dict(sorted(operation_counts.items())),
        receipt_statuses=dict(sorted(receipt_statuses.items())),
        rejection_reasons=dict(sorted(rejection_reasons.items())),
        no_op_runs=run_outcomes.get("no_op", 0),
        candidate=CurationCandidateMetricsPayload(
            total=len(candidate_rows),
            dispositions=dict(sorted(candidate_dispositions.items())),
            no_op_candidates=candidate_no_op_count,
            cooldown_candidates=cooldown_count,
        ),
        specialist_routes=_build_specialist_routes(routes),
        provider_disclosure=CurationProviderDisclosurePayload(
            mutation_events_with_provider=sum(1 for row in events if row.get("provider_id") is not None),
            mutation_events_without_provider=sum(1 for row in events if row.get("provider_id") is None),
            by_provider=dict(sorted(provider_counts.items())),
        ),
        history=CurationHistoryMetricsPayload(
            event_count=history_event_count,
            applied_event_count=sum(1 for row in events if row.get("status") == "applied"),
            restorable_event_count=restorable_event_count,
            restore_request_count=len(restores),
            restore_requests_by_status=dict(
                sorted(Counter(_text(row.get("status"), "unknown") for row in restores).items())
            ),
            restore_available=restorable_event_count > 0,
        ),
        retrieval_quality=CurationQualityMetricsPayload(
            sampled_action_count=len(quality),
            evaluated_action_count=len(evaluated_quality),
            quality_observed_action_count=len(quality_observed),
            no_query_action_count=sum(1 for row in quality if row.get("status") == "no_query"),
            structural_only_action_count=len(structural_only_quality),
            retrieval_regression_count=sum(
                _integer(row.get("retrieval_regression_count")) for row in evaluated_quality
            ),
            zero_result_change=sum(_integer(row.get("zero_result_change")) for row in evaluated_quality),
            payload_size_change=sum(_integer(row.get("payload_size_change")) for row in evaluated_quality),
            useful_work_count=useful_work_count,
            useful_work_observed_action_count=len(useful_work_observed),
            useful_work_rate=_ratio(useful_work_count, len(useful_work_observed)),
            content_evaluated_action_count=len(content_quality),
            content_quality_improved_count=sum(
                1 for row in content_quality if _number(row.get("content_quality_delta")) > 0
            ),
            content_quality_neutral_count=sum(
                1 for row in content_quality if _number(row.get("content_quality_delta")) == 0
            ),
            content_quality_regression_count=sum(
                1 for row in content_quality if _number(row.get("content_quality_delta")) < 0
            ),
            content_quality_delta=sum(
                _number(row.get("content_quality_delta")) for row in content_quality
            ),
        ),
        verified_yield=verified_yield,
        verified_receipt_count=verified_receipts,
        terminal_receipt_count=terminal_receipts,
        valid_plan_count=valid_plan_count,
        valid_plan_rate=_ratio(valid_plan_count, len(runs)),
        no_op_rate=_ratio(run_outcomes.get("no_op", 0), len(runs)),
        accepted_mutation_count=accepted_mutation_count,
        verification_failure_count=verification_failure_count,
        provider_failure_count=provider_failure_count,
        retry_count=retry_count,
        mutation_categories=dict(sorted(mutation_categories.items())),
    )


def _build_specialist_routes(rows: Sequence[Mapping[str, object]]) -> CurationSpecialistRouteMetricsPayload:
    by_family: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    for row in rows:
        payload = _json_mapping(row.get("payload_json"))
        family = _text(payload.get("primary_family"), _text(row.get("family_key"), "unknown"))
        reason = _text(payload.get("reason_code"), "unknown")
        by_family[family] += 1
        by_reason[reason] += 1
        by_status[_text(row.get("status"), "unknown")] += 1
    return CurationSpecialistRouteMetricsPayload(
        total=len(rows),
        by_family=dict(sorted(by_family.items())),
        by_reason=dict(sorted(by_reason.items())),
        by_status=dict(sorted(by_status.items())),
    )


def _is_specialist_route(value: object) -> bool:
    return _json_mapping(value).get("route_kind") == "specialist_route"


def _mutation_category(operation: str) -> str:
    return _MUTATION_CATEGORY_BY_OPERATION.get(operation, "other")


def _retry_count(row: Mapping[str, object]) -> int:
    usage = _json_mapping(row.get("budget_usage_json"))
    attempts = _json_non_negative_int(usage.get("planner_attempts"))
    if attempts > 1:
        return attempts - 1
    return 1 if row.get("retry_reason") else 0


def _json_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
        return [item for item in parsed if isinstance(item, str) and item] if isinstance(parsed, list) else []
    return []


def _in_window(value: object, cutoff: float) -> bool:
    timestamp = _timestamp(value)
    return timestamp is not None and timestamp >= cutoff


def _is_after(value: object, now: float) -> bool:
    timestamp = _timestamp(value)
    return timestamp is not None and timestamp > now


def _timestamp(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.timestamp()
    except ValueError:
        return None


def _text(value: object, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _integer(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if not isinstance(value, (int, float, str)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _number(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _json_non_negative_int(value: object) -> int:
    result = _integer(value)
    return max(result, 0)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 4)

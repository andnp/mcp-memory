from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from mcp_memory.management.models import (
    CurationCandidateMetricsPayload,
    CurationHistoryMetricsPayload,
    CurationMetricsPayload,
    CurationProviderDisclosurePayload,
    CurationSpecialistRouteMetricsPayload,
)
from mcp_memory.management.query_runner import ManagementQueryRunner


_RESTORABLE_OPERATIONS = frozenset({"normalize_memory", "create_link"})


def build_curation_metrics(
    db_manager,
    *,
    config=None,
    window_hours: int = 24,
    now: float | None = None,
) -> CurationMetricsPayload:
    """Build curation metrics exclusively from durable curation evidence."""
    curation_config = getattr(config, "curation", None)
    shadow_mode_enabled = bool(getattr(curation_config, "shadow_mode_enabled", False))
    normalize_execution_enabled = bool(getattr(curation_config, "normalize_execution_enabled", False))
    create_link_execution_enabled = bool(getattr(curation_config, "create_link_execution_enabled", False))
    generated_at = time.time() if now is None else now
    cutoff = generated_at - (max(window_hours, 0) * 3600)
    runner = ManagementQueryRunner(db_manager)
    if not runner.available:
        return CurationMetricsPayload(
            window_hours=window_hours,
            shadow_mode_enabled=shadow_mode_enabled,
            normalize_execution_enabled=normalize_execution_enabled,
            create_link_execution_enabled=create_link_execution_enabled,
        )

    run_rows = runner.fetchall(
        """
        SELECT run_id, state, outcome, rejection_codes_json, created_at
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
    terminal_receipts = sum(
        1
        for row in receipts
        if row.get("status") in {"verified", "verification_failed", "rejected", "stale", "failed"}
    )
    verified_yield = _ratio(verified_receipts, len(receipts))

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
        verified_yield=verified_yield,
        verified_receipt_count=verified_receipts,
        terminal_receipt_count=terminal_receipts,
        shadow_mode_enabled=shadow_mode_enabled,
        normalize_execution_enabled=normalize_execution_enabled,
        create_link_execution_enabled=create_link_execution_enabled,
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
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
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


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else round(numerator / denominator, 4)

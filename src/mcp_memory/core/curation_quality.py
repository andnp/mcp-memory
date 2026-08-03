"""Sampled, non-instrumenting retrieval quality evidence for curation."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from pydantic import Field

from mcp_memory.core.curation_evaluation import (
    ReplayCase,
    ReplayCaseReport,
    ReplayResult,
    ReplaySnapshot,
    evaluate_query_replay,
)
from mcp_memory.core.curation_models import (
    CampaignHypothesis,
    CampaignRetrievalProblem,
    CampaignTargetMode,
    CurationModel,
)
from mcp_memory.core.ports.curation import (
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
    CurationRepository,
    CurationReceiptState,
    CurationRun,
)

class CurationQualityEvidence(CurationModel):
    run_id: UUID
    action_id: UUID
    operation: str
    affected_memory_ids: list[UUID] = Field(default_factory=list)
    policy_version: str
    query_id: str | None = None
    status: str
    before_ranked_memory_ids: list[UUID] = Field(default_factory=list)
    after_ranked_memory_ids: list[UUID] = Field(default_factory=list)
    retrieval_regression_count: int | None = None
    zero_result_change: int | None = None
    payload_size_change: int | None = None
    useful_work: bool | None = None
    retrieval_utility_delta: float | None = None
    acceptance_met: bool | None = None
    neutral_reason: str | None = None
    created_at: datetime


class CurationQualityRepository(Protocol):
    def put_quality_evidence(self, evidence: CurationQualityEvidence) -> CurationQualityEvidence: ...

    def list_quality_evidence(
        self,
        *,
        run_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[CurationQualityEvidence]: ...


class CurationQualitySearch(Protocol):
    def search_memories_for_maintenance(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> Sequence[Any]: ...


class CurationQualitySampler:
    """Capture one replay case per sampled, genuinely applied action."""

    def __init__(
        self,
        *,
        db_manager: Any,
        search: CurationQualitySearch,
        repository: CurationQualityRepository,
        candidate_repository: CurationRepository | None = None,
        sample_rate: float = 0.25,
        max_actions: int = 8,
        top_k: int = 5,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0 and 1")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        self._db_manager = db_manager
        self._search = search
        self._repository = repository
        self._candidate_repository = candidate_repository
        self._sample_rate = sample_rate
        self._max_actions = max_actions
        self._top_k = top_k
        self._clock = clock or (lambda: datetime.now(UTC))

    def evaluate(
        self,
        *,
        run: CurationRun,
        receipts: Sequence[CurationActionReceipt],
        campaign_hypothesis: CampaignHypothesis | None = None,
    ) -> tuple[CurationQualityEvidence, ...]:
        selected = [
            receipt
            for receipt in receipts
            if receipt.status is CurationReceiptState.VERIFIED
            and receipt.mutation_event_id is not None
            and receipt.affected_ids
            and self._is_sampled(run.run_id, receipt.action_id)
        ][: self._max_actions]
        evidence = tuple(
            self._evaluate_action(run, receipt, campaign_hypothesis)
            for receipt in selected
        )
        for item in evidence:
            self._repository.put_quality_evidence(item)
        if self._candidate_repository is not None:
            for item, receipt in zip(evidence, selected, strict=True):
                self._escalate_quality_regression(run, item, receipt)
        return evidence

    def _escalate_quality_regression(
        self,
        run: CurationRun,
        evidence: CurationQualityEvidence,
        receipt: CurationActionReceipt,
    ) -> None:
        candidate_repository = self._candidate_repository
        if candidate_repository is None:
            return
        if evidence.status == "evaluated" and (
            (evidence.retrieval_regression_count or 0) > 0
            or (evidence.zero_result_change or 0) > 0
        ):
            reason = (
                "retrieval_regression"
                if (evidence.retrieval_regression_count or 0) > 0
                else "zero_result_regression"
            )
        elif evidence.status == "evaluated" and evidence.acceptance_met is False:
            reason = "acceptance_not_met"
        else:
            return
        for memory_id in self._intended_memory_ids(receipt):
            previous = candidate_repository.get_candidate_state(memory_id)
            coverage_evidence = (
                {} if previous is None else dict(previous.coverage_evidence_json)
            )
            coverage_evidence["quality_regression"] = {
                "reason": reason,
                "retrieval_regression_count": evidence.retrieval_regression_count or 0,
                "zero_result_change": evidence.zero_result_change or 0,
            }
            candidate_repository.put_candidate_state(
                CurationCandidateState(
                    memory_id=memory_id,
                    last_observed_revision_token=(
                        None if previous is None else previous.last_observed_revision_token
                    ),
                    disposition=CandidateDisposition.ESCALATED,
                    consecutive_no_op_count=0,
                    cooldown_until=None,
                    last_disposition_reason=reason,
                    last_frontier_key=run.frontier_key,
                    last_run_id=run.run_id,
                    escalation_count=(0 if previous is None else previous.escalation_count) + 1,
                    last_escalated_strategy=run.selector_strategy,
                    last_considered_at=(
                        None if previous is None else previous.last_considered_at
                    ),
                    last_considered_strategy=(
                        None if previous is None else previous.last_considered_strategy
                    ),
                    last_mutation_family=(
                        None if previous is None else previous.last_mutation_family
                    ),
                    last_mutated_at=(
                        None if previous is None else previous.last_mutated_at
                    ),
                    coverage_evidence_json=coverage_evidence,
                )
            )

    def _evaluate_action(
        self,
        run: CurationRun,
        receipt: CurationActionReceipt,
        campaign_hypothesis: CampaignHypothesis | None,
    ) -> CurationQualityEvidence:
        common = {
            "run_id": run.run_id,
            "action_id": receipt.action_id,
            "operation": receipt.operation,
            "affected_memory_ids": list(receipt.affected_ids),
            "policy_version": run.policy_version,
            "created_at": self._clock(),
        }
        if receipt.operation in {"create_link", "remove_link"}:
            return CurationQualityEvidence(status="structural_only", **common)

        explicit_hypothesis = (
            campaign_hypothesis if _is_explicit(campaign_hypothesis) else None
        )
        query = self._find_historical_query(receipt, explicit_hypothesis)
        if query is None:
            return CurationQualityEvidence(
                status="no_query",
                neutral_reason="no_trusted_query",
                **common,
            )

        query_id, historical_query_text, before_ids, replay_complete = query
        query_text = (
            explicit_hypothesis.query
            if (
                explicit_hypothesis is not None
                and explicit_hypothesis.target_mode is CampaignTargetMode.ZERO_RESULTS
                and explicit_hypothesis.query
            )
            else historical_query_text
        )
        neutral_reason = _neutral_query_reason(
            query_text,
            None if explicit_hypothesis is None else explicit_hypothesis.query,
            replay_complete,
        )
        if neutral_reason is not None:
            return CurationQualityEvidence(
                status=f"neutral_{neutral_reason}",
                query_id=query_id,
                before_ranked_memory_ids=[UUID(value) for value in before_ids],
                retrieval_regression_count=0,
                zero_result_change=0,
                useful_work=None,
                neutral_reason=neutral_reason,
                **common,
            )
        intended_ids = (
            list(explicit_hypothesis.expected_memory_ids)
            if explicit_hypothesis is not None and explicit_hypothesis.expected_memory_ids
            else self._intended_memory_ids(receipt)
        )
        top_k = (
            explicit_hypothesis.top_k
            if explicit_hypothesis is not None
            else self._top_k
        )
        after_contexts = self._search.search_memories_for_maintenance(
            query_text,
            limit=top_k,
        )
        after_results = tuple(
            ReplayResult(_context_memory_id(context))
            for context in after_contexts
        )
        before_results = tuple(
            ReplayResult(memory_id)
            for memory_id in before_ids
        )
        report = evaluate_query_replay(
            [
                ReplayCase(
                    query_id=query_id,
                    intended_memory_ids=tuple(str(value) for value in intended_ids),
                    before=ReplaySnapshot(before_results),
                    after=ReplaySnapshot(after_results),
                    top_k=top_k,
                    query_text=query_text,
                    expected_query=(
                        None
                        if explicit_hypothesis is None
                        else explicit_hypothesis.query
                    ),
                    replay_complete=replay_complete,
                )
            ]
        )
        case = report.cases[0]
        acceptance_met = (
            None
            if explicit_hypothesis is None
            else _acceptance_met(case, explicit_hypothesis)
        )
        return CurationQualityEvidence(
            status=(
                "evaluated"
                if case.neutral_reason is None
                else f"neutral_{case.neutral_reason}"
            ),
            query_id=query_id,
            before_ranked_memory_ids=[UUID(value) for value in before_ids],
            after_ranked_memory_ids=[
                UUID(result.memory_id) for result in after_results[:top_k]
            ],
            retrieval_regression_count=(
                report.retrieval_regression_count
                if case.neutral_reason is None
                else 0
            ),
            zero_result_change=(
                report.zero_result_change if case.neutral_reason is None else 0
            ),
            payload_size_change=None,
            useful_work=(
                report.useful_work_count > 0 if case.neutral_reason is None else None
            ),
            retrieval_utility_delta=case.retrieval_utility_delta,
            acceptance_met=acceptance_met,
            neutral_reason=case.neutral_reason,
            **common,
        )

    def _intended_memory_ids(self, receipt: CurationActionReceipt) -> list[UUID]:
        rows = _fetch_rows(
            self._db_manager,
            """
            SELECT memory_id, after_exists, after_snapshot
            FROM memory_record_revisions
            WHERE event_id = ?
            """,
            (str(receipt.mutation_event_id),),
        )
        snapshots = {
            str(row[0]): (row[1], row[2])
            for row in rows
            if row[0] is not None
        }
        intended: list[UUID] = []
        for memory_id in receipt.affected_ids:
            after_exists, snapshot = snapshots.get(str(memory_id), (True, None))
            if not after_exists:
                continue
            if isinstance(snapshot, str):
                try:
                    snapshot = json.loads(snapshot)
                except ValueError:
                    snapshot = None
            if isinstance(snapshot, Mapping):
                record_snapshot = snapshot.get("record", snapshot)
                if (
                    isinstance(record_snapshot, Mapping)
                    and record_snapshot.get("status") == "archived"
                ):
                    continue
            intended.append(memory_id)
        return intended if snapshots else list(receipt.affected_ids)

    def _is_sampled(self, run_id: UUID, action_id: UUID) -> bool:
        if self._sample_rate >= 1.0:
            return True
        digest = hashlib.sha256(f"{run_id}:{action_id}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self._sample_rate

    def _find_historical_query(
        self,
        receipt: CurationActionReceipt,
        explicit_hypothesis: CampaignHypothesis | None = None,
    ) -> tuple[str, str, list[str], bool] | None:
        if explicit_hypothesis is not None and explicit_hypothesis.target_mode is CampaignTargetMode.ZERO_RESULTS:
            return self._find_explicit_zero_result_query(receipt, explicit_hypothesis.query)
        target_ids = [str(value) for value in receipt.affected_ids]
        placeholders = ", ".join("?" for _ in target_ids)
        rows = _fetch_rows(
            self._db_manager,
            f"""
            SELECT invocation_id, query_text, memory_id, result_rank, result_count, created_at
            FROM memory_tool_events
            WHERE event_kind = 'search'
              AND caller_kind IN ('external', 'operator', 'user')
              AND memory_id IN ({placeholders})
            ORDER BY created_at DESC, id DESC
            LIMIT 1000
            """,
            target_ids,
        )
        target_id_set = set(target_ids)
        cutoff = receipt.applied_at or self._clock()
        grouped: dict[str, list[Any]] = defaultdict(list)
        for row in rows:
            invocation_id, _, memory_id, _, _, created_at = row
            event_time = _timestamp(created_at)
            if (
                str(memory_id) in target_id_set
                and event_time is not None
                and event_time < cutoff
            ):
                grouped[str(invocation_id)].append(row)
        if not grouped:
            return None
        invocation_id = next(iter(grouped))
        return self._hydrate_historical_query(invocation_id)

    def _find_explicit_zero_result_query(
        self,
        receipt: CurationActionReceipt,
        query_text: str | None,
    ) -> tuple[str, str, list[str], bool] | None:
        if not query_text or not query_text.strip():
            return None
        cutoff = receipt.applied_at or self._clock()
        rows = _fetch_rows(
            self._db_manager,
            """
            SELECT invocation_id, created_at
            FROM memory_tool_events
            WHERE event_kind = 'search'
              AND caller_kind IN ('external', 'operator', 'user')
              AND memory_id IS NULL
              AND result_count = 0
              AND query_text = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1000
            """,
            (query_text,),
        )
        for row in rows:
            event_time = _timestamp(row[1])
            if event_time is not None and event_time < cutoff:
                return self._hydrate_historical_query(str(row[0]))
        return None

    def _hydrate_historical_query(
        self,
        invocation_id: str,
    ) -> tuple[str, str, list[str], bool] | None:
        rows = _fetch_rows(
            self._db_manager,
            """
            SELECT invocation_id, query_text, memory_id, result_rank, result_count
            FROM memory_tool_events
            WHERE event_kind = 'search'
              AND caller_kind IN ('external', 'operator', 'user')
              AND invocation_id = ?
            ORDER BY result_rank ASC
            LIMIT 50
            """,
            (invocation_id,),
        )
        if not rows:
            return None
        return (
            invocation_id,
            str(rows[0][1] or ""),
            [str(row[2]) for row in rows if row[2] is not None],
            max(
                (int(str(row[4])) for row in rows if row[4] is not None),
                default=len(rows),
            )
            <= len(rows),
        )


def _is_explicit(hypothesis: CampaignHypothesis | None) -> bool:
    if hypothesis is None:
        return False
    return bool(
        hypothesis.expected_memory_ids
        or (hypothesis.query and hypothesis.query.strip())
        or hypothesis.retrieval_problem is not CampaignRetrievalProblem.HEURISTIC
        or hypothesis.minimum_improvement > 0
        or hypothesis.target_mode.value != "heuristic"
    )


def _acceptance_met(case: ReplayCaseReport, hypothesis: CampaignHypothesis) -> bool | None:
    if case.neutral_reason is not None:
        return None
    if case.intended_top_k_loss or case.zero_result_change > 0:
        return False
    if hypothesis.target_mode is CampaignTargetMode.RANK:
        improvement = case.rank_improvement
    elif hypothesis.target_mode is CampaignTargetMode.TOP_K:
        improvement = case.top_k_improvement
    elif hypothesis.target_mode is CampaignTargetMode.ZERO_RESULTS:
        improvement = case.zero_results_improvement
    else:
        improvement = case.retrieval_utility_delta
    return improvement >= hypothesis.minimum_improvement


def _neutral_query_reason(
    query_text: str,
    expected_query: str | None,
    replay_complete: bool,
) -> str | None:
    if not query_text.strip():
        return "blank_query"
    if expected_query and not _query_terms_overlap(query_text, expected_query):
        return "irrelevant_query"
    if not replay_complete:
        return "incomplete_replay"
    return None


def _query_terms_overlap(query: str, expected_query: str) -> bool:
    return bool(
        set(query.casefold().split()) & set(expected_query.casefold().split())
    )


def _fetch_rows(
    db_manager: Any,
    query: str,
    params: Sequence[object] = (),
) -> list[tuple[object, ...]]:
    if hasattr(db_manager, "get_connection"):
        connection = db_manager.get_connection()
        return [tuple(row) for row in connection.execute(query, list(params)).fetchall()]
    if hasattr(db_manager, "open_connection"):
        with db_manager.open_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query.replace("?", "%s"), tuple(params))
            return [tuple(row) for row in cursor.fetchall()]
    raise TypeError("quality sampling requires a supported database manager")


def _timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _context_memory_id(context: Any) -> str:
    return str(context.record.id)

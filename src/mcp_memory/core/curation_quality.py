"""Sampled, non-instrumenting retrieval quality evidence for curation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

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
from mcp_memory.core.curation_quality_consistency import assess_replay_consistency
from mcp_memory.core.curation_quality_inputs import (
    CurationQualityMutation,
    direct_action_id,
    mutation_from_receipt,
)
from mcp_memory.core.curation_quality_policy import (
    QualityOutcome,
    classify_quality_outcome,
    should_escalate_quality,
)
from mcp_memory.core.curation_quality_provenance import (
    QualityQuery,
    QueryProvenance,
)
from mcp_memory.core.curation_quality_structural import evaluate_structural_mutation
from mcp_memory.core.ports.curation import (
    CandidateDisposition,
    CurationActionReceipt,
    CurationCandidateState,
    CurationRepository,
    CurationRun,
)


class CurationQualityEvidence(CurationModel):
    run_id: UUID
    action_id: UUID
    evidence_id: str | None = None
    operation: str
    affected_memory_ids: list[UUID] = Field(default_factory=list)
    policy_version: str
    query_id: str | None = None
    query_text: str | None = None
    status: str
    before_ranked_memory_ids: list[UUID] = Field(default_factory=list)
    after_ranked_memory_ids: list[UUID] = Field(default_factory=list)
    retrieval_regression_count: int | None = None
    zero_result_change: int | None = None
    payload_size_change: int | None = None
    useful_work: bool | None = None
    retrieval_utility_delta: float | None = None
    content_quality_score_before: float | None = None
    content_quality_score_after: float | None = None
    content_quality_delta: float | None = None
    content_quality_improved: bool | None = None
    engagement_utility_delta: float | None = None
    engagement_evidence: dict[str, object] = Field(default_factory=dict)
    acceptance_met: bool | None = None
    neutral_reason: str | None = None
    wave_id: UUID | None = None
    wave_action_ids: list[UUID] = Field(default_factory=list)
    wave_status: str | None = None
    collateral_regression_count: int | None = None
    productive_mutation_count: int = 0
    override_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    override_reason: str | None = None
    override_judge_evidence: dict[str, object] = Field(default_factory=dict)
    override_outcome: str | None = None
    created_at: datetime


def quality_productive_mutation_count(
    quality_evidence: Iterable[CurationQualityEvidence],
) -> int:
    return sum(
        item.status
        in {
            QualityOutcome.PRODUCTIVE.value,
            QualityOutcome.STRUCTURAL_ONLY.value,
        }
        for item in quality_evidence
    )


@dataclass(frozen=True, slots=True)
class _ContentQualityComparison:
    before: float | None = None
    after: float | None = None
    delta: float | None = None
    improved: bool | None = None


def _total_utility_delta(
    *,
    retrieval: float | None,
    content: float | None,
    engagement: float | None,
) -> float:
    return sum(value or 0.0 for value in (retrieval, content, engagement))


class CurationQualityRepository(Protocol):
    def put_quality_evidence(self, evidence: CurationQualityEvidence) -> CurationQualityEvidence: ...

    def list_quality_evidence(
        self,
        *,
        run_id: UUID | None = None,
        limit: int = 100,
    ) -> Sequence[CurationQualityEvidence]: ...


class CurationQualityQueryPort(Protocol):
    def search_events_for_memory_ids(
        self,
        memory_ids: Sequence[str],
        *,
        before_epoch: float,
    ) -> Sequence[tuple[object, ...]]: ...

    def search_events_for_invocations(
        self,
        invocation_ids: Sequence[str],
        *,
        before_epoch: float,
    ) -> Sequence[tuple[object, ...]]: ...

    def read_events_for_memory_ids(
        self,
        memory_ids: Sequence[str],
        *,
        after_epoch: float,
        before_epoch: float,
    ) -> Sequence[tuple[object, ...]]: ...

    def revision_snapshots(self, event_id: str) -> Sequence[tuple[object, ...]]: ...

    def revision_states(self, event_id: str) -> Sequence[tuple[object, ...]]: ...

    def historical_search_events(self, memory_ids: Sequence[str]) -> Sequence[tuple[object, ...]]: ...

    def search_invocations_for_query(self, query_text: str) -> Sequence[tuple[object, ...]]: ...

    def search_events_for_invocation(self, invocation_id: str) -> Sequence[tuple[object, ...]]: ...


class CurationQualitySearch(Protocol):
    def search_memories_for_maintenance(
        self,
        query: str,
        *,
        limit: int = 50,
    ) -> Sequence[Any]: ...


class CurationQualitySampler:
    """Capture one replay case per sampled, genuinely applied action."""

    _QUALITY_REGRESSION_COOLDOWN = timedelta(hours=6)
    _ENGAGEMENT_ATTRIBUTION_WINDOW_SECONDS = 900.0

    def __init__(
        self,
        *,
        quality_query: CurationQualityQueryPort,
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
        self._quality_query = quality_query
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
        receipts: Sequence[CurationActionReceipt] = (),
        mutations: Sequence[CurationQualityMutation] | None = None,
        campaign_hypothesis: CampaignHypothesis | None = None,
    ) -> tuple[CurationQualityEvidence, ...]:
        selected_mutations = tuple(
            mutations
            if mutations is not None
            else (mutation_from_receipt(receipt) for receipt in receipts)
        )
        selected = [
            mutation
            for mutation in selected_mutations
            if mutation.applied
            and mutation.affected_memory_ids
            and self._is_sampled(run.run_id, mutation.mutation_id)
        ][: self._max_actions]
        wave_id = uuid4()
        evidence = self._evaluate_wave(
            run,
            selected,
            campaign_hypothesis,
            wave_id=wave_id,
        )
        for item in evidence:
            self._repository.put_quality_evidence(item)
        if self._candidate_repository is not None:
            for item, mutation in zip(evidence, selected, strict=True):
                self._escalate_quality_regression(run, item, mutation)
        return evidence

    def _evaluate_wave(
        self,
        run: CurationRun,
        mutations: Sequence[CurationQualityMutation | CurationActionReceipt],
        campaign_hypothesis: CampaignHypothesis | None,
        *,
        wave_id: UUID,
    ) -> tuple[CurationQualityEvidence, ...]:
        if not mutations:
            return ()
        normalized = tuple(_coerce_mutation(mutation) for mutation in mutations)
        identity_errors = _action_identity_errors(normalized)
        raw = [
            self._evaluate_action_with_identity(
                run,
                original,
                mutation,
                campaign_hypothesis,
                identity_error=identity_errors[index],
            )
            for index, (original, mutation) in enumerate(zip(mutations, normalized, strict=True))
        ]
        evaluated = [
            item
            for item in raw
            if item.status in {"evaluated", "productive", "verified_only", "regressed"}
            and item.query_id
        ]
        explicit = campaign_hypothesis if _is_explicit(campaign_hypothesis) else None
        complete_receipts = all(
            mutation.applied and mutation.verified for mutation in normalized
        )
        collateral = sum(item.retrieval_regression_count or 0 for item in evaluated)
        goal_improved = any(
            item.acceptance_met is True
            or bool(item.useful_work)
            or item.content_quality_improved is True
            for item in raw
        )
        content_evaluated = any(
            item.content_quality_improved is not None for item in raw
        )
        quality_observations = [
            item
            for item in raw
            if item in evaluated or item.content_quality_improved is not None
        ]
        aggregate_retrieval_utility = sum(
            item.retrieval_utility_delta or 0.0 for item in evaluated
        )
        aggregate_content_quality = sum(
            item.content_quality_delta or 0.0
            for item in raw
            if item.content_quality_delta is not None
        )
        aggregate_engagement = sum(
            item.engagement_utility_delta or 0.0 for item in raw
        )
        aggregate_quality_utility = (
            aggregate_retrieval_utility + aggregate_content_quality + aggregate_engagement
        )
        wave_acceptance = bool(
            (evaluated or content_evaluated)
            and complete_receipts
            and (
                (
                    aggregate_quality_utility > 0.0
                )
                if explicit is None
                else (
                    collateral == 0
                    and goal_improved
                    and all(item.acceptance_met is True for item in evaluated)
                )
            )
        )
        heuristic_regression = any(
            (item.retrieval_regression_count or 0) > 0
            or (item.zero_result_change or 0) > 0
            or (item.content_quality_delta is not None and item.content_quality_delta < 0)
            for item in raw
        )

        if wave_acceptance:
            wave_status = "accepted"
        elif not quality_observations or (explicit is None and not heuristic_regression):
            wave_status = "neutral"
        else:
            wave_status = "rejected"
        productive = sum(
            item.status in {
                QualityOutcome.PRODUCTIVE.value,
                QualityOutcome.STRUCTURAL_ONLY.value,
            }
            for item in raw
        )

        action_ids = [mutation.mutation_id for mutation in normalized]
        return tuple(
            item.model_copy(
                update={
                    "wave_id": wave_id,
                    "wave_action_ids": action_ids,
                    "wave_status": wave_status,
                    "collateral_regression_count": collateral,
                    "productive_mutation_count": productive,
                    "acceptance_met": (
                        wave_acceptance
                        if explicit is not None and item.status in {
                            "evaluated",
                            "productive",
                            "verified_only",
                            "regressed",
                        }
                        else item.acceptance_met
                    ),
                }
            )
            for item in raw
        )

    def _evaluate_action_with_identity(
        self,
        run: CurationRun,
        original: CurationQualityMutation | CurationActionReceipt,
        mutation: CurationQualityMutation,
        campaign_hypothesis: CampaignHypothesis | None,
        *,
        identity_error: str | None,
    ) -> CurationQualityEvidence:
        if identity_error is None:
            return self._evaluate_action(run, original, campaign_hypothesis)
        return self._evaluate_action(
            run,
            mutation,
            campaign_hypothesis,
            identity_error=identity_error,
        )

    def _escalate_quality_regression(
        self,
        run: CurationRun,
        evidence: CurationQualityEvidence,
        mutation: CurationQualityMutation,
    ) -> None:
        candidate_repository = self._candidate_repository
        if candidate_repository is None:
            return
        if mutation.evidence_id is not None:
            try:
                outcome = QualityOutcome(str(evidence.status))
            except ValueError:
                return
            if not should_escalate_quality(outcome, mutation_verified=mutation.verified):
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
        for memory_id in mutation.affected_memory_ids:
            compare_and_set = getattr(candidate_repository, "compare_and_set_candidate_state", None)
            for _ in range(3):
                previous = candidate_repository.get_candidate_state(memory_id)
                coverage_evidence = (
                    {} if previous is None else dict(previous.coverage_evidence_json)
                )
                coverage_evidence["quality_regression"] = {
                    "reason": reason,
                    "retrieval_regression_count": evidence.retrieval_regression_count or 0,
                    "zero_result_change": evidence.zero_result_change or 0,
                    "acceptance_met": evidence.acceptance_met,
                    "neutral_reason": evidence.neutral_reason,
                }
                updated = CurationCandidateState(
                    memory_id=memory_id,
                    last_observed_revision_token=(
                        None if previous is None else previous.last_observed_revision_token
                    ),
                    disposition=CandidateDisposition.ESCALATED,
                    consecutive_no_op_count=0,
                    cooldown_until=self._clock() + self._QUALITY_REGRESSION_COOLDOWN,
                    last_disposition_reason=reason,
                    last_frontier_key=run.frontier_key,
                    last_run_id=run.run_id,
                    escalation_count=(0 if previous is None else previous.escalation_count) + 1,
                    last_escalated_strategy=run.selector_strategy,
                    last_considered_at=(
                        None if previous is None else previous.last_considered_at
                    ),
                    last_considered_strategy=(
                        run.selector_strategy
                        if previous is None or previous.last_considered_strategy is None
                        else previous.last_considered_strategy
                    ),
                    last_mutation_family=(
                        None if previous is None else previous.last_mutation_family
                    ),
                    last_mutated_at=(
                        None if previous is None else previous.last_mutated_at
                    ),
                    coverage_evidence_json=coverage_evidence,
                )
                if callable(compare_and_set):
                    if compare_and_set(
                        memory_id,
                        None if previous is None else previous.last_observed_revision_token,
                        updated,
                    ) is not None:
                        break
                    continue
                candidate_repository.put_candidate_state(updated)
                break

    def _evaluate_action(
        self,
        run: CurationRun,
        mutation: CurationQualityMutation | CurationActionReceipt,
        campaign_hypothesis: CampaignHypothesis | None,
        *,
        after_results_by_query: Mapping[str, tuple[ReplayResult, ...]] | None = None,
        identity_error: str | None = None,
    ) -> CurationQualityEvidence:
        mutation = _coerce_mutation(mutation)
        common = {
            "run_id": run.run_id,
            "action_id": mutation.mutation_id,
            "evidence_id": mutation.evidence_id,
            "operation": mutation.operation,
            "affected_memory_ids": list(mutation.affected_memory_ids),
            "policy_version": run.policy_version,
            "created_at": self._clock(),
        }
        if identity_error is not None:
            return CurationQualityEvidence(
                status=QualityOutcome.UNOBSERVED.value,
                useful_work=None,
                neutral_reason=f"action_identity_{identity_error}",
                engagement_evidence={
                    "action_identity_error": identity_error,
                    "mutation_evidence_id": mutation.evidence_id,
                },
                **common,
            )
        content = self._content_quality(mutation)
        if mutation.evidence_id is not None and not mutation.verified:
            return CurationQualityEvidence(
                status=QualityOutcome.UNVERIFIED.value,
                useful_work=None,
                neutral_reason="mutation_evidence_unverified",
                engagement_evidence={"mutation_evidence_id": mutation.evidence_id},
                **common,
            )
        structural = evaluate_structural_mutation(
            mutation.operation, mutation.structural_deltas
        )
        if structural.relevant:
            return CurationQualityEvidence(
                status=(
                    QualityOutcome.STRUCTURAL_ONLY.value
                    if structural.verified
                    else QualityOutcome.UNVERIFIED.value
                ),
                useful_work=True if structural.verified and mutation.structural_deltas else None,
                neutral_reason=structural.reason,
                engagement_evidence={
                    "structural_verified": structural.verified,
                    "structural_details": dict(structural.details),
                },
                **common,
            )

        explicit_hypothesis = (
            campaign_hypothesis if _is_explicit(campaign_hypothesis) else None
        )
        query = self._find_historical_query(mutation, explicit_hypothesis)
        if query is None and explicit_hypothesis is None:
            query = _synthetic_query(mutation)
        if query is None:
            outcome = classify_quality_outcome(
                mutation_verified=mutation.verified,
                query_trusted=False,
                consistency_verified=True,
                structural_only=False,
                utility_delta=None,
                quality_observed=False,
                content_quality_delta=content.delta,
            )
            return CurationQualityEvidence(
                status=outcome.value,
                neutral_reason=(
                    None
                    if content.improved is not None
                    else "no_trusted_query"
                ),
                useful_work=(
                    None
                    if content.improved is None
                    else content.improved
                    and outcome is not QualityOutcome.UNOBSERVED
                ),
                content_quality_score_before=content.before,
                content_quality_score_after=content.after,
                content_quality_delta=content.delta,
                content_quality_improved=content.improved,
                **common,
            )

        query_id = query.query_id
        historical_query_text = query.query_text
        before_ids = query.before_memory_ids
        replay_complete = query.replay_complete
        query_text = (
            explicit_hypothesis.query
            if (
                explicit_hypothesis is not None
                and explicit_hypothesis.target_mode is CampaignTargetMode.ZERO_RESULTS
                and explicit_hypothesis.query
            )
            else historical_query_text
        )
        consistency_payload = mutation.payload.get("quality_consistency", {})
        consistency_data = (
            consistency_payload if isinstance(consistency_payload, Mapping) else {}
        )
        captured_after_context = consistency_data.get("after_query_context")
        after_query_context = (
            dict(captured_after_context)
            if isinstance(captured_after_context, Mapping)
            else dict(query.query_context)
        )
        neutral_reason = _neutral_query_reason(
            query_text,
            None if explicit_hypothesis is None else explicit_hypothesis.query,
            replay_complete,
        )
        if neutral_reason is not None:
            return CurationQualityEvidence(
                status=QualityOutcome.UNOBSERVED.value,
                query_id=query_id,
                query_text=query_text,
                before_ranked_memory_ids=[UUID(value) for value in before_ids],
                retrieval_regression_count=0,
                zero_result_change=0,
                useful_work=None,
                content_quality_score_before=content.before,
                content_quality_score_after=content.after,
                content_quality_delta=content.delta,
                content_quality_improved=content.improved,
                neutral_reason=neutral_reason,
                engagement_evidence={
                    "query_provenance": query.provenance.value,
                    "query_trusted": query.trusted,
                    **_replay_provenance(query, after_query_context, {}),
                },
                **common,
            )
        intended_ids = (
            list(explicit_hypothesis.expected_memory_ids)
            if explicit_hypothesis is not None and explicit_hypothesis.expected_memory_ids
            else list(mutation.affected_memory_ids)
        )
        top_k = (
            explicit_hypothesis.top_k
            if explicit_hypothesis is not None
            else self._top_k
        )
        engagement = (
            {}
            if explicit_hypothesis is not None
            else self._read_backed_engagement(
                intended_ids,
                cutoff=mutation.applied_at or self._clock(),
            )
        )
        engagement_delta = (
            None
            if not engagement
            else float(cast(float, engagement["utility_delta"]))
        )
        if after_results_by_query is None:
            after_contexts = self._search.search_memories_for_maintenance(
                query_text,
                limit=top_k,
            )
            after_results = tuple(
                ReplayResult(_context_memory_id(context))
                for context in after_contexts
            )
        else:
            after_results = after_results_by_query.get(query_text, ())
        before_results = tuple(
            ReplayResult(memory_id)
            for memory_id in before_ids
        )
        after_epochs = _search_epochs(self._search)
        consistency = assess_replay_consistency(
            before_context=query.query_context,
            after_context=after_query_context,
            before_epochs=query.before_search_epochs,
            after_epochs=after_epochs,
            replay_complete=replay_complete,
            unrelated_write_count=_non_negative_int(
                consistency_data.get("unrelated_write_count")
            ),
            write_pollution=bool(consistency_data.get("write_pollution", False)),
            index_lag=bool(consistency_data.get("index_lag", False)),
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
                    trusted_query=query.trusted,
                    replay_complete=replay_complete,
                    before_query_context=query.query_context,
                    after_query_context=after_query_context,
                    before_search_epochs=query.before_search_epochs,
                    after_search_epochs=after_epochs,
                    consistency_flags=consistency.flags,
                    affected_memory_clusters=tuple(
                        tuple(str(value) for value in cluster)
                        for cluster in mutation.affected_memory_clusters
                    ),
                )
            ]
        )
        case = report.cases[0]
        if not consistency.verified:
            return CurationQualityEvidence(
                status="unverified",
                query_id=query_id,
                query_text=query_text,
                before_ranked_memory_ids=[UUID(value) for value in before_ids],
                after_ranked_memory_ids=[
                    UUID(result.memory_id) for result in after_results[:top_k]
                ],
                retrieval_regression_count=0,
                zero_result_change=0,
                useful_work=None,
                retrieval_utility_delta=None,
                neutral_reason=consistency.flags[0],
                engagement_evidence={
                    "query_provenance": query.provenance.value,
                    "query_trusted": query.trusted,
                    **_replay_provenance(query, after_query_context, after_epochs),
                    "consistency_flags": list(consistency.flags),
                },
                **common,
            )
        acceptance_met = (
            None
            if explicit_hypothesis is None
            else _acceptance_met(case, explicit_hypothesis)
        )
        outcome = classify_quality_outcome(
            mutation_verified=mutation.verified,
            query_trusted=query.trusted,
            consistency_verified=True,
            structural_only=False,
            utility_delta=case.retrieval_utility_delta,
            quality_observed=case.neutral_reason is None,
            content_quality_delta=content.delta,
        )
        return CurationQualityEvidence(
            status=(
                outcome.value
                if mutation.evidence_id is not None
                else "evaluated"
                if case.neutral_reason is None
                else QualityOutcome.UNOBSERVED.value
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
                (
                    report.useful_work_count > 0
                    or content.improved is True
                    or _total_utility_delta(
                        retrieval=case.retrieval_utility_delta,
                        content=content.delta,
                        engagement=engagement_delta,
                    )
                    > 0.0
                )
                if case.neutral_reason is None
                else content.improved
            ),
            retrieval_utility_delta=case.retrieval_utility_delta,
            engagement_utility_delta=engagement_delta,
            engagement_evidence={
                **(
                    {}
                    if not engagement
                    else dict(cast(Mapping[str, object], engagement["evidence"]))
                ),
                "query_provenance": query.provenance.value,
                "query_trusted": query.trusted,
                **_replay_provenance(query, after_query_context, after_epochs),
                "cluster_utility_delta": case.cluster_utility_delta,
                "duplicate_density_change": case.duplicate_density_change,
            },
            content_quality_score_before=content.before,
            content_quality_score_after=content.after,
            content_quality_delta=content.delta,
            content_quality_improved=content.improved,
            acceptance_met=acceptance_met,
            neutral_reason=case.neutral_reason,
            **common,
        )

    def _read_backed_engagement(
        self,
        memory_ids: Sequence[UUID],
        *,
        cutoff: datetime,
    ) -> dict[str, object]:
        target_ids = {str(value) for value in memory_ids}
        if not target_ids:
            return {}
        cutoff_epoch = cutoff.timestamp()
        search_rows = self._quality_query.search_events_for_memory_ids(
            sorted(target_ids), before_epoch=cutoff_epoch
        )
        if not search_rows:
            return {
                "utility_delta": 0.0,
                "evidence": {
                    "positive": 0,
                    "conditional_negative": 0,
                    "weak_negative": 0,
                    "unknown": 0,
                    "exposures": 0,
                    "attribution_window_seconds": self._ENGAGEMENT_ATTRIBUTION_WINDOW_SECONDS,
                },
            }
        invocations = {str(row[0]) for row in search_rows}
        all_search_rows = self._quality_query.search_events_for_invocations(
            sorted(invocations), before_epoch=cutoff_epoch
        )
        episode_ids: dict[str, set[str]] = defaultdict(set)
        episode_search_times: dict[str, list[tuple[str, float]]] = defaultdict(list)
        episode_exposures: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for row in all_search_rows:
            invocation_id = str(row[0])
            memory_id = row[1]
            created_at = _epoch(row[2])
            if memory_id is None or created_at is None:
                continue
            episode_ids[invocation_id].add(str(memory_id))
            episode_search_times[invocation_id].append((str(memory_id), created_at))
            if str(memory_id) in target_ids:
                episode_exposures[invocation_id].append((str(memory_id), created_at))
        candidate_ids = sorted(set().union(*episode_ids.values()))
        search_epochs = [
            created_at
            for row in all_search_rows
            if (created_at := _epoch(row[2])) is not None
        ]
        read_rows: Sequence[tuple[object, ...]] = ()
        if candidate_ids:
            read_rows = self._quality_query.read_events_for_memory_ids(
                candidate_ids,
                after_epoch=min(search_epochs),
                before_epoch=cutoff_epoch,
            )
        reads_by_memory: dict[str, list[float]] = defaultdict(list)
        for row in read_rows:
            created_at = _epoch(row[1])
            if created_at is not None and row[0] is not None:
                reads_by_memory[str(row[0])].append(created_at)
        exposures = positives = conditional = 0
        target_exposure_counts: Counter[str] = Counter()
        for invocation_id, entries in episode_exposures.items():
            co_reads = {
                memory_id
                for memory_id in episode_ids[invocation_id]
                if any(
                    search_time <= read_time <= min(
                        cutoff_epoch,
                        search_time + self._ENGAGEMENT_ATTRIBUTION_WINDOW_SECONDS,
                    )
                    for search_memory, search_time in episode_search_times[invocation_id]
                    if search_memory == memory_id
                    for read_time in reads_by_memory.get(memory_id, ())
                )
            }
            for memory_id, search_time in entries:
                exposures += 1
                target_exposure_counts[memory_id] += 1
                target_read = any(
                    search_time <= read_time <= min(
                        cutoff_epoch,
                        search_time + self._ENGAGEMENT_ATTRIBUTION_WINDOW_SECONDS,
                    )
                    for read_time in reads_by_memory.get(memory_id, ())
                )
                if target_read:
                    positives += 1
                elif co_reads - {memory_id}:
                    conditional += 1
        weak = max(exposures - positives - conditional, 0)
        repeated = sum(max(count - 1, 0) for count in target_exposure_counts.values())
        weak = min(weak, repeated)
        unknown = exposures - positives - conditional - weak
        utility_delta = round(positives * 0.1 - conditional * 0.05 - weak * 0.02, 3)
        return {
            "utility_delta": utility_delta,
            "evidence": {
                "positive": positives,
                "conditional_negative": conditional,
                "weak_negative": weak,
                "unknown": unknown,
                "exposures": exposures,
                "attribution_window_seconds": self._ENGAGEMENT_ATTRIBUTION_WINDOW_SECONDS,
            },
        }

    def _content_quality(self, mutation: CurationQualityMutation | CurationActionReceipt) -> _ContentQualityComparison:
        mutation = _coerce_mutation(mutation)
        if mutation.operation not in {"normalize_memory", "rewrite_memory"}:
            return _ContentQualityComparison()
        rows = self._quality_query.revision_snapshots(str(mutation.mutation_event_id))
        scores = [
            (_content_quality_score(row[0]), _content_quality_score(row[1]))
            for row in rows
        ]
        if not scores or any(before is None or after is None for before, after in scores):
            return _ContentQualityComparison()
        before = sum(score[0] for score in scores if score[0] is not None) / len(scores)
        after = sum(score[1] for score in scores if score[1] is not None) / len(scores)
        delta = round(after - before, 3)
        return _ContentQualityComparison(
            before=before,
            after=after,
            delta=delta,
            improved=delta > 0,
        )

    def _intended_memory_ids(
        self,
        mutation: CurationQualityMutation | CurationActionReceipt,
    ) -> list[UUID]:
        mutation = _coerce_mutation(mutation)
        if mutation.mutation_event_id is None:
            return list(mutation.affected_memory_ids)
        rows = self._quality_query.revision_states(str(mutation.mutation_event_id))
        snapshots = {
            str(row[0]): (row[1], row[2])
            for row in rows
            if row[0] is not None
        }
        intended: list[UUID] = []
        for memory_id in mutation.affected_memory_ids:
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
        return intended if snapshots else list(mutation.affected_memory_ids)

    def _is_sampled(self, run_id: UUID, action_id: UUID) -> bool:
        if self._sample_rate >= 1.0:
            return True
        digest = hashlib.sha256(f"{run_id}:{action_id}".encode("ascii")).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self._sample_rate

    def _find_historical_query(
        self,
        mutation: CurationQualityMutation,
        explicit_hypothesis: CampaignHypothesis | None = None,
    ) -> QualityQuery | None:
        if explicit_hypothesis is not None and explicit_hypothesis.query:
            return self._find_explicit_query(mutation, explicit_hypothesis.query)
        if explicit_hypothesis is not None and explicit_hypothesis.target_mode is CampaignTargetMode.ZERO_RESULTS:
            return None
        target_ids = [str(value) for value in mutation.affected_memory_ids]
        if not target_ids:
            return None
        rows = self._quality_query.historical_search_events(target_ids)
        target_id_set = set(target_ids)
        cutoff = mutation.applied_at or self._clock()
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
        return self._hydrate_historical_query(
            invocation_id, provenance=QueryProvenance.REAL_USER_SEARCH
        )

    def _find_explicit_query(
        self,
        mutation: CurationQualityMutation,
        query_text: str | None,
    ) -> QualityQuery | None:
        if not query_text or not query_text.strip():
            return None
        cutoff = mutation.applied_at or self._clock()
        rows = self._quality_query.search_invocations_for_query(query_text)
        for row in rows:
            event_time = _timestamp(row[1])
            if event_time is not None and event_time < cutoff:
                return self._hydrate_historical_query(
                    str(row[0]), provenance=QueryProvenance.CAMPAIGN_HYPOTHESIS
                )
        return None

    def _hydrate_historical_query(
        self,
        invocation_id: str,
        *,
        provenance: QueryProvenance = QueryProvenance.REAL_USER_SEARCH,
    ) -> QualityQuery | None:
        rows = self._quality_query.search_events_for_invocation(invocation_id)
        if not rows:
            return None
        return QualityQuery(
            query_id=invocation_id,
            query_text=str(rows[0][1] or ""),
            before_memory_ids=tuple(str(row[2]) for row in rows if row[2] is not None),
            replay_complete=max(
                (int(str(row[4])) for row in rows if row[4] is not None),
                default=len(rows),
            )
            <= len(rows),
            provenance=provenance,
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


def _synthetic_query(mutation: CurationQualityMutation) -> QualityQuery | None:
    for snapshot in mutation.after_entities.values():
        terms = [
            str(snapshot.get(field, "")).strip()
            for field in ("title", "summary", "content")
            if str(snapshot.get(field, "")).strip()
        ]
        if terms:
            return QualityQuery(
                query_id=f"synthetic:{mutation.mutation_id}",
                query_text=" ".join(terms)[:240],
                before_memory_ids=(),
                replay_complete=False,
                provenance=QueryProvenance.SYNTHETIC_PROBE,
            )
    return None


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


def _content_quality_score(snapshot: Any) -> float | None:
    if isinstance(snapshot, str):
        try:
            snapshot = json.loads(snapshot)
        except ValueError:
            return None
    if not isinstance(snapshot, Mapping):
        return None
    record = snapshot.get("record", snapshot)
    if not isinstance(record, Mapping):
        return None
    content = str(record.get("content") or "").strip()
    title = str(record.get("title") or "").strip()
    summary = str(record.get("summary") or "").strip().lower()
    score = 0.0
    if title:
        score += 1.0
    if summary and not summary.startswith(("covers ", "added ", "updated ")):
        score += 1.0
    content_length = len(content)
    if 0 < content_length <= 1600:
        score += 1.0
    elif content_length <= 3000:
        score += 0.5
    if re.search(r"(?:/[\w.-]+|#[0-9]+|[A-Z]{2,}-[0-9]+|\b(?:commit|error|exception)\b)", content, re.IGNORECASE):
        score += 1.0
    return round(score / 4.0, 3)


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


def _epoch(value: object) -> float | None:
    timestamp = _timestamp(value)
    return None if timestamp is None else timestamp.timestamp()


def _context_memory_id(context: Any) -> str:
    return str(context.record.id)


def _search_epochs(search: Any) -> dict[str, int]:
    reader = getattr(search, "get_search_epochs", None)
    if not callable(reader):
        return {}
    values = reader()
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): int(value)
        for key, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _replay_provenance(
    query: QualityQuery,
    after_context: Mapping[str, object],
    after_epochs: Mapping[str, int],
) -> dict[str, object]:
    return {
        "before_query_context": dict(query.query_context),
        "after_query_context": dict(after_context),
        "before_search_epochs": dict(query.before_search_epochs),
        "after_search_epochs": dict(after_epochs),
    }


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and value > 0 else 0


def _coerce_mutation(
    value: CurationQualityMutation | CurationActionReceipt,
) -> CurationQualityMutation:
    if isinstance(value, CurationQualityMutation):
        return value
    return mutation_from_receipt(value)


def _action_identity_errors(
    mutations: Sequence[CurationQualityMutation],
) -> tuple[str | None, ...]:
    evidence_counts = Counter(
        mutation.evidence_id
        for mutation in mutations
        if mutation.action_identity_required and mutation.evidence_id is not None
    )
    missing_identity_action_counts = Counter(
        mutation.mutation_id
        for mutation in mutations
        if mutation.action_identity_required and mutation.evidence_id is None
    )
    errors: list[str | None] = []
    for mutation in mutations:
        if not mutation.action_identity_required:
            errors.append(None)
        elif mutation.evidence_id is None:
            errors.append(
                "ambiguous" if missing_identity_action_counts[mutation.mutation_id] > 1 else "missing"
            )
        elif evidence_counts[mutation.evidence_id] > 1:
            errors.append("ambiguous")
        elif mutation.mutation_id != direct_action_id(mutation.evidence_id):
            errors.append("mismatch")
        else:
            errors.append(None)
    return tuple(errors)

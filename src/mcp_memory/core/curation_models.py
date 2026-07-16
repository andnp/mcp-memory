"""Provider- and repository-neutral contracts for curation planning."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CurationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class LinkAssertion(CurationModel):
    source_id: UUID
    target_id: UUID
    link_type: str
    context: str | None = None


class EvidenceRef(CurationModel):
    memory_id: UUID | None = None
    link: LinkAssertion | None = None
    revision_token: str | None = None
    excerpt: str | None = None

    @model_validator(mode="after")
    def has_reference(self) -> EvidenceRef:
        if self.memory_id is None and self.link is None:
            raise ValueError("evidence must reference a memory or link")
        return self


class ActionPreconditions(CurationModel):
    record_tokens: dict[UUID, str] = Field(default_factory=dict)
    required_statuses: dict[UUID, MemoryStatus] = Field(default_factory=dict)
    required_links: list[LinkAssertion] = Field(default_factory=list)
    absent_links: list[LinkAssertion] = Field(default_factory=list)


class OmittedMaterial(CurationModel):
    material: str
    reason: Literal[
        "duplicate_wording",
        "routine_execution_trace",
        "formatting_or_boilerplate",
        "moved_to_split_child",
        "explicitly_superseded",
    ]


class ClaimMapping(CurationModel):
    output: str
    source_memory_ids: list[UUID] = Field(default_factory=list)


class ClaimManifest(CurationModel):
    preserved_claims: list[str] = Field(default_factory=list)
    transformed_claims: list[str] = Field(default_factory=list)
    omitted_material: list[OmittedMaterial] = Field(default_factory=list)
    source_mapping: list[ClaimMapping] = Field(default_factory=list)
    unresolved_tensions: list[str] = Field(default_factory=list)


class RetentionReason(StrEnum):
    ALREADY_FOCUSED = "already_focused"
    DISTINCT_CLAIMS = "distinct_claims"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROJECT_BOUNDARY_RISK = "project_boundary_risk"
    RELATIONSHIP_IS_INTENTIONAL = "relationship_is_intentional"
    DESTRUCTIVE_CHANGE_NOT_JUSTIFIED = "destructive_change_not_justified"
    NEEDS_DIFFERENT_SPECIALIST = "needs_different_specialist"


class RetentionDecision(CurationModel):
    memory_id: UUID
    reason: RetentionReason
    rationale: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class ActionEnvelope(CurationModel):
    action_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    preconditions: ActionPreconditions = Field(default_factory=ActionPreconditions)


class NormalizeMemoryAction(ActionEnvelope):
    operation: Literal["normalize_memory"] = "normalize_memory"
    target_id: UUID
    title: str | None = None
    summary: str | None = None
    tags: list[str] | None = None

    @model_validator(mode="after")
    def requires_metadata_change(self) -> NormalizeMemoryAction:
        if self.title is None and self.summary is None and self.tags is None:
            raise ValueError("normalize_memory requires at least one metadata field")
        return self


class RewriteMemoryAction(ActionEnvelope):
    operation: Literal["rewrite_memory"] = "rewrite_memory"
    target_id: UUID
    title: str | None = None
    content: str
    summary: str | None = None
    claim_manifest: ClaimManifest


class CreateLinkAction(ActionEnvelope):
    operation: Literal["create_link"] = "create_link"
    source_id: UUID
    target_id: UUID
    link_type: str
    context: str | None = None


class RemoveLinkAction(ActionEnvelope):
    operation: Literal["remove_link"] = "remove_link"
    source_id: UUID
    target_id: UUID
    link_type: str
    context: str | None = None


class MergeMemoriesAction(ActionEnvelope):
    operation: Literal["merge_memories"] = "merge_memories"
    canonical_id: UUID
    source_ids: list[UUID] = Field(min_length=1)
    title: str | None = None
    content: str
    summary: str | None = None
    claim_manifest: ClaimManifest


class SplitMemoryAction(ActionEnvelope):
    operation: Literal["split_memory"] = "split_memory"
    target_id: UUID
    children: list[ClaimMapping] = Field(min_length=1)
    claim_manifest: ClaimManifest


class ArchiveMemoryAction(ActionEnvelope):
    operation: Literal["archive_memory"] = "archive_memory"
    target_id: UUID
    claim_manifest: ClaimManifest


CurationAction = Annotated[
    NormalizeMemoryAction
    | RewriteMemoryAction
    | CreateLinkAction
    | RemoveLinkAction
    | MergeMemoriesAction
    | SplitMemoryAction
    | ArchiveMemoryAction,
    Field(discriminator="operation"),
]


class CurationPolicySummary(CurationModel):
    allowed_operations: list[str] = Field(default_factory=list)
    policy_version: str = "1"


class CurationContextPacket(CurationModel):
    seed_memory_ids: list[UUID] = Field(default_factory=list)
    support_memory_ids: list[UUID] = Field(default_factory=list)
    context_fingerprint: str


class CurationPlanningRequest(CurationModel):
    schema_version: Literal[1] = 1
    run_id: UUID
    plan_id: UUID
    frontier_key: str
    context_fingerprint: str
    policy_summary: CurationPolicySummary = Field(default_factory=CurationPolicySummary)
    context: CurationContextPacket | None = None


class CurationPlan(CurationModel):
    schema_version: Literal[1] = 1
    plan_id: UUID
    run_id: UUID
    frontier_key: str
    context_fingerprint: str
    actions: list[CurationAction] = Field(default_factory=list)
    retained: list[RetentionDecision] = Field(default_factory=list)
    rationale: str
    seed_memory_ids: list[UUID] = Field(default_factory=list)

    @model_validator(mode="after")
    def validates_seed_dispositions(self) -> CurationPlan:
        action_ids = {
            target
            for action in self.actions
            for target in _action_memory_ids(action)
        }
        retained_ids = {decision.memory_id for decision in self.retained}
        missing = set(self.seed_memory_ids) - action_ids - retained_ids
        if missing:
            raise ValueError(f"missing seed dispositions: {sorted(missing, key=str)}")
        if not self.actions and not self.retained:
            raise ValueError("an empty plan requires a retention decision")
        if action_ids & retained_ids:
            raise ValueError("a memory cannot be both acted on and retained")
        return self


class CurationBudgetUsage(CurationModel):
    seed_records: int = 0
    support_records: int = 0
    context_characters: int = 0
    read_tool_calls: int = 0
    records_returned: int = 0
    proposed_actions: int = 0
    accepted_mutations: int = 0
    planner_attempts: int = 0
    premium_requests: int = 0
    wall_clock_seconds: float = 0.0
    token_usage: int | None = None
    token_usage_source: str | None = None


class ReceiptStatus(StrEnum):
    APPLIED_UNVERIFIED = "applied_unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    STALE = "stale"
    FAILED = "failed"


class MutationReceipt(CurationModel):
    run_id: UUID
    action_id: UUID
    operation: str
    affected_ids: list[UUID] = Field(default_factory=list)
    status: ReceiptStatus
    before_token: str | None = None
    after_token: str | None = None
    error_code: str | None = None
    applied_at: str | None = None
    verified_at: str | None = None


class CurationRunOutcome(StrEnum):
    APPLIED = "applied"
    PARTIALLY_APPLIED = "partially_applied"
    NO_OP = "no_op"
    INVALID_PLAN = "invalid_plan"
    STALE_PLAN = "stale_plan"
    VERIFICATION_FAILED = "verification_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEFERRED = "deferred"
    PROVIDER_FAILED = "provider_failed"
    CANCELLED = "cancelled"
    NO_CANDIDATES = "no_candidates"


class CurationRunResult(CurationModel):
    run_id: UUID
    outcome: CurationRunOutcome
    plan_id: UUID | None = None
    receipts: list[MutationReceipt] = Field(default_factory=list)
    rejection_codes: list[str] = Field(default_factory=list)
    retry_reason: str | None = None
    budget_usage: CurationBudgetUsage = Field(default_factory=CurationBudgetUsage)
    verified_action_count: int = 0
    affected_memory_count: int = 0


def _action_memory_ids(action: CurationAction) -> set[UUID]:
    if isinstance(action, MergeMemoriesAction):
        return set(action.source_ids) | {action.canonical_id}
    if isinstance(action, (CreateLinkAction, RemoveLinkAction)):
        return {action.source_id, action.target_id}
    return {action.target_id}

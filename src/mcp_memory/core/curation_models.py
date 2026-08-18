"""Provider- and repository-neutral contracts for curation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

CanonicalLinkType = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]


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


class AbsentLinkAssertion(CurationModel):
    source_id: UUID
    target_id: UUID
    link_type: str


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
    absent_links: list[AbsentLinkAssertion] = Field(default_factory=list)


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


class ActionEnvelope(CurationModel):
    action_id: UUID
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    preconditions: ActionPreconditions = Field(default_factory=ActionPreconditions)


class NormalizeMemoryAction(ActionEnvelope):
    operation: Literal["normalize_memory"] = "normalize_memory"
    target_id: UUID = Field(description="An existing visible memory ID from the curator context.")
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
    target_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    title: str | None = None
    content: str
    summary: str | None = None
    claim_manifest: ClaimManifest


class CreateLinkAction(ActionEnvelope):
    operation: Literal["create_link"] = "create_link"
    source_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    target_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    link_type: CanonicalLinkType
    context: str | None = None

    @model_validator(mode="after")
    def requires_exact_evidence(self) -> CreateLinkAction:
        context = (self.context or "").strip()
        if len(context.split()) < 2:
            raise ValueError("create_link requires descriptive relationship context")
        return self


class RemoveLinkAction(ActionEnvelope):
    operation: Literal["remove_link"] = "remove_link"
    source_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    target_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    link_type: CanonicalLinkType
    context: str | None = None


class MergeMemoriesAction(ActionEnvelope):
    operation: Literal["merge_memories"] = "merge_memories"
    canonical_id: UUID = Field(
        description="An existing visible memory ID to retain; merge never creates a record."
    )
    source_ids: list[UUID] = Field(
        min_length=1,
        description="Existing visible memory IDs to merge into canonical_id; do not invent IDs.",
    )
    title: str | None = None
    content: str
    summary: str | None = None
    claim_manifest: ClaimManifest


class SplitMemoryAction(ActionEnvelope):
    operation: Literal["split_memory"] = "split_memory"
    target_id: UUID = Field(description="An existing visible memory ID from the curator context.")
    children: list[ClaimMapping] = Field(
        min_length=1,
        description="Typed child content only; child record IDs are assigned by execution, not invented here.",
    )
    claim_manifest: ClaimManifest


class ArchiveMemoryAction(ActionEnvelope):
    operation: Literal["archive_memory"] = "archive_memory"
    target_id: UUID
    claim_manifest: ClaimManifest


VerificationOperation = Literal[
    "normalize_memory",
    "rewrite_memory",
    "create_link",
    "remove_link",
    "merge_memories",
    "split_memory",
    "archive_memory",
]
VerificationStatus = Literal["active", "stale", "degraded", "archived"]

_MAX_VERIFICATION_TARGET_IDS = 64
_MAX_VERIFICATION_SOURCE_IDS = 64
_MAX_VERIFICATION_CHILD_IDS = 64
_MAX_VERIFICATION_CONTEXT_LENGTH = 512
_MAX_VERIFICATION_SPLIT_GROUP_LENGTH = 128


class CurationVerificationDescriptor(CurationModel):
    """Bounded, content-free postcondition data for crash recovery."""

    schema_version: Literal[1] = 1
    operation: VerificationOperation
    target_ids: list[UUID] = Field(default_factory=list, max_length=_MAX_VERIFICATION_TARGET_IDS)
    target_status: VerificationStatus | None = None
    source_id: UUID | None = None
    target_id: UUID | None = None
    link_type: CanonicalLinkType | None = None
    context: str | None = Field(default=None, max_length=_MAX_VERIFICATION_CONTEXT_LENGTH)
    exists: bool | None = None
    canonical_id: UUID | None = None
    source_ids: list[UUID] = Field(default_factory=list, max_length=_MAX_VERIFICATION_SOURCE_IDS)
    child_ids: list[UUID] = Field(default_factory=list, max_length=_MAX_VERIFICATION_CHILD_IDS)
    split_group_id: str | None = Field(default=None, max_length=_MAX_VERIFICATION_SPLIT_GROUP_LENGTH)
    child_count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_shape(self) -> CurationVerificationDescriptor:
        allowed: set[str]
        if self.operation in {"normalize_memory", "rewrite_memory"}:
            allowed = {"target_ids", "target_status"}
        elif self.operation in {"create_link", "remove_link"}:
            allowed = {"target_ids", "source_id", "target_id", "link_type", "context", "exists"}
        elif self.operation == "merge_memories":
            allowed = {"target_ids", "canonical_id", "source_ids"}
        else:
            allowed = {"target_ids", "target_id", "child_ids", "split_group_id", "child_count"}
            if self.operation == "archive_memory":
                allowed = {"target_ids", "target_status"}
        values = {
            "target_ids": self.target_ids,
            "target_status": self.target_status,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "link_type": self.link_type,
            "context": self.context,
            "exists": self.exists,
            "canonical_id": self.canonical_id,
            "source_ids": self.source_ids,
            "child_ids": self.child_ids,
            "split_group_id": self.split_group_id,
            "child_count": self.child_count,
        }
        if any(
            name not in allowed and value not in (None, [])
            for name, value in values.items()
        ):
            raise ValueError(f"verification fields do not match operation {self.operation!r}")
        for name, ids in (
            ("target_ids", self.target_ids),
            ("source_ids", self.source_ids),
            ("child_ids", self.child_ids),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"{name} must not contain duplicate IDs")
        if self.operation in {"normalize_memory", "rewrite_memory", "archive_memory"}:
            if len(self.target_ids) != 1 or self.target_status is None:
                raise ValueError("record verification requires one target ID and status")
            if self.operation == "archive_memory" and self.target_status != "archived":
                raise ValueError("archive verification requires archived target status")
        elif self.operation in {"create_link", "remove_link"}:
            if (
                len(self.target_ids) != 2
                or self.source_id is None
                or self.target_id is None
                or self.link_type is None
                or self.context is None
                or self.exists is None
                or set(self.target_ids) != {self.source_id, self.target_id}
            ):
                raise ValueError("link verification requires an exact endpoint assertion")
        elif self.operation == "merge_memories":
            if (
                self.canonical_id is None
                or not self.source_ids
                or self.canonical_id in self.source_ids
                or set(self.target_ids) != {self.canonical_id, *self.source_ids}
            ):
                raise ValueError("merge verification requires canonical and source IDs")
        elif self.operation == "split_memory":
            if (
                len(self.target_ids) != 1
                or self.target_id is None
                or self.target_ids[0] != self.target_id
                or not self.child_ids
                or self.split_group_id is None
                or self.child_count != len(self.child_ids)
            ):
                raise ValueError("split verification requires child/group IDs and count")
        return self


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


class CampaignRetrievalProblem(StrEnum):
    HEURISTIC = "heuristic"
    RETRIEVAL_QUALITY = "retrieval_quality"


class CampaignTargetMode(StrEnum):
    HEURISTIC = "heuristic"
    RANK = "rank"
    TOP_K = "top_k"
    ZERO_RESULTS = "zero_results"


class CampaignAcceptance(CurationModel):
    target_mode: CampaignTargetMode = CampaignTargetMode.HEURISTIC
    top_k: int = Field(default=5, ge=1)
    minimum_improvement: float = Field(default=0.0, ge=0.0)


class CampaignHypothesis(CurationModel):
    """Typed retrieval objective carried by curator campaigns."""

    schema_version: Literal[1] = 1
    retrieval_problem: CampaignRetrievalProblem = CampaignRetrievalProblem.HEURISTIC
    query: str | None = None
    expected_memory_ids: list[UUID] = Field(default_factory=list)
    target_mode: CampaignTargetMode = CampaignTargetMode.HEURISTIC
    top_k: int = Field(default=5, ge=1)
    minimum_improvement: float = Field(default=0.0, ge=0.0)
    acceptance: CampaignAcceptance = Field(default_factory=CampaignAcceptance)

    @model_validator(mode="before")
    @classmethod
    def flatten_acceptance(cls, value: Any) -> Any:
        if not isinstance(value, Mapping) or "acceptance" not in value:
            return value
        data = dict(value)
        acceptance = CampaignAcceptance.model_validate(data.pop("acceptance"))
        data["acceptance"] = acceptance
        for name in ("target_mode", "top_k", "minimum_improvement"):
            data.setdefault(name, getattr(acceptance, name))
        return data

    @model_validator(mode="after")
    def unique_expected_memory_ids(self) -> CampaignHypothesis:
        if len(self.expected_memory_ids) != len(set(self.expected_memory_ids)):
            raise ValueError("expected_memory_ids must not contain duplicates")
        self.acceptance = CampaignAcceptance(
            target_mode=self.target_mode,
            top_k=self.top_k,
            minimum_improvement=self.minimum_improvement,
        )
        return self

    @classmethod
    def legacy(cls) -> CampaignHypothesis:
        return cls()

CurationCampaignHypothesis = CampaignHypothesis
CurationCampaignAcceptance = CampaignAcceptance
RetrievalProblem = CampaignRetrievalProblem
TargetMode = CampaignTargetMode


def campaign_hypothesis_from_payload(payload: Mapping[str, Any] | None) -> CampaignHypothesis:
    if not payload or payload.get("campaign_hypothesis") is None:
        return CampaignHypothesis.legacy()
    return CampaignHypothesis.model_validate(payload["campaign_hypothesis"])


class CurationBudgetUsage(CurationModel):
    seed_records: int = 0
    support_records: int = 0
    exploratory_records: int = 0
    context_characters: int = 0
    read_tool_calls: int = 0
    records_returned: int = 0
    proposed_actions: int = 0
    accepted_mutations: int = 0
    # Retained so historical typed-curation payloads remain readable.
    planner_attempts: int = 0
    premium_requests: int = 0
    provider_calls: int = 0
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
    QUALITY_REJECTED = "quality_rejected"
    NO_OP = "no_op"
    # Historical outcomes retained for reading older typed-curation runs.
    INVALID_PLAN = "invalid_plan"
    STALE_PLAN = "stale_plan"
    VERIFICATION_FAILED = "verification_failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DEFERRED = "deferred"
    PROVIDER_FAILED = "provider_failed"
    CANCELLED = "cancelled"
    NO_CANDIDATES = "no_candidates"
    QUALITY_OVERRIDE = "quality_override"

"""Provider-free executors for accepted curation operations.

This module deliberately contains no production orchestration.  It turns a
typed action into the callback expected by a backend action transaction; the
transaction primitive remains responsible for atomic history, receipts, and
idempotency.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import NoReturn, cast
from uuid import UUID

from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    ClaimMapping,
    CreateLinkAction,
    CurationVerificationDescriptor,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
    VerificationStatus,
)
from mcp_memory.core.curation_policy import PolicyDecision
from mcp_memory.core.ports.curation import (
    CurationActionContractError,
    CurationActionFatalError,
    CurationActionReceipt,
    CurationActionStore,
    CurationTransaction,
    MutationResult,
)
from mcp_memory.mutation_history import ProtectionMode


class CurationPolicyRejection(CurationActionFatalError):
    """Compatibility error for callers that still report policy diagnostics."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        codes = ", ".join(code.value for code in decision.rejection_codes)
        super().__init__(f"{decision.operation} rejected by policy: {codes}")


class CurationExecutor:
    """Execute individually accepted curation actions through an action store."""

    def __init__(self, action_store: CurationActionStore) -> None:
        self._action_store = action_store

    def execute_normalize(
        self,
        action: NormalizeMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: Iterable[ProtectionMode | str] = (),
        expected_token: str | None = None,
    ) -> CurationActionReceipt:
        """Apply title/summary/tags only and return the compact transaction receipt.

        The record token comes from the typed action preconditions.  The
        optional ``expected_token`` is a convenience for callers that keep
        preconditions separately; supplying both values must agree.
        """
        record_token = action.preconditions.record_tokens.get(action.target_id)
        if record_token is not None and expected_token is not None and record_token != expected_token:
            _reject_missing_or_conflicting_token("expected record token conflicts with action preconditions")
        token = expected_token or record_token
        if not token:
            _reject_missing_or_conflicting_token("normalize_memory requires an expected record token")

        changes: dict[str, object] = {}
        if action.title is not None:
            changes["title"] = action.title
        if action.summary is not None:
            changes["summary"] = action.summary
        if action.tags is not None:
            changes["tags"] = action.tags

        def apply(transaction: CurationTransaction) -> MutationResult:
            # Do not add content here: normalize is intentionally metadata-only.
            transaction.update_memory(str(action.target_id), **changes)
            return MutationResult(action.operation, [action.target_id], _record_verification_descriptor(action))

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.target_id)],
            expected_tokens={str(action.target_id): token},
            preconditions=action.preconditions,
            operation=action.operation,
            payload={"title": action.title, "summary": action.summary, "tags": action.tags},
            apply=apply,
        )

    def execute_create_link(
        self,
        action: CreateLinkAction,
        *,
        run_id: UUID,
        source_type: str | None = None,
        target_type: str | None = None,
        memory_types: Mapping[UUID, str] | None = None,
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Create one typed edge through the action transaction.

        Link creation is deliberately stricter than the generic policy check:
        the plan must carry exact link evidence, a record token for each
        endpoint, and an exact absent-link assertion.  Those checks keep a
        generic relationship rationale from becoming mutation authority.
        """
        link_type = action.link_type.strip()
        context = (action.context or "").strip()
        _require_exact_link_evidence(action, link_type=link_type, context=context)
        _require_create_link_preconditions(action, link_type=link_type)

        endpoint_ids = (action.source_id, action.target_id)
        expected_tokens: dict[str, str] = {}
        for endpoint_id in endpoint_ids:
            token = action.preconditions.record_tokens.get(endpoint_id)
            if not token:
                raise CurationActionContractError(
                    f"create_link requires an expected record token for {endpoint_id}",
                    code="missing_record_token",
                )
            expected_tokens[str(endpoint_id)] = token

        def apply(transaction: CurationTransaction) -> MutationResult:
            transaction.add_link(
                str(action.source_id),
                str(action.target_id),
                link_type,
                context,
            )
            return MutationResult(
                action.operation,
                endpoint_ids,
                CurationVerificationDescriptor(
                    operation=action.operation,
                    target_ids=list(endpoint_ids),
                    source_id=action.source_id,
                    target_id=action.target_id,
                    link_type=link_type,
                    context=context,
                    exists=True,
                ),
            )

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(endpoint_id) for endpoint_id in endpoint_ids],
            expected_tokens=expected_tokens,
            preconditions=action.preconditions,
            operation=action.operation,
            payload={"link_type": link_type, "context": context},
            apply=apply,
        )

    def execute_rewrite(
        self,
        action: RewriteMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Rewrite one record through the same transaction boundary."""
        token = action.preconditions.record_tokens.get(action.target_id)
        if not token:
            _reject_missing_or_conflicting_token("rewrite_memory requires an expected record token")

        changes: dict[str, object] = {"content": action.content}
        if action.title is not None:
            changes["title"] = action.title
        if action.summary is not None:
            changes["summary"] = action.summary

        def apply(transaction: CurationTransaction) -> MutationResult:
            transaction.update_memory(str(action.target_id), **changes)
            return MutationResult(action.operation, [action.target_id], _record_verification_descriptor(action))

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.target_id)],
            expected_tokens={str(action.target_id): token},
            preconditions=action.preconditions,
            operation=action.operation,
            payload=action.model_dump(mode="json"),
            apply=apply,
        )

    def execute_remove_link(
        self,
        action: RemoveLinkAction,
        *,
        run_id: UUID,
        source_type: str | None = None,
        target_type: str | None = None,
        memory_types: Mapping[UUID, str] | None = None,
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Remove one typed edge through the same transaction boundary."""
        source_token = action.preconditions.record_tokens.get(action.source_id)
        target_token = action.preconditions.record_tokens.get(action.target_id)
        if not source_token or not target_token:
            raise CurationActionContractError(
                "remove_link requires expected record tokens for both endpoints",
                code="missing_record_token",
            )

        link_type = action.link_type.strip()
        context = (action.context or "").strip()

        def apply(transaction: CurationTransaction) -> MutationResult:
            transaction.remove_link(str(action.source_id), str(action.target_id), link_type)
            return MutationResult(
                action.operation,
                [action.source_id, action.target_id],
                CurationVerificationDescriptor(
                    operation=action.operation,
                    target_ids=[action.source_id, action.target_id],
                    source_id=action.source_id,
                    target_id=action.target_id,
                    link_type=link_type,
                    context=context,
                    exists=False,
                ),
            )

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.source_id), str(action.target_id)],
            expected_tokens={str(action.source_id): source_token, str(action.target_id): target_token},
            preconditions=action.preconditions,
            operation=action.operation,
            payload={"link_type": link_type, "context": context},
            apply=apply,
        )

    def execute_merge(
        self,
        action: MergeMemoriesAction,
        *,
        run_id: UUID,
        memory_types: Mapping[UUID, str] | None = None,
        contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset(),
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Merge source records into an existing canonical record."""
        merge_ids = _canonical_merge_ids(action.canonical_id, action.source_ids)
        if action.canonical_id in action.source_ids:
            raise CurationActionContractError(
                "merge_memories source_ids must not include canonical_id",
                code="invalid_merge_targets",
            )

        expected_tokens: dict[str, str] = {}
        for memory_id in merge_ids:
            token = action.preconditions.record_tokens.get(memory_id)
            if not token:
                raise CurationActionContractError(
                    f"merge_memories requires an expected record token for {memory_id}",
                    code="missing_record_token",
                )
            expected_tokens[str(memory_id)] = token

        def apply(transaction: CurationTransaction) -> MutationResult:
            canonical = transaction.get_memory(str(action.canonical_id))
            if canonical is None:
                raise CurationActionFatalError(f"canonical memory {action.canonical_id!r} is missing")
            sources = []
            for source_id in merge_ids[1:]:
                source = transaction.get_memory(str(source_id))
                if source is None:
                    raise CurationActionFatalError(f"source memory {source_id!r} is missing")
                sources.append(source)

            merged_tags = _sorted_unique([*canonical.tags, *(tag for source in sources for tag in source.tags)])
            merged_workspaces = _sorted_unique(
                [*canonical.workspace_ids, *(workspace for source in sources for workspace in source.workspace_ids)]
            )
            merged_metadata = dict(canonical.metadata)
            existing_merged_source_ids = merged_metadata.get("merged_source_ids")
            merged_metadata["merged_source_ids"] = _sorted_unique(
                [
                    *map(str, existing_merged_source_ids if isinstance(existing_merged_source_ids, list) else []),
                    *(source.id for source in sources),
                ]
            )
            transaction.update_memory(
                str(action.canonical_id),
                title=action.title or canonical.title,
                content=action.content,
                summary=action.summary if action.summary is not None else canonical.summary,
                tags=merged_tags,
                workspace_ids=merged_workspaces,
                metadata=merged_metadata,
            )
            for source in sources:
                transaction.add_link(str(action.canonical_id), str(source.id), "SUPERSEDES", _MERGE_LINK_CONTEXT)
                transaction.update_memory(str(source.id), status="archived")
            return MutationResult(
                action.operation,
                [action.canonical_id, *[source.id for source in sources]],
                CurationVerificationDescriptor(
                    operation=action.operation,
                    target_ids=list(merge_ids),
                    canonical_id=action.canonical_id,
                    source_ids=list(action.source_ids),
                ),
            )

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(memory_id) for memory_id in merge_ids],
            expected_tokens=expected_tokens,
            preconditions=action.preconditions,
            operation=action.operation,
            payload=action.model_dump(mode="json"),
            apply=apply,
        )

    def execute_split(
        self,
        action: SplitMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Split one record into typed child records through the same transaction boundary."""
        token = action.preconditions.record_tokens.get(action.target_id)
        if not token:
            _reject_missing_or_conflicting_token("split_memory requires an expected record token")

        child_specs = _normalized_split_children(action.children)

        def apply(transaction: CurationTransaction) -> MutationResult:
            original = transaction.get_memory(str(action.target_id))
            if original is None:
                raise CurationActionFatalError(f"split target {action.target_id!r} is missing")

            split_group_id = str(action.action_id)
            created_children = []
            for index, child in enumerate(child_specs, start=1):
                created = transaction.create_memory(
                    title=child["title"],
                    content=child["content"],
                    summary=None,
                    memory_type=original.type,
                    status="active",
                    workspace_ids=list(original.workspace_ids),
                    tags=list(original.tags),
                    metadata={
                        "split_from_memory_id": original.id,
                        "split_from_memory_title": original.title,
                        "split_group_id": split_group_id,
                        "split_part_index": index,
                        "split_part_count": len(child_specs),
                    },
                )
                transaction.add_link(str(created.id), str(original.id), _SPLIT_LINK_TYPE, _SPLIT_LINK_CONTEXT)
                created_children.append(created)

            child_ids = [child.id for child in created_children]
            for child in created_children:
                sibling_ids = [child_id for child_id in child_ids if child_id != child.id]
                transaction.update_memory(
                    str(child.id),
                    metadata={
                        **child.metadata,
                        "split_from_memory_id": original.id,
                        "split_from_memory_title": original.title,
                        "split_group_id": split_group_id,
                        "split_part_index": child_ids.index(child.id) + 1,
                        "split_part_count": len(child_ids),
                        "split_child_memory_ids": child_ids,
                        "split_sibling_memory_ids": sibling_ids,
                    },
                )
            transaction.update_memory(
                str(original.id),
                metadata={
                    **original.metadata,
                    "split_group_id": split_group_id,
                    "split_child_memory_ids": child_ids,
                    "split_child_count": len(child_ids),
                },
            )
            return MutationResult(
                action.operation,
                [action.target_id, *child_ids],
                CurationVerificationDescriptor(
                    operation=action.operation,
                    target_ids=[action.target_id],
                    target_id=action.target_id,
                    child_ids=[UUID(child_id) for child_id in child_ids],
                    split_group_id=split_group_id,
                    child_count=len(child_ids),
                ),
            )

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.target_id)],
            expected_tokens={str(action.target_id): token},
            preconditions=action.preconditions,
            operation=action.operation,
            payload=action.model_dump(mode="json"),
            apply=apply,
        )

    def execute_archive(
        self,
        action: ArchiveMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: Iterable[ProtectionMode | str] = (),
    ) -> CurationActionReceipt:
        """Archive one record through the same transaction boundary."""
        token = action.preconditions.record_tokens.get(action.target_id)
        if not token:
            _reject_missing_or_conflicting_token("archive_memory requires an expected record token")

        def apply(transaction: CurationTransaction) -> MutationResult:
            transaction.update_memory(str(action.target_id), status="archived")
            return MutationResult(
                action.operation,
                [action.target_id],
                CurationVerificationDescriptor(
                    operation=action.operation,
                    target_ids=[action.target_id],
                    target_status="archived",
                ),
            )

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.target_id)],
            expected_tokens={str(action.target_id): token},
            preconditions=action.preconditions,
            operation=action.operation,
            payload=action.model_dump(mode="json"),
            apply=apply,
        )


def execute_normalize_memory(
    action_store: CurationActionStore,
    action: NormalizeMemoryAction,
    *,
    run_id: UUID,
    memory_type: str,
    protections: Iterable[ProtectionMode | str] = (),
    expected_token: str | None = None,
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral normalize execution."""
    return CurationExecutor(action_store).execute_normalize(
        action,
        run_id=run_id,
        memory_type=memory_type,
        protections=protections,
        expected_token=expected_token,
    )


def execute_create_link(
    action_store: CurationActionStore,
    action: CreateLinkAction,
    *,
    run_id: UUID,
    source_type: str | None = None,
    target_type: str | None = None,
    memory_types: Mapping[UUID, str] | None = None,
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral create-link execution."""
    return CurationExecutor(action_store).execute_create_link(
        action,
        run_id=run_id,
        source_type=source_type,
        target_type=target_type,
        memory_types=memory_types,
        protections=protections,
    )


def execute_rewrite_memory(
    action_store: CurationActionStore,
    action: RewriteMemoryAction,
    *,
    run_id: UUID,
    memory_type: str,
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral rewrite execution."""
    return CurationExecutor(action_store).execute_rewrite(
        action,
        run_id=run_id,
        memory_type=memory_type,
        protections=protections,
    )


def execute_remove_link(
    action_store: CurationActionStore,
    action: RemoveLinkAction,
    *,
    run_id: UUID,
    source_type: str | None = None,
    target_type: str | None = None,
    memory_types: Mapping[UUID, str] | None = None,
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral remove-link execution."""
    return CurationExecutor(action_store).execute_remove_link(
        action,
        run_id=run_id,
        source_type=source_type,
        target_type=target_type,
        memory_types=memory_types,
        protections=protections,
    )


def execute_merge_memories(
    action_store: CurationActionStore,
    action: MergeMemoriesAction,
    *,
    run_id: UUID,
    memory_types: Mapping[UUID, str] | None = None,
    contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset(),
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral merge execution."""
    return CurationExecutor(action_store).execute_merge(
        action,
        run_id=run_id,
        memory_types=memory_types,
        contradictory_memory_ids=contradictory_memory_ids,
        protections=protections,
    )


def execute_split_memory(
    action_store: CurationActionStore,
    action: SplitMemoryAction,
    *,
    run_id: UUID,
    memory_type: str,
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral split execution."""
    return CurationExecutor(action_store).execute_split(
        action,
        run_id=run_id,
        memory_type=memory_type,
        protections=protections,
    )


def execute_archive_memory(
    action_store: CurationActionStore,
    action: ArchiveMemoryAction,
    *,
    run_id: UUID,
    memory_type: str,
    protections: Iterable[ProtectionMode | str] = (),
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral archive execution."""
    return CurationExecutor(action_store).execute_archive(
        action,
        run_id=run_id,
        memory_type=memory_type,
        protections=protections,
    )


def _record_verification_descriptor(
    action: NormalizeMemoryAction | RewriteMemoryAction,
) -> CurationVerificationDescriptor | None:
    status = action.preconditions.required_statuses.get(action.target_id)
    if status is None:
        return None
    return CurationVerificationDescriptor(
        operation=action.operation,
        target_ids=[action.target_id],
        target_status=cast(VerificationStatus, str(status)),
    )


def _reject_missing_or_conflicting_token(message: str) -> NoReturn:
    raise CurationActionContractError(message, code="invalid_record_token")


def _sorted_unique(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values}, key=lambda value: value.encode("utf-8"))


def _canonical_merge_ids(canonical_id: UUID, source_ids: Sequence[UUID]) -> list[UUID]:
    unique_source_ids: list[UUID] = []
    seen = {canonical_id}
    for source_id in source_ids:
        if source_id in seen:
            continue
        seen.add(source_id)
        unique_source_ids.append(source_id)
    return [canonical_id, *sorted(unique_source_ids, key=lambda value: str(value).encode("utf-8"))]


def _normalized_split_children(children: Sequence[ClaimMapping]) -> list[dict[str, str]]:
    normalized_children: list[dict[str, str]] = []
    for child in children:
        output = child.output.strip()
        if not output:
            raise CurationActionContractError(
                "split_memory requires non-empty child output",
                code="invalid_split_child",
            )
        title = output.splitlines()[0].strip() or output
        normalized_children.append({"title": title, "content": output})
    return normalized_children


_MERGE_LINK_CONTEXT = "Merged into canonical memory by provider-free curation."
_SPLIT_LINK_TYPE = "DEPENDS_ON"
_SPLIT_LINK_CONTEXT = "Derived from a provider-free memory split."


def _require_exact_link_evidence(action: CreateLinkAction, *, link_type: str, context: str) -> None:
    for evidence in action.evidence:
        link = evidence.link
        if link is None:
            continue
        if (
            link.source_id == action.source_id
            and link.target_id == action.target_id
            and link.link_type.strip() == link_type
            and (link.context or "").strip() == context
        ):
            return
    raise CurationActionContractError(
        "create_link requires exact endpoint, type, and context evidence",
        code="missing_link_evidence",
    )


def _require_create_link_preconditions(
    action: CreateLinkAction,
    *,
    link_type: str,
) -> None:
    normalized_type = _normalize_link_type(link_type)
    for assertion in action.preconditions.absent_links:
        if getattr(assertion, "context", None) is not None:
            raise CurationActionContractError(
                "create_link absent-link precondition must omit relationship context",
                code="invalid_absent_link_precondition",
            )
        if (
            assertion.source_id == action.source_id
            and assertion.target_id == action.target_id
            and _normalize_link_type(assertion.link_type) == normalized_type
        ):
            return
    raise CurationActionContractError(
        "create_link requires an exact absent-link precondition",
        code="missing_absent_link_precondition",
    )


def _normalize_link_type(link_type: str) -> str:
    return re.sub(r"[\s-]+", "_", link_type.strip()).upper()


__all__ = [
    "CurationExecutor",
    "CurationPolicyRejection",

    "execute_archive_memory",
    "execute_create_link",
    "execute_merge_memories",
    "execute_normalize_memory",
    "execute_remove_link",
    "execute_rewrite_memory",
    "execute_split_memory",
]

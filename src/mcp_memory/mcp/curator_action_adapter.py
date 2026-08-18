from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_investigation import CURATOR_MUTATION_TOOLS
from mcp_memory.core.curation_quality_inputs import direct_action_id
from mcp_memory.core.curator_evidence import current_curator_execution
from mcp_memory.core.direct_mutation_evidence import DirectMutationEvidence
from mcp_memory.curation_action_store import (
    CurationActionContractError,
    CurationTransaction,
    MutationResult,
)
from mcp_memory.curation_store import CurationActionReceipt

MutationService = Callable[[ApplicationContext, dict[str, Any]], dict[str, Any]]

_OPERATIONS = {
    "internal_archive_memory_record": "archive_memory",
    "internal_create_memory_link": "create_link",
    "internal_delete_memory_link": "remove_link",
    "internal_merge_memory_into_canonical": "merge_memories",
    "internal_split_memory_record": "split_memory",
}


def is_curator_mutation_tool(name: str) -> bool:
    return name in CURATOR_MUTATION_TOOLS


def execute_curator_mutation(
    ctx: ApplicationContext,
    name: str,
    arguments: dict[str, Any],
    service: MutationService,
    evidence: DirectMutationEvidence,
) -> dict[str, Any]:
    execution = current_curator_execution()
    action_store = getattr(ctx, "curation_action_store", None)
    if execution is None or execution.run_id is None or action_store is None:
        return service(ctx, arguments)

    repository = ctx.repository
    if repository is None:
        return service(ctx, arguments)

    target_ids = _target_ids(name, arguments)
    action_id = direct_action_id(evidence.evidence_id)
    operation = _operation(name, arguments)
    receipt_store = getattr(ctx, "curation", None)
    get_receipt = getattr(receipt_store, "get_receipt", None)
    existing = (
        cast(CurationActionReceipt | None, get_receipt(execution.run_id, action_id))
        if callable(get_receipt)
        else None
    )
    if existing is not None:
        if existing.operation != operation:
            raise CurationActionContractError("curation action identity collision", code="action_identity_collision")
        return {"status": "ok", "curation_receipt": existing.model_dump(mode="json")}
    expected_tokens = {
        memory_id: record_token(record)
        for memory_id in target_ids
        if not memory_id.startswith("ext:")
        for record in [repository.get_memory(memory_id)]
        if record is not None
    }
    captured: dict[str, Any] = {}

    def apply(transaction: CurationTransaction) -> MutationResult:
        transaction_ctx = replace(
            ctx,
            repository=_TransactionRepository(transaction),
            task_queue=None,
        )
        payload = service(transaction_ctx, arguments)
        if payload.get("status") == "error":
            error = str(payload.get("error", "mutation_rejected"))
            detail = payload.get("detail")
            raise CurationActionContractError(
                str(detail) if detail is not None else error,
                code=error,
            )
        captured["payload"] = payload
        return MutationResult(operation, _affected_ids(payload, target_ids))

    receipt = action_store.execute_action(
        run_id=execution.run_id,
        action_id=action_id,
        target_ids=target_ids,
        expected_tokens=expected_tokens,
        apply=apply,
        operation=operation,
        payload=arguments,
        idempotency_key=evidence.idempotency_key,
    )
    receipt_payload = {"curation_receipt": receipt.model_dump(mode="json")}
    if "payload" in captured:
        return {**captured["payload"], **receipt_payload}
    return {"status": "ok", **receipt_payload}


def _operation(name: str, arguments: dict[str, Any]) -> str:
    if name == "internal_update_memory_record":
        return "rewrite_memory" if "content" in arguments else "normalize_memory"
    return _OPERATIONS[name]


def _target_ids(name: str, arguments: dict[str, Any]) -> list[str]:
    keys = {
        "internal_archive_memory_record": ("memory_id",),
        "internal_create_memory_link": ("source_id", "target_id"),
        "internal_delete_memory_link": ("source_id", "target_id"),
        "internal_merge_memory_into_canonical": ("canonical_memory_id", "source_memory_id"),
        "internal_split_memory_record": ("memory_id",),
        "internal_update_memory_record": ("memory_id",),
    }[name]
    return list(dict.fromkeys(str(arguments[key]) for key in keys if isinstance(arguments.get(key), str)))


def _affected_ids(payload: dict[str, Any], target_ids: list[str]) -> list[str]:
    values = list(target_ids)

    def visit(value: object) -> None:
        if isinstance(value, dict):
            memory_id = value.get("id")
            if isinstance(memory_id, str):
                values.append(memory_id)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return list(dict.fromkeys(value for value in values if not value.startswith("ext:")))


class _TransactionRepository:
    def __init__(self, transaction: CurationTransaction) -> None:
        self._transaction = transaction

    def get_memory(self, memory_id: str):
        return self._transaction.get_memory(memory_id)

    def update_memory(self, memory_id: str, **changes: object):
        changes = {key: value for key, value in changes.items() if value is not None}
        return self._transaction.update_memory(memory_id, **changes)

    def create_memory(self, **values: object):
        return self._transaction.create_memory(**values)

    def delete_memory(self, memory_id: str):
        return self._transaction.delete_memory(memory_id)

    def add_link(self, source_id: str, target_id: str, link_type: str, context: str = ""):
        return self._transaction.add_link(source_id, target_id, link_type, context)

    def remove_link(self, source_id: str, target_id: str, link_type: str):
        return self._transaction.remove_link(source_id, target_id, link_type)

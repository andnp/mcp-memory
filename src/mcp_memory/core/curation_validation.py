"""Direct-curator mutation budgets and specialist-routing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mcp_memory.core.curation_models import CurationAction
from mcp_memory.core.curation_routing import MaintenanceFamily


@dataclass(frozen=True, slots=True)
class CurationMutationBudget:
    """Independent limits for direct-curator mutation calls."""

    max_proposed_actions: int = 256
    max_accepted_mutations: int = 256

    def __post_init__(self) -> None:
        for name in ("max_proposed_actions", "max_accepted_mutations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class CurationValidationReasonCode(StrEnum):
    NEEDS_DIFFERENT_SPECIALIST = "needs_different_specialist"


@dataclass(frozen=True, slots=True)
class CurationSpecialistRoute:
    action: CurationAction
    family: MaintenanceFamily
    reason_code: CurationValidationReasonCode = CurationValidationReasonCode.NEEDS_DIFFERENT_SPECIALIST

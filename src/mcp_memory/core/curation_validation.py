"""Direct-curator specialist-routing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mcp_memory.core.curation_models import CurationAction
from mcp_memory.core.curation_routing import MaintenanceFamily


class CurationValidationReasonCode(StrEnum):
    NEEDS_DIFFERENT_SPECIALIST = "needs_different_specialist"


@dataclass(frozen=True, slots=True)
class CurationSpecialistRoute:
    action: CurationAction
    family: MaintenanceFamily
    reason_code: CurationValidationReasonCode = CurationValidationReasonCode.NEEDS_DIFFERENT_SPECIALIST

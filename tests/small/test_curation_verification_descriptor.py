from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from mcp_memory.core.curation_models import CurationVerificationDescriptor

pytestmark = pytest.mark.small


def test_descriptor_rejects_operation_specific_shape_mismatch() -> None:
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor(
            operation="create_link",
            target_ids=[uuid4()],
            source_id=uuid4(),
            target_id=uuid4(),
            link_type="SUPPORTS",
            context="exact",
            exists=True,
        )
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor(
            operation="archive_memory",
            target_ids=[uuid4()],
            target_status="archived",
            source_id=uuid4(),
        )


def test_descriptor_rejects_unknown_fields_and_invalid_split_count() -> None:
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor.model_validate(
            {
                "operation": "split_memory",
                "target_ids": [str(uuid4())],
                "target_id": str(uuid4()),
                "child_ids": [str(uuid4())],
                "split_group_id": str(uuid4()),
                "child_count": 2,
                "provider": "forbidden",
            }
        )


def test_descriptor_bounds_and_duplicate_ids() -> None:
    target_id = uuid4()
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor(
            operation="archive_memory",
            target_ids=[target_id, target_id],
            target_status="archived",
        )
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor(
            operation="create_link",
            target_ids=[uuid4(), uuid4()],
            source_id=uuid4(),
            target_id=uuid4(),
            link_type="SUPPORTS",
            context="x" * 513,
            exists=True,
        )
    with pytest.raises(ValidationError):
        CurationVerificationDescriptor(
            operation="split_memory",
            target_ids=[target_id],
            target_id=target_id,
            child_ids=[uuid4()] * 65,
            split_group_id="group",
            child_count=65,
        )

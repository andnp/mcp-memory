"""Contract tests for the provider-neutral memory pipeline boundary."""

import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.context import MemoryReadCapabilities
from mcp_memory.core.pipeline import MemoryPipeline
from mcp_memory.core.ports.memory import MemoryQueriesPort
from mcp_memory.core.retrieval import MemoryRetrievalPort

pytestmark = pytest.mark.small


def test_pipeline_preserves_injected_query_and_retrieval_capabilities() -> None:
    """Injected provider capabilities remain identity-preserving pipeline fields."""
    queries = object()
    retrieval = object()

    pipeline = MemoryPipeline.from_context(
        MemoryReadCapabilities(repository=object()),
        SimpleNamespace(has_runtime=True, client_count=2),
        memory_queries=cast(MemoryQueriesPort, queries),
        retrieval=cast(MemoryRetrievalPort, retrieval),
    )

    assert pipeline.memory_queries is queries
    assert pipeline.retrieval is retrieval


def test_pipeline_import_does_not_load_provider_adapters() -> None:
    """Importing core pipeline code does not load relational integration adapters."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys

import mcp_memory.core.pipeline

assert "mcp_memory.integrations.memory_retrieval" not in sys.modules
assert "mcp_memory.relational.queries" not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""

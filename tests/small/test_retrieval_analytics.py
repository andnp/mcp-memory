from __future__ import annotations

import time
from pathlib import Path

import pytest

from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.mcp.internal_services import internal_search_memory_records_service
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.provider_usage_store import ProviderUsageRepository


pytestmark = pytest.mark.small


def test_build_nerd_metrics_includes_retrieval_analytics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None

        alpha = runtime.repository.create_memory(
            title="Alpha retrieval memory",
            content="alpha phrase for retrieval analytics",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
            tags=["alpha-tag", "common-tag"],
        )
        beta = runtime.repository.create_memory(
            title="Beta retrieval memory",
            content="beta phrase for retrieval analytics",
            workspace_ids=[runtime.workspace_id],
            memory_type="plan",
            status="stale",
            tags=["beta-tag", "common-tag"],
        )
        assert alpha is not None
        assert beta is not None

        alpha_search = search_memory_records_service(runtime, {"query": "alpha phrase", "limit": 5, "status": "active"})
        beta_search = search_memory_records_service(runtime, {"query": "beta phrase", "limit": 5, "status": "stale"})
        missing_search = search_memory_records_service(runtime, {"query": "alpha phrase", "limit": 5, "status": "archived"})
        internal_beta_search = internal_search_memory_records_service(runtime, {"query": "beta phrase", "limit": 5, "status": "stale"})
        alpha_read = read_memory_record_service(runtime, {"memory_id": alpha.id})
        second_alpha_read = read_memory_record_service(runtime, {"memory_id": alpha.id})
        beta_read = read_memory_record_service(runtime, {"memory_id": beta.id})

        assert alpha_search["status"] == "ok"
        assert beta_search["status"] == "ok"
        assert missing_search["status"] == "ok"
        assert internal_beta_search["status"] == "ok"
        assert alpha_read["status"] == "ok"
        assert second_alpha_read["status"] == "ok"
        assert beta_read["status"] == "ok"

        payload = build_nerd_metrics(
            db_manager=runtime.db_manager,
            workspace_id=runtime.workspace_id,
            task_queue=runtime.task_queue,
            provider_usage_repo=ProviderUsageRepository(runtime.db_manager, workspace_id=runtime.workspace_id),
            config=runtime.config,
            ai_json_provider=runtime.ai_json_provider,
            ai_agent_provider=runtime.ai_agent_provider,
            ai_provider_registry=runtime.ai_provider_registry,
            relational_search=runtime.relational_search,
            window_hours=24,
            bucket_minutes=60,
            now=time.time() + 1.0,
        )

        assert payload.retrieval.summary.search_invocations == 4
        assert payload.retrieval.summary.search_hits == 3
        assert payload.retrieval.summary.zero_result_searches == 1
        assert payload.retrieval.summary.read_events == 3
        assert payload.retrieval.summary.unique_search_memories == 2
        assert payload.retrieval.summary.unique_read_memories == 2

        assert [row.memory_id for row in payload.retrieval.top_read_memories[:2]] == [alpha.id, beta.id]
        assert [row.read_count for row in payload.retrieval.top_read_memories[:2]] == [2, 1]
        assert [row.memory_id for row in payload.retrieval.top_search_memories[:2]] == [beta.id, alpha.id]
        assert [row.search_count for row in payload.retrieval.top_search_memories[:2]] == [2, 1]

        top_tag_keys = {row.key for row in payload.retrieval.top_tags}
        assert {"alpha-tag", "beta-tag", "common-tag"}.issubset(top_tag_keys)
        common_tag = next(row for row in payload.retrieval.top_tags if row.key == "common-tag")
        assert common_tag.read_count == 3
        assert common_tag.search_count == 3
        assert common_tag.total_count == 6

        timeline_keys = {series.key for series in payload.retrieval.tag_timelines}
        assert {"alpha-tag", "beta-tag", "common-tag"}.issubset(timeline_keys)
        common_timeline = next(series for series in payload.retrieval.tag_timelines if series.key == "common-tag")
        assert sum(bucket.count for bucket in common_timeline.read_buckets) == 3
        assert sum(bucket.count for bucket in common_timeline.search_buckets) == 3
    finally:
        runtime.close()
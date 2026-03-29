from mcp_memory.management.agent_run_reporting import build_agent_run_history_payload, decode_run_result


def test_decode_run_result_accepts_postgres_jsonb_mapping() -> None:
    raw_result = {
        "claimed_entry_ids": [101, 102],
        "processed_entry_ids": [101],
        "released_entry_ids": [102],
        "meaningful_actions": 1,
        "mutations": 1,
        "entry_dispositions": [
            {
                "entry_id": 101,
                "disposition": "created",
                "finalization_status": "recoverable",
                "memory_id": "memory-101",
            }
        ],
    }

    result = decode_run_result(raw_result)
    payload = build_agent_run_history_payload(
        task_id="task-1",
        task_name="ingest-system1",
        status="completed",
        started_at=10.0,
        completed_at=12.0,
        duration_seconds=2.0,
        error_text=None,
        result=result,
        detail_level="full",
    )

    assert payload.result == raw_result
    assert payload.ingest_audit.claimed_count == 2
    assert payload.ingest_audit.handled_count == 1
    assert payload.ingest_audit.released_count == 1
    assert payload.ingest_audit.meaningful_actions == 1
    assert payload.ingest_audit.mutations == 1
    assert payload.ingest_audit.entry_dispositions[0].memory_id == "memory-101"

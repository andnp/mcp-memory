#!/usr/bin/env bash
# Pin a corpus snapshot for replay evaluation.
#
# Replay compares ranker variants against each other, which is only valid when
# every variant sees identical data. Production changes under a multi-hour run,
# so the corpus is copied once into a local store and every variant reads that.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

COMPOSE_FILE=compose.replay-snapshot.yml
ARTIFACTS=benchmarks/replay_eval/artifacts
DUMP="${ARTIFACTS}/corpus-snapshot.dump"
FINGERPRINT="${ARTIFACTS}/corpus-snapshot.json"

REPLAY_PORT="${MCP_MEMORY_REPLAY_PORT:-5455}"
REPLAY_USER="${MCP_MEMORY_REPLAY_USER:-replay}"
REPLAY_PASSWORD="${MCP_MEMORY_REPLAY_PASSWORD:-replay}"
REPLAY_DB="${MCP_MEMORY_REPLAY_DB:-mcp_memory}"
SNAPSHOT_DSN="postgresql://${REPLAY_USER}:${REPLAY_PASSWORD}@127.0.0.1:${REPLAY_PORT}/${REPLAY_DB}"

# The source DSN carries a credential, so it is read from the operator's own
# configuration at run time and never written into the repository.
SOURCE_DSN="$(uv run python -c 'from mcp_memory.config import load_config; print(load_config().storage.postgres.dsn)')"
if [[ -z "${SOURCE_DSN}" ]]; then
  echo "no postgres dsn configured; a replay snapshot needs the live store" >&2
  exit 1
fi

mkdir -p "${ARTIFACTS}"

# Tables excluded below are not read by search: conversations and task history
# are unrelated, and telemetry is already extracted to labels beforehand.
# Their schemas are still dumped so the application sees a complete database.
echo "==> dumping corpus from the live store"
pg_dump \
  --format=custom \
  --no-owner \
  --no-privileges \
  --exclude-table-data=ai_conversations \
  --exclude-table-data=memory_tool_events \
  --exclude-table-data=task_runs \
  --file="${DUMP}" \
  "${SOURCE_DSN}"

echo "==> starting the snapshot store"
docker compose -f "${COMPOSE_FILE}" up -d --wait

echo "==> restoring into the snapshot store"
pg_restore \
  --dbname="${SNAPSHOT_DSN}" \
  --no-owner \
  --no-privileges \
  --clean \
  --if-exists \
  --exit-on-error \
  "${DUMP}"

echo "==> recording the snapshot fingerprint"
psql --dbname="${SNAPSHOT_DSN}" --quiet --tuples-only --no-align --command "
  SELECT json_build_object(
    'captured_at', now(),
    'source_server_version', current_setting('server_version'),
    'memories', (SELECT count(*) FROM memories),
    'embeddings', (SELECT count(*) FROM embeddings),
    'search_documents', (SELECT count(*) FROM memory_search_documents)
  );" > "${FINGERPRINT}"

echo "snapshot ready at ${SNAPSHOT_DSN}"
cat "${FINGERPRINT}"

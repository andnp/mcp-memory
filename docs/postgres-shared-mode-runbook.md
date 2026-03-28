# Postgres Shared-Mode Runbook

This runbook is the operator path for using `mcp-memory` with Postgres as the authoritative backend.

Use this mode when you want one shared memory store across machines.
Do **not** use Syncthing or similar live file-sync tools against an active SQLite store for that job.

## 1. Bring up Postgres locally with Docker Compose

From the repo root:

```bash
docker compose -f compose.postgres.yml up -d
```

Check readiness:

```bash
docker compose -f compose.postgres.yml ps
docker compose -f compose.postgres.yml logs postgres --tail=50
```

Default container settings from `compose.postgres.yml`:

- database: `mcp_memory`
- user: `mcp_memory`
- password: `change-me`
- host port: `5432`

Override them with environment variables if needed:

- `MCP_MEMORY_POSTGRES_DB`
- `MCP_MEMORY_POSTGRES_USER`
- `MCP_MEMORY_POSTGRES_PASSWORD`
- `MCP_MEMORY_POSTGRES_PORT`

## 2. Point `mcp-memory` at Postgres

Edit `~/.config/mcp-memory/config.toml`:

```toml
[storage]
backend = "postgres"

[storage.postgres]
dsn = "postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory"
pool_min = 1
pool_max = 10
statement_timeout_ms = 30000
lock_timeout_ms = 5000
application_name = "mcp-memory"
```

Important invariants:

- when `storage.backend = "postgres"`, Postgres is authoritative
- the runtime must not silently fall back to SQLite
- local filesystem state may still exist for sockets, logs, embeddings, and runtime support, but it is not the source of truth

## 3. Migrate an existing SQLite corpus

Dry run first:

```bash
uv run mcp-memory migrate-sqlite-to-postgres --dry-run --postgres-dsn 'postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory'
```

Then perform the import:

```bash
uv run mcp-memory migrate-sqlite-to-postgres --postgres-dsn 'postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory'
```

Current migration scope covers the core relational memory graph:

- memories
- workspace mappings
- tags
- memory/tag mappings
- links

It does **not** currently migrate every operational table or every piece of telemetry.
That is intentional for the first dogfooding slice.

## 4. Boot the runtime and smoke it

Start or restart the daemon:

```bash
uv run mcp-memory daemon-restart
```

Check health:

```bash
uv run mcp-memory health --json
uv run mcp-memory stats
uv run mcp-memory search health
```

Recommended smoke flow:

1. confirm `uv run mcp-memory health --json` reports `storage_backend` as `postgres`
2. record a new thought through your MCP client or CLI
3. search for that memory
4. read the memory back
5. confirm the dashboard/health surface still looks sane

Useful direct commands:

```bash
uv run mcp-memory stash "postgres shared-mode smoke test"
uv run mcp-memory health --json
uv run mcp-memory stats
```

## 5. Backups and restore

In shared mode, Postgres owns durability.
SQLite snapshot guidance is no longer the primary backup story.

A simple logical backup:

```bash
docker compose -f compose.postgres.yml exec -T postgres pg_dump -U mcp_memory -d mcp_memory > mcp-memory-postgres.sql
```

A simple restore into a fresh database:

```bash
cat mcp-memory-postgres.sql | docker compose -f compose.postgres.yml exec -T postgres psql -U mcp_memory -d mcp_memory
```

For a remote server, prefer provider-native backups or scheduled `pg_dump` over pretending the local filesystem cache is your backup plan.

## 6. Failure guidance

### Postgres unavailable at startup

Expected behavior: startup should fail clearly.
It must not degrade into implicit SQLite.

Check:

```bash
docker compose -f compose.postgres.yml logs postgres --tail=100
uv run mcp-memory health --json
```

### Authentication or DSN mistakes

Check the configured DSN carefully:

- username
- password
- host
- port
- database name
- `postgresql://` scheme

### Remote server deployment

For remote Postgres, keep the database private if possible.
Prefer one of:

- private network/VPN
- SSH tunnel
- firewall rules limited to your client hosts

Exposing `5432` naked to the public internet is a great way to meet chaos sooner than planned.

## 7. Shut down the local dev database

```bash
docker compose -f compose.postgres.yml down
```

Remove the volume too if you want a clean reset:

```bash
docker compose -f compose.postgres.yml down -v
```
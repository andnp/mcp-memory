# Specification: Testing Strategy

## 1. Overview
The `mcp-memory` testing suite follows the "Small/Medium/Large" organization pattern. We strictly avoid open-ended, fragile mocks (e.g., `unittest.mock.MagicMock` on internal classes). Instead, because SQLite is incredibly fast and local, we use a real, temporary database for almost all tests.

## 2. Test Categories

### 2.1 Small Tests (`< 10ms`)
- **Location**: `tests/small/`
- **Scope**: Isolated functions, pure algorithms, and fast SQLite interactions.
- **Rules**:
  - NO network calls.
  - NO AI model loading or CLI subprocess execution.
  - DO use a real, in-memory or `tempfile` SQLite database (via a `pytest` fixture).
- **Examples**:
  - Testing the `access_score` decay math.
  - Testing the `Strategy Roulette` selection logic (mocking the `random` seed).
  - Testing deterministic agents (e.g., `Project Manager` executing its SQL update).

### 2.2 Medium Tests (`< 1s`)
- **Location**: `tests/medium/`
- **Scope**: Integration between multiple components, but still within a single process.
- **Rules**:
  - NO external network calls (unless hitting a local mock server).
  - FAISS index interactions are allowed (using small, deterministic dummy embeddings).
  - AI Providers MUST be stubbed out with "Fake Providers" that return deterministic JSON or strings, rather than executing real models.
- **Examples**:
  - Testing the `search_memories` pipeline (RRF fusion + recency boosting) using a seeded database and FAISS index.
  - Testing the `Ingestor` workflow (using a Fake AI Provider that returns a hardcoded JSON payload) to ensure it correctly updates the DB and FAISS.

### 2.3 Large Tests (`> 1s`)
- **Location**: `tests/large/`
- **Scope**: Full End-to-End (E2E) system tests involving the daemon, ZeroMQ IPC, and the MCP proxies.
- **Rules**:
  - Permitted to spin up the actual background daemon process.
  - Permitted to use real CLI tools (e.g., `ollama` or `copilot-cli`) if a specific "integration-enabled" flag is passed, though default runs should still use a loopback mock.
- **Examples**:
  - Boot race condition tests (spawning 3 proxies simultaneously to ensure the `.lock` file holds and only one daemon starts).
  - Full MCP tool execution (Client -> Proxy -> ZMQ -> Daemon -> SQLite -> Daemon -> ZMQ -> Proxy -> Client).

## 3. The "Dummy Data" Strategy
To avoid brittle mocks, we will rely on a robust set of deterministic fixtures and "Fakes".

### 3.1 The Seeded Database Fixture
We will maintain a `tests/fixtures/seed_data.json` file containing ~50 realistic memories, thoughts, and links (covering edge cases like orphaned memories, stale plans, and contradictions). 
A `pytest` fixture (`seeded_db`) will quickly load this data into a temporary SQLite DB before a test runs, providing a rich, realistic graph to query against without mocking the ORM/SQL layer.

### 3.2 The Fake AI Provider
Instead of mocking `provider.generate_json()`, we implement a concrete `FakeAIAssistantProvider` class that implements the `AIAssistantProvider` protocol.
- It can be initialized with canned responses: `FakeAIAssistantProvider(responses=[{"action": "merge"}])`.
- It acts exactly like the real provider but returns instantly, allowing us to test the *logic* of the background agents (how they parse the JSON and update the DB) without testing the LLM itself.

### 3.3 Dummy Embeddings
Loading `sentence-transformers` takes seconds and requires downloading weights. For tests, we use a `DummyEmbeddingProvider` that returns random (but seeded) or hardcoded float arrays of the correct dimension, ensuring FAISS logic can be tested in `< 10ms`.

# AI Chatbot Backend

A conversational API that collects structured service desk requests through natural language, with semantic duplicate detection.

## Docs

- [Local setup](docs/local-setup.md) — get running in minutes
- [Architecture](docs/architecture.md) — how the system works
- [Project structure](docs/project-structure.md) — what lives where
- [Design decisions](docs/decisions.md) — why LangGraph, why pgvector, how duplicate detection works
- [Deployment](docs/deployment.md) — production deployment and CI/CD
- [Evaluation Plan](docs/evaluation-plan.md) — future evaluation strategy for agent reliability and duplicate detection quality
- [Observability Plan](docs/observability-plan.md) — metrics (Prometheus + Grafana), logs (Fluent Bit + Elasticsearch + Kibana), and tracing (LangSmith)

## Quick start

**Option A — full Docker stack (no local Python required):**

```bash
./scripts/setup-local-env.sh
docker compose up -d
```

**Option B — local API with hot reload (for development):**

```bash
uv sync
./scripts/setup-local-env.sh
docker compose up -d postgres
uv run uvicorn src.api.main:app --reload
```

Then open `http://localhost:8000/docs`.

### Try it interactively

Once the API is running (either option), use the interactive test REPL to chat with the bot from your terminal:

```bash
./scripts/interactive_test.sh
```

Commands inside the REPL:

| Input | Action |
|---|---|
| Any text | Send a message to the bot |
| `health` | Check API health |
| `state` | Show current session state |
| `quit` / `exit` | End the session |

The session ID is auto-generated on each run. Override the defaults with env vars if needed:

```bash
API_URL=http://localhost:8000 API_KEY=your-key ./scripts/interactive_test.sh
```

## Tests

```bash
uv run pytest tests/unit/ -v
```

# AI Chatbot Backend

A conversational API that collects structured service desk requests through natural language, with semantic duplicate detection.

## Docs

- [Local setup](docs/local-setup.md) — get running in minutes
- [Architecture](docs/architecture.md) — how the system works
- [Project structure](docs/project-structure.md) — what lives where
- [Design decisions](docs/decisions.md) — why LangGraph, why pgvector, how duplicate detection works
- [Deployment](docs/deployment.md) — production deployment and CI/CD

## Quick start

```bash
uv sync
./scripts/setup-local-env.sh
docker compose up -d postgres redis
uv run uvicorn src.api.main:app --reload
```

Then open `http://localhost:8000/docs`.

## Tests

```bash
uv run pytest tests/unit/ -v
```

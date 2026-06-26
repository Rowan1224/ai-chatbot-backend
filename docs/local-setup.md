# Local Setup

## Prerequisites

- [uv](https://docs.astral.sh/uv/) — Python package manager
- Docker or Podman
- An API key for OpenAI, Azure OpenAI, or Anthropic

## Steps

**1. Install dependencies**

```bash
uv sync
```

**2. Create your `.env`**

```bash
chmod +x scripts/setup-local-env.sh
./scripts/setup-local-env.sh
```

This walks you through provider selection, API keys, and generates a secure `API_KEY`. It writes `.env` from `.env.example`.

**3. Start the full stack**

```bash
docker compose up -d
```

This starts Postgres, Redis, and the API together. API is at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

> **Iterating on code?** Skip the API container and run it locally with hot reload instead:
> ```bash
> docker compose up -d postgres redis
> uv run uvicorn src.api.main:app --reload
> ```
> This way code changes take effect immediately without rebuilding the image.

## Try it

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-1", "message": "I need infrastructure provisioning"}'
```

Or use the interactive REPL:

```bash
./scripts/interactive_test.sh
```

## Run tests

```bash
# Unit only (fast, no infra)
uv run pytest tests/unit/ -v

# With coverage
./scripts/run_tests.sh

# Integration (spins up Postgres + Redis automatically)
uv run pytest tests/integration/ -v -m integration
```

## Change behaviour without code

- Edit `src/config/prompt_config.yaml` to change what the bot asks for or how it responds.
- Edit `src/config/app_config.yaml` to change model, thresholds, or feature flags.
- Restart the API — no code changes needed.

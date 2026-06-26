# Deployment Guide

## Project structure

```
docker-compose.yml          # Local development — full stack in containers
docker-compose.prod.yml     # Production — API only, external DB/Redis, no .env file
scripts/setup-local-env.sh  # One-time local .env generator
src/config/app_config.yaml  # Behaviour config (models, thresholds, feature flags)
src/config/prompt_config.yaml # System prompt and version
.env.example                # Template — documents all required env vars
```

**Config split:**
- **`.env`** (local) / **pipeline secrets** (production) — secrets and infrastructure URLs only
- **`app_config.yaml`** — model selection, duplicate detection tuning, privacy, logging. Safe to commit.
- **`prompt_config.yaml`** — system prompt and config version. Safe to commit.

---

## Local Development

### First-time setup

```bash
# 1. Generate .env interactively
chmod +x scripts/setup-local-env.sh
./scripts/setup-local-env.sh

# 2. Start Postgres + Redis
docker compose up -d postgres redis

# 3. Start the API (with hot reload)
uv run uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Swagger docs at `http://localhost:8000/docs`.

### Run the full local stack (API in container)

```bash
docker compose up -d
```

### Run tests

```bash
# Unit tests (no infrastructure required)
uv run pytest tests/unit/ -v

# Integration tests (spins up Postgres + Redis via testcontainers)
uv run pytest tests/integration/ -v -m integration

# Full build + all tests + Docker image
./deploy/build.sh
```

### Generate a secure API key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Production Deployment

### How it works

`docker-compose.prod.yml` contains no `env_file:`. All secrets are injected
by the CI/CD pipeline as shell environment variables before `docker compose up`
runs. The app reads them the same way it reads `.env` locally — `pydantic-settings`
checks real environment variables first, then falls back to `.env` if present.

There is no `.env` file on the production server.

### Required pipeline secrets

Set these in your secrets store (GitHub Actions, Azure Key Vault, AWS Secrets
Manager, etc.) and export them before the deploy step:

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai`, `azure`, or `anthropic` |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |
| `AZURE_OPENAI_API_KEY` | Required when `LLM_PROVIDER=azure` |
| `AZURE_OPENAI_ENDPOINT` | Required when `LLM_PROVIDER=azure` |
| `ANTHROPIC_API_KEY` | Required when `LLM_PROVIDER=anthropic` |
| `POSTGRESQL_URL` | Full connection string including credentials |
| `REDIS_URL` | Redis connection URL |
| `API_KEY` | Secret key for `X-API-Key` header |
| `IMAGE_NAME` | Container image name (default: `ai-chatbot-api`) |
| `IMAGE_TAG` | Image tag to deploy (default: `latest`) |

### Deploy

```bash
# Build and push image (from CI/CD)
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
docker push ${IMAGE_NAME}:${IMAGE_TAG}

# Deploy (all ${VAR} are already exported by the pipeline)
docker compose -f docker-compose.prod.yml up -d
```

### Example: GitHub Actions

```yaml
- name: Deploy
  env:
    LLM_PROVIDER: ${{ vars.LLM_PROVIDER }}
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    POSTGRESQL_URL: ${{ secrets.POSTGRESQL_URL }}
    REDIS_URL: ${{ secrets.REDIS_URL }}
    API_KEY: ${{ secrets.API_KEY }}
    IMAGE_NAME: my-registry/ai-chatbot-api
    IMAGE_TAG: ${{ github.sha }}
  run: docker compose -f docker-compose.prod.yml up -d
```

### Example: Azure Pipelines

```yaml
- task: Bash@3
  env:
    LLM_PROVIDER: $(LLM_PROVIDER)
    OPENAI_API_KEY: $(OPENAI_API_KEY)
    POSTGRESQL_URL: $(POSTGRESQL_URL)
    REDIS_URL: $(REDIS_URL)
    API_KEY: $(API_KEY)
    IMAGE_NAME: $(IMAGE_NAME)
    IMAGE_TAG: $(Build.BuildId)
  inputs:
    script: docker compose -f docker-compose.prod.yml up -d
```

---

## Updating behaviour config without a redeploy

`app_config.yaml` and `prompt_config.yaml` are committed to git and baked into
the image at build time. To change model selection, thresholds, or the system
prompt:

1. Edit the relevant YAML file
2. Rebuild and redeploy the image

No secrets need to be rotated and no environment variables change.

---

## Health check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "mongodb": "connected", "redis": "connected"}
```

---

## Troubleshooting

### API won't start — missing required field

`pydantic-settings` raises a `ValidationError` on startup if a required variable
is absent. Check the container logs:

```bash
docker compose logs api
```

The error message names the missing field exactly.

### Test the API key

```bash
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-1", "message": "hello"}'
```

A `401` means the key doesn't match `API_KEY`. A `422` means the request body
is malformed.

---

**Made with Bob**

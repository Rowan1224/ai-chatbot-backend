# Project Structure

```
ai-chatbot-backend/
│
├── src/
│   ├── api/
│   │   ├── main.py          # FastAPI app, endpoints, lifespan
│   │   └── models.py        # Request/response Pydantic models
│   │
│   ├── config/
│   │   ├── settings.py      # Env var loading (secrets + infra)
│   │   ├── app_config.yaml  # Behaviour config — models, thresholds, feature flags
│   │   └── prompt_config.yaml # System prompt and config version
│   │
│   └── core/
│       ├── workflow.py      # LangGraph graph definition and nodes
│       ├── guardrails.py    # Input guardrail node — PII redaction + injection detection
│       ├── models.py        # Pydantic state models (ChatResponse, ConversationState, …)
│       ├── prompts.py       # Internal LLM prompt constants
│       ├── schema.py        # Extraction schema (ExtractedRequest)
│       ├── database.py      # PostgreSQL client and query functions
│       └── llm.py           # LLM and embedding model factory
│
├── tests/
│   ├── unit/                # Fast, no infrastructure — mocked dependencies
│   ├── integration/         # Postgres via testcontainers
│   └── e2e/                 # Full stack against the built Docker image
│
├── deploy/
│   ├── build.sh             # CI/CD pipeline — test, build, tag
│   └── docker-compose.prod.yml  # Production compose — no .env, vars from pipeline
│
├── scripts/
│   ├── setup-local-env.sh   # One-time local .env generator
│   ├── run_tests.sh         # Unit tests with coverage report
│   └── interactive_test.sh  # Manual API test REPL
│
├── docs/                    # You are here
│
├── docker-compose.yml       # Local dev stack (Postgres + API)
├── Dockerfile               # Multi-stage image build
├── .env.example             # Documents all required environment variables
└── pyproject.toml           # Dependencies and project metadata
```

## Config split

| File | What lives there | Safe to commit? |
|---|---|---|
| `.env` | API keys, DB URLs, infra credentials | **No** |
| `app_config.yaml` | Model names, thresholds, feature flags | Yes |
| `prompt_config.yaml` | System prompt, config version | Yes |

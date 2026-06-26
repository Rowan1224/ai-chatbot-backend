# Architecture

## What it does

A conversational API that collects structured service desk requests through natural language. The user chats with the bot, the bot extracts structured data when it has enough information, checks for duplicates, and saves the request.

## Request flow

```
User message
     │
     ▼
┌─────────────┐
│  chat node  │  Conversation — collects all required fields
└──────┬──────┘
       │ is_ready = true
       ▼
┌─────────────┐
│ extract node│  Structures the conversation into typed fields
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ duplicate_check  │  Fuzzy → vector → LLM judge (see decisions.md)
└──────┬───────────┘
       │ no duplicates          │ duplicates found
       ▼                        ▼
┌─────────────┐         ┌──────────────────────┐
│  save node  │         │ await user decision   │
└─────────────┘         │ modify / proceed /    │
                        │ cancel                │
                        └──────────────────────┘
```

Each node returns a state update. LangGraph merges updates and decides the next node based on conditional edges.

## Session persistence

Every conversation is a LangGraph thread identified by `session_id`. The full message history and state are checkpointed to Redis after each turn. If the server restarts, a returning user's session is resumed from exactly where it left off.

In local dev (`USE_REDIS_CHECKPOINTER=false`) an in-memory checkpointer is used instead — sessions are lost on restart.

## Data storage

PostgreSQL is the only database. It stores:

- `requests` table — the structured data extracted from each conversation, including a vector embedding of the non-PII fields
- LangGraph checkpoint tables — conversation state managed by the Redis checkpointer

PII fields (`name`, `employee_id`) are stored in the request record but are **never** included in the vector embedding. They are listed in `app_config.yaml` under `privacy.pii_fields`.

## Config layers

```
.env                 ← secrets (API keys, DB URLs) — per environment
app_config.yaml      ← behaviour (models, thresholds) — same across environments
prompt_config.yaml   ← domain (system prompt) — changes the bot's persona/purpose
```

Changing `prompt_config.yaml` to describe a different domain (e.g. HR requests instead of IT requests) requires no code changes and no database migration. Historical records are preserved as-is.

## API

| Endpoint | Purpose |
|---|---|
| `POST /chat` | Send a message, get a response |
| `GET /session/{id}` | Inspect full session state (debug) |
| `GET /health` | Liveness check |

Authentication: `X-API-Key` header on all endpoints except `/health`.

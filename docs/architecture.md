# Architecture

## What it does

A conversational API that collects structured service desk requests through natural language. The user chats with the bot, the bot extracts structured data when it has enough information, checks for duplicates, and saves the request.

## Request flow

```
User message
     │
     ▼
┌──────────────┐
│ guardrail    │  PII redaction + prompt-injection detection
└──────┬───────┘
       │ cleared                │ injection blocked
       ▼                        ▼ (refusal message, turn ends)
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

## Input guardrails

Every user message passes through the `guardrail` node before any other node runs. Two independent checks are applied (each togglable in `app_config.yaml`):

**PII redaction** — emails are replaced with `[REDACTED_EMAIL]` and credit-card numbers are masked to `****-****-****-XXXX`. The redacted message is written back into state so the LLM never sees raw PII values. Uses LangChain's `PIIMiddleware` detector functions directly (the middleware hook points have no attachment surface on a hand-built `StateGraph`).

**Prompt-injection detection** — common jailbreak and instruction-override patterns (e.g. "ignore previous instructions", "you are now DAN") are matched with compiled regexes. A detected message is blocked immediately: the turn ends with a refusal reply and `is_ready` is set to `False`; nothing reaches the LLM.

> **LLM provider as a second line of defence.** OpenAI, Azure OpenAI, and Anthropic all enforce their own content policies server-side. Attempts that slip past the regex layer — novel jailbreak phrasing, obfuscated injections — will typically be caught and rejected by the provider (the `chat_node` handles `BadRequestError` from the provider for exactly this case). Because the provider enforces content policy unconditionally, the regex layer only needs to cover the most common, well-known patterns; exhaustive coverage is not required.

## Session persistence

Every conversation is a LangGraph thread identified by `session_id`. The full message history and state are checkpointed to PostgreSQL (via `AsyncPostgresSaver`) after each turn. If the server restarts, a returning user's session is resumed from exactly where it left off.

In local dev (`USE_POSTGRES_CHECKPOINTER=false`) an in-memory checkpointer is used instead — sessions are lost on restart, but no database is required.

## Data storage

PostgreSQL is the only database. It stores:

- `requests` table — the structured data extracted from each conversation, including a vector embedding of the non-PII fields
- LangGraph checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_migrations`) — conversation state managed by `AsyncPostgresSaver`

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

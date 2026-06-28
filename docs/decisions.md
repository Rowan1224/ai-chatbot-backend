# Design Decisions

## LangGraph for conversation management

Managing multi-turn conversations requires tracking state across HTTP requests — the bot needs to remember what it has already collected, whether it is waiting for a duplicate decision, and the full message history.

LangGraph handles this with a graph of nodes and explicit state. Each node returns a partial state update; LangGraph merges it and persists the full state to PostgreSQL (via `AsyncPostgresSaver`). The routing logic (`should_extract`, `after_duplicate_decision`, etc.) is just Python functions returning strings — easy to read and test.

The alternative would be manually serialising and deserialising conversation state on every request. LangGraph removes that boilerplate and makes the flow visible as a graph.

## PostgreSQL instead of MongoDB

MongoDB was the first choice — document storage is a natural fit for the schema-free `additional_data` field. However, MongoDB's kernel compatibility requirement (kernel ≥ 5.1 for AVX support) makes it incompatible with Linux kernel 6.1 environments where AVX instructions behave differently. The containerised MongoDB instance failed to start reliably in that environment.

PostgreSQL with the `pgvector` extension covers everything MongoDB would have provided:
- JSONB for schema-free document storage
- `pgvector` for vector similarity search
- `pg_trgm` for fuzzy text matching
- A single database for both application data and LangGraph checkpoints (no separate Redis service needed)

The schema-free design is preserved: `additional_data` is stored as JSONB. Adding new fields to the prompt requires no migration.

## Why not exact-match duplicate detection?

Exact matching on structured fields sounds simple but fails in practice for natural language input:

- "infra provisioning" vs "infrastructure-provisioning" — same intent, different string
- Two requests for "deploy service X to production" submitted on different days — likely duplicates but field values are identical so there is no way to distinguish them by exact match alone
- Minor typos or rephrasing produces no match at all

The three-stage pipeline addresses this:

**Stage 1 — Fuzzy pre-filter (pg_trgm)**
Trigram similarity on `request_type` only — the one non-PII field common to all requests. Catches typos and variations. Returns a small candidate set (≤50 rows by default) without scanning the full table. Fast.

**Stage 2 — Vector similarity**
Cosine similarity on embeddings of the non-PII fields. Catches semantic equivalence — "need more servers" and "infrastructure provisioning for scaling" score highly. Scoped to the candidate set from Stage 1, so it is not a full-table scan.

**Stage 3 — LLM judge**
For each candidate that passed the vector threshold, the LLM compares the new request and the candidate side-by-side and decides if they represent the same intent. This eliminates false positives where vector similarity is high but a meaningful field differs (e.g. development vs production environment).

Only candidates the LLM confirms as duplicates reach the user. The cost is one LLM call per vector candidate — typically zero or one in practice.

## Scaling duplicate detection

The current pipeline scales in two directions:

**More requests in the table** — the fuzzy pre-filter keeps Stage 2 and Stage 3 bounded regardless of table size. Add a composite index on `(request_type, created_at)` if `find_fuzzy_candidates` becomes slow.

**Higher throughput** — Stage 3 LLM calls are independent per candidate and can be parallelised with `asyncio.gather`. The current implementation is sequential because the candidate count is typically small (≤5 after Stage 2 filtering). If that changes, parallelising is a one-function change in `duplicate_check_node`.

**Tuning without code changes** — all thresholds (`similarity_threshold`, `fuzzy_threshold`, `lookback_days`, `candidate_limit`) live in `app_config.yaml`. Tighten or loosen them and restart.

## Secrets management: environment variables vs Docker secrets

The app currently reads all credentials via environment variables (`pydantic-settings` with `env_file=".env"`). Docker/Podman secrets (files mounted under `/run/secrets/`) were considered as an alternative.

**Why env vars are the right default here:**

- GitHub Actions and Azure Pipelines — the documented CI/CD targets — manage secrets securely and inject them as environment variables. They do not write to files.
- Env vars are the universal lowest-common-denominator: they work identically in local dev, CI runners, VMs, Swarm, and Kubernetes without any infrastructure-specific code.
- The real env-var exposure risk (`docker inspect` leaking `Config.Env`) is mitigated by restricting Docker socket access on the host — standard practice on hardened CI runners and production VMs.

**When Docker / Kubernetes secrets are the better choice:**

- **Docker Swarm**: Docker secrets are the idiomatic approach. Values are never stored in `Config.Env` and are only mounted into containers that explicitly declare them.
- **Kubernetes**: Secrets can be projected as files or injected via `envFrom`. File-based projection avoids env-var visibility entirely.
- **High-compliance environments** where `docker inspect` access cannot be restricted, or where audit requirements mandate secrets never appear in process environment.

**How to adopt file-based secrets with zero application code changes:**

`pydantic-settings` natively reads from a secrets directory. Adding `secrets_dir` to [`Settings`](../src/config/settings.py) is the only change needed:

```python
model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    secrets_dir="/run/secrets",   # reads /run/secrets/openai_api_key, etc.
    extra="ignore",
)
```

`pydantic-settings` checks environment variables first and falls back to the secrets directory, so both mechanisms work simultaneously. This makes the migration to Swarm or Kubernetes a infrastructure change only — no Python changes required.

---

## Input guardrails: regex layer + LLM provider defence-in-depth

Two input safety concerns are handled before any user message reaches the chat node:

**PII redaction** uses LangChain's `PIIMiddleware` detector functions (`detect_email`, `detect_credit_card`, `apply_strategy`) directly rather than through the middleware. The middleware's `before_model` / `after_model` hooks have no attachment surface on a hand-built `StateGraph`, so the detectors are consumed at the function level instead. The behaviour is identical; the integration path is different.

**Prompt-injection detection** uses compiled regexes that match the most common jailbreak and instruction-override patterns (role override, ignore-instructions phrasing, DAN, developer mode, etc.). A matched message is blocked immediately — the turn ends with a refusal reply and the graph never proceeds to the `chat` node.

### Why only common patterns — not exhaustive coverage

The regex layer does not attempt to catch every possible injection variant. The reason is that **all three supported LLM providers** (OpenAI, Azure OpenAI, Anthropic) enforce their own content and safety policies server-side. Any injection attempt that is sophisticated enough to bypass the regex patterns will still encounter the provider's own guardrails. The `chat_node` already handles the `BadRequestError` (HTTP 400) that providers return when a message is rejected by their content policy — it responds with a refusal message identical in tone to the local rejection.

This means the regex layer and the provider layer are complementary:

| Layer | What it catches | Latency cost |
|---|---|---|
| Regex (local) | Common, well-known patterns | Zero — no network call |
| LLM provider (remote) | Novel, obfuscated, or context-dependent injections | Already paid for every turn |

Because the provider acts as a reliable backstop, keeping the regex list focused on high-frequency patterns is the correct trade-off. Trying to enumerate all injection variants in regexes would be an arms race with diminishing returns and a high false-positive risk for legitimate messages.

### Configuration

Both checks are independently togglable in `app_config.yaml` under `guardrails:` — `pii_redaction_enabled` and `injection_detection_enabled`. A master `enabled` flag disables the entire `guardrail_node` without removing it from the graph topology.

---

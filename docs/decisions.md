# Design Decisions

## LangGraph for conversation management

Managing multi-turn conversations requires tracking state across HTTP requests — the bot needs to remember what it has already collected, whether it is waiting for a duplicate decision, and the full message history.

LangGraph handles this with a graph of nodes and explicit state. Each node returns a partial state update; LangGraph merges it and persists the full state to Redis. The routing logic (`should_extract`, `after_duplicate_decision`, etc.) is just Python functions returning strings — easy to read and test.

The alternative would be manually serialising and deserialising conversation state on every request. LangGraph removes that boilerplate and makes the flow visible as a graph.

## PostgreSQL instead of MongoDB

MongoDB was the first choice — document storage is a natural fit for the schema-free `additional_data` field. However, MongoDB's kernel compatibility requirement (kernel ≥ 5.1 for AVX support) makes it incompatible with Linux kernel 6.1 environments where AVX instructions behave differently. The containerised MongoDB instance failed to start reliably in that environment.

PostgreSQL with the `pgvector` extension covers everything MongoDB would have provided:
- JSONB for schema-free document storage
- `pgvector` for vector similarity search
- `pg_trgm` for fuzzy text matching
- A single database for both application data and LangGraph checkpoints

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

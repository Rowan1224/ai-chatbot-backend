"""Database connections for PostgreSQL (pgvector) and Redis."""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector
from redis.asyncio import Redis

from src.config.settings import (
    app_config,
    prompt_config,
    settings,
)

logger = logging.getLogger(__name__)

# DDL executed once on first connect
_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS requests (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    config_version TEXT NOT NULL,
    data        JSONB NOT NULL,
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_requests_created_at
    ON requests (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_requests_config_version
    ON requests (config_version);

CREATE INDEX IF NOT EXISTS idx_requests_embedding
    ON requests USING hnsw (embedding vector_cosine_ops);
"""


class PostgreSQLClient:
    """Async PostgreSQL client with pgvector support."""

    def __init__(self) -> None:
        """Initialize PostgreSQL client."""
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """
        Bootstrap sequence:

        1. Open a plain connection (no codec registered yet).
        2. Run ``CREATE EXTENSION IF NOT EXISTS vector`` so the
           pgvector type exists before any codec registration.
        3. Run remaining DDL (table + indexes).
        4. Close the plain connection.
        5. Create the pool — its ``init`` hook now safely calls
           ``register_vector`` because the type already exists.
        """
        try:
            # Steps 1-4: DDL on a raw connection so the vector
            # extension is installed before pool init fires.
            bootstrap = await asyncpg.connect(
                settings.postgresql_url
            )
            try:
                await bootstrap.execute(_SCHEMA_SQL)
            finally:
                await bootstrap.close()

            # Step 5: pool init= is now safe.
            self.pool = await asyncpg.create_pool(
                settings.postgresql_url,
                min_size=2,
                max_size=10,
                init=_init_connection,
            )
            logger.info(
                "Connected to PostgreSQL and schema ready"
            )
        except Exception as e:
            logger.error(
                f"Failed to connect to PostgreSQL: {e}"
            )
            raise

    async def close(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def ping(self) -> bool:
        """Health-check: returns True if the pool is alive."""
        if not self.pool:
            return False
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return True


async def _init_connection(conn: asyncpg.Connection) -> None:
    """
    Per-connection hook — registers pgvector codec.

    Only called after pool creation, by which point
    CREATE EXTENSION vector has already been executed.
    """
    await register_vector(conn)


# ---------------------------------------------------------------------------
# Public helpers used by the workflow
# ---------------------------------------------------------------------------


async def save_request(
    pg_client: PostgreSQLClient,
    data: dict[str, Any],
    embedding: list[float],
) -> str:
    """
    Insert a new request row and return its UUID string.

    Args:
        pg_client: PostgreSQL client instance
        data: Request data dictionary
        embedding: Vector embedding for duplicate detection

    Returns:
        Inserted row UUID as string
    """
    async with pg_client.pool.acquire() as conn:
        row_id = await conn.fetchval(
            """
            INSERT INTO requests
                (config_version, data, embedding, created_at)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            prompt_config.config_version,
            json.dumps(data),
            embedding,
            datetime.utcnow(),
        )

    request_id = str(row_id)
    logger.info(f"Saved request with ID: {request_id}")
    return request_id


async def find_fuzzy_candidates(
    pg_client: PostgreSQLClient,
    request_type: str,
    lookback_days: int = 90,
    fuzzy_threshold: float = 0.3,
    limit: int = 50,
) -> list[str]:
    """
    Stage 1 — fuzzy pre-filter using pg_trgm similarity.

    Matches only on ``request_type`` (the one non-PII field
    common to all generic requests) using trigram similarity
    so minor typos / variations still produce candidates.
    Returns only row UUIDs — no PII or embeddings fetched.

    Args:
        pg_client: PostgreSQL client instance
        request_type: The request type string to match against
        lookback_days: How many days to look back (wide window)
        fuzzy_threshold: pg_trgm similarity floor (0–1)
        limit: Maximum number of candidate IDs to return

    Returns:
        List of UUID strings for rows that pass the fuzzy filter
    """
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    async with pg_client.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id,
                   similarity(data->>'request_type', $1)
                       AS rt_sim
            FROM   requests
            WHERE  created_at >= $2
              AND  similarity(data->>'request_type', $1) >= $3
            ORDER  BY similarity(data->>'request_type', $1) DESC
            LIMIT  $4
            """,
            request_type,
            cutoff,
            fuzzy_threshold,
            limit,
        )

    candidate_ids = [str(row["id"]) for row in rows]
    logger.info(
        f"Fuzzy pre-filter found {len(candidate_ids)} "
        f"candidate(s) for request_type='{request_type}'"
    )
    return candidate_ids


async def find_similar_requests(
    pg_client: PostgreSQLClient,
    embedding: list[float],
    candidate_ids: list[str],
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    """
    Stage 2 — vector similarity search scoped to candidates.

    Runs cosine similarity only against the candidate set
    produced by ``find_fuzzy_candidates``, so the vector index
    never scans unrelated request types.

    Args:
        pg_client: PostgreSQL client instance
        embedding: Query embedding vector
        candidate_ids: UUID strings from the fuzzy pre-filter
        threshold: Cosine similarity threshold (0–1)

    Returns:
        List of similar rows with ``similarity_score``,
        sorted descending by score (top 5).
    """
    if not candidate_ids:
        return []

    if app_config.vector_search_provider == "pgvector":
        try:
            return await _pgvector_search(
                pg_client, embedding,
                candidate_ids, threshold,
            )
        except Exception as e:
            logger.warning(
                f"pgvector search failed: {e}, "
                "falling back to local similarity"
            )
            if app_config.duplicate_detection_enabled:
                return await _local_similarity_search(
                    pg_client, embedding,
                    candidate_ids, threshold,
                )
            return []
    else:
        return await _local_similarity_search(
            pg_client, embedding,
            candidate_ids, threshold,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _pgvector_search(
    pg_client: PostgreSQLClient,
    embedding: list[float],
    candidate_ids: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """
    pgvector HNSW cosine search scoped to ``candidate_ids``.

    ``1 - (embedding <=> query)`` converts cosine distance to
    similarity so the threshold is consistent with the fallback.
    """
    uuids = [uuid.UUID(cid) for cid in candidate_ids]

    async with pg_client.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id,
                   config_version,
                   data,
                   created_at,
                   1 - (embedding <=> $1) AS similarity_score
            FROM   requests
            WHERE  id = ANY($2)
              AND  1 - (embedding <=> $1) >= $3
            ORDER  BY embedding <=> $1
            LIMIT  5
            """,
            embedding,
            uuids,
            threshold,
        )

    results = [_row_to_dict(r) for r in rows]
    logger.info(
        f"pgvector search found {len(results)} "
        f"similar request(s) in candidate set"
    )
    return results


async def _local_similarity_search(
    pg_client: PostgreSQLClient,
    embedding: list[float],
    candidate_ids: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    """
    Fallback: in-process cosine similarity scoped to candidates.

    Fetches only the candidate rows (already limited by the
    fuzzy pre-filter) so no OOM risk.

    Requires numpy and scikit-learn, which are dev-only dependencies
    (not present in the production image).  If either is missing this
    function logs a clear error and returns empty — the same safe
    default used when pgvector itself raises an exception.
    To enable this path in production, install the extras:
        pip install numpy scikit-learn
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as exc:
        logger.error(
            f"Local similarity fallback unavailable: {exc}. "
            "Install numpy and scikit-learn to enable it, or set "
            "vector_search_provider=pgvector in app_config.yaml."
        )
        return []

    uuids = [uuid.UUID(cid) for cid in candidate_ids]

    async with pg_client.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, config_version, data,
                   created_at, embedding
            FROM   requests
            WHERE  id = ANY($1)
              AND  embedding IS NOT NULL
            """,
            uuids,
        )

    query_vec = np.array(embedding).reshape(1, -1)
    similar: list[dict[str, Any]] = []

    for row in rows:
        doc_vec = np.array(
            list(row["embedding"])
        ).reshape(1, -1)
        score = float(
            cosine_similarity(query_vec, doc_vec)[0][0]
        )
        if score >= threshold:
            doc = _row_to_dict(row)
            doc["similarity_score"] = score
            similar.append(doc)

    similar.sort(
        key=lambda x: x["similarity_score"], reverse=True
    )
    logger.info(
        f"Local similarity search found "
        f"{len(similar)} similar request(s) in candidate set"
    )
    return similar[:5]


def _row_to_dict(row: asyncpg.Record) -> dict[str, Any]:
    """Convert an asyncpg Record to a plain dict."""
    d = dict(row)
    # Normalise id to string
    if "id" in d and isinstance(d["id"], uuid.UUID):
        d["_id"] = str(d.pop("id"))
    # data is stored as a JSON string in asyncpg
    if "data" in d and isinstance(d["data"], str):
        d["data"] = json.loads(d["data"])
    return d


class RedisClient:
    """Redis client for LangGraph checkpointing."""

    def __init__(self) -> None:
        """Initialize Redis client."""
        self.client: Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self.client = Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                decode_responses=True,
                protocol=2,
            )
            await self.client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    async def get_client(self) -> Redis:
        """Get Redis client instance."""
        if not self.client:
            await self.connect()
        return self.client


# Note: Instances are created in main.py lifespan, not here.

# Made with Bob

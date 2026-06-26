"""
Shared fixtures for integration tests.

Starts a real PostgreSQL+pgvector container and a real Redis container
once per test session via testcontainers, then tears them down after the
session ends.

DATABASE ISOLATION STRATEGY
----------------------------
asyncpg pools are bound to the event loop they were created on.
pytest-asyncio 1.4 creates a new loop per test function by default.
Rather than fighting that, we expose the raw DSN as a session fixture
and let each test (or a function-scoped helper) create its own
PostgreSQLClient that lives exactly one test.  This is safe because
testcontainers keeps the Postgres process running for the whole session.

REDIS
-----
LangGraph's AsyncRedisSaver and the plain redis.asyncio client are
per-invocation objects created inside each test function, so they also
naturally avoid cross-loop issues.

Run with:  pytest tests/integration/ -v

Podman note
-----------
Set DOCKER_HOST to your Podman socket before running, e.g.:
  export DOCKER_HOST=unix:///var/folders/.../podman-machine-default-api.sock
  export TESTCONTAINERS_RYUK_DISABLED=true
Or set PODMAN_SOCKET_PATH and this conftest will detect it automatically.
"""

import asyncio
import os

import pytest
import pytest_asyncio
from testcontainers.postgres import PostgresContainer
from testcontainers.redis import RedisContainer

import src.config.settings as _settings_mod
from src.core.database import PostgreSQLClient, RedisClient

# ---------------------------------------------------------------------------
# Podman / Docker socket auto-detection
# ---------------------------------------------------------------------------

_PODMAN_GLOB = (
    "/var/folders/*/*/T/podman/podman-machine-default-api.sock"
)

if not os.environ.get("DOCKER_HOST"):
    _custom = os.environ.get("PODMAN_SOCKET_PATH")
    if _custom and os.path.exists(_custom):
        os.environ["DOCKER_HOST"] = f"unix://{_custom}"
        os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    else:
        import glob as _glob
        _candidates = _glob.glob(_PODMAN_GLOB)
        if _candidates:
            os.environ["DOCKER_HOST"] = f"unix://{_candidates[0]}"
            os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

# ---------------------------------------------------------------------------
# Container images — match docker-compose.yml exactly
# ---------------------------------------------------------------------------

PG_IMAGE = "pgvector/pgvector:pg16"
REDIS_IMAGE = "redis/redis-stack-server:latest"

# ---------------------------------------------------------------------------
# Session-scoped containers (started once, reused across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_container():
    """
    Start a pgvector/pgvector:pg16 container once for the session.
    The container process stays alive; individual tests connect to it
    via their own function-scoped clients.
    """
    with PostgresContainer(
        image=PG_IMAGE,
        username="test",
        password="test",
        dbname="test_chatbot",
    ) as container:
        yield container


@pytest.fixture(scope="session")
def pg_dsn(postgres_container):
    """
    Return the asyncpg-compatible DSN for the running container.
    This is session-scoped (just a string) and safe to share.
    """
    url = postgres_container.get_connection_url()
    return url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "psycopg2://", "postgresql://"
    )


@pytest.fixture(scope="session")
def redis_container():
    """
    Start a redis/redis-stack-server container once for the session.
    Redis Stack is required by langgraph-checkpoint-redis.
    """
    with RedisContainer(image=REDIS_IMAGE) as container:
        yield container


@pytest.fixture(scope="session")
def redis_url(redis_container):
    """Return the redis:// URL for the running container."""
    host = redis_container.get_container_host_ip()
    port = redis_container.get_exposed_port(6379)
    return f"redis://{host}:{port}"


# ---------------------------------------------------------------------------
# Function-scoped pg_client (new pool per test → no cross-loop issues)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pg_client(pg_dsn):
    """
    Function-scoped PostgreSQLClient — creates a fresh asyncpg pool
    on the current test's event loop and tears it down after the test.

    Schema bootstrap is idempotent (CREATE IF NOT EXISTS) so running it
    once per test is safe and simple.
    """
    _settings_mod.settings.postgresql_url = pg_dsn
    client = PostgreSQLClient()
    await client.connect()
    yield client
    await client.close()


@pytest_asyncio.fixture
async def clean_db(pg_client):
    """
    Truncate the requests table before each test to ensure isolation.
    """
    async with pg_client.pool.acquire() as conn:
        await conn.execute(
            "TRUNCATE TABLE requests RESTART IDENTITY CASCADE"
        )
    yield


# Made with Bob

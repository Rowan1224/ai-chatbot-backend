"""
Integration tests for PostgreSQLClient — real pgvector/pgvector:pg16 container.

Verifies:
- Schema bootstrap creates the required table and extensions
- save_request persists a row with correct JSON data and embedding
- update_request modifies the row and sets updated_at
- update_request returns False for unknown IDs
- ping returns True on a live connection
"""

import json
import uuid

import pytest
import pytest_asyncio

from src.core.database import PostgreSQLClient, save_request, update_request


pytestmark = pytest.mark.integration

# Fixed 1536-dim embedding — no real model needed
EMBEDDING = [0.1] * 1536
EMBEDDING_ALT = [0.2] * 1536


# ---------------------------------------------------------------------------
# Schema bootstrap
# ---------------------------------------------------------------------------


class TestSchemaBootstrap:
    """Verify that connect() correctly bootstraps the database schema."""

    @pytest.mark.asyncio
    async def test_requests_table_exists(self, pg_client):
        """The ``requests`` table must exist after connect()."""
        async with pg_client.pool.acquire() as conn:
            result = await conn.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'requests'
                )
                """
            )
        assert result is True, "requests table was not created"

    @pytest.mark.asyncio
    async def test_vector_extension_installed(self, pg_client):
        """The pgvector extension must be installed."""
        async with pg_client.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
        assert result is True, "vector extension was not installed"

    @pytest.mark.asyncio
    async def test_pg_trgm_extension_installed(self, pg_client):
        """The pg_trgm extension must be installed for fuzzy search."""
        async with pg_client.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
            )
        assert result is True, "pg_trgm extension was not installed"

    @pytest.mark.asyncio
    async def test_embedding_column_is_vector_type(self, pg_client):
        """The embedding column must be declared as vector(1536)."""
        async with pg_client.pool.acquire() as conn:
            result = await conn.fetchrow(
                """
                SELECT udt_name
                FROM information_schema.columns
                WHERE table_name = 'requests'
                  AND column_name = 'embedding'
                """
            )
        assert result is not None, "embedding column not found"
        assert result["udt_name"] == "vector"


# ---------------------------------------------------------------------------
# save_request
# ---------------------------------------------------------------------------


class TestSaveRequest:
    """save_request inserts a row and returns its UUID."""

    @pytest.mark.asyncio
    async def test_returns_uuid_string(self, pg_client, clean_db):
        """save_request must return a UUID string."""
        data = {
            "request_type": "infrastructure-provisioning",
            "target_environment": "production",
            "business_justification": "Need servers",
            "name": "Jane",
            "employee_id": "EMP001",
        }
        request_id = await save_request(pg_client, data, EMBEDDING)

        assert isinstance(request_id, str)
        # Must be parseable as UUID
        uuid.UUID(request_id)

    @pytest.mark.asyncio
    async def test_row_exists_in_database(self, pg_client, clean_db):
        """The saved row must be retrievable from the database."""
        data = {
            "request_type": "service-deployment",
            "business_justification": "Deploy v2",
            "name": "Bob",
            "employee_id": "EMP002",
        }
        request_id = await save_request(pg_client, data, EMBEDDING)

        async with pg_client.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, data FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        assert row is not None, "saved row not found in database"

    @pytest.mark.asyncio
    async def test_data_json_persisted_correctly(self, pg_client, clean_db):
        """The data dict must round-trip through JSONB without loss."""
        data = {
            "request_type": "access-grant",
            "target_environment": "staging",
            "business_justification": "Need read access",
            "name": "Carol",
            "employee_id": "EMP003",
        }
        request_id = await save_request(pg_client, data, EMBEDDING)

        async with pg_client.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT data FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        # asyncpg returns JSONB as a string
        stored = json.loads(raw) if isinstance(raw, str) else raw
        assert stored["request_type"] == "access-grant"
        assert stored["target_environment"] == "staging"
        assert stored["name"] == "Carol"

    @pytest.mark.asyncio
    async def test_embedding_is_stored_and_not_null(self, pg_client, clean_db):
        """The embedding column must be non-NULL after save."""
        data = {"request_type": "pipeline-change", "name": "Dave", "employee_id": "EMP004"}
        request_id = await save_request(pg_client, data, EMBEDDING)

        async with pg_client.pool.acquire() as conn:
            emb = await conn.fetchval(
                "SELECT embedding FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        assert emb is not None, "embedding was not stored"

    @pytest.mark.asyncio
    async def test_created_at_is_set(self, pg_client, clean_db):
        """created_at must be populated on insert."""
        data = {"request_type": "incident-fix", "name": "Eve", "employee_id": "EMP005"}
        request_id = await save_request(pg_client, data, EMBEDDING)

        async with pg_client.pool.acquire() as conn:
            created_at = await conn.fetchval(
                "SELECT created_at FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        assert created_at is not None


# ---------------------------------------------------------------------------
# update_request
# ---------------------------------------------------------------------------


class TestUpdateRequest:
    """update_request modifies an existing row."""

    @pytest.mark.asyncio
    async def test_returns_true_when_row_updated(self, pg_client, clean_db):
        """update_request returns True when the row exists and is updated."""
        data = {"request_type": "access-grant", "name": "Frank", "employee_id": "EMP006"}
        request_id = await save_request(pg_client, data, EMBEDDING)

        updated_data = {**data, "target_environment": "production"}
        result = await update_request(pg_client, request_id, updated_data, EMBEDDING_ALT)

        assert result is True

    @pytest.mark.asyncio
    async def test_data_is_changed_after_update(self, pg_client, clean_db):
        """The updated data must be reflected when the row is re-read."""
        data = {"request_type": "pipeline-change", "name": "Gina", "employee_id": "EMP007"}
        request_id = await save_request(pg_client, data, EMBEDDING)

        updated_data = {**data, "business_justification": "Updated reason"}
        await update_request(pg_client, request_id, updated_data, EMBEDDING_ALT)

        async with pg_client.pool.acquire() as conn:
            raw = await conn.fetchval(
                "SELECT data FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        stored = json.loads(raw) if isinstance(raw, str) else raw
        assert stored.get("business_justification") == "Updated reason"

    @pytest.mark.asyncio
    async def test_updated_at_is_set_after_update(self, pg_client, clean_db):
        """updated_at must be non-NULL after an update."""
        data = {"request_type": "service-deployment", "name": "Hank", "employee_id": "EMP008"}
        request_id = await save_request(pg_client, data, EMBEDDING)

        await update_request(pg_client, request_id, data, EMBEDDING_ALT)

        async with pg_client.pool.acquire() as conn:
            updated_at = await conn.fetchval(
                "SELECT updated_at FROM requests WHERE id = $1",
                uuid.UUID(request_id),
            )

        assert updated_at is not None, "updated_at was not set after update"

    @pytest.mark.asyncio
    async def test_returns_false_for_unknown_id(self, pg_client, clean_db):
        """update_request returns False when the UUID does not exist."""
        missing_id = str(uuid.uuid4())
        data = {"request_type": "access-grant", "name": "Ian", "employee_id": "EMP009"}

        result = await update_request(pg_client, missing_id, data, EMBEDDING)

        assert result is False


# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_returns_true_on_live_connection(self, pg_client):
        """ping() must return True when the pool is alive."""
        result = await pg_client.ping()
        assert result is True


# Made with Bob

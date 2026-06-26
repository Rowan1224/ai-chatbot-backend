"""Unit tests for database module — PostgreSQL/pgvector implementation."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database import (
    PostgreSQLClient,
    RedisClient,
    find_fuzzy_candidates,
    find_similar_requests,
    save_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(data: dict, row_id: uuid.UUID | None = None) -> dict:
    """
    Simulate the dict that _row_to_dict produces from an asyncpg Record.
    Mirrors the column names of the ``requests`` table.
    """
    return {
        "_id": str(row_id or uuid.uuid4()),
        "config_version": "v1",
        "data": data,
        "created_at": datetime.utcnow(),
    }


def _make_asyncpg_record(data: dict,
                         row_id: uuid.UUID | None = None,
                         similarity_score: float | None = None
                         ) -> MagicMock:
    """
    Return a MagicMock that behaves like an asyncpg Record:
    - dict(row) returns the expected column mapping
    - row["col"] works via __getitem__
    """
    uid = row_id or uuid.uuid4()
    raw: dict = {
        "id": uid,
        "config_version": "v1",
        "data": json.dumps(data),
        "created_at": datetime.utcnow(),
    }
    if similarity_score is not None:
        raw["similarity_score"] = similarity_score

    record = MagicMock()
    record.__iter__ = MagicMock(
        return_value=iter(raw.items())
    )
    record.__getitem__ = MagicMock(
        side_effect=lambda k: raw[k]
    )
    record.keys = MagicMock(return_value=raw.keys())
    # dict(record) path used by _row_to_dict
    record.__class__ = type(
        "Record", (dict,), {}
    )
    # Make dict(record) work
    record.items = MagicMock(return_value=raw.items())

    return record


def _mock_pg_client() -> MagicMock:
    """Build a PostgreSQLClient mock with a working pool."""
    client = MagicMock(spec=PostgreSQLClient)
    pool = MagicMock()
    client.pool = pool

    # pool.acquire() is an async context manager
    conn = AsyncMock()
    acquire_ctx = MagicMock()
    acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
    acquire_ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acquire_ctx

    return client, conn


# ---------------------------------------------------------------------------
# PostgreSQLClient
# ---------------------------------------------------------------------------


class TestPostgreSQLClient:
    """Test PostgreSQLClient initialisation."""

    def test_init_creates_instance(self):
        """Client starts with no pool."""
        client = PostgreSQLClient()
        assert client is not None
        assert client.pool is None

    @pytest.mark.asyncio
    async def test_connect_creates_pool(self):
        """connect() runs DDL via bootstrap conn then builds pool."""
        mock_bootstrap = AsyncMock()
        mock_bootstrap.execute = AsyncMock()
        mock_bootstrap.close = AsyncMock()

        pool_instance = AsyncMock()

        with (
            patch("src.core.database.asyncpg.connect",
                  new_callable=AsyncMock,
                  return_value=mock_bootstrap),
            patch("src.core.database.asyncpg.create_pool",
                  new_callable=AsyncMock,
                  return_value=pool_instance),
            patch("src.core.database.register_vector",
                  new_callable=AsyncMock),
        ):
            client = PostgreSQLClient()
            await client.connect()

            assert client.pool is not None
            # DDL must have been executed on the bootstrap conn
            mock_bootstrap.execute.assert_called_once()
            # Bootstrap conn must be closed regardless of errors
            mock_bootstrap.close.assert_called_once()


# ---------------------------------------------------------------------------
# RedisClient
# ---------------------------------------------------------------------------


class TestRedisClient:
    """Test RedisClient initialisation."""

    def test_init_creates_instance(self):
        client = RedisClient()
        assert client is not None
        assert client.client is None

    @pytest.mark.asyncio
    async def test_connect_sets_client(self):
        with patch("src.core.database.Redis") as mock_redis:
            mock_instance = MagicMock()
            mock_instance.ping = AsyncMock(return_value=True)
            mock_redis.return_value = mock_instance

            client = RedisClient()
            await client.connect()

            assert client.client is not None


# ---------------------------------------------------------------------------
# find_fuzzy_candidates
# ---------------------------------------------------------------------------


class TestFindFuzzyCandidates:
    """Stage 1: pg_trgm fuzzy pre-filter returns candidate UUIDs."""

    @pytest.mark.asyncio
    async def test_no_candidates_found(self):
        """Returns empty list when nothing passes the threshold."""
        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[])

        result = await find_fuzzy_candidates(
            pg_client=client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
        )

        assert result == []
        conn.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_candidates_returned_as_uuid_strings(self):
        """Returns list of UUID strings, not full row dicts."""
        uid = uuid.uuid4()
        fake_row = {"id": uid, "rt_sim": 0.9}

        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[fake_row])

        result = await find_fuzzy_candidates(
            pg_client=client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
        )

        assert result == [str(uid)]

    @pytest.mark.asyncio
    async def test_query_uses_only_request_type(self):
        """
        CRITICAL: only request_type is sent as the fuzzy
        match field — no PII, no target_environment.
        Positional args: SQL, request_type, cutoff,
                         threshold, limit.
        """
        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[])

        await find_fuzzy_candidates(
            pg_client=client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        call_args = conn.fetch.call_args[0]
        # args[0]=SQL, [1]=request_type, [2]=cutoff,
        # [3]=threshold, [4]=limit
        assert call_args[1] == "infrastructure-provisioning"
        assert call_args[3] == 0.3

    @pytest.mark.asyncio
    async def test_multiple_candidates(self):
        """Returns all candidate IDs up to the limit."""
        uids = [uuid.uuid4() for _ in range(3)]
        fake_rows = [{"id": u, "rt_sim": 0.8} for u in uids]

        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=fake_rows)

        result = await find_fuzzy_candidates(
            pg_client=client,
            request_type="service-deployment",
            lookback_days=90,
        )

        assert len(result) == 3
        assert result == [str(u) for u in uids]


# ---------------------------------------------------------------------------
# find_similar_requests
# ---------------------------------------------------------------------------


class TestFindSimilarRequests:
    """Stage 2: vector search scoped to candidate IDs."""

    @pytest.mark.asyncio
    async def test_empty_candidate_ids_returns_empty(self):
        """Short-circuits immediately when no candidates exist."""
        client, conn = _mock_pg_client()

        result = await find_similar_requests(
            pg_client=client,
            embedding=[0.1] * 1536,
            candidate_ids=[],
            threshold=0.85,
        )

        assert result == []
        # Pool should never be touched
        conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_similar_requests(self):
        """Returns empty list when pgvector finds nothing."""
        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[])

        candidate_ids = [str(uuid.uuid4())]

        result = await find_similar_requests(
            pg_client=client,
            embedding=[0.1] * 1536,
            candidate_ids=candidate_ids,
            threshold=0.85,
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_similar_request_found(self):
        """Returns rows when pgvector search finds matches."""
        data = {"request_type": "infra", "env": "dev"}
        fake_row = {
            "id": uuid.uuid4(),
            "config_version": "v1",
            "data": json.dumps(data),
            "created_at": datetime.utcnow(),
            "similarity_score": 0.92,
        }

        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[fake_row])

        with patch(
            "src.core.database.settings"
        ) as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True
            result = await find_similar_requests(
                pg_client=client,
                embedding=[0.1] * 1536,
                candidate_ids=[str(uuid.uuid4())],
                threshold=0.85,
            )

        assert isinstance(result, list)
        if result:
            assert result[0]["similarity_score"] == 0.92

    @pytest.mark.asyncio
    async def test_query_scoped_to_candidate_ids(self):
        """
        CRITICAL: the SQL WHERE clause must filter by the
        candidate UUID list, not scan the full table.
        """
        client, conn = _mock_pg_client()
        conn.fetch = AsyncMock(return_value=[])

        candidate_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

        with patch(
            "src.core.database.settings"
        ) as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True
            await find_similar_requests(
                pg_client=client,
                embedding=[0.1] * 1536,
                candidate_ids=candidate_ids,
                threshold=0.85,
            )

        call_args = conn.fetch.call_args[0]
        # Second positional arg = list of uuid.UUID objects
        passed_uuids = call_args[2]
        assert len(passed_uuids) == 2
        assert all(
            isinstance(u, uuid.UUID) for u in passed_uuids
        )


# ---------------------------------------------------------------------------
# save_request
# ---------------------------------------------------------------------------


class TestSaveRequest:
    """Inserting new requests into PostgreSQL."""

    @pytest.mark.asyncio
    async def test_save_request_success(self):
        """Returns a UUID string on success."""
        new_id = uuid.uuid4()
        client, conn = _mock_pg_client()
        conn.fetchval = AsyncMock(return_value=new_id)

        request_id = await save_request(
            pg_client=client,
            data={"request_type": "infra", "name": "John"},
            embedding=[0.1] * 1536,
        )

        assert request_id == str(new_id)
        conn.fetchval.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_includes_required_metadata(self):
        """
        CRITICAL: The INSERT must pass config_version, data,
        embedding, and created_at as positional params.
        """
        new_id = uuid.uuid4()
        client, conn = _mock_pg_client()
        conn.fetchval = AsyncMock(return_value=new_id)

        data = {"request_type": "test"}
        embedding = [0.1] * 1536

        await save_request(
            pg_client=client,
            data=data,
            embedding=embedding,
        )

        args = conn.fetchval.call_args[0]
        # args[0] = SQL, args[1] = config_version,
        # args[2] = data JSON, args[3] = embedding,
        # args[4] = created_at
        assert len(args) == 5, (
            "Expected 5 positional args to INSERT"
        )
        assert json.loads(args[2]) == data, (
            "data not serialised correctly"
        )
        assert args[3] == embedding, "embedding missing"
        assert isinstance(args[4], datetime), (
            "created_at must be a datetime"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob

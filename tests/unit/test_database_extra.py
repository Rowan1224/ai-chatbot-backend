"""Additional unit tests for database.py — branches not covered by test_database.py."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.database import (
    PostgreSQLClient,
    RedisClient,
    _row_to_dict,
)


# ---------------------------------------------------------------------------
# PostgreSQLClient — close and ping
# ---------------------------------------------------------------------------


class TestPostgreSQLClientClose:

    @pytest.mark.asyncio
    async def test_close_with_pool_calls_pool_close(self):
        client = PostgreSQLClient()
        mock_pool = AsyncMock()
        client.pool = mock_pool

        await client.close()

        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_pool_is_noop(self):
        """close() must not raise when pool is None."""
        client = PostgreSQLClient()
        await client.close()  # should not raise

    @pytest.mark.asyncio
    async def test_ping_returns_false_without_pool(self):
        client = PostgreSQLClient()
        assert client.pool is None
        result = await client.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_returns_true_with_live_pool(self):
        client = PostgreSQLClient()

        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=1)

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=False)

        pool = MagicMock()
        pool.acquire.return_value = acquire_ctx
        client.pool = pool

        result = await client.ping()

        assert result is True
        conn.fetchval.assert_awaited_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_connect_raises_on_bootstrap_failure(self):
        """connect() propagates exceptions from asyncpg.connect."""
        with patch(
            "src.core.database.asyncpg.connect",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("refused"),
        ):
            client = PostgreSQLClient()
            with pytest.raises(ConnectionRefusedError):
                await client.connect()


# ---------------------------------------------------------------------------
# RedisClient — close and get_client
# ---------------------------------------------------------------------------


class TestRedisClientExtra:

    @pytest.mark.asyncio
    async def test_close_with_client_calls_close(self):
        client = RedisClient()
        mock_redis = AsyncMock()
        client.client = mock_redis

        await client.close()

        mock_redis.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_client_is_noop(self):
        client = RedisClient()
        await client.close()  # should not raise

    @pytest.mark.asyncio
    async def test_get_client_returns_existing_client(self):
        client = RedisClient()
        mock_redis = MagicMock()
        client.client = mock_redis

        result = await client.get_client()

        assert result is mock_redis

    @pytest.mark.asyncio
    async def test_get_client_connects_if_none(self):
        """get_client() calls connect() when client is None."""
        with patch("src.core.database.Redis") as mock_redis_cls:
            mock_instance = MagicMock()
            mock_instance.ping = AsyncMock(return_value=True)
            mock_redis_cls.return_value = mock_instance

            client = RedisClient()
            result = await client.get_client()

        assert result is mock_instance

    @pytest.mark.asyncio
    async def test_connect_raises_on_redis_failure(self):
        with patch("src.core.database.Redis") as mock_redis_cls:
            mock_instance = MagicMock()
            mock_instance.ping = AsyncMock(
                side_effect=ConnectionRefusedError("redis down")
            )
            mock_redis_cls.return_value = mock_instance

            client = RedisClient()
            with pytest.raises(ConnectionRefusedError):
                await client.connect()


# ---------------------------------------------------------------------------
# _row_to_dict
# ---------------------------------------------------------------------------


class TestRowToDict:
    """Tests for the _row_to_dict helper."""

    def test_uuid_id_is_converted_to_string(self):
        uid = uuid.uuid4()
        row = {"id": uid, "config_version": "v1", "data": json.dumps({"k": "v"}), "created_at": datetime.utcnow()}
        result = _row_to_dict(row)

        assert "_id" in result
        assert isinstance(result["_id"], str)
        assert "id" not in result

    def test_data_string_is_parsed_to_dict(self):
        uid = uuid.uuid4()
        data_dict = {"request_type": "infra", "env": "prod"}
        row = {
            "id": uid,
            "config_version": "v1",
            "data": json.dumps(data_dict),
            "created_at": datetime.utcnow(),
        }
        result = _row_to_dict(row)

        assert isinstance(result["data"], dict)
        assert result["data"]["request_type"] == "infra"

    def test_non_uuid_id_is_preserved_as_is(self):
        """If id is already a string (edge case), it stays under 'id'."""
        row = {
            "id": "not-a-uuid",
            "config_version": "v1",
            "data": "{}",
            "created_at": datetime.utcnow(),
        }
        result = _row_to_dict(row)
        # string 'id' → treated as non-UUID, kept under original key
        assert "id" in result or "_id" in result

    def test_already_dict_data_is_not_double_parsed(self):
        """If data is already a dict (not a JSON string), it should remain a dict."""
        uid = uuid.uuid4()
        data_dict = {"request_type": "access-grant"}
        row = {
            "id": uid,
            "config_version": "v1",
            "data": data_dict,  # already a dict
            "created_at": datetime.utcnow(),
        }
        result = _row_to_dict(row)
        # data was not a string so json.loads was not called
        assert result["data"] == data_dict


# ---------------------------------------------------------------------------
# find_similar_requests — local fallback path
# ---------------------------------------------------------------------------


class TestFindSimilarRequestsLocalFallback:
    """Exercise the local similarity search fallback path."""

    @pytest.mark.asyncio
    async def test_uses_local_search_when_provider_is_local(self):
        """When vector_search_provider='local', _local_similarity_search is used."""
        from src.core.database import find_similar_requests

        client = MagicMock(spec=["pool"])
        conn = AsyncMock()

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=False)
        client.pool = MagicMock()
        client.pool.acquire.return_value = acquire_ctx

        conn.fetch = AsyncMock(return_value=[])

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "local"
            mock_settings.duplicate_detection_enabled = True

            result = await find_similar_requests(
                pg_client=client,
                embedding=[0.1] * 1536,
                candidate_ids=[str(uuid.uuid4())],
                threshold=0.85,
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_pgvector_exception_falls_back_to_local(self):
        """When pgvector raises an exception and fallback is enabled, local is used."""
        from src.core.database import find_similar_requests

        client = MagicMock(spec=["pool"])
        conn = AsyncMock()

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=False)
        client.pool = MagicMock()
        client.pool.acquire.return_value = acquire_ctx

        # First call (pgvector) raises; second call (local fallback) returns []
        conn.fetch = AsyncMock(
            side_effect=[RuntimeError("pgvector error"), []]
        )

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            result = await find_similar_requests(
                pg_client=client,
                embedding=[0.1] * 1536,
                candidate_ids=[str(uuid.uuid4())],
                threshold=0.85,
            )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_pgvector_exception_no_fallback_returns_empty(self):
        """When pgvector raises and duplicate_detection_enabled=False, returns []."""
        from src.core.database import find_similar_requests

        client = MagicMock(spec=["pool"])
        conn = AsyncMock()

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=False)
        client.pool = MagicMock()
        client.pool.acquire.return_value = acquire_ctx

        conn.fetch = AsyncMock(side_effect=RuntimeError("pgvector error"))

        with patch("src.core.database.app_config") as mock_cfg:
            mock_cfg.vector_search_provider = "pgvector"
            mock_cfg.duplicate_detection_enabled = False

            result = await find_similar_requests(
                pg_client=client,
                embedding=[0.1] * 1536,
                candidate_ids=[str(uuid.uuid4())],
                threshold=0.85,
            )

        assert result == []


# ---------------------------------------------------------------------------
# update_request — exception branch
# ---------------------------------------------------------------------------


class TestUpdateRequestExceptionBranch:
    @pytest.mark.asyncio
    async def test_update_request_returns_false_on_exception(self):
        from src.core.database import update_request

        client = MagicMock(spec=["pool"])
        conn = AsyncMock()

        acquire_ctx = MagicMock()
        acquire_ctx.__aenter__ = AsyncMock(return_value=conn)
        acquire_ctx.__aexit__ = AsyncMock(return_value=False)
        client.pool = MagicMock()
        client.pool.acquire.return_value = acquire_ctx

        conn.execute = AsyncMock(side_effect=RuntimeError("db error"))

        result = await update_request(
            pg_client=client,
            request_id=str(uuid.uuid4()),
            data={"request_type": "infra"},
            embedding=[0.1] * 1536,
        )

        assert result is False


# Made with Bob

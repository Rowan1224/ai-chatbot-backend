"""
Integration tests for FastAPI HTTP endpoints.

Uses httpx.AsyncClient with ASGITransport to run the real FastAPI app
in-process.  LLM and embeddings are mocked; PostgreSQL is real.

The lifespan (startup/shutdown) is bypassed by injecting pre-built
app.state directly, so no real Redis is needed for most tests.
"""

import math
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langgraph.checkpoint.memory import InMemorySaver

from src.core.schema import DataField, ExtractedRequest
from src.core.workflow import (
    ChatResponse as WorkflowChatResponse,
)
from src.core.workflow import (
    ConversationWorkflow,
    DuplicateJudgement,
)

pytestmark = pytest.mark.integration

VALID_API_KEY = "test-api-key"
HEADERS = {"X-API-Key": VALID_API_KEY}

_DIM = 1536
_SCALE = 1.0 / math.sqrt(_DIM)
EMBED = [_SCALE] * _DIM


# ---------------------------------------------------------------------------
# Fixture — real app with injected state
# ---------------------------------------------------------------------------


def _build_workflow(pg_client) -> ConversationWorkflow:
    """Create a ConversationWorkflow with mocked LLM/embeddings."""
    mock_llm = MagicMock()
    mock_embed = AsyncMock()
    mock_embed.aembed_query = AsyncMock(return_value=EMBED)

    with (
        patch("src.core.workflow.get_llm", return_value=mock_llm),
        patch("src.core.workflow.get_embedding_model", return_value=mock_embed),
    ):
        wf = ConversationWorkflow(pg_client=pg_client)

    wf.llm_chat = MagicMock()
    wf.llm_extract = MagicMock()
    wf.llm_duplicate_judge = AsyncMock()
    wf.llm_duplicate_decision = AsyncMock()
    wf.embedding_model = mock_embed
    return wf


@pytest.fixture
def app_state(pg_client):
    """
    Returns the FastAPI app with pg_client and a compiled workflow
    injected into app.state.  Bypasses lifespan so no real Redis is
    needed.
    """
    from src.api.main import app

    wf = _build_workflow(pg_client)
    wf.llm_chat.ainvoke = AsyncMock(
        return_value=WorkflowChatResponse(response="How can I help?", is_ready=False)
    )
    wf.llm_duplicate_judge.ainvoke = AsyncMock(
        return_value=DuplicateJudgement(is_duplicate=False, reasoning="No match")
    )

    compiled = wf.compile(InMemorySaver())

    app.state.pg_client = pg_client
    app.state.redis_client = AsyncMock()
    app.state.conversation_app = compiled

    return app, wf


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:

    @pytest.mark.asyncio
    async def test_health_returns_200(self, app_state):
        app, _ = app_state
        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = False
            mock_settings.api_key = VALID_API_KEY
            mock_settings.log_level = "INFO"

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_shows_connected_when_pg_alive(self, app_state):
        """When PostgreSQL is reachable, mongodb field must be 'connected'."""
        app, _ = app_state
        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = False
            mock_settings.api_key = VALID_API_KEY
            mock_settings.log_level = "INFO"

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

        body = resp.json()
        assert body["postgresql"] == "connected"

    @pytest.mark.asyncio
    async def test_health_redis_disabled_in_dev_mode(self, app_state):
        """When use_redis_checkpointer=False, redis field must say 'disabled'."""
        app, _ = app_state
        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = False
            mock_settings.api_key = VALID_API_KEY
            mock_settings.log_level = "INFO"

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/health")

        body = resp.json()
        assert "disabled" in body["redis"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_422(self, app_state):
        app, _ = app_state
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/chat",
                json={"session_id": "s1", "message": "hi"},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_401(self, app_state):
        app, _ = app_state
        with patch("src.api.main.settings") as mock_settings:
            mock_settings.api_key = VALID_API_KEY

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/chat",
                    json={"session_id": "s1", "message": "hi"},
                    headers={"X-API-Key": "wrong-key"},
                )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_api_key_is_accepted(self, app_state):
        app, _ = app_state
        with patch("src.api.main.settings") as mock_settings:
            mock_settings.api_key = VALID_API_KEY

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/chat",
                    json={"session_id": "s1", "message": "hi"},
                    headers=HEADERS,
                )
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------


class TestChatEndpoint:

    @pytest.mark.asyncio
    async def test_chat_returns_200_and_valid_shape(self, app_state):
        """POST /chat must return 200 with the expected response fields."""
        app, wf = app_state
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(
                response="What environment do you need?", is_ready=False
            )
        )

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.api_key = VALID_API_KEY

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/chat",
                    json={"session_id": "chat-s1", "message": "I need infra"},
                    headers=HEADERS,
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "chat-s1"
        assert "response" in body
        assert "is_ready" in body
        assert "is_complete" in body

    @pytest.mark.asyncio
    async def test_chat_saves_row_when_workflow_completes(
        self, pg_client, app_state, clean_db
    ):
        """
        When the workflow runs to completion (is_complete=True),
        exactly one row must be saved in PostgreSQL.
        """
        app, wf = app_state
        wf.llm_chat.ainvoke = AsyncMock(
            return_value=WorkflowChatResponse(
                response="Got everything.", is_ready=True
            )
        )
        wf.llm_extract.ainvoke = AsyncMock(
            return_value=ExtractedRequest(
                request_type="infrastructure-provisioning",
                name="Jane",
                employee_id="EMP001",
                additional_data=[
                    DataField(key="target_environment", value="production"),
                    DataField(
                        key="business_justification", value="Need servers"
                    ),
                ],
            )
        )
        wf.llm_duplicate_judge.ainvoke = AsyncMock(
            return_value=DuplicateJudgement(
                is_duplicate=False, reasoning="No match"
            )
        )

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.api_key = VALID_API_KEY

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/chat",
                    json={
                        "session_id": "chat-save",
                        "message": "Need infrastructure",
                    },
                    headers=HEADERS,
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["is_complete"] is True

        async with pg_client.pool.acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM requests")
        assert count == 1

    @pytest.mark.asyncio
    async def test_chat_invalid_payload_returns_422(self, app_state):
        """Missing required fields must return 422."""
        app, _ = app_state
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/chat",
                json={"message": "no session id"},
                headers=HEADERS,
            )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session endpoint
# ---------------------------------------------------------------------------


class TestSessionEndpoint:

    @pytest.mark.asyncio
    async def test_get_session_returns_state_after_chat(self, app_state):
        """
        After a chat turn, GET /session/{id} must return the session state
        with at least the messages key populated.
        """
        app, wf = app_state
        wf.llm_chat.invoke = MagicMock(
            return_value=WorkflowChatResponse(
                response="I can help with that.", is_ready=False
            )
        )

        tid = f"sess-{uuid.uuid4().hex[:8]}"

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.api_key = VALID_API_KEY

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Send a chat message first
                await client.post(
                    "/chat",
                    json={"session_id": tid, "message": "Hello"},
                    headers=HEADERS,
                )

                # Then retrieve the session state
                resp = await client.get(
                    f"/session/{tid}",
                    headers=HEADERS,
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == tid
        assert "state" in body


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:

    @pytest.mark.asyncio
    async def test_root_returns_api_info(self, app_state):
        app, _ = app_state
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/")

        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "AI Chatbot Backend"
        assert "docs" in body


# Made with Bob

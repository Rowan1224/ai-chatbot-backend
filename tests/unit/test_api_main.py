"""Unit tests for FastAPI endpoints (src/api/main.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# App fixture — patches lifespan so no real DB/Redis is needed
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_mocked_state():
    """
    Import the FastAPI app and inject mocked state so no real
    infrastructure (PostgreSQL / Redis) is required.
    """
    from src.api.main import app

    # Build a fake conversation app
    fake_convo_app = AsyncMock()
    fake_convo_app.ainvoke = AsyncMock()
    fake_convo_app.aget_state = AsyncMock()

    # Inject directly into app.state (bypasses lifespan)
    app.state.conversation_app = fake_convo_app
    app.state.pg_client = AsyncMock()
    app.state.redis_client = AsyncMock()

    return app, fake_convo_app


@pytest.fixture
def client(app_with_mocked_state):
    app, _ = app_with_mocked_state
    return TestClient(app, raise_server_exceptions=False), app_with_mocked_state[1]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

VALID_HEADERS = {"X-API-Key": "test-api-key"}


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        tc, _ = client
        resp = tc.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "AI Chatbot Backend"

    def test_root_contains_docs_link(self, client):
        tc, _ = client
        resp = tc.get("/")
        assert "docs" in resp.json()


# ---------------------------------------------------------------------------
# API key authentication
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def test_missing_api_key_returns_422(self, client):
        """No X-API-Key header → 422 Unprocessable Entity."""
        tc, _ = client
        resp = tc.post("/chat", json={"session_id": "s1", "message": "hi"})
        assert resp.status_code == 422

    def test_wrong_api_key_returns_401(self, client):
        tc, _ = client
        resp = tc.post(
            "/chat",
            json={"session_id": "s1", "message": "hi"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_valid_api_key_does_not_return_401(self, client, app_with_mocked_state):
        tc, fake_app = client
        from langchain_core.messages import AIMessage

        fake_app.ainvoke = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Hello!")],
                "is_ready": False,
                "is_complete": False,
                "collected_data": None,
                "duplicate_warning": None,
            }
        )
        resp = tc.post(
            "/chat",
            json={"session_id": "s1", "message": "hi"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# /chat endpoint
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_chat_returns_200_with_valid_payload(self, client):
        tc, fake_app = client
        from langchain_core.messages import AIMessage

        fake_app.ainvoke = AsyncMock(
            return_value={
                "messages": [AIMessage(content="How can I help?")],
                "is_ready": False,
                "is_complete": False,
                "collected_data": None,
                "duplicate_warning": None,
            }
        )
        resp = tc.post(
            "/chat",
            json={"session_id": "sess-abc", "message": "Hello"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-abc"
        assert body["response"] == "How can I help?"
        assert body["is_ready"] is False
        assert body["is_complete"] is False

    def test_chat_propagates_is_complete_true(self, client):
        tc, fake_app = client
        from langchain_core.messages import AIMessage

        fake_app.ainvoke = AsyncMock(
            return_value={
                "messages": [AIMessage(content="Done!")],
                "is_ready": True,
                "is_complete": True,
                "collected_data": {"request_type": "infra"},
                "duplicate_warning": None,
            }
        )
        resp = tc.post(
            "/chat",
            json={"session_id": "s2", "message": "submit"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["is_complete"] is True
        assert body["collected_data"] == {"request_type": "infra"}

    def test_chat_returns_500_on_workflow_exception(self, client):
        tc, fake_app = client
        fake_app.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))

        resp = tc.post(
            "/chat",
            json={"session_id": "err-sess", "message": "oops"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 500

    def test_chat_empty_messages_falls_back_to_no_response(self, client):
        tc, fake_app = client
        fake_app.ainvoke = AsyncMock(
            return_value={
                "messages": [],
                "is_ready": False,
                "is_complete": False,
            }
        )
        resp = tc.post(
            "/chat",
            json={"session_id": "s3", "message": "hi"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 200
        assert resp.json()["response"] == "No response generated"

    def test_chat_missing_session_id_returns_422(self, client):
        tc, _ = client
        resp = tc.post(
            "/chat",
            json={"message": "hello"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 422

    def test_chat_missing_message_returns_422(self, client):
        tc, _ = client
        resp = tc.post(
            "/chat",
            json={"session_id": "s1"},
            headers=VALID_HEADERS,
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /session/{session_id} endpoint
# ---------------------------------------------------------------------------


class TestSessionEndpoint:
    def test_get_session_returns_200(self, client):
        tc, fake_app = client
        mock_state = MagicMock()
        mock_state.values = {"messages": [], "is_ready": False}
        fake_app.aget_state = AsyncMock(return_value=mock_state)

        resp = tc.get("/session/my-session", headers=VALID_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "my-session"
        assert "state" in body

    def test_get_session_returns_500_on_exception(self, client):
        tc, fake_app = client
        fake_app.aget_state = AsyncMock(side_effect=RuntimeError("fail"))

        resp = tc.get("/session/bad-session", headers=VALID_HEADERS)
        assert resp.status_code == 500

    def test_get_session_requires_api_key(self, client):
        tc, _ = client
        resp = tc.get("/session/my-session")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /health endpoint (no auth required)
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        tc, _ = client
        app_obj = tc.app  # the FastAPI app

        # Mock pg_client.ping
        app_obj.state.pg_client.ping = AsyncMock(return_value=True)

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = False
            mock_settings.log_level = "INFO"
            mock_settings.api_key = "test-api-key"

            resp = tc.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "mongodb" in body
        assert "redis" in body

    def test_health_shows_disconnected_on_pg_failure(self, client):
        tc, _ = client
        app_obj = tc.app

        app_obj.state.pg_client.ping = AsyncMock(
            side_effect=RuntimeError("no connection")
        )

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = False
            mock_settings.log_level = "INFO"
            mock_settings.api_key = "test-api-key"

            resp = tc.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["mongodb"] == "disconnected"

    def test_health_shows_redis_connected_when_checkpointer_enabled(self, client):
        tc, _ = client
        app_obj = tc.app

        app_obj.state.pg_client.ping = AsyncMock(return_value=True)

        mock_redis_client_instance = AsyncMock()
        mock_redis_client_instance.ping = AsyncMock(return_value=True)

        app_obj.state.redis_client.get_client = AsyncMock(
            return_value=mock_redis_client_instance
        )

        with patch("src.api.main.settings") as mock_settings:
            mock_settings.use_redis_checkpointer = True
            mock_settings.redis_url = "redis://localhost:6379"
            mock_settings.log_level = "INFO"
            mock_settings.api_key = "test-api-key"

            resp = tc.get("/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["redis"] == "connected"


# Made with Bob

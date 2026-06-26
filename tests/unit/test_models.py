"""Unit tests for API models (src/api/models.py)."""

import pytest
from pydantic import ValidationError

from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionResponse,
)


class TestChatRequest:
    """Tests for ChatRequest model."""

    def test_valid_chat_request(self):
        req = ChatRequest(session_id="sess-1", message="Hello")
        assert req.session_id == "sess-1"
        assert req.message == "Hello"

    def test_missing_session_id_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello")  # type: ignore[call-arg]

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="sess-1")  # type: ignore[call-arg]

    def test_empty_session_id_raises(self):
        """session_id min_length=1 — empty string must be rejected."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="", message="hello")

    def test_invalid_session_id_chars_raises(self):
        """session_id rejects path-traversal characters."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="../admin", message="hello")

    def test_valid_session_id_formats(self):
        """session_id accepts UUIDs, prefixed IDs, and simple slugs."""
        for sid in ["e2e-abc123", "sess_001", "a1B2c3"]:
            req = ChatRequest(session_id=sid, message="hi")
            assert req.session_id == sid


class TestChatResponseModel:
    """Tests for the API ChatResponse model (src/api/models.py)."""

    def test_required_fields_only(self):
        resp = ChatResponse(session_id="s1", response="Hi")
        assert resp.session_id == "s1"
        assert resp.response == "Hi"
        assert resp.is_ready is False
        assert resp.is_complete is False
        assert resp.collected_data is None
        assert resp.duplicate_warning is None

    def test_all_fields_populated(self):
        resp = ChatResponse(
            session_id="s2",
            response="Done",
            is_ready=True,
            is_complete=True,
            collected_data={"request_type": "infra"},
            duplicate_warning=[{"id": "abc", "similarity_score": 0.9}],
        )
        assert resp.is_ready is True
        assert resp.is_complete is True
        assert resp.collected_data == {"request_type": "infra"}
        assert len(resp.duplicate_warning) == 1

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            ChatResponse(session_id="s3")  # type: ignore[call-arg]

    def test_defaults_for_optional_fields(self):
        resp = ChatResponse(session_id="s4", response="test")
        assert resp.collected_data is None
        assert resp.duplicate_warning is None


class TestSessionResponse:
    """Tests for SessionResponse model."""

    def test_valid_session_response(self):
        resp = SessionResponse(session_id="s1", state={"messages": [], "is_ready": False})
        assert resp.session_id == "s1"
        assert resp.state == {"messages": [], "is_ready": False}

    def test_empty_state_dict(self):
        resp = SessionResponse(session_id="s1", state={})
        assert resp.state == {}

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            SessionResponse(session_id="s1")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            SessionResponse(state={})  # type: ignore[call-arg]


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_healthy_response(self):
        resp = HealthResponse(
            status="healthy", postgresql="connected", redis="connected"
        )
        assert resp.status == "healthy"
        assert resp.postgresql == "connected"
        assert resp.redis == "connected"

    def test_unhealthy_response(self):
        resp = HealthResponse(
            status="unhealthy",
            postgresql="disconnected",
            redis="disabled (dev mode)",
        )
        assert resp.status == "unhealthy"

    def test_missing_fields_raises(self):
        with pytest.raises(ValidationError):
            HealthResponse(  # type: ignore[call-arg]
                status="healthy", postgresql="connected"
            )

    def test_serialisation_round_trip(self):
        """Model can be serialised to dict and reconstructed."""
        original = HealthResponse(
            status="healthy", postgresql="connected", redis="connected"
        )
        data = original.model_dump()
        reconstructed = HealthResponse(**data)
        assert reconstructed == original


# Made with Bob

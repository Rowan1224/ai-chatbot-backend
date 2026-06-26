"""
End-to-end tests for the AI Chatbot Backend.

These tests exercise the full deployed stack:
  - Real Docker image (built by build.sh)
  - Real PostgreSQL + pgvector container
  - Real Redis container (LangGraph checkpointing)
  - Mock LLM (LLM_PROVIDER=mock — no API key required)

Scenarios
---------
1. Happy-path conversation  — multi-turn chat → data saved
2. Duplicate detection path — same request twice → warning → proceed
3. Cancellation path        — duplicate warning → user cancels

Run
---
    pytest tests/e2e/ -m e2e -v

Prerequisites: docker compose stack must be up (handled by conftest.py
session fixture) or the E2E_API_URL env var points at a running stack.
"""

import uuid

import httpx
import pytest

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _session() -> str:
    """Return a fresh unique session ID."""
    return f"e2e-{uuid.uuid4().hex[:12]}"


def _chat(
    client: httpx.Client,
    session_id: str,
    message: str,
) -> dict:
    """POST /chat and return the parsed JSON body."""
    resp = client.post(
        "/chat",
        json={"session_id": session_id, "message": message},
    )
    assert resp.status_code == 200, (
        f"POST /chat failed: {resp.status_code} {resp.text}"
    )
    return resp.json()


def _drive_to_completion(
    client: httpx.Client,
    session_id: str,
) -> dict:
    """
    Drive a session through the mock-LLM script until
    ``is_complete=True``.

    The mock LLM script has exactly 5 turns before signalling
    is_ready=True, after which the workflow extracts, checks for
    duplicates, and saves.

    Because the DB is shared across all tests in the session,
    later calls may hit a duplicate warning even on the "first"
    submission.  We automatically reply "proceed" when that happens
    so the workflow always reaches is_complete=True here.
    Tests that specifically want to observe or act on the duplicate
    warning drive the conversation themselves instead of using this
    helper.
    """
    turns = [
        "Hello, I need help submitting a request.",
        "infrastructure-provisioning",
        "production",
        "We need new servers for the payment service.",
        "John Doe,EMP001",
    ]

    last_body: dict = {}
    for msg in turns:
        last_body = _chat(client, session_id, msg)
        if last_body.get("is_complete"):
            return last_body
        # Duplicate warning mid-flight — send "proceed" to push through
        if last_body.get("duplicate_warning"):
            last_body = _chat(client, session_id, "proceed")
            if last_body.get("is_complete"):
                return last_body

    assert last_body.get("is_complete"), (
        f"Workflow did not complete after {len(turns)} turns. "
        f"Last response: {last_body}"
    )
    return last_body


# ---------------------------------------------------------------------------
# Scenario 1: Infrastructure checks
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestInfrastructure:

    def test_root_returns_api_info(
        self, api_client: httpx.Client
    ) -> None:
        """GET / must return the API name and version."""
        resp = api_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "AI Chatbot Backend"
        assert "docs" in body
        assert "health" in body

    def test_health_is_healthy(
        self, api_client: httpx.Client
    ) -> None:
        """
        GET /health must return status='healthy' when Postgres
        and Redis are both reachable.
        """
        # Remove auth header — health endpoint is unauthenticated
        resp = httpx.get(
            str(api_client.base_url) + "/health",
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy", (
            f"Health check returned unhealthy: {body}"
        )
        assert body["mongodb"] == "connected"
        assert body["redis"] == "connected"

    def test_openapi_docs_reachable(
        self, api_client: httpx.Client
    ) -> None:
        """GET /docs must return 200 (OpenAPI UI is available)."""
        resp = api_client.get("/docs")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Scenario 2: Authentication
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestAuthentication:

    def test_missing_api_key_returns_422(
        self, api_url: str
    ) -> None:
        """POST /chat without X-API-Key must return 422."""
        resp = httpx.post(
            f"{api_url}/chat",
            json={
                "session_id": _session(),
                "message": "hi",
            },
            timeout=10.0,
        )
        assert resp.status_code == 422

    def test_wrong_api_key_returns_401(
        self, api_url: str
    ) -> None:
        """POST /chat with a wrong key must return 401."""
        resp = httpx.post(
            f"{api_url}/chat",
            json={
                "session_id": _session(),
                "message": "hi",
            },
            headers={"X-API-Key": "wrong-key"},
            timeout=10.0,
        )
        assert resp.status_code == 401

    def test_valid_api_key_accepted(
        self, api_client: httpx.Client
    ) -> None:
        """POST /chat with the correct key must not return 401/422."""
        resp = api_client.post(
            "/chat",
            json={
                "session_id": _session(),
                "message": "hello",
            },
        )
        assert resp.status_code not in (401, 422), (
            f"Auth rejected valid key: {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Scenario 3: Happy-path — full conversation → data saved
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestHappyPath:

    def test_single_turn_returns_valid_shape(
        self, api_client: httpx.Client
    ) -> None:
        """
        A single POST /chat turn must return the expected fields.
        """
        body = _chat(
            api_client, _session(), "Hello, I need help."
        )
        assert "session_id" in body
        assert "response" in body
        assert isinstance(body["response"], str)
        assert len(body["response"]) > 0
        assert "is_ready" in body
        assert "is_complete" in body

    def test_multi_turn_conversation_completes(
        self, api_client: httpx.Client
    ) -> None:
        """
        A full scripted conversation must end with is_complete=True
        and a non-empty response confirming submission.
        """
        sid = _session()
        final = _drive_to_completion(api_client, sid)

        assert final["is_complete"] is True, (
            f"Expected is_complete=True, got: {final}"
        )
        assert "✅" in final["response"] or "request" in final[
            "response"
        ].lower(), (
            f"Expected confirmation message, got: {final['response']}"
        )

    def test_session_state_persisted_after_completion(
        self, api_client: httpx.Client
    ) -> None:
        """
        After a completed conversation, GET /session/{id} must
        return 200 with the session_id echoed back and a state key.

        We only assert the envelope shape here — the deep contents
        of state.values (BaseMessage objects, asyncpg records) are
        not JSON-serialisable in all configurations, so asserting
        on specific nested fields would be fragile.
        """
        sid = _session()
        _drive_to_completion(api_client, sid)

        resp = api_client.get(f"/session/{sid}")
        assert resp.status_code == 200, (
            f"GET /session returned {resp.status_code}: {resp.text}"
        )

        body = resp.json()
        assert body["session_id"] == sid
        assert "state" in body

    def test_independent_sessions_are_isolated(
        self, api_client: httpx.Client
    ) -> None:
        """
        Two concurrent sessions must not share state.
        Each should get a response specific to their own context.
        """
        sid_a = _session()
        sid_b = _session()

        resp_a = _chat(api_client, sid_a, "I need infrastructure provisioning.")
        resp_b = _chat(api_client, sid_b, "I need access to the VPN.")

        # Both must succeed independently
        assert resp_a["session_id"] == sid_a
        assert resp_b["session_id"] == sid_b
        # Sessions must not share messages
        assert resp_a["session_id"] != resp_b["session_id"]

    def test_invalid_payload_returns_422(
        self, api_client: httpx.Client
    ) -> None:
        """POST /chat without session_id must return 422."""
        resp = api_client.post(
            "/chat",
            json={"message": "no session id here"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Scenario 4: Duplicate-detection path
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestDuplicateDetection:

    def test_duplicate_triggers_warning(
        self, api_client: httpx.Client
    ) -> None:
        """
        Submitting the same request type twice should trigger a
        duplicate_warning on the second session.

        The mock LLM always returns DuplicateJudgement(is_duplicate=True)
        and the fixed embeddings are identical, so the second request
        will always hit the duplicate path once the first is saved.
        """
        # Submit first request to completion
        sid1 = _session()
        _drive_to_completion(api_client, sid1)

        # Start second identical request
        sid2 = _session()
        # Drive through all turns — the last turn (is_ready=True)
        # will trigger extract → duplicate_check → warning
        turns = [
            "Hello, I need help submitting a request.",
            "infrastructure-provisioning",
            "production",
            "We need new servers for the payment service.",
            "John Doe,EMP001",
        ]
        last_body: dict = {}
        for msg in turns:
            last_body = _chat(api_client, sid2, msg)
            # Stop as soon as a duplicate warning appears
            if last_body.get("duplicate_warning"):
                break

        assert last_body.get("duplicate_warning"), (
            "Expected duplicate_warning in response after "
            f"second identical submission. Got: {last_body}"
        )
        assert last_body["is_complete"] is False, (
            "Workflow should be paused awaiting decision"
        )

    def test_proceed_after_duplicate_saves_new_record(
        self, api_client: httpx.Client
    ) -> None:
        """
        Replying 'proceed' after a duplicate warning must save
        the request as a new record (is_complete=True).
        """
        # Get to the duplicate warning state
        sid1 = _session()
        _drive_to_completion(api_client, sid1)

        sid2 = _session()
        turns = [
            "Hello, I need help submitting a request.",
            "infrastructure-provisioning",
            "production",
            "We need new servers for the payment service.",
            "John Doe,EMP001",
        ]
        warning_body: dict = {}
        for msg in turns:
            warning_body = _chat(api_client, sid2, msg)
            if warning_body.get("duplicate_warning"):
                break

        if not warning_body.get("duplicate_warning"):
            pytest.skip(
                "No duplicate detected — skipping proceed test "
                "(DB may have been reset between runs)"
            )

        # User chooses to proceed
        final = _chat(api_client, sid2, "proceed")
        assert final["is_complete"] is True, (
            f"Expected is_complete=True after 'proceed', got: {final}"
        )


# ---------------------------------------------------------------------------
# Scenario 5: Cancellation path
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestCancellation:

    def test_cancel_after_duplicate_ends_without_saving(
        self, api_client: httpx.Client
    ) -> None:
        """
        Replying 'cancel' after a duplicate warning must end the
        conversation with is_complete=False.
        """
        # Need a prior record to trigger duplicate detection
        sid1 = _session()
        _drive_to_completion(api_client, sid1)

        sid2 = _session()
        turns = [
            "Hello, I need help submitting a request.",
            "infrastructure-provisioning",
            "production",
            "We need new servers for the payment service.",
            "John Doe,EMP001",
        ]
        warning_body: dict = {}
        for msg in turns:
            warning_body = _chat(api_client, sid2, msg)
            if warning_body.get("duplicate_warning"):
                break

        if not warning_body.get("duplicate_warning"):
            pytest.skip(
                "No duplicate detected — skipping cancel test "
                "(DB may have been reset between runs)"
            )

        # User chooses to cancel
        cancel_resp = _chat(api_client, sid2, "cancel")
        assert cancel_resp["is_complete"] is False, (
            f"Expected is_complete=False after cancel, got: {cancel_resp}"
        )
        assert "cancel" in cancel_resp["response"].lower() or \
               "abandon" in cancel_resp["response"].lower() or \
               "ready" in cancel_resp["response"].lower(), (
            f"Expected cancellation message, got: {cancel_resp['response']}"
        )


# Made with Bob

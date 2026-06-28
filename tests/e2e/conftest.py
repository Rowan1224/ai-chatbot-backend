"""
Shared fixtures for E2E tests.

These tests boot the real Docker image produced by the build and
send HTTP requests over a real TCP socket — exactly as a client
(e.g. Postman) would.

Prerequisites
-------------
- Docker or Podman must be running.
- The image ai-chatbot-api:latest must have been built by build.sh.
- Run from the ai-chatbot-backend/ directory:

    pytest tests/e2e/ -m e2e -v

The fixtures bring the full stack up once per session (Compose),
wait for the API to become healthy, yield an httpx.Client, then
tear everything down.

Environment variables
---------------------
E2E_API_URL   Override the base URL (default: http://localhost:8001)
E2E_API_KEY   Override the API key  (default: e2e-test-key)
E2E_TIMEOUT   Health-poll timeout in seconds (default: 120)
"""

import os
import shlex
import subprocess
import time

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BASE_URL = os.environ.get("E2E_API_URL", "http://localhost:8001")
_API_KEY = os.environ.get("E2E_API_KEY", "e2e-test-key")
_TIMEOUT = int(os.environ.get("E2E_TIMEOUT", "120"))

HEADERS = {"X-API-Key": _API_KEY}

# Compose files relative to ai-chatbot-backend/
_COMPOSE_BASE = "docker-compose.yml"
_COMPOSE_E2E = "tests/e2e/docker-compose.e2e.yml"

# build.sh exports E2E_COMPOSE_CMD with the resolved runtime
# (e.g. "podman compose" or "docker compose" or "podman-compose").
# When running tests directly without build.sh, default to
# "docker compose" — override via the env var if needed.
_compose_cmd_str = os.environ.get(
    "E2E_COMPOSE_CMD", "docker compose"
)
_COMPOSE_CMD = shlex.split(_compose_cmd_str) + [
    "-f", _COMPOSE_BASE,
    "-f", _COMPOSE_E2E,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compose(*args: str) -> subprocess.CompletedProcess:
    """Run a docker compose subcommand, streaming output."""
    cmd = _COMPOSE_CMD + list(args)
    return subprocess.run(
        cmd,
        check=True,
        text=True,
    )


def _wait_healthy(
    base_url: str,
    timeout: int,
    interval: float = 3.0,
) -> None:
    """
    Poll GET /health until status=='healthy' or timeout expires.

    Raises RuntimeError if the service does not become healthy in
    time.
    """
    deadline = time.monotonic() + timeout
    last_err: Exception = RuntimeError("Never started")

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(
                f"{base_url}/health", timeout=5.0
            )
            if resp.status_code == 200:
                body = resp.json()
                if body.get("status") == "healthy":
                    return
                # Service is up but not yet healthy — keep polling
        except Exception as exc:
            last_err = exc

        time.sleep(interval)

    raise RuntimeError(
        f"API at {base_url} did not become healthy within "
        f"{timeout}s. Last error: {last_err}"
    )


# ---------------------------------------------------------------------------
# Session-scoped stack fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_stack():
    """
    Bring the full Docker Compose stack up once for the session.

    Yields the base URL once the API is healthy.
    Tears down (and removes volumes) after all E2E tests finish.
    """
    # Bring up in detached mode, building the api image if needed
    _compose("up", "--build", "-d", "--wait")

    try:
        _wait_healthy(_BASE_URL, _TIMEOUT)
        yield _BASE_URL
    finally:
        _compose("down", "--volumes", "--remove-orphans")


# ---------------------------------------------------------------------------
# Per-test HTTP client
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(e2e_stack):
    """
    httpx.Client pre-configured with the base URL and API key header.

    Using a synchronous client keeps tests straightforward; the API
    itself is async but that is transparent to the HTTP client.
    """
    with httpx.Client(
        base_url=e2e_stack,
        headers=HEADERS,
        timeout=30.0,
    ) as client:
        yield client


@pytest.fixture
def api_url(e2e_stack) -> str:
    """Return the base URL of the running E2E stack."""
    return e2e_stack




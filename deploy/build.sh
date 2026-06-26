#!/bin/bash
# Build script for AI Chatbot Backend
#
# Phases
# ------
# 1. Dependency installation
# 2. Unit tests         (fast, no infra)
# 3. Integration tests  (testcontainers — Postgres + Redis)
# 4. Build Docker image
# 5. E2E tests          (full stack via Docker Compose)
# 6. Tag image with version
#
# The script exits on the first failure so a broken step never
# produces a tagged image.

set -euo pipefail

# Always run from the project root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[build] $*"; }
log "Working directory: ${ROOT_DIR}"

# ---------------------------------------------------------------------------
# Select container runtime
# ---------------------------------------------------------------------------

# CONTAINER_RUNTIME can be set in the environment (e.g. by CI) to skip the
# interactive prompt.  Accepted values: "docker" | "podman" (default: "docker")
if [ -n "${CONTAINER_RUNTIME:-}" ]; then
    _choice="${CONTAINER_RUNTIME}"
    log "Container runtime from environment: ${_choice}"
elif [ -t 0 ]; then
    # stdin is a terminal — ask interactively
    echo ""
    echo "Select your container runtime:"
    echo "  1) docker        (Docker Desktop / Docker CLI)"
    echo "  2) podman        (Podman CLI, compose via 'podman compose')"
    echo ""
    read -rp "Enter choice [1-2] (default: 1): " _choice
    echo ""
else
    # Non-interactive and no env var — fall back to docker
    _choice="docker"
    log "Non-interactive mode: defaulting to docker"
fi

case "${_choice:-docker}" in
    1|docker)
        CONTAINER_CMD="docker"
        COMPOSE_CMD="docker compose"
        ;;
    2|podman)
        CONTAINER_CMD="podman"
        COMPOSE_CMD="podman compose"
        ;;
    *)
        echo "[build] ERROR: invalid CONTAINER_RUNTIME '${_choice}'. Use 'docker' or 'podman'." >&2
        exit 1
        ;;
esac

log "Container runtime: ${CONTAINER_CMD}"
log "Compose command:   ${COMPOSE_CMD}"

IMAGE_NAME="ai-chatbot-api"
VERSION=$(date +%Y%m%d-%H%M%S)

# ---------------------------------------------------------------------------
# Phase 1: Install dependencies
# ---------------------------------------------------------------------------

log "Phase 1 — Installing dependencies..."
uv sync

# ---------------------------------------------------------------------------
# Phase 2: Lint (ruff — fast, no infra required)
# ---------------------------------------------------------------------------

log "Phase 2 — Running linter (ruff)..."
uv run --with ruff ruff check src/ tests/
log "Linter passed"

# ---------------------------------------------------------------------------
# Phase 3: Unit tests (no external services required)
# ---------------------------------------------------------------------------

log "Phase 3 — Running unit tests..."
uv run pytest tests/unit/ -v -m "not integration and not e2e"

# ---------------------------------------------------------------------------
# Phase 4: Integration tests (testcontainers — spins up Postgres + Redis)
# ---------------------------------------------------------------------------

log "Phase 4 — Running integration tests..."
uv run pytest tests/integration/ -v -m integration

# ---------------------------------------------------------------------------
# Phase 5: Build Docker image
# ---------------------------------------------------------------------------

log "Phase 5 — Building Docker image: ${IMAGE_NAME}:latest ..."
# Podman defaults to OCI format which drops HEALTHCHECK; force Docker format.
# The --format flag is Podman-only — Docker always uses Docker format already.
if [ "${CONTAINER_CMD}" = "podman" ]; then
    ${CONTAINER_CMD} build --format docker -t "${IMAGE_NAME}:latest" .
else
    ${CONTAINER_CMD} build -t "${IMAGE_NAME}:latest" .
fi
log "Image built: ${IMAGE_NAME}:latest"

# ---------------------------------------------------------------------------
# Phase 6: End-to-end tests against the built image
#
# The compose override (tests/e2e/docker-compose.e2e.yml):
#   - Uses LLM_PROVIDER=mock (no API key required)
#   - Exposes the API on port 8001 to avoid dev stack conflicts
#   - Mounts a dedicated e2e_postgres_data volume
#
# The E2E test session fixture handles compose up / down, so we just
# run pytest normally.  The --no-header flag keeps CI output clean.
# ---------------------------------------------------------------------------

log "Phase 6 — Running E2E tests against built image..."

# Export so conftest.py and the compose override can read them
export E2E_API_URL="http://localhost:8001"
export E2E_API_KEY="e2e-test-key"
export E2E_TIMEOUT="120"
# Pass the resolved compose command so conftest.py can use it
export E2E_COMPOSE_CMD="${COMPOSE_CMD}"

uv run pytest tests/e2e/ -v -m e2e --no-header

# ---------------------------------------------------------------------------
# Phase 7: Tag image with build timestamp
# ---------------------------------------------------------------------------

log "Phase 7 — Tagging image: ${IMAGE_NAME}:${VERSION} ..."
${CONTAINER_CMD} tag "${IMAGE_NAME}:latest" "${IMAGE_NAME}:${VERSION}"
log "Tagged: ${IMAGE_NAME}:${VERSION}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log "Build completed successfully."
log "Images: ${IMAGE_NAME}:latest, ${IMAGE_NAME}:${VERSION}"

# Made with Bob

#!/bin/bash
# setup-local-env.sh
# ─────────────────────────────────────────────────────────────────────────────
# Interactive helper to generate a local .env file from .env.example.
# Run once when setting up the project locally.
#
# Usage:
#   chmod +x scripts/setup-local-env.sh
#   ./scripts/setup-local-env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
EXAMPLE_FILE="${ROOT_DIR}/.env.example"

echo ""
echo "AI Chatbot Backend — Local environment setup"
echo "─────────────────────────────────────────────"

# Guard: don't overwrite an existing .env without confirmation
if [[ -f "${ENV_FILE}" ]]; then
    echo ""
    echo "⚠  .env already exists."
    read -rp "   Overwrite it? [y/N]: " _overwrite
    if [[ "${_overwrite:-N}" != "y" && "${_overwrite:-N}" != "Y" ]]; then
        echo "Aborted. Existing .env was not changed."
        exit 0
    fi
fi

# Copy example as base
cp "${EXAMPLE_FILE}" "${ENV_FILE}"
echo ""
echo "Copied .env.example → .env"
echo ""

# ── LLM Provider ─────────────────────────────────────────────────────────────

echo "LLM Provider"
echo "  Options: openai, azure, anthropic"
read -rp "  Provider [openai]: " _provider
_provider="${_provider:-openai}"
sed -i.bak "s|^LLM_PROVIDER=.*|LLM_PROVIDER=${_provider}|" "${ENV_FILE}"

case "${_provider}" in
    openai)
        read -rp "  OPENAI_API_KEY: " _key
        sed -i.bak "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${_key}|" "${ENV_FILE}"
        ;;
    azure)
        read -rp "  AZURE_OPENAI_API_KEY: " _key
        read -rp "  AZURE_OPENAI_ENDPOINT: " _endpoint
        read -rp "  OPENAI_API_VERSION [2024-12-01-preview]: " _version
        _version="${_version:-2024-12-01-preview}"
        sed -i.bak "s|^# AZURE_OPENAI_API_KEY=.*|AZURE_OPENAI_API_KEY=${_key}|" "${ENV_FILE}"
        sed -i.bak "s|^# AZURE_OPENAI_ENDPOINT=.*|AZURE_OPENAI_ENDPOINT=${_endpoint}|" "${ENV_FILE}"
        sed -i.bak "s|^# OPENAI_API_VERSION=.*|OPENAI_API_VERSION=${_version}|" "${ENV_FILE}"
        ;;
    anthropic)
        read -rp "  ANTHROPIC_API_KEY: " _key
        sed -i.bak "s|^# ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${_key}|" "${ENV_FILE}"
        ;;
    *)
        echo "Unknown provider '${_provider}'. Edit .env manually."
        ;;
esac

# ── PostgreSQL ────────────────────────────────────────────────────────────────

echo ""
echo "PostgreSQL"
echo "  Default: postgresql://chatbot:chatbot_password@localhost:5432/chatbot"
read -rp "  POSTGRESQL_URL [press Enter for default]: " _pg_url
if [[ -n "${_pg_url}" ]]; then
    sed -i.bak "s|^POSTGRESQL_URL=.*|POSTGRESQL_URL=${_pg_url}|" "${ENV_FILE}"
fi

# ── API Key ───────────────────────────────────────────────────────────────────

echo ""
echo "API Key"
_generated=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))" 2>/dev/null || echo "")
if [[ -n "${_generated}" ]]; then
    echo "  Generated: ${_generated}"
    read -rp "  Use this key? [Y/n]: " _use_generated
    if [[ "${_use_generated:-Y}" == "Y" || "${_use_generated:-Y}" == "y" ]]; then
        _api_key="${_generated}"
    else
        read -rp "  API_KEY: " _api_key
    fi
else
    read -rp "  API_KEY: " _api_key
fi
sed -i.bak "s|^API_KEY=.*|API_KEY=${_api_key}|" "${ENV_FILE}"

# ── LangSmith (optional) ──────────────────────────────────────────────────────

echo ""
read -rp "Enable LangSmith tracing? [y/N]: " _ls
if [[ "${_ls:-N}" == "y" || "${_ls:-N}" == "Y" ]]; then
    read -rp "  LANGSMITH_API_KEY: " _ls_key
    read -rp "  LANGSMITH_PROJECT [default]: " _ls_project
    _ls_project="${_ls_project:-default}"
    sed -i.bak "s|^LANGSMITH_TRACING=.*|LANGSMITH_TRACING=true|" "${ENV_FILE}"
    sed -i.bak "s|^# LANGSMITH_API_KEY=.*|LANGSMITH_API_KEY=${_ls_key}|" "${ENV_FILE}"
    sed -i.bak "s|^# LANGSMITH_PROJECT=.*|LANGSMITH_PROJECT=${_ls_project}|" "${ENV_FILE}"
fi

# ── Cleanup backup files created by sed -i.bak ───────────────────────────────
rm -f "${ENV_FILE}.bak"

echo ""
echo "✅ .env created at ${ENV_FILE}"
echo ""
echo "Next steps:"
echo "  docker compose up -d        # start Postgres + API"
echo "  uv run uvicorn src.api.main:app --reload  # start API (hot reload)"
echo ""

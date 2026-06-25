#!/bin/bash
# Build script for AI Chatbot Backend
# Exits on first failure (tests, build, etc.)

set -e

echo "Starting build process..."

# Install dependencies
echo "Installing dependencies..."
uv sync

# Run tests (build fails if tests fail)
echo "Running tests..."
uv run pytest tests/unit/ -v

# Build podman image
echo "Building podman image..."
podman build -t ai-chatbot-api:latest .

# Tag with version
VERSION=$(date +%Y%m%d-%H%M%S)
podman tag ai-chatbot-api:latest ai-chatbot-api:$VERSION

echo "Build completed successfully"
echo "Images: ai-chatbot-api:latest, ai-chatbot-api:$VERSION"

# Made with Bob
